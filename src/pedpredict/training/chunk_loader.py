"""Crash-safe chunk prefetch loader — resolves B9.

Replaces the hand-rolled multiprocessing prefetch scattered through OLD ``train.py:367-498`` (the
``mp.Queue(maxsize=3)`` + ``processes``/``results`` bookkeeping, the preload window of 3, the
``queue.get(timeout=300)`` skip-on-timeout, ``wait_for_memory`` before each spawn, the per-epoch
``random.shuffle``, and the ``finally:`` terminate-all + drain) together with the warm worker
``scripts/train_utils.py:80-98`` (``mp_async_load``).

Design — two layers, all queue/process state hidden:

* :func:`warm_lmdb_chunk` — the module-level warm worker (EXACT port of ``mp_async_load``). It does **not**
  load data: it opens the LMDB read-only, reads one ``_meta`` key to warm the OS page cache, then returns
  the *path* for the main process to instantiate. Kept top-level so it is picklable under ``spawn``/Windows.
* :class:`ChunkLoaderIterator` — one warm-ahead traversal of a fixed chunk list. Owns the
  ``mp.Queue`` + the live warm processes + the N-ahead window, and exposes the
  ``start`` / ``__next__`` / ``close`` / ``__enter__`` / ``__exit__`` lifecycle. ``close()`` is idempotent
  and terminates+joins every live process and drains the queue, so no child leaks on early break,
  exhaustion, or an exception in the consumer.
* :class:`ChunkPrefetcher` — satisfies :class:`~pedpredict.training.trainer.ChunkProvider`. Builds the
  per-chunk :class:`~pedpredict.data.lmdb_dataset.LMDBChunkDataset` ``DataLoader`` (with the optional
  online :class:`~torch.utils.data.WeightedRandomSampler`, 1.6) and wraps each pass in a context-managed
  ``ChunkLoaderIterator`` so the Trainer just iterates ``epoch_loaders`` / ``val_loaders``.

Behavior preserved vs OLD (B9 is orchestration, not math — no tensor changes):

* preload depth, per-spawn RAM gating, per-epoch reshuffle, and the timeout/err **skip** semantics
  (a skipped chunk shrinks the warm window exactly as the legacy ``continue`` did — no replacement spawn).
  C1 hardening on top: every skip now emits a loud ``RuntimeWarning`` naming the chunk (the legacy path
  was fully silent), and **validation** uses ``skip_policy="raise"`` — a partial val set would make
  ``val_loss``/val metrics silently non-comparable across epochs, so it is a hard error instead.
* One intentional, behavior-neutral reorder: the next chunk's warm is spawned *before* the current
  loader is yielded (so warming overlaps the current chunk's training), where OLD spawned it after
  ``train_one_chunk``. Warming is an unobservable OS-cache side effect; only its timing moves.

Coupling to the Dataset (1.5): the dataset is built in the **main** process (as OLD did). The warm
processes here and the DataLoader worker processes never share an LMDB env; the dataset's own
pid-keyed env + picklable ``__getstate__`` (1.5) handle the worker boundary independently. Sampler
weights are built from the dataset's own ``seq_ids`` via the shared :class:`LabelScanCache`, so the
per-sample weight order matches the dataset's iteration order (1.6).
"""

from __future__ import annotations

import multiprocessing as mp
import random
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from queue import Empty

from torch.utils.data import DataLoader

from pedpredict.config.schema import RootCfg
from pedpredict.data.collate import build_collate
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.lmdb_warm import WarmResult, warm_lmdb_chunk
from pedpredict.data.sampler import LabelScanCache, build_weighted_sampler
from pedpredict.paths import resolve_paths
from pedpredict.utils.memory import wait_for_memory

# ``warm_lmdb_chunk`` lives in the torch-free ``data.lmdb_warm`` so ``spawn``ed warm children don't import
# torch; re-exported here as the package's prefetch API. ``torch.cuda`` is imported lazily where needed.
__all__ = ["warm_lmdb_chunk", "ChunkLoaderIterator", "ChunkPrefetcher", "gather_lmdb_chunks"]

#: ``path -> ready DataLoader`` (built in the MAIN process, exactly like OLD).
BuildLoader = Callable[[str], DataLoader]

#: A warm worker: ``(idx, path, queue) -> None``. Top-level (picklable under ``spawn``); a test seam.
WarmWorker = Callable[[int, str, "mp.Queue[WarmResult]"], None]


# --------------------------------------------------------------------------- warm-ahead iterator


class ChunkLoaderIterator:
    """One warm-ahead traversal of ``chunk_paths``; yields ready DataLoaders in list order.

    Encapsulates ALL the OLD queue/process bookkeeping. Lifecycle: :meth:`start` spawns the first
    ``preload_depth`` warmers; each :meth:`__next__` drains the queue to the next ready chunk, spawns the
    warmer ``preload_depth`` ahead, and returns ``build_loader(path)``; :meth:`close` (idempotent) tears
    every process down. Use as a context manager so teardown runs on early break / exception.
    """

    def __init__(
        self,
        chunk_paths: Sequence[str],
        build_loader: BuildLoader,
        *,
        preload_depth: int,
        ram_threshold: float,
        mem_interval: float,
        mem_timeout: float | None,
        queue_timeout: float,
        mp_context: mp.context.BaseContext | None = None,
        warm_fn: WarmWorker = warm_lmdb_chunk,
        skip_policy: str = "warn",
    ) -> None:
        if skip_policy not in ("warn", "raise"):
            raise ValueError(f"skip_policy must be 'warn' or 'raise'; got {skip_policy!r}")
        self._paths = list(chunk_paths)
        self._build_loader = build_loader
        self._warm_fn = warm_fn
        self._skip_policy = skip_policy        # C1: "warn" = legacy skip but LOUD; "raise" = hard error (val)
        self._preload = max(1, min(preload_depth, len(self._paths))) if self._paths else 0
        self._ram_threshold = ram_threshold
        self._mem_interval = mem_interval
        self._mem_timeout = mem_timeout
        self._queue_timeout = queue_timeout
        self._ctx = mp_context if mp_context is not None else mp.get_context("spawn")

        self._queue: mp.Queue[WarmResult] | None = None
        self._procs: dict[int, mp.process.BaseProcess] = {}
        self._results: dict[int, tuple[str, str]] = {}   # idx -> (status, payload)
        self._cursor = 0                                  # next chunk index to attempt
        self._started = False
        self._closed = False

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Spawn the first ``preload_depth`` warmers (OLD train.py:379-384). Idempotent."""
        if self._started or not self._paths:
            self._started = True
            return
        self._queue = self._ctx.Queue(maxsize=self._preload)
        for idx in range(self._preload):
            self._spawn(idx)
        self._started = True

    def __iter__(self) -> ChunkLoaderIterator:
        return self

    def __next__(self) -> DataLoader:
        if not self._started:
            self.start()
        while True:
            if self._cursor >= len(self._paths):
                self.close()
                raise StopIteration
            idx = self._cursor
            self._cursor += 1
            status, payload = self._await_chunk(idx)
            if status != "ok":           # timed out or warm error (OLD silently `continue`d — C1: loud)
                reason = (
                    f"warm worker did not report within queue_timeout={self._queue_timeout}s"
                    if status == "timeout"
                    else f"warm worker error: {payload}"
                )
                msg = f"[ChunkLoaderIterator] SKIPPING chunk {self._paths[idx]!r} — {reason}"
                if self._skip_policy == "raise":
                    self.close()
                    raise RuntimeError(msg + " (skip_policy='raise': a partial pass is not acceptable)")
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                continue
            self._spawn(idx + self._preload)   # warm the next chunk before training this one
            return self._build_loader(payload)

    def __enter__(self) -> ChunkLoaderIterator:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Terminate + join every live warmer and drain the queue (OLD :484-493). Idempotent."""
        if self._closed:
            return
        self._closed = True
        for proc in self._procs.values():
            if proc.is_alive():
                proc.terminate()
            proc.join()
        self._procs.clear()
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except (Empty, OSError, ValueError):
                    break
            close = getattr(self._queue, "close", None)   # mp.Queue has it; a plain stdlib queue does not
            if close is not None:
                close()
            self._queue = None

    # ----------------------------------------------------------------- internals

    def _spawn(self, idx: int) -> None:
        """RAM-gate then launch one warm process for chunk ``idx`` (OLD :380-384, :477-482)."""
        if idx >= len(self._paths) or idx in self._procs or self._queue is None:
            return
        wait_for_memory(self._ram_threshold, self._mem_interval, timeout=self._mem_timeout)
        proc = self._ctx.Process(
            target=self._warm_fn, args=(idx, self._paths[idx], self._queue), daemon=True
        )
        proc.start()
        self._procs[idx] = proc

    def _await_chunk(self, idx: int) -> tuple[str, str | None]:
        """Drain the queue until ``idx`` is ready; return ``(status, payload)``.

        ``("ok", path)`` when warmed; ``("timeout", None)`` when the warmer never reported within
        ``queue_timeout`` (the stuck process is terminated); ``("err", message)`` when the warm worker
        reported a failure. The caller (``__next__``) decides skip-vs-raise per ``skip_policy`` (C1).
        A skipped chunk does NOT trigger a replacement spawn — the warm window shrinks exactly as the
        legacy ``continue`` did.
        """
        assert self._queue is not None
        try:
            while idx not in self._results:
                got_idx, status, payload = self._queue.get(timeout=self._queue_timeout)
                self._results[got_idx] = (status, payload)
        except Empty:
            self._reap(idx, terminate=True)
            return "timeout", None
        status, payload = self._results.pop(idx)
        self._reap(idx, terminate=False)
        return ("ok", payload) if status == "ok" else ("err", payload)

    def _reap(self, idx: int, *, terminate: bool) -> None:
        """Join (optionally terminate) the warmer for ``idx`` and drop it from the live set."""
        proc = self._procs.pop(idx, None)
        if proc is None:
            return
        if terminate and proc.is_alive():
            proc.terminate()
        proc.join()


# --------------------------------------------------------------------------- chunk gathering


def gather_lmdb_chunks(dirs: Sequence[str | Path]) -> list[str]:
    """Collect ``*.lmdb`` chunks from one or more dirs — EXACT semantics of OLD ``gather_chunks``.

    Missing dirs are skipped (so the opt-in augmented dir is optional on a fresh machine); raises only
    when no dir yields a chunk. Within each dir, chunks are sorted for a stable val order.
    """
    all_files: list[str] = []
    missing: list[str] = []
    for folder in dirs:
        path = Path(folder)
        if not path.is_dir():
            missing.append(str(folder))
            continue
        all_files.extend(sorted(str(p) for p in path.iterdir() if p.name.endswith(".lmdb")))
    if missing:
        print(f"gather_lmdb_chunks: skipping missing folder(s): {missing}")
    if not all_files:
        raise FileNotFoundError(f"gather_lmdb_chunks: no .lmdb chunks found in any of {list(dirs)}")
    return all_files


# --------------------------------------------------------------------------- ChunkProvider impl


class ChunkPrefetcher:
    """Warm-ahead :class:`~pedpredict.training.trainer.ChunkProvider` over LMDB chunks (4.2, B9).

    Each :meth:`epoch_loaders` / :meth:`val_loaders` call wraps a fresh :class:`ChunkLoaderIterator` in a
    ``with`` block, so the warm processes for one pass are always torn down before the next — even if the
    Trainer breaks out early. The shared :class:`LabelScanCache` is the same instance the Trainer uses for
    the global class-weight scan (1.6), so each chunk is scanned at most once across both levers.
    """

    def __init__(
        self,
        cfg: RootCfg,
        train_lmdb_paths: Sequence[str],
        val_lmdb_paths: Sequence[str],
        *,
        scan_cache: LabelScanCache | None = None,
        pin_memory: bool | None = None,
        mp_context: mp.context.BaseContext | None = None,
        shuffle_rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.train_lmdb_paths: list[str] = list(train_lmdb_paths)
        self.val_lmdb_paths: list[str] = list(val_lmdb_paths)
        self.scan_cache = scan_cache if scan_cache is not None else LabelScanCache()
        if pin_memory is None:
            import torch  # local: keep the module import light for spawn children
            pin_memory = torch.cuda.is_available()
        self._pin = pin_memory
        self._ctx = mp_context if mp_context is not None else mp.get_context("spawn")
        # ``random`` (module) and ``random.Random`` both expose ``.shuffle``; default to global RNG (OLD).
        self._rng = shuffle_rng if shuffle_rng is not None else random
        self._collate = build_collate(cfg.data)
        self._closed = False

    @classmethod
    def from_config(cls, cfg: RootCfg, **kwargs: object) -> ChunkPrefetcher:
        """Gather train (base + opt-in aug) and val chunks from ``paths.yaml`` (replaces OLD gather_chunks)."""
        resolved = resolve_paths(cfg.paths)
        train = gather_lmdb_chunks(resolved.lmdb_train)
        val = gather_lmdb_chunks([resolved.lmdb_val])
        return cls(cfg, train, val, **kwargs)  # type: ignore[arg-type]

    # ----------------------------------------------------------------- ChunkProvider surface

    def epoch_loaders(self, epoch: int) -> Iterator[DataLoader]:
        """Yield reshuffled train-chunk loaders for one epoch (OLD :375 per-epoch ``random.shuffle``).

        Train skips stay skip-tolerant (a slow warm shouldn't kill a long run) but are loudly
        warned (C1) — watch for ``SKIPPING chunk`` warnings on slow disks.
        """
        paths = list(self.train_lmdb_paths)
        self._rng.shuffle(paths)
        with ChunkLoaderIterator(
            paths, self._build_train_loader, skip_policy="warn", **self._iter_kwargs
        ) as iterator:
            yield from iterator

    def val_loaders(self) -> Iterator[DataLoader]:
        """Yield validation-chunk loaders in stable (sorted) order; no sampler, no shuffle.

        C1: a skipped val chunk is a HARD ERROR — ``val_loss`` selects best.pth and stops training,
        so it must never be computed over a silently varying subset of validation.
        """
        with ChunkLoaderIterator(
            self.val_lmdb_paths, self._build_val_loader, skip_policy="raise", **self._iter_kwargs
        ) as iterator:
            yield from iterator

    def close(self) -> None:
        """No long-lived resources (each pass is context-managed); mark closed for symmetry."""
        self._closed = True

    # ----------------------------------------------------------------- internals

    @property
    def _iter_kwargs(self) -> dict[str, object]:
        t = self.cfg.train
        return {
            "preload_depth": t.chunk_preload_depth,
            "ram_threshold": t.chunk_warm_ram_threshold,
            "mem_interval": t.chunk_warm_mem_interval,
            "mem_timeout": t.chunk_warm_mem_timeout,
            "queue_timeout": t.chunk_queue_timeout,
            "mp_context": self._ctx,
        }

    def _loader_kwargs(self) -> dict[str, object]:
        """Common DataLoader kwargs (OLD train.py:414-423).

        Q2: ``persistent_workers`` is on (with workers) so a loader's workers survive iterator
        exhaustion. NOTE the structural limit: loaders are still built per chunk, so workers are
        respawned per chunk regardless — the real fix is the standard-sharding rework (backlog 7);
        this only stops same-loader respawns and costs nothing.
        """
        t = self.cfg.train
        kwargs: dict[str, object] = {
            "batch_size": t.batch_size,
            "num_workers": t.num_workers,
            "collate_fn": self._collate,
            "pin_memory": self._pin,
            "persistent_workers": t.num_workers > 0,
        }
        if t.num_workers > 0:
            kwargs["prefetch_factor"] = t.dataloader_prefetch_factor
        return kwargs

    def _build_train_loader(self, path: str) -> DataLoader:
        """Train loader with the optional online ``WeightedRandomSampler`` (OLD train.py:413-454)."""
        dataset = LMDBChunkDataset.from_config(path, self.cfg.data)
        kwargs = self._loader_kwargs()
        if self.cfg.train.use_weighted_sampler:
            scan = self.scan_cache.get(path, dataset.seq_ids)   # aligns weights to dataset order (1.6)
            kwargs["sampler"] = build_weighted_sampler(scan, self.cfg.train)
            kwargs["shuffle"] = False
        else:
            kwargs["shuffle"] = True
        return DataLoader(dataset, **kwargs)

    def _build_val_loader(self, path: str) -> DataLoader:
        """Validation loader: stable order, no sampler (OLD validate path)."""
        dataset = LMDBChunkDataset.from_config(path, self.cfg.data)
        return DataLoader(dataset, shuffle=False, **self._loader_kwargs())

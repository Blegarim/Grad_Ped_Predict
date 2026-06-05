"""Prompt 4.2 — chunk prefetch loader (B9): crash-safety + warm-ahead orchestration.

This layer adds NO math (it warms OS cache and builds DataLoaders), so parity here is structural /
behavioral rather than a tensor fixture (same class as the 0.3 infra rows):

  * WARM WORKER — ``warm_lmdb_chunk`` reproduces OLD ``mp_async_load`` (ok->path / err->message) and is
    picklable under ``spawn``.
  * TRAVERSAL — chunks are yielded in list order; per-epoch reshuffle is deterministic under a seed.
  * SKIP semantics — a queue timeout or a warm error skips that chunk (OLD ``continue``) without hanging.
  * NO LEAKED PROCESSES — a full pass, an early break, and an exception each return ``active_children``
    to baseline (the headline crash-safety guarantee), exercised with REAL spawned processes.
  * WIRING — ``ChunkPrefetcher`` satisfies ``ChunkProvider``, attaches the online ``WeightedRandomSampler``
    (1.6) via the shared ``LabelScanCache``, and ``gather_lmdb_chunks`` matches OLD ``gather_chunks``.

To keep the suite fast on Windows (where every ``spawn`` re-imports the parent), the order/skip/wiring
tests run the warm worker through an in-process ``_InlineCtx`` that exercises the exact same iterator
queue/skip/window logic synchronously; only the leak tests pay for real processes.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import queue as queue_mod
import random
import time
from pathlib import Path

import lmdb
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from pedpredict.config.schema import RootCfg
from pedpredict.data.sampler import LabelScanCache
from pedpredict.training import Trainer
from pedpredict.training.chunk_loader import (
    ChunkLoaderIterator,
    ChunkPrefetcher,
    gather_lmdb_chunks,
    warm_lmdb_chunk,
)
from pedpredict.training.trainer import ChunkProvider

_BASE_KW = {"preload_depth": 2, "ram_threshold": 100.0, "mem_interval": 0.01, "mem_timeout": 1.0}


# --------------------------------------------------------------------------- in-process context


class _InlineProcess:
    """A ``Process`` look-alike whose ``start()`` runs the target synchronously (no real spawn)."""

    def __init__(self, target, args) -> None:
        self._target, self._args = target, args

    def start(self) -> None:
        self._target(*self._args)

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
        pass

    def terminate(self) -> None:
        pass


class _InlineCtx:
    """Minimal ``mp`` context: a stdlib queue + synchronous processes. Drives the iterator without spawn."""

    def Queue(self, maxsize: int = 0) -> queue_mod.Queue:  # noqa: N802 (mp API name)
        return queue_mod.Queue(maxsize=maxsize)

    def Process(self, target, args=(), daemon=None) -> _InlineProcess:  # noqa: N802, ARG002
        return _InlineProcess(target, args)


def _silent_warm(idx: int, path: str, queue) -> None:  # noqa: ARG001
    """A warm worker that never reports — drives the queue-timeout skip path (top-level for spawn)."""
    time.sleep(30)


# --------------------------------------------------------------------------- helpers


def _write_label_lmdb(path: str, n: int = 3) -> None:
    """Tiny label-only LMDB: ``<i>_meta`` pickles with the three task labels (enough for warm + scan)."""
    env = lmdb.open(path, map_size=8 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                rec = {"actions": i % 2, "looks": i % 2, "crosses": int(i == 0)}
                txn.put(f"{i}_meta".encode(), pickle.dumps(rec))
    finally:
        env.close()


def _make_chunks(tmp_path: Path, count: int) -> list[str]:
    """Create ``count`` tiny label LMDBs under ``tmp_path``; return their paths in creation order."""
    paths = []
    for c in range(count):
        p = str(tmp_path / f"chunk{c}.lmdb")
        _write_label_lmdb(p)
        paths.append(p)
    return paths


def _inline_iter(paths, build_loader=lambda p: p, **over):
    """An iterator wired to the in-process context (fast, deterministic)."""
    kw = {**_BASE_KW, "queue_timeout": 1.0, "mp_context": _InlineCtx(), **over}
    return ChunkLoaderIterator(paths, build_loader, **kw)


def _settle(active_before: int, *, tries: int = 60) -> int:
    """Poll until child-process count settles (join should suffice; guards spawn-teardown races)."""
    for _ in range(tries):
        n = len(mp.active_children())
        if n <= active_before:
            return n
        time.sleep(0.05)
    return len(mp.active_children())


# --------------------------------------------------------------------------- warm worker


def test_warm_chunk_ok_returns_path(tmp_path) -> None:
    p = str(tmp_path / "ok.lmdb")
    _write_label_lmdb(p)
    q: queue_mod.Queue = queue_mod.Queue()
    warm_lmdb_chunk(7, p, q)
    assert q.get(timeout=5) == (7, "ok", p)


def test_warm_chunk_err_on_bad_path(tmp_path) -> None:
    q: queue_mod.Queue = queue_mod.Queue()
    warm_lmdb_chunk(3, str(tmp_path / "does_not_exist.lmdb"), q)
    idx, status, _payload = q.get(timeout=5)
    assert (idx, status) == (3, "err")


def test_warm_target_is_picklable() -> None:
    assert pickle.loads(pickle.dumps(warm_lmdb_chunk)) is warm_lmdb_chunk


# --------------------------------------------------------------------------- traversal order / skip


def test_chunk_order_preserved(tmp_path) -> None:
    paths = _make_chunks(tmp_path, 5)
    with _inline_iter(paths) as it:
        assert list(it) == paths            # build_loader is identity -> yielded == input order


def test_err_status_skips_chunk(tmp_path) -> None:
    good = _make_chunks(tmp_path, 2)
    bad = str(tmp_path / "missing.lmdb")    # never written -> warm reports 'err'
    paths = [good[0], bad, good[1]]
    with _inline_iter(paths, preload_depth=3) as it:
        assert list(it) == [good[0], good[1]]


def test_timeout_skips_chunk(tmp_path) -> None:
    """A warmer that never reports is skipped after ``queue_timeout`` — no hang, no yield."""
    paths = _make_chunks(tmp_path, 2)
    with _inline_iter(paths, warm_fn=_skip_warm, queue_timeout=0.3, preload_depth=2) as it:
        assert list(it) == []


def _skip_warm(idx: int, path: str, queue) -> None:  # noqa: ARG001
    """Inline warm worker that intentionally puts nothing -> the parent must time out and skip."""
    return


def test_close_is_idempotent(tmp_path) -> None:
    paths = _make_chunks(tmp_path, 3)
    it = _inline_iter(paths)
    it.start()
    next(it)
    it.close()
    it.close()                               # second close is a no-op, never raises


# --------------------------------------------------------------------------- crash-safety / leaks (REAL spawn)


def test_no_leaked_processes_full_pass(tmp_path) -> None:
    base = len(mp.active_children())
    paths = _make_chunks(tmp_path, 3)
    with ChunkLoaderIterator(paths, lambda p: p, **_BASE_KW, queue_timeout=30.0) as it:
        list(it)
    assert _settle(base) <= base


def test_no_leaked_processes_on_early_break(tmp_path) -> None:
    base = len(mp.active_children())
    paths = _make_chunks(tmp_path, 3)
    with ChunkLoaderIterator(paths, lambda p: p, **_BASE_KW, queue_timeout=30.0) as it:
        for _ in it:
            break                            # abandon mid-iteration
    assert _settle(base) <= base


def test_no_leaked_processes_on_exception(tmp_path) -> None:
    base = len(mp.active_children())
    paths = _make_chunks(tmp_path, 3)
    with pytest.raises(RuntimeError):
        with ChunkLoaderIterator(paths, lambda p: p, **_BASE_KW, queue_timeout=30.0) as it:
            for _ in it:
                raise RuntimeError("boom")   # __exit__ must still tear warmers down
    assert _settle(base) <= base


def test_no_leaked_processes_on_real_timeout(tmp_path) -> None:
    """A real spawned warmer that hangs is terminated on timeout — the OLD ``proc.terminate()`` path."""
    base = len(mp.active_children())
    paths = _make_chunks(tmp_path, 2)
    with ChunkLoaderIterator(
        paths, lambda p: p, **_BASE_KW, queue_timeout=0.5, warm_fn=_silent_warm
    ) as it:
        assert list(it) == []                # every warmer hangs -> all skipped
    assert _settle(base) <= base


# --------------------------------------------------------------------------- ChunkPrefetcher wiring


def test_prefetcher_satisfies_chunkprovider(tmp_path) -> None:
    paths = _make_chunks(tmp_path, 2)
    pf = ChunkPrefetcher(RootCfg(), paths, paths)
    assert isinstance(pf, ChunkProvider)
    assert pf.train_lmdb_paths == paths


def test_epoch_reshuffle_deterministic(tmp_path) -> None:
    paths = _make_chunks(tmp_path, 5)
    inline = _InlineCtx()
    pf_a = ChunkPrefetcher(RootCfg(), paths, paths, shuffle_rng=random.Random(123), mp_context=inline)
    pf_b = ChunkPrefetcher(RootCfg(), paths, paths, shuffle_rng=random.Random(123), mp_context=inline)
    pf_a._build_train_loader = lambda p: p   # bypass real dataset build; we only assert path order
    pf_b._build_train_loader = lambda p: p

    order_a0 = list(pf_a.epoch_loaders(0))
    order_b0 = list(pf_b.epoch_loaders(0))
    order_a1 = list(pf_a.epoch_loaders(1))

    assert order_a0 == order_b0               # same seed -> reproducible
    assert sorted(order_a0) == sorted(paths)  # a permutation of the chunk set
    assert order_a1 != order_a0               # successive epochs reshuffle


def test_weighted_sampler_attached_via_scan_cache(tmp_path) -> None:
    p = str(tmp_path / "c.lmdb")
    _write_label_lmdb(p, n=6)
    cache = LabelScanCache()
    pf = ChunkPrefetcher(RootCfg(), [p], [p], scan_cache=cache)

    loader = pf._build_train_loader(p)        # default use_weighted_sampler=True
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert p in cache._store                  # the per-chunk scan was cached (shared with the loss lever)


def test_prefetcher_shares_scan_cache_with_trainer(tmp_path) -> None:
    """The Trainer takes the provider's scan_cache so one scan per chunk serves both levers (1.6)."""
    paths = _make_chunks(tmp_path, 2)
    pf = ChunkPrefetcher(RootCfg(), paths, paths)
    model = torch.nn.Linear(1, 1)             # the sharing seam is independent of the real model
    trainer = Trainer(RootCfg(), model, torch.device("cpu"), pf, scan_cache=pf.scan_cache)
    assert trainer.scan_cache is pf.scan_cache


# --------------------------------------------------------------------------- gather_lmdb_chunks


def test_gather_sorted_and_skips_missing(tmp_path) -> None:
    d = tmp_path / "train"
    d.mkdir()
    for name in ("b.lmdb", "a.lmdb"):
        (d / name).mkdir()
    (d / "note.txt").write_text("x")
    chunks = gather_lmdb_chunks([d, tmp_path / "absent"])   # missing dir skipped, not fatal
    assert [Path(c).name for c in chunks] == ["a.lmdb", "b.lmdb"]


def test_gather_raises_when_no_chunks(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no .lmdb chunks"):
        gather_lmdb_chunks([tmp_path / "absent"])

"""LMDB page-cache warm worker — deliberately torch-free.

Isolated from ``training/chunk_loader.py`` for one reason: it is the target of a ``spawn``ed process.
Under ``spawn`` (the Windows/macOS default) the child re-imports the module that defines the target
callable; if that module pulled in ``torch`` (as ``chunk_loader`` does, transitively via ``DataLoader``),
every warm process would pay torch's multi-second import before it could read a single key. Keeping the
worker here — importing only ``lmdb`` + stdlib — lets the warm children start fast.

Q1 fix over the OLD ``mp_async_load`` port: the legacy worker read exactly ONE ``_meta`` key — the whole
spawn/queue/RAM-gate apparatus pre-loaded a few KB of a multi-GB chunk, so the prefetcher delivered
~none of its intended benefit on an HDD. The worker now walks the full cursor (keys AND values), which
faults every used page into the OS cache in B-tree order — for a chunk written in one append pass that
is a near-sequential read of the real payload (~2-3 GB), exactly what an HDD wants.
"""

from __future__ import annotations

import multiprocessing as mp

import lmdb

__all__ = ["WarmResult", "walk_chunk", "warm_lmdb_chunk"]

#: ``(idx, status, payload)`` put on the warm queue — ``status`` is ``"ok"`` (payload=path) or ``"err"``.
WarmResult = tuple[int, str, str]


def warm_lmdb_chunk(idx: int, path: str, queue: mp.Queue[WarmResult], meta_only: bool = False) -> None:
    """Warm one LMDB chunk's OS page cache for real (Q1), then report its path.

    Opens the chunk read-only and iterates the full cursor — py-lmdb materializes each value as
    ``bytes``, touching every used page — then puts ``(idx, 'ok', path)``; any failure puts
    ``(idx, 'err', <message>)`` so the parent can skip/raise per its skip policy (C1).
    Must stay module-level (and in a torch-free module): ``spawn``/Windows pickles the target by reference.

    ``meta_only`` (pixel-free read, ``data.visual_input=none``): touch only the ``_meta`` values. A full walk
    reads every JPEG crop in the chunk — the whole store, every epoch — for a reader that never decodes one.
    Page cache only: what the model sees is identical either way. Defaults off, so a caller that passes
    three positional args (an older parent under ``spawn``) keeps the full walk.
    """
    try:
        env = lmdb.open(path, readonly=True, lock=False, readahead=not meta_only)
        with env.begin(write=False) as txn:
            walk_chunk(txn, meta_only=meta_only)
        env.close()
        queue.put((idx, "ok", path))
    except Exception as exc:  # noqa: BLE001 — any failure is reported, never raised in the child
        queue.put((idx, "err", str(exc)))


def walk_chunk(txn: lmdb.Transaction, *, meta_only: bool) -> int:
    """Materialize the values a reader will need; return how many were read.

    Full: every value (keys + values cursor). ``meta_only``: a keys-only cursor, then a ``get`` per ``_meta``
    key — the keys-only cursor never copies a value, so the image blobs are not read.
    """
    if not meta_only:
        return sum(1 for _ in txn.cursor())
    n = 0
    for key in txn.cursor().iternext(keys=True, values=False):
        if key.endswith(b"_meta"):
            txn.get(key)
            n += 1
    return n

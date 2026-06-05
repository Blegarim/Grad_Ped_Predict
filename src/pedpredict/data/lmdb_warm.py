"""LMDB page-cache warm worker (Prompt 4.2, B9) — deliberately torch-free.

Isolated from ``training/chunk_loader.py`` for one reason: it is the target of a ``spawn``ed process.
Under ``spawn`` (the Windows/macOS default) the child re-imports the module that defines the target
callable; if that module pulled in ``torch`` (as ``chunk_loader`` does, transitively via ``DataLoader``),
every warm process would pay torch's multi-second import before it could read a single key. Keeping the
worker here — importing only ``lmdb`` + stdlib — lets the warm children start fast.

This is the EXACT port of OLD ``scripts/train_utils.py:80-98`` ``mp_async_load``.
"""

from __future__ import annotations

import multiprocessing as mp

import lmdb

__all__ = ["WarmResult", "warm_lmdb_chunk"]

#: ``(idx, status, payload)`` put on the warm queue — ``status`` is ``"ok"`` (payload=path) or ``"err"``.
WarmResult = tuple[int, str, str]


def warm_lmdb_chunk(idx: int, path: str, queue: mp.Queue[WarmResult]) -> None:
    """Warm one LMDB chunk's OS page cache, then return its path — EXACT port of ``mp_async_load``.

    Opens the chunk read-only, reads a single ``_meta`` key to encourage OS file caching, closes, and
    puts ``(idx, 'ok', path)``; any failure puts ``(idx, 'err', <message>)`` so the parent can skip it.
    Must stay module-level (and in a torch-free module): ``spawn``/Windows pickles the target by reference.
    """
    try:
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                if key.decode().endswith("_meta"):
                    _ = txn.get(key)
                    break
        env.close()
        queue.put((idx, "ok", path))
    except Exception as exc:  # noqa: BLE001 — any failure is reported, never raised in the child
        queue.put((idx, "err", str(exc)))

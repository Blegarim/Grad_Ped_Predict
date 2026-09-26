"""Global shuffle-rewrite of the training LMDB chunks — fixes the block-ordered visit order.

The training loader does a **full pass over one chunk before touching the next**
(``training/chunk_loader.ChunkPrefetcher.epoch_loaders`` shuffles chunk *order* only), and chunks are
written as contiguous slices of the record list (``lmdb_writer.write_dataset_chunks_from``) from two
dirs with opposite class composition: ``preprocessed_train`` (~2.9% crossing) and
``preprocessed_train_aug`` (minority records only — see ``data/augment`` module docstring). One chunk in
three is therefore nearly all-positive, the model's output bias is set by whichever chunk ran last, and
``best.pth`` is selected on a validation metric that samples that lottery at the epoch boundary.

This module rewrites those dirs into ONE dir whose chunks are a uniform random sample of the union, so
every chunk carries the global class mix and chunk-visit order stops mattering.

**Verbatim copy, no re-encode.** JPEG blobs and the pickled meta move byte-for-byte under new keys
(``{seq_id}_*`` -> ``{j}_*``), so the window population, labels, counts, S1 onset fields and pose
keypoints are all unchanged — no Dataset Statistics re-pin, no golden-fixture change, no re-backfill.
Only storage order changes. The source dirs are never modified.

Val/test need no equivalent pass: the validation loader iterates chunks in stable order with no sampler
and the metric accumulates across all of them, so order cannot bias it.

**Metadata-only output** (``meta_only=True``): copies each sample's ``_meta`` record verbatim and nothing
else. A pixel-free run (``data.visual_input=none``) reads nothing but those records, so this is the same
training data at a few percent of the disk — 54 GB of JPEG crops never get duplicated. Name such a dir with
:data:`META_ONLY_DIR_MARKER` so config validation refuses it to any model that decodes images.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import lmdb

from pedpredict.config.schema import DataCfg
from pedpredict.data.lmdb_writer import compute_map_size

__all__ = [
    "META_ONLY_DIR_MARKER",
    "SampleRef",
    "enumerate_dir",
    "enumerate_dirs",
    "plan_shuffle",
    "chunk_positive_rates",
    "write_shuffled",
]

#: Suffix marking a per-sample metadata key; the part before it is the sample's in-chunk id.
_META_SUFFIX = "_meta"
#: A train dir whose name contains this holds ``_meta`` records only (no crops): pixel-free models only.
META_ONLY_DIR_MARKER = "_metaonly"


@dataclass(frozen=True)
class SampleRef:
    """One stored sample: which chunk holds it, its in-chunk id, and its crossing label."""

    chunk: Path
    seq_id: str
    crosses: int


def _chunk_paths(directory: Path) -> list[Path]:
    """The ``chunk_*.lmdb`` environments in ``directory``, in sorted (stable) order."""
    paths = sorted(p for p in directory.iterdir() if p.name.startswith("chunk_"))
    if not paths:
        raise FileNotFoundError(f"no chunk_*.lmdb found in {directory}")
    return paths


def enumerate_dir(directory: Path) -> list[SampleRef]:
    """Every sample in one LMDB dir, read from the ``_meta`` keys only (no image blob is touched)."""
    refs: list[SampleRef] = []
    for chunk in _chunk_paths(directory):
        env = lmdb.open(str(chunk), readonly=True, lock=False, readahead=False)
        try:
            with env.begin(write=False) as txn:
                # Keys only: a values-too cursor would materialize every JPEG blob in the store just to
                # reach the handful of _meta keys, turning a label scan into a full-dataset read.
                names = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)]
                for name in names:
                    if not name.endswith(_META_SUFFIX):
                        continue
                    meta = pickle.loads(txn.get(name.encode()))
                    refs.append(SampleRef(chunk, name[: -len(_META_SUFFIX)], int(meta["crosses"])))
        finally:
            env.close()
    return refs


def enumerate_dirs(dirs: list[Path]) -> list[SampleRef]:
    """Every sample across several LMDB dirs (the union that ``paths.lmdb_train`` trains on)."""
    refs: list[SampleRef] = []
    for directory in dirs:
        refs.extend(enumerate_dir(directory))
    return refs


def plan_shuffle(refs: list[SampleRef], seed: int) -> list[SampleRef]:
    """A seeded global permutation of ``refs`` — the whole point of the pass, kept pure for testing."""
    out = list(refs)
    random.Random(seed).shuffle(out)
    return out


def chunk_positive_rates(refs: list[SampleRef], chunk_size: int) -> list[float]:
    """Crossing rate of each ``chunk_size`` group, in order — the before/after proof of the fix.

    On the source ordering this alternates between ~0.03 and ~1.0; after :func:`plan_shuffle` every
    entry sits at the global rate. ``scripts/reshuffle_train_lmdb.py`` gates on the spread.
    """
    rates: list[float] = []
    for i in range(0, len(refs), chunk_size):
        group = refs[i : i + chunk_size]
        rates.append(sum(r.crosses for r in group) / len(group))
    return rates


class _EnvCache:
    """Read-only LMDB envs kept open across the rewrite (the reads are scattered by construction)."""

    def __init__(self) -> None:
        self._envs: dict[Path, lmdb.Environment] = {}

    def get(self, path: Path) -> lmdb.Environment:
        if path not in self._envs:
            self._envs[path] = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        return self._envs[path]

    def close(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs.clear()


def _copy_sample(
    src_txn: lmdb.Transaction, dst_txn: lmdb.Transaction, seq_id: str, j: int, *, meta_only: bool = False
) -> int:
    """Copy every key of one sample (or only its ``_meta`` record) under its new in-chunk index ``j``.
    Returns the key count."""
    if meta_only:
        meta = src_txn.get(f"{seq_id}{_META_SUFFIX}".encode())
        if meta is None:
            raise KeyError(f"sample {seq_id!r} has no meta record")
        dst_txn.put(f"{j}{_META_SUFFIX}".encode(), meta)
        return 1
    prefix = f"{seq_id}_".encode()
    cursor = src_txn.cursor()
    if not cursor.set_range(prefix):
        raise KeyError(f"sample {seq_id!r} has no keys")
    n = 0
    for key, value in cursor:
        if not key.startswith(prefix):
            break
        dst_txn.put(f"{j}_".encode() + key[len(prefix) :], value)
        n += 1
    if n == 0:
        raise KeyError(f"sample {seq_id!r} has no keys")
    return n


def write_shuffled(
    refs: list[SampleRef], out_dir: Path, cfg: DataCfg, *, chunk_size: int | None = None, meta_only: bool = False
) -> list[Path]:
    """Write ``refs`` (already permuted) into ``chunk_*.lmdb`` files under ``out_dir``.

    Blobs are copied verbatim, so this never decodes or re-encodes an image; ``meta_only`` copies the
    ``_meta`` records alone (see the module docstring). Returns the chunk paths.
    """
    size = cfg.chunk_size if chunk_size is None else chunk_size
    out_dir.mkdir(parents=True, exist_ok=True)
    envs = _EnvCache()
    written: list[Path] = []
    try:
        for start in range(0, len(refs), size):
            group = refs[start : start + size]
            path = out_dir / f"chunk_{start:06d}.lmdb"
            dst = lmdb.open(str(path), map_size=compute_map_size(len(group), cfg))
            try:
                with dst.begin(write=True) as dst_txn:
                    for j, ref in enumerate(group):
                        with envs.get(ref.chunk).begin(write=False) as src_txn:
                            _copy_sample(src_txn, dst_txn, ref.seq_id, j, meta_only=meta_only)
            finally:
                dst.close()
            written.append(path)
    finally:
        envs.close()
    return written

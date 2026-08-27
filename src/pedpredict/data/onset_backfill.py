"""Backfill the S1 onset keys into LMDB chunks that were built before S1 (metadata-only).

``pie_sequences`` has always written ``onset_offset`` / ``future_observed`` / ``track_crosses`` into the
sequence pkls, but ``lmdb_writer.pack_meta`` used to drop them, so the chunks that training and eval
actually read never carried them. The writer now packs them; this module upgrades the chunks that
already exist, so the onset head can be trained without a rebuild.

**Only the meta value is rewritten.** Crops live under their own ``{j}_{t}_tight`` / ``{j}_{t}_context``
keys and are never touched, so this is a fast pass over the small pickled part of each entry — minutes,
not the hours a rebuild costs.

Sample-to-record matching is **positional**, which is exactly how the writer laid the chunks down:
``write_dataset_chunks`` slices ``records[i : i + chunk_size]`` into ``chunk_{i:06d}.lmdb``, so the file
name carries the record offset and the in-chunk index ``j`` is the offset from it. Position alone is a
silent-corruption risk, so every sample is verified against its record on ``track_id`` **and**
``crosses`` before anything is written; the first mismatch aborts the chunk with both values named.

Offline-augmented dirs (``preprocessed_train_aug``) are deliberately **not** backfillable: augmentation
oversamples, so sample count != record count and the positional map does not hold. The verification
above turns that into a loud abort rather than a wrong answer. Their route is to backfill the base dir
and re-run ``scripts/augment_dataset.py`` — the augment path reads through ``read_raw_sample`` and
rebuilds meta through ``pack_meta``, both of which now carry the onset keys, so the copies inherit them.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lmdb

from pedpredict.data.pie_sequences import ONSET_FIELDS, SequenceRecord

__all__ = [
    "CHUNK_PATTERN",
    "BackfillReport",
    "chunk_record_offset",
    "iter_chunk_paths",
    "backfill_chunk",
    "backfill_dir",
    "format_reports",
]

#: ``chunk_{start_index:06d}.lmdb`` — the writer's name, whose number IS the record offset.
CHUNK_PATTERN = re.compile(r"^chunk_(\d+)\.lmdb$")


@dataclass(frozen=True)
class BackfillReport:
    """Per-chunk outcome. ``written + already_present == samples`` on a clean pass."""

    chunk: str
    samples: int
    written: int
    already_present: int


def chunk_record_offset(path: Path) -> int:
    """Record index the chunk starts at, parsed from ``chunk_{i:06d}.lmdb``."""
    match = CHUNK_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"[onset_backfill] {path.name!r} is not a writer-produced chunk name (chunk_NNNNNN.lmdb); "
            f"refusing to guess its record offset."
        )
    return int(match.group(1))


def iter_chunk_paths(chunk_dir: Path) -> list[Path]:
    """Every ``chunk_*.lmdb`` under ``chunk_dir``, in record order."""
    paths = sorted((p for p in chunk_dir.iterdir() if CHUNK_PATTERN.match(p.name)), key=chunk_record_offset)
    if not paths:
        raise FileNotFoundError(f"[onset_backfill] no chunk_*.lmdb found under {chunk_dir}")
    return paths


def _seq_ids(txn: lmdb.Transaction) -> list[str]:
    """In-chunk sample ids in numeric order (the cursor's lexicographic order is not numeric)."""
    ids = [k.decode().split("_")[0] for k, _ in txn.cursor() if k.decode().endswith("_meta")]
    return sorted(ids, key=int)


def _verify(meta: dict, record: SequenceRecord, *, chunk: str, seq_id: str, record_idx: int) -> None:
    """Abort unless this stored sample really is this record (positional-map integrity check)."""
    if meta.get("track_id") != record["track_id"]:
        raise ValueError(
            f"[onset_backfill] {chunk} sample {seq_id}: track_id {meta.get('track_id')!r} != record "
            f"{record_idx} track_id {record['track_id']!r}. The chunk dir and the sequence pkl do not "
            f"correspond — check the pkl, and note that augmented dirs cannot be backfilled positionally."
        )
    if int(meta["crosses"]) != int(record["crosses"]):
        raise ValueError(
            f"[onset_backfill] {chunk} sample {seq_id}: crosses {int(meta['crosses'])} != record "
            f"{record_idx} crosses {int(record['crosses'])} (same track_id). Chunk and pkl are "
            f"different builds of the same split — rebuild, or supply the pkl this chunk came from."
        )


def _map_size(path: Path) -> int:
    """The chunk's own map_size, so reopening for write neither shrinks nor regrows the file."""
    env = lmdb.open(str(path), readonly=True, lock=False)
    try:
        return int(env.info()["map_size"])
    finally:
        env.close()


def _open_for_backfill(path: Path, *, dry_run: bool) -> lmdb.Environment:
    """Open the chunk for the backfill, translating the Windows in-use failure into an instruction.

    Windows refuses a write-open while any process holds a memory-mapped section of the file, so a
    live ``LMDBChunkDataset`` — a running trainer, an open notebook — blocks the backfill with a
    message that names neither the cause nor the cure.
    """
    try:
        return lmdb.open(str(path), map_size=_map_size(path), readonly=dry_run, lock=not dry_run)
    except lmdb.Error as exc:
        raise RuntimeError(
            f"[onset_backfill] cannot open {path.name} for writing: {exc}. On Windows this means "
            f"something still has the chunk memory-mapped — stop any running training/eval job and "
            f"close any LMDBChunkDataset (call .close()) before backfilling. '--dry-run' opens "
            f"read-only and works while readers are live."
        ) from exc


def backfill_chunk(
    chunk_path: Path, records: Sequence[SequenceRecord], *, dry_run: bool = False
) -> BackfillReport:
    """Write the three S1 keys into every meta in one chunk. Idempotent; one transaction.

    ``records`` is the FULL split record list — the chunk's own slice is taken by file-name offset.
    With ``dry_run`` every sample is still read and verified, but nothing is written.
    """
    offset = chunk_record_offset(chunk_path)
    env = _open_for_backfill(chunk_path, dry_run=dry_run)
    written = present = 0
    try:
        with env.begin(write=not dry_run) as txn:
            ids = _seq_ids(txn)
            if offset + len(ids) > len(records):
                raise ValueError(
                    f"[onset_backfill] {chunk_path.name} holds {len(ids)} samples at record offset "
                    f"{offset}, past the end of a {len(records)}-record pkl. Wrong pkl for this dir."
                )
            for seq_id in ids:
                key = f"{seq_id}_meta".encode()
                meta = pickle.loads(txn.get(key))
                record_idx = offset + int(seq_id)
                record = records[record_idx]
                _verify(meta, record, chunk=chunk_path.name, seq_id=seq_id, record_idx=record_idx)
                if all(field in meta for field in ONSET_FIELDS):
                    present += 1
                    continue
                meta.update({field: int(record[field]) for field in ONSET_FIELDS})  # type: ignore[literal-required]
                if not dry_run:
                    txn.put(key, pickle.dumps(meta))
                written += 1
        if not dry_run:
            env.sync()
    finally:
        env.close()
    return BackfillReport(chunk_path.name, len(ids), written, present)


def backfill_dir(
    chunk_dir: Path, records: Sequence[SequenceRecord], *, dry_run: bool = False
) -> list[BackfillReport]:
    """Run :func:`backfill_chunk` over every chunk in ``chunk_dir``, in record order."""
    return [backfill_chunk(path, records, dry_run=dry_run) for path in iter_chunk_paths(chunk_dir)]


def format_reports(reports: Sequence[BackfillReport], *, dry_run: bool = False) -> str:
    """Per-chunk lines plus a total, for the CLI."""
    verb = "would write" if dry_run else "wrote"
    lines = [
        f"  {r.chunk:<24} {r.samples:>7} samples  {verb} {r.written:>7}  already present {r.already_present:>7}"
        for r in reports
    ]
    samples = sum(r.samples for r in reports)
    written = sum(r.written for r in reports)
    present = sum(r.already_present for r in reports)
    lines.append(f"  {'TOTAL':<24} {samples:>7} samples  {verb} {written:>7}  already present {present:>7}")
    return "\n".join(lines)

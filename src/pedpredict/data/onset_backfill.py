"""Backfill the S1 onset keys into LMDB chunks that were built before S1 (metadata-only).

``pie_sequences`` has always written ``onset_offset`` / ``future_observed`` / ``track_crosses`` into the
sequence pkls, but ``lmdb_writer.pack_meta`` used to drop them, so the chunks that training and eval
actually read never carried them. The writer now packs them; this module upgrades the chunks that
already exist, so the onset head can be trained without a rebuild.

**Only the meta value is rewritten.** Crops live under their own ``{j}_{t}_tight`` / ``{j}_{t}_context``
keys and are never touched, so this is a fast pass over the small pickled part of each entry — minutes,
not the hours a rebuild costs.

**Capacity.** Rewriting a meta is copy-on-write: LMDB allocates fresh pages for the new value and
cannot reclaim the old ones until the transaction commits, so rewriting a whole chunk in one
transaction transiently needs room for a second copy of every meta in it. Chunks built with a tight
``data.lmdb_map_size_bytes`` (setup.md step 0's disk knob) have no such room, and used to fail with
``MDB_MAP_FULL``. Writes are therefore committed in batches of :data:`BATCH_SIZE`, which bounds the
transient cost to one batch; if even that does not fit, the chunk's ``map_size`` grows a step at a time
(:data:`GROWTH_STEP_BYTES`) and the batch is retried. Growth costs real disk immediately on Windows,
which pre-allocates, so it is bounded and reported rather than unlimited.

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
    "BATCH_SIZE",
    "CHUNK_PATTERN",
    "GROWTH_STEP_BYTES",
    "MAX_GROWTH_STEPS",
    "BackfillReport",
    "chunk_record_offset",
    "iter_chunk_paths",
    "backfill_chunk",
    "backfill_dir",
    "format_reports",
]

#: ``chunk_{start_index:06d}.lmdb`` — the writer's name, whose number IS the record offset.
CHUNK_PATTERN = re.compile(r"^chunk_(\d+)\.lmdb$")

#: Samples per write transaction. Bounds the copy-on-write cost of the pass (see the module docstring)
#: to one batch instead of a whole chunk. Lower it only if a chunk's metas are unusually large.
BATCH_SIZE = 512

#: One step of ``map_size`` growth when a batch still does not fit. LMDB PRE-ALLOCATES on Windows, so
#: every byte is real disk the moment it is asked for — grow in fixed steps, never by doubling.
GROWTH_STEP_BYTES = 64 * 1024**2

#: Growth steps allowed per chunk before giving up, so a wrong pkl cannot silently eat the volume.
MAX_GROWTH_STEPS = 8


@dataclass(frozen=True)
class BackfillReport:
    """Per-chunk outcome. ``written + already_present == samples`` on a clean pass.

    ``rewrite_bytes`` is what the pass has to write, ``free_bytes`` the room the chunk had left at the
    end. The two do not compare directly — a write transaction also needs page, freelist and meta-page
    overhead, so ``free_bytes`` comfortably above ``rewrite_bytes`` can still hit ``MDB_MAP_FULL``.
    They are a report, not a predictor; the dry-run warning flags chunks with under one growth step of
    room rather than pretending to compute the exact need.
    """

    chunk: str
    samples: int
    written: int
    already_present: int
    #: Capacity accounting, for the pre-flight warning and the growth note.
    map_size: int = 0
    free_bytes: int = 0
    rewrite_bytes: int = 0
    grown_bytes: int = 0


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


MAP_FULL_HINT = (
    "[onset_backfill] {chunk}: still out of room after growing {grown} MiB. The chunk was built with a "
    "map_size that leaves no headroom (setup.md step 0's tight-disk knob does exactly that). Free space "
    "on the volume holding the chunks and re-run — the pass is idempotent, so it resumes where it "
    "stopped, rewriting only the metas that are still missing the keys."
)


def _env_capacity(env: lmdb.Environment) -> tuple[int, int]:
    """``(map_size, free_bytes)``, from the environment's own page accounting."""
    info, stat = env.info(), env.stat()
    used = (int(info["last_pgno"]) + 1) * int(stat["psize"])
    map_size = int(info["map_size"])
    return map_size, max(map_size - used, 0)


def _grow_map(env: lmdb.Environment, *, chunk: str, grown: int) -> int:
    """Add one :data:`GROWTH_STEP_BYTES` step to the chunk's ``map_size``; returns the bytes added."""
    current = int(env.info()["map_size"])
    try:
        env.set_mapsize(current + GROWTH_STEP_BYTES)
    except (lmdb.Error, OSError) as exc:
        raise RuntimeError(
            f"[onset_backfill] {chunk}: cannot grow map_size past {current / 1024 ** 3:.2f} GiB "
            f"(grew {grown / 1024 ** 2:.0f} MiB already): {exc}. LMDB pre-allocates the file on "
            f"Windows, so this is normally free disk on the volume holding the chunks."
        ) from exc
    return GROWTH_STEP_BYTES


def _run_batch(
    env: lmdb.Environment,
    ids: Sequence[str],
    records: Sequence[SequenceRecord],
    offset: int,
    *,
    chunk: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """Read, verify and (unless ``dry_run``) rewrite one batch of metas in a single transaction.

    Returns ``(written, already_present, rewrite_bytes)``. A ``MapFullError`` propagates with the
    transaction aborted, so the caller can grow the map and call again with the same ids.
    """
    written = present = rewrite_bytes = 0
    with env.begin(write=not dry_run) as txn:
        for seq_id in ids:
            key = f"{seq_id}_meta".encode()
            meta = pickle.loads(txn.get(key))
            record_idx = offset + int(seq_id)
            record = records[record_idx]
            _verify(meta, record, chunk=chunk, seq_id=seq_id, record_idx=record_idx)
            if all(field in meta for field in ONSET_FIELDS):
                present += 1
                continue
            meta.update({field: int(record[field]) for field in ONSET_FIELDS})  # type: ignore[literal-required]
            blob = pickle.dumps(meta)
            rewrite_bytes += len(blob)
            if not dry_run:
                txn.put(key, blob)
            written += 1
    return written, present, rewrite_bytes


def _run_batch_growing(
    env: lmdb.Environment,
    ids: Sequence[str],
    records: Sequence[SequenceRecord],
    offset: int,
    *,
    chunk: str,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    """:func:`_run_batch`, growing the chunk's map and retrying the batch when it runs out of room."""
    grown = 0
    for attempt in range(MAX_GROWTH_STEPS + 1):
        try:
            written, present, rewrite_bytes = _run_batch(
                env, ids, records, offset, chunk=chunk, dry_run=dry_run
            )
        except lmdb.MapFullError as exc:
            if dry_run or attempt == MAX_GROWTH_STEPS:
                raise RuntimeError(MAP_FULL_HINT.format(chunk=chunk, grown=grown // 1024 ** 2)) from exc
            grown += _grow_map(env, chunk=chunk, grown=grown)
            continue
        return written, present, rewrite_bytes, grown
    raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover


def backfill_chunk(
    chunk_path: Path,
    records: Sequence[SequenceRecord],
    *,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> BackfillReport:
    """Write the three S1 keys into every meta in one chunk. Idempotent; batched transactions.

    ``records`` is the FULL split record list — the chunk's own slice is taken by file-name offset.
    With ``dry_run`` every sample is still read and verified, but nothing is written. Batching and the
    ``map_size`` growth that backs it are explained in the module docstring.
    """
    offset = chunk_record_offset(chunk_path)
    env = _open_for_backfill(chunk_path, dry_run=dry_run)
    written = present = rewrite_bytes = grown = 0
    try:
        with env.begin() as txn:
            ids = _seq_ids(txn)
        if offset + len(ids) > len(records):
            raise ValueError(
                f"[onset_backfill] {chunk_path.name} holds {len(ids)} samples at record offset "
                f"{offset}, past the end of a {len(records)}-record pkl. Wrong pkl for this dir."
            )
        for begin in range(0, len(ids), batch_size):
            batch = _run_batch_growing(
                env, ids[begin: begin + batch_size], records, offset,
                chunk=chunk_path.name, dry_run=dry_run,
            )
            written, present = written + batch[0], present + batch[1]
            rewrite_bytes, grown = rewrite_bytes + batch[2], grown + batch[3]
        if not dry_run:
            env.sync()
        map_size, free_bytes = _env_capacity(env)
    finally:
        env.close()
    return BackfillReport(
        chunk_path.name, len(ids), written, present, map_size, free_bytes, rewrite_bytes, grown
    )


def backfill_dir(
    chunk_dir: Path,
    records: Sequence[SequenceRecord],
    *,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> list[BackfillReport]:
    """Run :func:`backfill_chunk` over every chunk in ``chunk_dir``, in record order."""
    return [
        backfill_chunk(path, records, dry_run=dry_run, batch_size=batch_size)
        for path in iter_chunk_paths(chunk_dir)
    ]


def format_reports(reports: Sequence[BackfillReport], *, dry_run: bool = False) -> str:
    """Per-chunk lines, the total, and the capacity note.

    In a dry run the note is the pre-flight warning — which chunks have less free map than the pass
    must rewrite, i.e. the ones a real run will have to grow. In a real run it is what was grown.
    """
    verb = "would write" if dry_run else "wrote"
    lines = [
        f"  {r.chunk:<24} {r.samples:>7} samples  {verb} {r.written:>7}  already present {r.already_present:>7}"
        for r in reports
    ]
    samples = sum(r.samples for r in reports)
    written = sum(r.written for r in reports)
    present = sum(r.already_present for r in reports)
    lines.append(f"  {'TOTAL':<24} {samples:>7} samples  {verb} {written:>7}  already present {present:>7}")
    lines.extend(_capacity_lines(reports, dry_run=dry_run))
    return "\n".join(lines)


def _capacity_lines(reports: Sequence[BackfillReport], *, dry_run: bool) -> list[str]:
    """The map_size note: what a real pass will have to grow, or what this pass did grow."""
    if not dry_run:
        grown = sum(r.grown_bytes for r in reports)
        chunks = sum(1 for r in reports if r.grown_bytes)
        return [f"  grew map_size by {grown / 1024 ** 2:.0f} MiB across {chunks} chunk(s) — LMDB "
                f"pre-allocates on Windows, so that is disk now in use"] if grown else []
    tight = [r for r in reports if r.free_bytes < GROWTH_STEP_BYTES]
    if not tight:
        return []
    step = GROWTH_STEP_BYTES // 1024 ** 2
    return [
        f"  NOTE {len(tight)} chunk(s) have under {step} MiB of free map. A real run grows map_size in "
        f"{step} MiB steps as it needs to — up to {len(tight) * step} MiB of extra disk here, taken "
        f"immediately on Windows, which pre-allocates:",
        *(f"    {r.chunk:<24} free {r.free_bytes / 1024 ** 2:>8.1f} MiB   "
          f"rewrite {r.rewrite_bytes / 1024 ** 2:>8.1f} MiB" for r in tight[:5]),
    ]

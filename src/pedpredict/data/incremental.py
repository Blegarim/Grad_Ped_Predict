"""Disk-bounded, resumable LMDB build: extract frames per-video, crop, delete, advance.

The full-split build (:func:`pedpredict.data.lmdb_writer.write_dataset_chunks`) assumes every frame
a split references is already on disk. For ``train`` (set01+02+04) that peaks at hundreds of GB and
is what fills a storage-limited disk. This module never holds more than the videos overlapping the
*current* chunk window:

    for each chunk [a, b):
        extract the videos records[a:b] reference (only the frame ids they use)  ->  build chunk a
        delete every video whose last referencing record index < b

Because the sequence pkl is ordered set -> video -> track (PIE iterates deterministically), a video
is referenced by one contiguous record range, so "delete once we pass its last record" is exact and
peak disk stays at ~the videos straddling one chunk boundary plus the growing LMDB.

The plan (:func:`iter_build_steps`) is pure and unit-tested; extraction (:func:`extract_video_frames`,
cv2 BGR ``VideoCapture`` -> ``imwrite``, byte-identical to ``PIE.extract_and_save_images``) and the
per-chunk build are the only I/O. Resumes from existing ``chunk_*.lmdb`` dirs, so a crashed full build
continues where it stopped (delete the partial final chunk first).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pedpredict.data.pie_sequences import SequenceRecord

__all__ = [
    "VideoKey",
    "BuildStep",
    "parse_frame_path",
    "next_chunk_start",
    "iter_build_steps",
    "extract_video_frames",
]

VideoKey = tuple[str, str]  # (set_id, video_id), e.g. ("set01", "video_0001")

_SET_RE = re.compile(r"^set\d+$")
_VIDEO_RE = re.compile(r"^video_\d+$")
_CHUNK_RE = re.compile(r"^chunk_(\d+)\.lmdb$")


def parse_frame_path(path: str) -> tuple[str, str, int]:
    """``.../set01/video_0001/00123.png`` -> ``("set01", "video_0001", 123)`` (separator-agnostic)."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if _SET_RE.match(part) and i + 2 < len(parts) and _VIDEO_RE.match(parts[i + 1]):
            return part, parts[i + 1], int(Path(parts[i + 2]).stem)
    raise ValueError(f"cannot parse set/video/frame from image path: {path!r}")


def next_chunk_start(existing_starts: list[int], chunk_size: int) -> int:
    """Resume index: one chunk past the highest existing ``chunk_<start>.lmdb`` (0 if none)."""
    return 0 if not existing_starts else max(existing_starts) + chunk_size


def existing_chunk_starts(out_dir: Path) -> list[int]:
    """Start indices of the ``chunk_*.lmdb`` dirs already present in ``out_dir`` (empty if none)."""
    if not out_dir.is_dir():
        return []
    return sorted(int(m.group(1)) for p in out_dir.iterdir() if (m := _CHUNK_RE.match(p.name)))


@dataclass(slots=True)
class BuildStep:
    """One chunk's worth of work: extract these videos, build ``[chunk_start, chunk_end)``, then delete."""

    chunk_start: int
    chunk_end: int
    extract: dict[VideoKey, dict[int, str]] = field(default_factory=dict)  # newly-needed: fid -> on-disk path
    delete: list[VideoKey] = field(default_factory=list)                   # videos finished after this chunk


def _index_records(records: list[SequenceRecord]) -> tuple[
    list[set[VideoKey]], dict[VideoKey, dict[int, str]], dict[VideoKey, int]
]:
    """Per-record video sets, per-video {frame_id -> path}, and per-video last referencing record index."""
    per_record: list[set[VideoKey]] = []
    frames: dict[VideoKey, dict[int, str]] = {}
    last_idx: dict[VideoKey, int] = {}
    for i, rec in enumerate(records):
        vids: set[VideoKey] = set()
        for path in rec["images"]:
            set_id, vid, fid = parse_frame_path(path)
            key = (set_id, vid)
            vids.add(key)
            frames.setdefault(key, {})[fid] = path
            last_idx[key] = i
        per_record.append(vids)
    return per_record, frames, last_idx


def iter_build_steps(
    records: list[SequenceRecord], chunk_size: int, start_idx: int = 0
) -> Iterator[BuildStep]:
    """Plan the extract/build/delete steps for chunks from ``start_idx`` to the end (pure, no I/O).

    Videos consumed entirely before ``start_idx`` are never scheduled for extraction, so a resumed
    build only touches frames it still needs.
    """
    per_record, frames, last_idx = _index_records(records)
    extracted: set[VideoKey] = set()
    for a in range(start_idx, len(records), chunk_size):
        b = min(a + chunk_size, len(records))
        in_window: set[VideoKey] = set().union(*per_record[a:b]) if b > a else set()
        step = BuildStep(chunk_start=a, chunk_end=b)
        for key in sorted(in_window):
            if key not in extracted:
                step.extract[key] = frames[key]
                extracted.add(key)
        step.delete = sorted(k for k in extracted if last_idx[k] < b)
        extracted.difference_update(step.delete)
        yield step


def extract_video_frames(
    clips_dir: Path, root: Path, set_id: str, video_id: str, frames: dict[int, str]
) -> int:
    """Decode ``clips_dir/set_id/video_id.mp4`` and write the requested ``{fid: path}`` frames.

    cv2 ``VideoCapture`` (BGR) + ``imwrite`` reproduces ``PIE.extract_and_save_images`` byte-for-byte.
    Skips frames already on disk; ``path`` strings are resolved under ``root`` when relative. Returns
    the number of frames written. Reads only up to the highest requested frame id.
    """
    import cv2  # local import: only the extraction path needs opencv

    targets = {fid: (p if (p := Path(path)).is_absolute() else root / path) for fid, path in frames.items()}
    pending = {fid for fid, dst in targets.items() if not dst.exists()}
    if not pending:
        return 0
    video_path = clips_dir / set_id / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open clip: {video_path}")
    written, last_needed = 0, max(pending)
    try:
        frame_num = 0
        ok, image = cap.read()
        while ok and frame_num <= last_needed:
            if frame_num in pending:
                dst = targets[frame_num]
                dst.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst), image)
                written += 1
            ok, image = cap.read()
            frame_num += 1
    finally:
        cap.release()
    if written != len(pending):
        raise RuntimeError(f"{set_id}/{video_id}: wrote {written}/{len(pending)} frames (clip too short?)")
    return written

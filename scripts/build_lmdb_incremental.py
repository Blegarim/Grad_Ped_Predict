"""Disk-bounded, resumable LMDB build for storage-limited machines.

Same output as ``build_lmdb.py`` (identical ``chunk_<start>.lmdb`` dirs) but never holds more than the
videos straddling the current chunk on disk: per chunk it extracts only the frames those records use
(cv2, byte-identical to PIE's extractor), builds the chunk, then deletes spent video frame dirs. The
sequence pkl must already exist (``make_sequences.py`` — annotations only, tiny). See
:mod:`pedpredict.data.incremental` for the plan and setup.md "Incremental extraction".

    # finish a crashed train build — a short (mid-write) final chunk is detected and refused (C2),
    # so delete the partial chunk dir the error names, then re-run:
    python scripts/build_lmdb_incremental.py --split train
    # explicit resume / fresh start / keep frames for inspection:
    python scripts/build_lmdb_incremental.py --split train --start-idx 20000 --keep-frames
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.incremental import (
    assert_resume_safe,
    existing_chunk_starts,
    extract_video_frames,
    iter_build_steps,
    next_chunk_start,
)
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.paths import ResolvedPaths, resolve_paths

_SPLITS = ("train", "val", "test")


def _out_dir_for(split: str, paths: ResolvedPaths) -> Path:
    """Split -> its configured LMDB dir (mirrors build_lmdb.py; train -> first unioned dir)."""
    if split == "train":
        return paths.lmdb_train[0]
    return paths.lmdb_val if split == "val" else paths.lmdb_test


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", choices=_SPLITS, default="train")
    parser.add_argument("--pkl", default=None, help="Override the input sequence pickle path.")
    parser.add_argument("--out-dir", default=None, help="Override the output LMDB directory.")
    parser.add_argument("--start-idx", type=int, default=None, help="Resume index (default: auto-detect).")
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete extracted frames.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    pkl = Path(args.pkl) if args.pkl else paths.sequences_dir / f"sequences_{args.split}.pkl"
    out_dir = Path(args.out_dir) if args.out_dir else _out_dir_for(args.split, paths)
    clips_dir = paths.pie_root / "PIE_clips"

    records = load_sequences(pkl)
    if args.start_idx is None:
        # C2 guard: a crashed build leaves a SHORT final chunk that auto-resume would skip forever.
        assert_resume_safe(out_dir, len(records), cfg.data.chunk_size)
    start_idx = args.start_idx if args.start_idx is not None else next_chunk_start(
        existing_chunk_starts(out_dir), cfg.data.chunk_size
    )
    if (partial := out_dir / f"chunk_{start_idx:06d}.lmdb").exists():
        raise SystemExit(f"{partial} already exists — delete the partial chunk before resuming at {start_idx}.")
    print(f"[{args.split}] {len(records)} records; resuming at index {start_idx} -> {out_dir}")

    frame_dirs: dict[tuple[str, str], Path] = {}
    for step in iter_build_steps(records, cfg.data.chunk_size, start_idx):
        for key, frames in step.extract.items():
            n = extract_video_frames(clips_dir, paths.root, key[0], key[1], frames)
            sample = next(iter(frames.values()))
            frame_dirs[key] = (p if (p := Path(sample)).is_absolute() else paths.root / sample).parent
            print(f"  extracted {n} frame(s) for {key[0]}/{key[1]}")
        write_dataset_chunks(records, out_dir, cfg.data, start_idx=step.chunk_start, end_idx=step.chunk_end)
        print(f"  built chunk [{step.chunk_start}, {step.chunk_end})")
        if not args.keep_frames:
            for key in step.delete:
                if (d := frame_dirs.pop(key, None)) and d.is_dir():
                    shutil.rmtree(d)
            if step.delete:
                print(f"  deleted frames for {len(step.delete)} finished video(s)")
    print(f"[{args.split}] done.")


if __name__ == "__main__":
    main()

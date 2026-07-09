"""Disk-bounded, resumable LMDB build for storage-limited machines.

Same output as ``build_lmdb.py`` (identical ``chunk_<start>.lmdb`` dirs) but never holds more than the
videos straddling the current chunk on disk: per chunk it extracts only the frames those records use
(cv2, byte-identical to PIE's extractor), builds the chunk, then deletes spent video frame dirs. The
sequence pkl must already exist (``make_sequences.py`` — annotations only, tiny). See
:mod:`pedpredict.data.incremental` for the plan and setup.md "Incremental extraction".

    # finish a crashed train build — a short (mid-write) final chunk is detected and refused (C2),
    # so delete the partial chunk dir the error names, then re-run:
    python scripts/build_lmdb_incremental.py --split train
    # anchored-protocol train (S1 pivot) — same disk-bounded extraction, reads sequences_train_benchmark.pkl:
    python scripts/build_lmdb_incremental.py --split train_benchmark
    # meta-splits — each member built disk-bounded, in sequence:
    python scripts/build_lmdb_incremental.py --split all            # streaming train, val, test
    python scripts/build_lmdb_incremental.py --split all_benchmark  # anchored train, val, test
    python scripts/build_lmdb_incremental.py --split exhaustive     # all + all_benchmark + train_aug
    # explicit resume / fresh start / keep frames for inspection:
    python scripts/build_lmdb_incremental.py --split train --start-idx 20000 --keep-frames
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.augment import augment_lmdb_dir
from pedpredict.data.incremental import (
    assert_resume_safe,
    existing_chunk_starts,
    extract_video_frames,
    iter_build_steps,
    next_chunk_start,
)
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.data.pose import PoseCache
from pedpredict.paths import ResolvedPaths, resolve_paths

_SPLITS = ("train", "val", "test")
# explicit-only anchored-protocol splits (S1 pivot); train_benchmark is the disk-bounded use case.
_EXTRA_SPLITS = ("train_benchmark", "val_benchmark", "test_benchmark")
# meta-splits: expand to concrete build steps (each still disk-bounded). ``train_aug`` is NOT a
# frame-extracted split — it is augmented from the built ``preprocessed_train`` LMDB (see
# ``_build_train_aug``), so it only appears in ``exhaustive`` and runs after the base train build.
_META_SPLITS = {
    "all": _SPLITS,
    "all_benchmark": _EXTRA_SPLITS,
    "exhaustive": (*_SPLITS, *_EXTRA_SPLITS, "train_aug"),
}


def _out_dir_for(split: str, paths: ResolvedPaths) -> Path:
    """Split -> its configured LMDB dir (mirrors build_lmdb.py; train -> first unioned dir;
    ``*_benchmark`` -> the anchored-protocol dirs)."""
    if split == "train":
        return paths.lmdb_train[0]
    if split == "train_benchmark":
        return paths.lmdb_train_benchmark[0]
    if split == "val_benchmark":
        return paths.lmdb_val_benchmark
    if split == "test_benchmark":
        return paths.lmdb_test_benchmark
    return paths.lmdb_val if split == "val" else paths.lmdb_test


def _build_split(split: str, cfg, paths: ResolvedPaths, *, pkl: Path, out_dir: Path,
                 start_idx: int | None, keep_frames: bool, pose_cache) -> None:
    """Disk-bounded build of one frame-extracted split: per chunk, extract only the frames its records
    use, build the chunk, then delete spent video frame dirs. Preserves the incremental storage-reclaim
    mechanism — the whole reason this script exists."""
    clips_dir = paths.pie_root / "PIE_clips"
    records = load_sequences(pkl)
    if start_idx is None:
        # C2 guard: a crashed build leaves a SHORT final chunk that auto-resume would skip forever.
        assert_resume_safe(out_dir, len(records), cfg.data.chunk_size)
    resume_at = start_idx if start_idx is not None else next_chunk_start(
        existing_chunk_starts(out_dir), cfg.data.chunk_size
    )
    if (partial := out_dir / f"chunk_{resume_at:06d}.lmdb").exists():
        raise SystemExit(f"{partial} already exists — delete the partial chunk before resuming at {resume_at}.")
    print(f"[{split}] {len(records)} records; resuming at index {resume_at} -> {out_dir}")

    frame_dirs: dict[tuple[str, str], Path] = {}
    for step in iter_build_steps(records, cfg.data.chunk_size, resume_at):
        for key, frames in step.extract.items():
            n = extract_video_frames(clips_dir, paths.root, key[0], key[1], frames)
            sample = next(iter(frames.values()))
            frame_dirs[key] = (p if (p := Path(sample)).is_absolute() else paths.root / sample).parent
            print(f"  extracted {n} frame(s) for {key[0]}/{key[1]}")
        write_dataset_chunks(
            records, out_dir, cfg.data,
            start_idx=step.chunk_start, end_idx=step.chunk_end, pose_cache=pose_cache,
        )
        print(f"  built chunk [{step.chunk_start}, {step.chunk_end})")
        if not keep_frames:
            for key in step.delete:
                if (d := frame_dirs.pop(key, None)) and d.is_dir():
                    shutil.rmtree(d)
            if step.delete:
                print(f"  deleted frames for {len(step.delete)} finished video(s)")
    print(f"[{split}] done.")


def _build_train_aug(cfg, paths: ResolvedPaths) -> None:
    """Offline minority-augmentation LMDB (``preprocessed_train_aug`` = ``lmdb_train[1]``). NOT a
    frame-extracted split: crops come from the already-built ``preprocessed_train`` LMDB, so this runs
    after the base train build and needs no PIE frames (mirrors ``scripts/augment_dataset.py``)."""
    if not cfg.augment.enabled:
        raise SystemExit(
            "augment.enabled is false — train_aug is opt-out. Re-run with '--set augment.enabled=true' "
            "to include it in the exhaustive build."
        )
    in_dir, out_dir = paths.lmdb_train[0], paths.lmdb_train[1]
    if not in_dir.is_dir():
        raise SystemExit(f"Missing base train LMDB {in_dir} — build the train split before train_aug.")
    print(f"[train_aug] augmenting {in_dir.name} -> {out_dir.name}")
    chunks = augment_lmdb_dir(in_dir, out_dir, cfg.data, cfg.augment)
    print(f"[train_aug] wrote {len(chunks)} chunk(s)")


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument(
        "--split", choices=(*_SPLITS, *_EXTRA_SPLITS, *_META_SPLITS), default="train"
    )
    parser.add_argument("--pkl", default=None, help="Override the input sequence pickle path.")
    parser.add_argument("--out-dir", default=None, help="Override the output LMDB directory.")
    parser.add_argument("--start-idx", type=int, default=None, help="Resume index (default: auto-detect).")
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete extracted frames.")
    args = parser.parse_args(argv)

    splits = _META_SPLITS.get(args.split, (args.split,))
    # Per-split overrides / resume are single-split only: a meta-split walks many out dirs, so a single
    # --pkl/--out-dir/--start-idx can't apply to each member.
    if (args.pkl or args.out_dir or args.start_idx is not None) and len(splits) != 1:
        raise SystemExit("--pkl / --out-dir / --start-idx require a single --split (not a meta-split)")

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    pose_cache = PoseCache.from_config(cfg)  # None unless pose.enabled

    for split in splits:
        if split == "train_aug":
            _build_train_aug(cfg, paths)
            continue
        pkl = Path(args.pkl) if args.pkl else paths.sequences_dir / f"sequences_{split}.pkl"
        out_dir = Path(args.out_dir) if args.out_dir else _out_dir_for(split, paths)
        _build_split(
            split, cfg, paths, pkl=pkl, out_dir=out_dir,
            start_idx=args.start_idx, keep_frames=args.keep_frames, pose_cache=pose_cache,
        )


if __name__ == "__main__":
    main()

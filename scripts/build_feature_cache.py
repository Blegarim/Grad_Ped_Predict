"""Build the recipe-v2 frozen-backbone feature cache (docs/RECIPE_V2_PLAN.md Task 6). Needs the LMDBs + a GPU.

Thin wrapper over :func:`pedpredict.data.feature_cache.build_chunk_cache`: decodes each chunk's context crops
once, runs the eval-mode timm backbone, and writes ``<paths.feature_cache_dir>/<LMDB dir>/<chunk>/``. Train
dirs (both protocols) get the flip + ``augment.cache_color_variants`` color-jitter variants; val/test dirs get
the clean variant only. Resumable: a chunk with a matching complete cache is skipped (``--overwrite`` rebuilds).

Usage (research PC):
    python scripts/build_feature_cache.py --split all
    python scripts/build_feature_cache.py --split val --limit-chunks 1      # timing smoke
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

from pedpredict.config import build_argparser, load_config, validate_config
from pedpredict.data.feature_cache import build_chunk_cache, verify_chunk_cache
from pedpredict.models.timm_backbone import TimmBackbone
from pedpredict.paths import resolve_paths
from pedpredict.training.chunk_loader import gather_lmdb_chunks
from pedpredict.utils.device import enable_perf_flags, get_device

_SPLITS = ("train", "val", "test", "train_benchmark", "val_benchmark", "test_benchmark")
# Gate on the 99.9th percentile of |cache - recompute|, not its max: the max is the worst of ~4e5
# AMP-noisy values and moved 5.4x between splits (6.3e-3 train, 3.4e-2 train_benchmark) on
# 2026-09-17 while the mean moved 11% and the inputs matched exactly. Observed p99.9: 3.7e-3 and
# 8.2e-3, so 2e-2 is ~2.4x headroom. A real preprocessing mismatch shifts the whole distribution
# to O(1), so this stays far more sensitive than a max threshold loose enough not to false-alarm.
_PARITY_TOL = 2e-2


def _split_dirs(cfg, split: str) -> tuple[list[Path], bool]:
    """``(LMDB dirs, augmented?)`` for one split name."""
    p = resolve_paths(cfg.paths)
    table = {
        "train": (list(p.lmdb_train), True),
        "val": ([p.lmdb_val], False),
        "test": ([p.lmdb_test], False),
        "train_benchmark": (list(p.lmdb_train_benchmark), True),
        "val_benchmark": ([p.lmdb_val_benchmark], False),
        "test_benchmark": ([p.lmdb_test_benchmark], False),
    }
    return table[split]


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--split", action="append", choices=[*_SPLITS, "all"], required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--sub-batch", type=int, default=512, help="Frames per backbone forward.")
    parser.add_argument("--limit-chunks", type=int, default=0, help="Stop after N chunks per split (timing smoke).")
    parser.add_argument(
        "--verify-windows", type=int, default=0,
        help="After each split, recompute N windows of its first chunk on the image path and fail if the "
             f"99.9th percentile of the difference exceeds {_PARITY_TOL} (real-data parity).",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)
    validate_config(cfg)
    if cfg.model.vit_backbone == "legacy":
        parser.error("the feature cache needs a timm backbone (model.vit_backbone != legacy)")

    device = get_device()
    enable_perf_flags(device)
    net = TimmBackbone.from_config(cfg.model, cfg.data.read_context_height).net.eval().to(device)
    splits = _SPLITS if "all" in args.split else tuple(dict.fromkeys(args.split))
    for split in splits:
        dirs, augmented = _split_dirs(cfg, split)
        chunks = gather_lmdb_chunks(dirs)
        if args.limit_chunks:
            chunks = chunks[: args.limit_chunks]
        start, built = time.time(), 0
        for i, chunk in enumerate(chunks, 1):
            t0 = time.time()
            out, did_build = build_chunk_cache(
                chunk, cfg, net, augmented=augmented, device=device, batch_windows=args.batch_windows,
                num_workers=args.num_workers, sub_batch=args.sub_batch, overwrite=args.overwrite,
            )
            built += did_build
            state = f"built in {time.time() - t0:.0f}s" if did_build else "up to date"
            print(f"[feature_cache] {split} {i}/{len(chunks)} {Path(chunk).name}: {state} -> {out}", flush=True)
        print(f"[feature_cache] {split}: {built} built, {len(chunks) - built} skipped, {time.time() - start:.0f}s")
        if args.verify_windows:
            stats = verify_chunk_cache(chunks[0], cfg, net, device=device, windows=args.verify_windows)
            verdict = "OK" if stats.p999 <= _PARITY_TOL else "MISMATCH"
            print(f"[feature_cache] {split} parity over {args.verify_windows} windows: {stats} {verdict}",
                  flush=True)
            if stats.p999 > _PARITY_TOL:
                return 1
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())

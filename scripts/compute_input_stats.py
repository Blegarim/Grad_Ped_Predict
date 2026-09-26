"""Compute the per-channel mean/std that ``pose.input_stats`` standardizes the pose read path with.

Thin wrapper over :mod:`pedpredict.data.input_stats`. Scans the training dirs (``paths.lmdb_train`` after
overrides — pass the same ``--set`` flags the run will train with) and writes a JSON. Metadata only; no
image is decoded. Needs a pose build, so pass the pose bundle:

    python scripts/compute_input_stats.py --set pose.enabled=true --set model.motion_norm=none \\
        --set data.motion_dim=58 --set model.motion_dim=58 \\
        --set "paths.lmdb_train=[preprocessed_train_shuffled]" \\
        --out outputs/input_stats/pose58_train_shuffled.json

Then train with ``--set pose.input_stats=outputs/input_stats/pose58_train_shuffled.json``. The numbers
are copied into the run's resolved config at load time, which is what eval inherits.
"""

from __future__ import annotations

import json
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.input_stats import compute_input_stats
from pedpredict.paths import resolve_paths


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--out", required=True, help="Output JSON path (parent dirs are created).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    if not cfg.pose.enabled:
        raise SystemExit("compute_input_stats needs a pose build — pass --set pose.enabled=true (+ bundle)")
    dirs = [Path(p) for p in resolve_paths(cfg.paths).lmdb_train]
    print(f"[input_stats] scanning {', '.join(d.name for d in dirs)} (metas only)...", flush=True)
    stats = compute_input_stats(dirs, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    std = stats["std"]
    print(f"[input_stats] {stats['n_windows']} windows, {stats['n_frames']} frames, width {stats['width']}, "
          f"{stats['nonfinite']} non-finite | std min {min(std):.2e} max {max(std):.2e} -> {out}")


if __name__ == "__main__":
    main()

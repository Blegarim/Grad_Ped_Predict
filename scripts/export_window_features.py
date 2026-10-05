#!/usr/bin/env python
"""Export one or more LMDB dirs' pixel-free model inputs + labels to a single ``.npz`` (research PC).

Thin wrapper over :func:`pedpredict.eval.window_features.export_windows`. Forces the ``pose_kinematics``
read path (pose on, 58-dim vector, no image decode) with standardization OFF — the export is raw by
contract. Only ``_meta`` records are read, so it runs in minutes and works on ``*_metaonly`` dirs.

    python scripts/export_window_features.py --lmdb preprocessed_test --out outputs/features/pose58/test.npz \\
        --check-dump outputs/diagnostics/<run>/onset_test.npz

``--lmdb`` dirs resolve against the project root like ``configs/paths.yaml`` entries. ``--check-dump``
refuses to keep an export whose labels/``track_id`` do not match that dump row for row.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from pedpredict.config import build_argparser, load_config, validate_config
from pedpredict.eval.onset_timing import load_dump, save_dump
from pedpredict.eval.window_features import channel_names, check_against_dump, export_windows
from pedpredict.paths import find_project_root
from pedpredict.training.chunk_loader import gather_lmdb_chunks

#: The pose_kinematics read path, raw (no pose.input_stats).
_PIXEL_FREE = ("eval.model_type=pose_kinematics", "pose.enabled=true", "model.motion_norm=none",
               "data.motion_dim=58", "model.motion_dim=58", "data.visual_input=none")


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--lmdb", nargs="+", required=True, help="LMDB dir(s), project-root-relative or absolute.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--check-dump", default="", help="Prediction dump the export must match row for row.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, [*_PIXEL_FREE, *args.overrides])
    validate_config(cfg)
    root = find_project_root()
    dirs = [str(p if Path(p).is_absolute() else root / p) for p in args.lmdb]
    chunks = gather_lmdb_chunks(dirs)
    start = time.time()
    arrays = export_windows(cfg, chunks, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"[export] {len(arrays['crosses'])} windows from {len(chunks)} chunks in {time.time() - start:.0f}s; "
          f"keys {sorted(arrays)}")
    if args.check_dump:
        dump, _ = load_dump(args.check_dump)
        problems = check_against_dump(arrays, dump)
        if problems:
            print(f"[export] REFUSED: does not match {args.check_dump}: {problems}")
            return 1
        print(f"[export] matches {args.check_dump} row for row")
    meta = {
        "lmdb_dirs": list(args.lmdb), "chunks": [Path(c).name for c in chunks], "n": int(len(arrays["crosses"])),
        "channels": list(channel_names(cfg.pose)), "seq_len": int(arrays["X"].shape[1]),
        "source_width": cfg.data.source_width, "source_height": cfg.data.source_height,
        "ego_speed_scale": cfg.model.ego_speed_scale,
        "pose": {"include_arms": cfg.pose.include_arms, "conf_channel": cfg.pose.conf_channel},
        "standardized": False, "git_commit": _git_commit(root), "overrides": list(args.overrides),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = save_dump(args.out, arrays, meta)
    print(f"[export] -> {path} ({path.stat().st_size / 1e6:.0f} MB); meta {json.dumps(meta)[:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())

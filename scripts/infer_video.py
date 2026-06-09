"""Video inference entry point — thin CLI over ``eval.inference.run_video_inference``.

Mirrors ``scripts/evaluate.py`` (reuses ``--config-dir`` / ``--set`` dotted overrides). The source may be
a video file OR a directory of frames (folds OLD ``extract_frames.py`` into ``DirFrameSource``).

Usage:
    python scripts/infer_video.py --video test_clip.mp4 \
        --checkpoint outputs/runs/<id>/checkpoints/best.pth --out out.mp4
    python scripts/infer_video.py --video qualitative_visualize/ --checkpoint <path> --out out.mp4
"""

from __future__ import annotations

import multiprocessing as mp
import sys

from pedpredict.config import build_argparser, load_config
from pedpredict.eval.inference import run_video_inference
from pedpredict.utils.device import get_device


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--video", required=True, help="Video file OR directory of frame images.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to run.")
    parser.add_argument("--out", default=None, help="Output annotated mp4 (skip rendering if omitted).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, args.overrides)
    res = run_video_inference(
        cfg, video=args.video, checkpoint=args.checkpoint, out_video=args.out, device=get_device()
    )
    print(
        f"Inference complete: {res.n_windows} windows over {res.n_frames} frames "
        f"({cfg.eval.model_type}) -> {res.out_video}"
    )
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())

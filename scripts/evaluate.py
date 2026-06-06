"""Evaluation entry point (Prompt 5.1) — thin CLI over ``eval.run_evaluation``.

Ports OLD ``test.py``'s ``main()`` argument surface (``--model_type`` -> ``--set eval.model_type``,
``--model_path`` -> ``--checkpoint``, ``--save_predictions`` -> ``--save-predictions``). Efficiency
metrics (FLOPs/latency/FPS) come from 5.2's ``benchmark`` behind ``--benchmark``.

Usage:
    python scripts/evaluate.py --set eval.model_type=full \
        --checkpoint outputs/runs/<run_id>/checkpoints/best.pth
    python scripts/evaluate.py --set eval.model_type=motion_only \
        --checkpoint <path> --save-predictions --save-temporal-weights
"""

from __future__ import annotations

import multiprocessing as mp
import sys

from pedpredict.config import build_argparser, load_config
from pedpredict.eval.evaluate import run_evaluation
from pedpredict.utils.device import get_device


def main(argv=None) -> int:
    parser = build_argparser()                       # reuse --config-dir / --set dotted overrides
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint to evaluate.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--save-predictions", action="store_true", help="Write plots/predictions.npz.")
    parser.add_argument(
        "--save-temporal-weights", action="store_true",
        help="Write plots/temporal_weights.npz (full model only).",
    )
    parser.add_argument("--benchmark", action="store_true", help="Also run 5.2 efficiency metrics.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)
    device = get_device()

    efficiency = None
    if args.benchmark:                               # optional 5.2 hook (absent until 5.2 lands)
        from pedpredict.eval.benchmark import measure_efficiency
        efficiency = measure_efficiency(cfg, device=device)

    report = run_evaluation(
        cfg,
        checkpoint=args.checkpoint,
        device=device,
        split=args.split,
        save_predictions=args.save_predictions,
        save_temporal_weights=args.save_temporal_weights,
        efficiency=efficiency,
    )
    m = report.artifacts.metrics
    print(f"Eval ({args.split}, {cfg.eval.model_type}) — {report.artifacts.n_samples} samples")
    print(f"  crosses: f1={m.per_task['crosses'].f1:.4f} auc={m.per_task['crosses'].auc:.4f}")
    print(f"  macro_f1={m.macro_f1:.4f} overall_acc={m.overall_accuracy:.4f}")
    print(f"  -> {report.eval_log_path}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())

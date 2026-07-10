"""Evaluation entry point — thin CLI over ``eval.run_evaluation``.

Ports OLD ``test.py``'s ``main()`` argument surface (``--model_type`` -> ``--set eval.model_type``,
``--model_path`` -> ``--checkpoint``, ``--save_predictions`` -> ``--save-predictions``). Efficiency
metrics (FLOPs/latency/FPS) come from 5.2's ``benchmark`` behind ``--benchmark``.

Threshold protocol (M2): run ``--split val`` FIRST — it sweeps per-task decision thresholds on val
and stores them in the run dir; the ``--split test`` pass then loads and applies them (``tuned_*``
columns). Test-swept numbers are logged as ``oracle_*`` only and must never be reported.

Model architecture is INHERITED from the checkpoint's own ``resolved_config.yaml`` (``<run_dir>/`` two
parents up from the checkpoint): ``eval.model_type`` + the ``model``/``pose`` bundle + ``data.motion_dim``
come from the run that trained the weights, so you never re-type the arch to eval a ``pose_full`` /
timm-backbone / non-``full`` checkpoint (mismatching it silently built ``full`` and the strict load failed
on a wall of key mismatches). Runtime choices (``data.protocol``, ``eval.batch_size``, paths, split) stay
with the caller, and an explicit ``--set model.X`` still wins. ``--no-config-inherit`` restores the legacy
"build purely from --config-dir + --set" behavior.

Usage:
    python scripts/evaluate.py --split val \
        --checkpoint outputs/runs/<run_id>/checkpoints/best.pth      # tune + store thresholds
    python scripts/evaluate.py --split test \
        --checkpoint outputs/runs/<run_id>/checkpoints/best.pth      # report at frozen thresholds
    python scripts/evaluate.py --split test --set data.protocol=anchored \
        --checkpoint outputs/runs/<pose_full_run>/checkpoints/best.pth  # arch inherited; no --set model.*
    python scripts/evaluate.py --set eval.model_type=ped_local --no-config-inherit \
        --checkpoint <path> --save-predictions --save-temporal-weights
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config, merge_eval_config, validate_config
from pedpredict.eval.evaluate import run_evaluation
from pedpredict.utils.device import get_device
from pedpredict.utils.seed import set_seed

#: Checkpoint layout is ``<run_dir>/checkpoints/<name>.pth`` (RunDir); the run's architecture snapshot
#: sits at ``<run_dir>/resolved_config.yaml`` (dump_config). Two parents up from the checkpoint.
_RESOLVED_CONFIG_NAME = "resolved_config.yaml"


def _checkpoint_run_config(checkpoint: str) -> Path | None:
    """The checkpoint's run-dir ``resolved_config.yaml`` if it exists (``<ckpt>/../../``), else None."""
    candidate = Path(checkpoint).resolve().parent.parent / _RESOLVED_CONFIG_NAME
    return candidate if candidate.exists() else None


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
    parser.add_argument(
        "--no-config-inherit", action="store_true",
        help="Do NOT inherit the model architecture from the checkpoint's resolved_config.yaml; build the "
             "model purely from --config-dir + --set (the legacy behavior).",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)

    # OQ7: build the model from the CHECKPOINT's architecture, not the ambient config defaults. The run dir
    # already snapshots the exact model_type / backbone / motion_dim / pose bundle the weights were trained
    # under; inheriting it means eval builds a matching module (otherwise a pose_full timm checkpoint is
    # rebuilt as `full` and the strict load fails on a wall of key mismatches). Explicit --set still wins.
    ckpt_config = None if args.no_config_inherit else _checkpoint_run_config(args.checkpoint)
    if ckpt_config is not None:
        cfg = merge_eval_config(cfg, ckpt_config, args.overrides)
        validate_config(cfg)        # the merge narrows a validated cfg; re-check the spliced architecture
        print(f"[evaluate] inherited model architecture from {ckpt_config} (model_type={cfg.eval.model_type}).")
    elif not args.no_config_inherit:
        print(
            f"[evaluate] WARNING: no {_RESOLVED_CONFIG_NAME} beside {args.checkpoint} — building the model "
            "from --config-dir + --set only. Pass --set eval.model_type=... and the full arch bundle if this "
            "checkpoint is not a plain `full` model, or the strict weight load will fail."
        )

    set_seed(cfg.train.seed)        # M7: symmetric with train.py (eval is deterministic anyway)
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

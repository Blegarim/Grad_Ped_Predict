"""Quantitative figure generation (Prompt 6.1) — thin CLI over ``viz.plots``.

Re-points OLD ``scripts/plot_results.py`` at the new run-dir artifacts (4.5 / 5.1). Per-run figures
(loss / F1 / PR / threshold / temporal) come from a resolved ``RunDir``; cross-run ablation bars come
from one ``eval_log.csv`` per model type (or auto-discovered from ``index.csv``).

Usage:
    # per-run figures (newest run under outputs/runs/ by default)
    python scripts/visualize.py
    python scripts/visualize.py --run-id 20260607_120000_full
    python scripts/visualize.py --run-dir outputs/runs/<run_id> --only loss,f1

    # cross-run ablation bars (written to outputs/runs/ablation/ by default)
    python scripts/visualize.py --ablation full=outputs/runs/A/eval_log.csv,motion_only=.../eval_log.csv
    python scripts/visualize.py --ablation-from-index
"""

from __future__ import annotations

import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.paths import resolve_paths
from pedpredict.utils.logging import RunDir
from pedpredict.viz.plots import (
    ablation_logs_from_index,
    generate_ablation_figures,
    generate_run_figures,
)

_ABLATION_SUBDIR = "ablation"


def _parse_ablation_arg(raw: str) -> dict[str, str]:
    """Parse ``name=path,name=path,...`` into a dict (OLD ``_parse_ablation_arg``)."""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            raise ValueError(f"Expected name=path, got: {pair}")
        name, path = pair.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _resolve_run_dir(runs_root: Path, *, run_dir: str | None, run_id: str | None) -> RunDir:
    """Resolve the target run from an explicit ``--run-dir`` / ``--run-id``, else the newest run dir."""
    if run_dir is not None:
        path = Path(run_dir)
        return RunDir(run_id=path.name, path=path)
    if run_id is not None:
        return RunDir(run_id=run_id, path=runs_root / run_id)
    candidates = [
        p for p in runs_root.iterdir() if p.is_dir() and p.name != _ABLATION_SUBDIR
    ] if runs_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs_root}.")
    newest = max(candidates, key=lambda p: p.name)  # run ids are timestamp-first -> name sorts by time
    return RunDir(run_id=newest.name, path=newest)


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()  # reuse --config-dir / --set dotted overrides
    parser.add_argument("--run-dir", default=None, help="Explicit run directory (overrides --run-id).")
    parser.add_argument("--run-id", default=None, help="Run id under paths.runs_dir.")
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated subset of {loss,f1,pr,threshold,temporal} (default: all available).",
    )
    parser.add_argument("--ablation", default=None, help="Comma-separated model_type=eval_log.csv pairs.")
    parser.add_argument(
        "--ablation-from-index", action="store_true",
        help="Discover ablation eval logs from index.csv (latest eval per model type).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Ablation output dir (default: paths.runs_dir/ablation). Per-run figures always go to the "
        "run's plots dir.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, args.overrides)
    runs_root = resolve_paths(cfg.paths).runs_dir
    generated: list[Path] = []

    # Cross-run ablation bars (independent of any single run).
    eval_logs: dict[str, str | Path] = {}
    if args.ablation:
        eval_logs.update(_parse_ablation_arg(args.ablation))
    if args.ablation_from_index:
        eval_logs.update(ablation_logs_from_index(runs_root))
    if eval_logs:
        out_dir = Path(args.out_dir) if args.out_dir else runs_root / _ABLATION_SUBDIR
        generated += generate_ablation_figures(eval_logs, out_dir)

    # Per-run figures (skipped if the invocation was ablation-only).
    if not (args.ablation or args.ablation_from_index) or args.run_dir or args.run_id:
        run = _resolve_run_dir(runs_root, run_dir=args.run_dir, run_id=args.run_id)
        which = [w.strip() for w in args.only.split(",")] if args.only else None
        generated += generate_run_figures(run, which=which)

    if not generated:
        print("No artifacts found to plot. Run training/eval first, or pass --ablation.")
        return 0
    print(f"Generated {len(generated)} figure(s):")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

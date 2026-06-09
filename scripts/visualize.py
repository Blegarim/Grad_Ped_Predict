"""Quantitative + qualitative figure generation (Prompts 6.1 + 6.2) — thin CLI over viz modules.

Re-points OLD ``scripts/plot_results.py`` at the new run-dir artifacts (4.5 / 5.1). Per-run figures
(loss / F1 / PR / threshold / temporal) come from a resolved ``RunDir``; cross-run ablation bars come
from one ``eval_log.csv`` per model type (or auto-discovered from ``index.csv``).

Qualitative MP4 overlays (6.2): pass ``--qualitative`` to generate GT / comparison / attention videos
in ``<run_dir>/plots/``.  Requires a ``--sequences-pkl`` pointing at the test-split pkl produced by
``scripts/make_sequences.py`` for the GT and comparison modes.

Usage:
    # per-run quantitative figures (newest run under outputs/runs/ by default)
    python scripts/visualize.py
    python scripts/visualize.py --run-id 20260607_120000_full
    python scripts/visualize.py --run-dir outputs/runs/<run_id> --only loss,f1

    # cross-run ablation bars (written to outputs/runs/ablation/ by default)
    python scripts/visualize.py --ablation full=outputs/runs/A/eval_log.csv,motion_only=.../eval_log.csv
    python scripts/visualize.py --ablation-from-index

    # qualitative videos (GT + comparison + attention)
    python scripts/visualize.py --qualitative --sequences-pkl data/sequences/sequences_test.pkl
    python scripts/visualize.py --qualitative attention --run-id 20260607_120000_full
    python scripts/visualize.py --qualitative comparison --sequences-pkl .../sequences_test.pkl \\
        --comparison-mode diff --max-sequences 30
"""

from __future__ import annotations

import pickle
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


def _load_sequences_pkl(pkl_path: str | None) -> list[dict] | None:
    """Load a sequences pickle produced by ``scripts/make_sequences.py``; return None if not given."""
    if pkl_path is None:
        return None
    path = Path(pkl_path)
    if not path.exists():
        print(f"Warning: sequences pkl not found: {path}; qualitative GT/comparison modes skipped.")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


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
    # ---- qualitative (6.2) ----
    parser.add_argument(
        "--qualitative",
        nargs="?",
        const="all",
        metavar="MODE",
        help=(
            "Generate qualitative MP4 video(s) in <run_dir>/plots/. "
            "MODE: gt, comparison, attention, or all (default). "
            "GT and comparison modes require --sequences-pkl."
        ),
    )
    parser.add_argument(
        "--sequences-pkl",
        default=None,
        metavar="PATH",
        help="Path to sequences .pkl (from scripts/make_sequences.py) required for "
             "--qualitative gt and --qualitative comparison.",
    )
    parser.add_argument(
        "--comparison-mode",
        default="both",
        choices=["gt", "pred", "both", "diff"],
        help="Comparison render variant used with --qualitative comparison (default: both).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=20,
        help="Max sequences to render per qualitative video (default: 20).",
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

    # Resolve target run (needed for per-run quant figures AND qualitative).
    need_run = (
        not (args.ablation or args.ablation_from_index)
        or args.run_dir
        or args.run_id
        or args.qualitative
    )
    run: RunDir | None = None
    if need_run:
        run = _resolve_run_dir(runs_root, run_dir=args.run_dir, run_id=args.run_id)

    # Per-run quantitative figures.
    if run is not None and not args.qualitative:
        which = [w.strip() for w in args.only.split(",")] if args.only else None
        generated += generate_run_figures(run, which=which)

    # Qualitative videos (6.2).
    if args.qualitative and run is not None:
        from pedpredict.viz.qualitative import ComparisonMode, generate_qualitative_figures

        qual_mode = None if args.qualitative == "all" else args.qualitative
        which_qual = None if qual_mode is None else [qual_mode]
        sequences = _load_sequences_pkl(args.sequences_pkl)

        if qual_mode == "comparison":
            from pedpredict.viz.qualitative import render_comparison
            import numpy as np

            preds_path = run.plots_dir / "predictions.npz"
            if sequences is not None and preds_path.exists():
                raw  = np.load(preds_path)
                preds = {k: raw[k] for k in raw.files}
                out  = render_comparison(
                    sequences, preds, cfg.data,
                    run.plots_dir / f"qualitative_comparison_{args.comparison_mode}.mp4",
                    mode=ComparisonMode(args.comparison_mode),
                    fps=5,
                )
                generated.append(out)
            else:
                generated += generate_qualitative_figures(
                    run, cfg, which=which_qual, sequences=sequences,
                    max_sequences=args.max_sequences, fps=5,
                )
        else:
            generated += generate_qualitative_figures(
                run, cfg, which=which_qual, sequences=sequences,
                max_sequences=args.max_sequences, fps=5,
            )

    if not generated:
        print("No artifacts found to plot. Run training/eval first, or pass --ablation.")
        return 0
    print(f"Generated {len(generated)} figure(s):")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Go/no-go gate for the pixel-free re-run campaign (``docs/RERUN_PLAN_2026-09-26.md`` §2).

Reads the binary arm's seeds — each run dir's ``train_log.csv`` + ``eval_log.csv``, and its streaming-test
dump at ``outputs/diagnostics/<run dir name>/onset_test.npz`` — and writes ``verdict.json`` + ``verdict.md``.

    python scripts/rerun_gate.py --out outputs/diagnostics/rerun_gate \\
        --run outputs/runs/<..>_pf_fix_s42 --run outputs/runs/<..>_pf_fix_s43 --run outputs/runs/<..>_pf_fix_s44

Exit code: 0 = GO, 1 = NO-GO, 2 = an input is missing (not ready, or a ladder job failed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from pedpredict.eval import detection_curve as dc
from pedpredict.eval.onset_timing import load_dump
from pedpredict.eval.rerun_gate import (
    Check,
    GateThresholds,
    curve_checks,
    read_train_log,
    seed_checks,
    selected_epoch_auc,
    streaming_test_auc,
)


def _selection_metric(run: Path) -> str:
    """The run's own ``train.selection_metric`` (what picked its best.pth); crosses_f1 if unrecorded."""
    snapshot = run / "resolved_config.yaml"
    if not snapshot.exists():
        return "crosses_f1"
    return yaml.safe_load(snapshot.read_text(encoding="utf-8")).get("train", {}).get("selection_metric", "crosses_f1")


def _rate_at(points: list[dc.DetectionPoint], budget: float) -> float:
    return next(p.detection_rate for p in points if p.budget_per_hour == budget)


def _gather(runs: list[Path], dumps_root: Path, th: GateThresholds) -> tuple[list[Check], list[str]]:
    """All L1–L5 checks, plus the list of missing inputs (non-empty -> exit 2)."""
    missing = [str(p) for r in runs for p in (r / "train_log.csv", r / "eval_log.csv",
                                             dumps_root / r.name / "onset_test.npz") if not p.exists()]
    if missing:
        return [], missing
    checks: list[Check] = []
    val_aucs, test_aucs, spread, level = [], [], [], []
    for run in runs:
        rows = read_train_log(run / "train_log.csv")
        checks += curve_checks(run.name, rows, th)
        val_aucs.append(selected_epoch_auc(rows, _selection_metric(run)))
        test_aucs.append(streaming_test_auc(run / "eval_log.csv"))
        arrays, _ = load_dump(dumps_root / run.name / "onset_test.npz")
        points = dc.detection_curve(arrays, "p_frame", budgets=(int(th.spread_budget), int(th.level_budget)))
        spread.append(_rate_at(points, th.spread_budget))
        level.append(_rate_at(points, th.level_budget))
    checks += seed_checks(val_aucs, test_aucs, spread, level, th)
    return checks, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="Binary-arm run dir (one per seed).")
    parser.add_argument("--dumps-root", default="outputs/diagnostics")
    parser.add_argument("--out", required=True, help="Directory for verdict.json + verdict.md.")
    args = parser.parse_args(argv)

    th = GateThresholds()
    checks, missing = _gather([Path(r) for r in args.run], Path(args.dumps_root), th)
    go = bool(checks) and all(c.passed for c in checks) and not missing
    verdict = "GO" if go else ("MISSING INPUTS" if missing else "NO-GO")
    lines = [f"# Re-run gate: **{verdict}**", "", *(f"- {'PASS' if c.passed else 'FAIL'} **{c.name}** — {c.detail}"
                                                 for c in checks), *(f"- missing: `{m}`" for m in missing)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "verdict.json").write_text(json.dumps({
        "verdict": verdict, "runs": args.run, "thresholds": asdict(th),
        "checks": [asdict(c) for c in checks], "missing": missing,
    }, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0 if go else (2 if missing else 1)


if __name__ == "__main__":
    sys.exit(main())

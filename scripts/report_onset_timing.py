"""Timing report from prediction dumps — runs anywhere (no data, no checkpoint, no GPU).

Thin wrapper over :func:`pedpredict.eval.onset_timing.timing_report`. ``--test`` is the dump to score;
``--val`` (optional) adds F1 at thresholds tuned on val, the same grid and tie-break ``evaluate.py`` uses.
``--eval-log`` (optional) checks the dump against the run's stored ``eval_log.csv`` row for the same
protocol + split: the binary head's AUC must match, and with ``--val`` its tuned F1 too. A mismatch means
the dump is not the model the stored numbers came from — the script exits nonzero.

Usage:
    python scripts/report_onset_timing.py --test onset_test.npz --val onset_val.npz \\
        --eval-log outputs/runs/<run>/eval_log.csv --out-dir outputs/diagnostics/<run>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pedpredict.eval import onset_timing as ot

_TOL = 1e-3


def _stored_row(eval_log: Path, meta: dict) -> dict[str, str] | None:
    name = Path(str(meta.get("checkpoint", ""))).name
    rows = [
        r for r in csv.DictReader(eval_log.open(encoding="utf-8"))
        if r.get("protocol") == meta.get("protocol") and r.get("split") == meta.get("split")
        and r.get("checkpoint", "").endswith(name)
    ]
    return rows[-1] if rows else None


def _parity(report: dict, stored: dict[str, str] | None) -> dict[str, object]:
    if stored is None:
        return {"checked": False, "reason": "no matching eval_log row"}
    checks = {"auc": (report["stored_label_auc"]["crosses_frame"], float(stored["crosses_auc"]))}
    tuned = report.get("tuned_f1_stored_label", {}).get("p_frame")
    if tuned is not None and stored.get("tuned_crosses_f1"):
        checks["tuned_f1"] = (tuned["f1"], float(stored["tuned_crosses_f1"]))
    diffs = {k: abs(a - b) for k, (a, b) in checks.items()}
    return {"checked": True, "passed": all(d <= _TOL for d in diffs.values()), "abs_diff": diffs}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="Dump to score (usually the test split).")
    parser.add_argument("--val", default="", help="Val dump — enables F1 at val-tuned thresholds.")
    parser.add_argument("--eval-log", default="", help="The run's eval_log.csv, for the parity check.")
    parser.add_argument("--horizons", default="16,32,64,96")
    parser.add_argument("--out-dir", default="", help="Write report.json + report.md here (else print only).")
    args = parser.parse_args(argv)

    test, meta = ot.load_dump(args.test)
    val = ot.load_dump(args.val)[0] if args.val else None
    horizons = tuple(int(h) for h in args.horizons.split(","))
    report = ot.timing_report(test, meta, val=val, horizons=horizons)
    if args.eval_log:
        report["parity_with_eval_log"] = _parity(report, _stored_row(Path(args.eval_log), meta))

    text = ot.format_report(report)
    if "parity_with_eval_log" in report:
        text += f"\nParity with eval_log.csv: `{json.dumps(report['parity_with_eval_log'])}`\n"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"timing_{meta.get('split', 'dump')}"
        (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / f"{stem}.md").write_text(text, encoding="utf-8")
    parity = report.get("parity_with_eval_log", {})
    return 1 if parity.get("checked") and not parity.get("passed") else 0


if __name__ == "__main__":
    sys.exit(main())

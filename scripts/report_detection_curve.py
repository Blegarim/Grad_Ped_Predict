"""Detection-latency curves from prediction dumps — runs anywhere (no data, no checkpoint, no GPU).

Thin wrapper over :mod:`pedpredict.eval.detection_curve`, the project's primary metric. Each ``--arm`` is
one arm to score, as ``label=path/to/onset_test.npz``; repeat the flag to compare arms in one table.
Repeat the SAME label to pool seeds — the arm is then reported as mean +- sample sd across them, which is
the pre-registered multi-seed form (docs/SEED_PLAN_2026-09-21.md).

**Pick the head per arm**, by appending ``#p_frame`` or ``#p_readout`` to that arm's path: ``p_readout``
is an onset model's horizon read-out, ``p_frame`` the binary crossing head. This matters — an onset arm's
dump carries *both*, and scoring a binary baseline by the read-out its objective never supervised (or an
onset arm by its degenerate frame head) silently compares the wrong things. Without a suffix, ``--score``
applies; its default ``auto`` prefers ``p_readout`` and falls back to ``p_frame``.

Usage:
    python scripts/report_detection_curve.py \\
        --arm "binary=outputs/diagnostics/r2s_s42/onset_test.npz#p_frame" \\
        --arm "binary=outputs/diagnostics/r2s_s43/onset_test.npz#p_frame" \\
        --arm "hazard=outputs/diagnostics/r3/onset_test.npz#p_readout" \\
        --out-dir outputs/diagnostics/detection
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from pedpredict.eval import detection_curve as dc
from pedpredict.eval.onset_timing import load_dump


def _pick_score(arrays: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    return "p_readout" if "p_readout" in arrays else "p_frame"


def _parse_arms(specs: list[str]) -> OrderedDict[str, list[tuple[str, str]]]:
    """``label=path[#score]`` -> ``{label: [(path, score_or_empty), ...]}``, preserving order."""
    arms: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for spec in specs:
        label, sep, rest = spec.partition("=")
        if not sep or not rest:
            raise SystemExit(f"--arm must be label=path[#score], got {spec!r}")
        path, _, score = rest.partition("#")
        if score and score not in ("p_readout", "p_frame"):
            raise SystemExit(f"--arm score must be p_readout or p_frame, got {score!r}")
        arms.setdefault(label, []).append((path, score))
    return arms


def _summary_table(results: dict[str, dict], budgets: tuple[int, ...]) -> str:
    """One row per budget, one column per arm: detection rate @ mean lead, with sd when seeds were pooled."""
    labels = list(results)
    lines = [
        "## Detection rate @ mean lead time, at matched false-alarm budgets",
        "",
        "| FA/hr | " + " | ".join(f"{k} (n={results[k]['n_seeds']})" for k in labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for i, budget in enumerate(budgets):
        cells = []
        for label in labels:
            row = results[label]["aggregate"][i]
            rate, sd = row["detection_rate_mean"] * 100, row["detection_rate_sd"] * 100
            lead = row["mean_lead_s_mean"]
            cells.append(f"{rate:.1f}% @ {lead:.2f} s" if np.isnan(sd) else f"{rate:.1f}±{sd:.1f}% @ {lead:.2f} s")
        lines.append(f"| {budget} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=DUMP",
                        help="Arm to score; repeat the same LABEL to pool seeds.")
    parser.add_argument("--score", default="auto", choices=("auto", "p_readout", "p_frame"))
    parser.add_argument("--budgets", default=",".join(str(b) for b in dc.DEFAULT_BUDGETS),
                        help="False-alarm budgets in alarms/hour.")
    parser.add_argument("--accounting", default="per_window", choices=("per_window", "per_track"),
                        help="per_window counts alarming windows; per_track counts nuisance pedestrians.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=3, help="Frames between windows (data.stride).")
    parser.add_argument("--out-dir", default="", help="Write curves.json + curves.md here (else print only).")
    args = parser.parse_args(argv)

    budgets = tuple(int(b) for b in args.budgets.split(","))
    clock = dc.DecisionClock(fps=args.fps, stride=args.stride)
    results: dict[str, dict] = {}
    text = ""

    for label, entries in _parse_arms(args.arm).items():
        curves, sources = [], []
        for path, per_arm_score in entries:
            arrays, meta = load_dump(path)
            score_key = _pick_score(arrays, per_arm_score or args.score)
            if score_key not in arrays:
                raise SystemExit(f"{path}: no {score_key!r} in dump (has {sorted(arrays)})")
            curves.append(dc.detection_curve(arrays, score_key, budgets=budgets,
                                             accounting=args.accounting, clock=clock))
            sources.append({"dump": path, "score": score_key, "checkpoint": meta.get("checkpoint", "?")})
            text += dc.format_curve(curves[-1], title=f"{label} — {Path(path).parent.name} ({score_key})") + "\n"
        results[label] = {
            "n_seeds": len(curves), "sources": sources,
            "aggregate": dc.aggregate_curves(curves),
            "curves": [[p.as_dict() for p in c] for c in curves],
        }

    text = _summary_table(results, budgets) + "\n" + text
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"accounting": args.accounting, "budgets": list(budgets),
                   "decisions_per_hour": clock.decisions_per_hour, "arms": results}
        (out / "curves.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out / "curves.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

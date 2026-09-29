#!/usr/bin/env python
"""Anchored "who" x streaming "when": the test pre-registered in outputs/diagnostics/combo_test/PREREG.md.

Combines, window by window, an anchored-trained model's score with a streaming-trained model's score, from
existing dumps only (no model is loaded). Seeds are paired (anchored ``c4_r2a_sN`` with ``c4_r2s_sN`` /
``c4_r3_sN``); only the fitted variant S2 reads the streaming-val dumps, and only to fit.

    python scripts/report_combination.py          # -> outputs/diagnostics/combo_test/{results.md, results.json}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pedpredict.eval import detection_curve as dc
from pedpredict.eval.campaign_report import (
    StackedCombiner,
    assert_aligned,
    compare_curves,
    intent_timing,
    mean_sd,
    smoothed_scores,
)
from pedpredict.eval.onset_timing import load_dump

SEEDS = (42, 43, 44)
BASELINE = "R2 alone (baseline)"
PRIMARY = "PRIMARY: anchored × R2"
ACCOUNTINGS = ("per_window", "per_track")


def dump(root: Path, stem: str, seed: int, split: str) -> dict[str, np.ndarray]:
    found = sorted(root.glob(f"*_{stem}_s{seed}/onset_{split}.npz"))
    if not found:
        raise SystemExit(f"no {split} dump for {stem}_s{seed} under {root}")
    return load_dump(found[-1])[0]


def seed_scores(root: Path, seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The test arrays (window metadata) and every pre-registered score for one paired seed."""
    anc, r2, r3 = (dump(root, s, seed, "test") for s in ("c4_r2a", "c4_r2s", "c4_r3"))
    v_anc, v_r2 = dump(root, "c4_r2a", seed, "val"), dump(root, "c4_r2s", seed, "val")
    assert_aligned(anc, r2)
    assert_aligned(anc, r3)
    assert_aligned(v_anc, v_r2)
    p_anc, p_r2, p_r3 = anc["p_frame"], r2["p_frame"], r3["p_readout"]
    stack = StackedCombiner().fit(v_anc["p_frame"], v_r2["p_frame"], v_anc["crosses"])
    scores = {
        BASELINE: p_r2,
        "anchored alone": p_anc,
        "R3 alone": p_r3,
        PRIMARY: p_anc * p_r2,
        "S1: causal intent (running mean of anchored) × R2": smoothed_scores(anc, p_anc, k=10**9) * p_r2,
        "S2: stacked on val (anchored, R2)": stack.score(p_anc, p_r2),
        "S3: anchored × R3": p_anc * p_r3,
    }
    return anc, scores


def evaluate(arrays: dict[str, np.ndarray], score: np.ndarray) -> dict[str, object]:
    arr = {**arrays, "combo": np.asarray(score, dtype=float)}
    return {
        "auc": intent_timing(arr, arr["combo"])["window"],
        **{acc: [p.as_dict() for p in dc.detection_curve(arr, "combo", accounting=acc)] for acc in ACCOUNTINGS},
    }


def fmt(values: list[float], digits: int = 1, scale: float = 100.0) -> str:
    m, s = mean_sd([v * scale for v in values])
    return f"{m:.{digits}f}" if np.isnan(s) else f"{m:.{digits}f} ± {s:.{digits}f}"


def render(results: dict[str, list[dict]]) -> str:
    names = list(results)
    budgets = [p["budget_per_hour"] for p in results[BASELINE][0]["per_window"]]
    lines = ["# Anchored who × streaming when — results (pre-registered: PREREG.md)", "",
             "Detection rate % (`per_window`, streaming test, mean ± sd over 3 paired seeds) and window AUC.", "",
             "| score | AUC | " + " | ".join(f"@{b:g}/hr" for b in budgets) + " |", "|---|---|" + "---|" * len(budgets)]
    for name in names:
        seeds = results[name]
        cells = " | ".join(fmt([s["per_window"][i]["detection_rate"] for s in seeds]) for i in range(len(budgets)))
        lines.append(f"| {name} | {fmt([s['auc'] for s in seeds], 3, 1.0)} | {cells} |")
    lines += ["", "## Pre-registered criterion vs R2 alone (beats R2 beyond the sd sum at >= 3 of 5 budgets)", ""]
    for name in names:
        if name == BASELINE:
            continue
        for acc in ACCOUNTINGS:
            verdict, _ = compare_curves([s[acc] for s in results[name]], [s[acc] for s in results[BASELINE]],
                                        arm_name=name, base_name=BASELINE)
            tag = " **(PRIMARY)**" if name == PRIMARY and acc == "per_window" else ""
            lines.append(f"- {name}, `{acc}`{tag}: **{verdict}**")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dumps", default="outputs/diagnostics")
    parser.add_argument("--out", default="outputs/diagnostics/combo_test")
    args = parser.parse_args(argv)
    results: dict[str, list[dict]] = {}
    for seed in SEEDS:
        arrays, scores = seed_scores(Path(args.dumps), seed)
        for name, score in scores.items():
            results.setdefault(name, []).append({"seed": seed, **evaluate(arrays, score)})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = render(results)
    (out / "results.md").write_text(text, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

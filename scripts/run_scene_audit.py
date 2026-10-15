#!/usr/bin/env python
"""Scene audit (``outputs/diagnostics/scene_audit/PREREG.md``): what information decides crossing onset?

Research PC (needs PIE annotations + the sequence files + the window exports):

    python scripts/run_scene_audit.py features   # gate: window <-> frame match; then scene_{split}.npz
    python scripts/run_scene_audit.py run        # tree arms base, R, T, S, RTS, H, E, RTS_H, RTS_E, RTS_HE
    python scripts/run_scene_audit.py report     # -> outputs/diagnostics/scene_audit/{report.md,results.json}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from pedpredict.eval import tree_study as ts
from pedpredict.eval.detection_curve import DEFAULT_BUDGETS
from pedpredict.eval.onset_timing import load_dump, save_dump
from pedpredict.eval.paired_bootstrap import paired_contrast
from pedpredict.eval.scene_features import (
    SCENE_GROUPS,
    check_alignment,
    load_attributes,
    match_windows,
    parse_video,
    scene_features,
    window_refs,
)
from pedpredict.eval.tree_study_report import N_BOOT, arm_summary, verdict

_PIE = Path("data")
_EXPORTS = Path("outputs/features/pose58")
_OUT = Path("outputs/diagnostics/scene_audit")
_TREE = Path("outputs/diagnostics/tree_study")
_SPLITS = ("train", "val", "test")
_ARMS = tuple(ts.ArmSpec(name, cues=cues, weighting_from="B", tags=("scene",)) for name, cues in (
    ("base", "all"), ("RTS", "all+R+T+S"), ("R", "all+R"), ("T", "all+T"), ("S", "all+S"), ("H", "all+H"),
    ("E", "all+E"), ("RTS_H", "all+R+T+S+H"), ("RTS_E", "all+R+T+S+E"), ("RTS_HE", "all+R+T+S+H+E")))
_SPEC = {a.name: a for a in _ARMS}


def cmd_features(args) -> int:
    attributes = load_attributes(_PIE / "annotations_attributes")
    scenes: dict[tuple[str, str], object] = {}
    summary = {}
    for split in _SPLITS:
        records = pickle.load(open(_PIE / "sequences" / f"sequences_{split}.pkl", "rb"))
        export, _ = load_dump(_EXPORTS / f"{split}.npz")
        perm = match_windows(window_refs(records), export)   # content match; the LMDB order is not the pkl's
        records = [records[j] for j in perm]
        refs = window_refs(records)
        problems = check_alignment(refs, export)
        if problems:
            print(f"[scene] {split}: export and sequence records DISAGREE: {problems} - stopping")
            return 1
        for key in {(r.set_id, r.video_id) for r in refs} - set(scenes):
            s, v = key
            scenes[key] = parse_video(_PIE / "annotations" / s / f"{v}_annt.xml",
                                      _PIE / "annotations_vehicle" / s / f"{v}_obd.xml")
        before = [(r["bboxes"][0][0] + r["bboxes"][0][2]) / 2.0 for r in records]
        feats = scene_features(refs, scenes, attributes, before_cx=before)
        meta = {"split": split, "n": len(refs), "groups": {g: list(n) for g, n in SCENE_GROUPS.items()},
                "aligned_with": str(_EXPORTS / f"{split}.npz"), "created_utc": time.strftime("%FT%TZ", time.gmtime())}
        save_dump(_EXPORTS / f"scene_{split}.npz", feats, meta)
        summary[split] = {"n": len(refs), "regular_light_visible": float((feats["r_n_lights"] > 0).mean()),
                          "pedestrian_light_visible": float(feats["p_visible"].mean()),
                          "crosswalk_visible": float(feats["s_cw_visible"].mean()),
                          "signalized_pedestrian": float((feats["s_signalized"] > 0).mean()),
                          "nan_share": {k: float(np.isnan(v).mean()) for k, v in feats.items()}}
        moved = int((perm != np.arange(len(perm))).sum())
        print(f"[scene] {split}: {len(refs)} windows matched ({moved} not in file order); "
              f"{json.dumps(summary[split])[:400]}")
    (_OUT / "features_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


def _load(split: str) -> ts.SplitData:
    base = ts.load_exports(_EXPORTS, [split])[split]
    scene, _ = load_dump(_EXPORTS / f"scene_{split}.npz")
    return base.with_groups({g: np.stack([scene[n] for n in names], axis=1) for g, names in SCENE_GROUPS.items()})


def cmd_run(args) -> int:
    data = {split: _load(split) for split in _SPLITS}
    chosen = {"B": json.loads((_TREE / "B" / "arm.json").read_text())["weighting"]}
    arms = [a for a in _ARMS if not args.arms or a.name in args.arms]
    for spec in arms:
        ts.run_arm(spec, data, _OUT, ts.TreeParams(), chosen)
        if spec.name == "base":   # gate: identical to tree-study B, seed for seed
            for seed in spec.seeds:
                new = load_dump(_OUT / "base" / f"s{seed}" / "onset_test.npz")[0]["p_frame"]
                old = load_dump(_TREE / "B" / f"s{seed}" / "onset_test.npz")[0]["p_frame"]
                if not np.array_equal(new, old):
                    print(f"[scene] GATE FAIL: base s{seed} differs from tree-study B (max {np.abs(new-old).max()})")
                    return 1
            print("[scene] gate: base reproduces tree-study B exactly at every seed")
    return 0


def _scores(name: str) -> list[np.ndarray]:
    spec = _SPEC[name]
    return [load_dump(_OUT / name / f"s{s}" / "onset_test.npz")[0]["p_frame"] for s in spec.seeds
            if (_OUT / name / f"s{s}" / "fit.json").exists()]


def _question(test: dict, a: str, b: str, n_boot: int) -> dict:
    arms = {a: _scores(a), b: _scores(b)}
    out = {"label": f"{b} - {a}"}
    for acc, budgets in (("per_window", DEFAULT_BUDGETS), ("per_track", DEFAULT_BUDGETS[1:])):
        est = paired_contrast(test, arms, {a: -1.0, b: 1.0}, budgets=budgets, accounting=acc, n_boot=n_boot)
        out[acc] = {"budgets": list(est.budgets), "rates": {k: v.tolist() for k, v in est.rates.items()},
                    "gap": est.gap.tolist(), "lo": est.lo.tolist(), "hi": est.hi.tolist(),
                    "verdict": verdict(est, a, b)}
    gap205 = out["per_window"]["gap"][list(DEFAULT_BUDGETS).index(205)]
    v = out["per_window"]["verdict"]
    out["reading"] = (("large" if gap205 >= 0.10 else "small") if v == f"{b} better"
                      else "none" if v.startswith("equivalent") else v)
    return out


def _subset(test: dict, mask_tracks: np.ndarray) -> dict:
    return {k: v[mask_tracks] for k, v in test.items()}


def cmd_report(args) -> int:
    test = {k: v for k, v in load_dump(_OUT / "base" / "s42" / "onset_test.npz")[0].items() if k != "p_frame"}
    scene, _ = load_dump(_EXPORTS / "scene_test.npz")
    results = {"arms": {a.name: arm_summary(_OUT, a) for a in _ARMS}, "questions": {}}
    q = results["questions"]
    q["Q1_RTS_vs_base"] = _question(test, "base", "RTS", args.n_boot)
    for g in ("R", "T", "S"):
        q[f"Q2_{g}_vs_base"] = _question(test, "base", g, args.n_boot)
    q["Q3_H_beyond_RTS"] = _question(test, "RTS", "RTS_H", args.n_boot)
    q["Q3_E_beyond_RTS"] = _question(test, "RTS", "RTS_E", args.n_boot)
    q["Q3_bound_RTS_HE_vs_base"] = _question(test, "base", "RTS_HE", args.n_boot)
    signalized = scene["s_signalized"] > 0
    for name, mask in (("signalized", signalized), ("unsignalized", ~signalized)):
        sub = _subset(test, mask)
        arms = {a: [s[mask] for s in _scores(a)] for a in ("base", "RTS")}
        est = paired_contrast(sub, arms, {"base": -1.0, "RTS": 1.0}, n_boot=args.n_boot)
        q[f"Q4_{name}"] = {"n_windows": int(mask.sum()), "rates": {k: v.tolist() for k, v in est.rates.items()},
                           "gap": est.gap.tolist(), "lo": est.lo.tolist(), "hi": est.hi.tolist(),
                           "verdict": verdict(est, "base", "RTS")}
    (_OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (_OUT / "report.md").write_text(_markdown(results), encoding="utf-8")
    print((_OUT / "report.md").read_text(encoding="utf-8"))
    return 0


def _markdown(results: dict) -> str:
    lines = ["# Scene audit - results", "", "Rules: PREREG.md (stamped 2026-10-06T05:46:08Z).", ""]
    for name, r in results["questions"].items():
        lines.append(f"## {name}: {r.get('label', 'base vs RTS')}" + (f" - reading: **{r['reading']}**"
                                                                       if "reading" in r else ""))
        blocks = [(acc, r[acc]) for acc in ("per_window", "per_track") if acc in r] or [("per_window", r)]
        for acc, b in blocks:
            arms = list(b["rates"])
            header = "| alarms/hr | " + " | ".join(arms) + " | gap | 95% CI |"
            lines += ["", f"`{acc}` - **{b['verdict']}**", "", header, "|---|" + "---|" * len(arms) + "---|---|"]
            for i, budget in enumerate(b.get("budgets", DEFAULT_BUDGETS)):
                lines.append(f"| {budget:g} | " + " | ".join(f"{100 * b['rates'][a][i]:.1f}" for a in arms)
                             + f" | {100 * b['gap'][i]:+.1f} | [{100 * b['lo'][i]:+.1f}, {100 * b['hi'][i]:+.1f}] |")
        lines.append("")
    lines += ["## Every arm (streaming test, mean over seeds)", "",
              "| arm | det @205 | det @41 | det @205 per_track | AUC | who | when "
              "| sep 0-1 s | 1-2 s | 2-3 s | 3-5 s | 5-10 s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in results["arms"].items():
        if s is None:
            continue
        pw = {r["budget_per_hour"]: r["detection_rate_mean"] for r in s["detection_per_window"]}
        pt = {r["budget_per_hour"]: r["detection_rate_mean"] for r in s["detection_per_track"]}
        sep = s["separability"]
        lines.append(f"| {name} | {100 * pw[205.0]:.1f} | {100 * pw[41.0]:.1f} | {100 * pt[205.0]:.1f} | "
                     f"{s['streaming']['auc'][0]:.3f} | {s['intent_timing']['who'][0]:.3f} | "
                     f"{s['intent_timing']['when'][0]:.3f} | "
                     + " | ".join(f"{sep[k][0]:.3f}" for k in ("0-1s", "1-2s", "2-3s", "3-5s", "5-10s")) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    sub.add_parser("run").add_argument("--arms", nargs="*", default=[])
    sub.add_parser("report").add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    return {"features": cmd_features, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

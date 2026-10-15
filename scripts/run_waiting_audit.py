#!/usr/bin/env python
"""Waiting audit (``outputs/diagnostics/waiting_audit/PREREG.md``): does behaviour history move "when"?

Research PC, after ``run_scene_audit.py features`` (reuses its scene features and window matching):

    python scripts/run_waiting_audit.py features   # history_{split}.npz (content-matched to the exports)
    python scripts/run_waiting_audit.py run        # arms base, W, WL, RTS, RTS_WL (5 seeds each)
    python scripts/run_waiting_audit.py report     # -> outputs/diagnostics/waiting_audit/{report.md,results.json}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pedpredict.eval import tree_study as ts
from pedpredict.eval.detection_curve import DEFAULT_BUDGETS, split_tracks
from pedpredict.eval.onset_timing import load_dump, save_dump
from pedpredict.eval.paired_bootstrap import paired_contrast
from pedpredict.eval.scene_features import (
    HISTORY_GROUPS,
    SCENE_GROUPS,
    check_alignment,
    history_features,
    match_windows,
    parse_pedestrians,
    window_refs,
)
from pedpredict.eval.tree_study_report import arm_summary, verdict

_PIE = Path("data")
_EXPORTS = Path("outputs/features/pose58")
_OUT = Path("outputs/diagnostics/waiting_audit")
_TREE = Path("outputs/diagnostics/tree_study")
_SCENE = Path("outputs/diagnostics/scene_audit")
_SPLITS = ("train", "val", "test")
_ARMS = tuple(ts.ArmSpec(n, cues=c, weighting_from="B", tags=("waiting",)) for n, c in (
    ("base", "all"), ("W", "all+W"), ("WL", "all+W+L"), ("RTS", "all+R+T+S"), ("RTS_WL", "all+R+T+S+W+L")))
_SPEC = {a.name: a for a in _ARMS}
_H = 32


def cmd_features(args) -> int:
    peds: dict = {}
    for split in _SPLITS:
        records = pickle.load(open(_PIE / "sequences" / f"sequences_{split}.pkl", "rb"))
        export, _ = load_dump(_EXPORTS / f"{split}.npz")
        refs = window_refs([records[j] for j in match_windows(window_refs(records), export)])
        if problems := check_alignment(refs, export):
            print(f"[wait] {split}: DISAGREE {problems}")
            return 1
        for key in {(r.set_id, r.video_id) for r in refs} - set(peds):
            peds[key] = parse_pedestrians(_PIE / "annotations" / key[0] / f"{key[1]}_annt.xml")
        feats = history_features(refs, peds)
        save_dump(_EXPORTS / f"history_{split}.npz", feats, {"split": split, "groups": HISTORY_GROUPS})
        print(f"[wait] {split}: {len(refs)} windows; standing at t {np.nanmean(feats['w_standing']):.2f}; "
              f"median standing bout {np.nanmedian(feats['w_stand_bout']):.0f} frames; "
              f"NaN share {np.mean([np.isnan(v).mean() for v in feats.values()]):.3f}")
    return 0


def _load(split: str) -> ts.SplitData:
    base = ts.load_exports(_EXPORTS, [split])[split]
    scene, _ = load_dump(_EXPORTS / f"scene_{split}.npz")
    hist, _ = load_dump(_EXPORTS / f"history_{split}.npz")
    groups = {g: np.stack([scene[n] for n in names], axis=1) for g, names in SCENE_GROUPS.items()}
    groups |= {g: np.stack([hist[n] for n in names], axis=1) for g, names in HISTORY_GROUPS.items()}
    return base.with_groups(groups)


def cmd_run(args) -> int:
    data = {split: _load(split) for split in _SPLITS}
    chosen = {"B": json.loads((_TREE / "B" / "arm.json").read_text())["weighting"]}
    for spec in _ARMS:
        ts.run_arm(spec, data, _OUT, ts.TreeParams(), chosen)
        ref = {"base": _TREE / "B", "RTS": _SCENE / "RTS"}.get(spec.name)
        if ref is not None:   # gate: the shared arms reproduce their earlier fits exactly
            for seed in spec.seeds:
                new = load_dump(_OUT / spec.name / f"s{seed}" / "onset_test.npz")[0]["p_frame"]
                old = load_dump(ref / f"s{seed}" / "onset_test.npz")[0]["p_frame"]
                if not np.array_equal(new, old):
                    print(f"[wait] GATE FAIL: {spec.name} s{seed} differs from {ref}")
                    return 1
            print(f"[wait] gate: {spec.name} reproduces {ref} exactly")
    return 0


def _scores(name: str) -> list[np.ndarray]:
    return [load_dump(_OUT / name / f"s{s}" / "onset_test.npz")[0]["p_frame"] for s in _SPEC[name].seeds]


def _when_contrast(test: dict, a: str, b: str, n_boot: int, seed: int = 0) -> dict:
    """Seed-averaged "when" AUC (crossers, crossing ahead: within 32 frames vs later) for b - a, with a 95%
    interval from resampling crossing pedestrians."""
    onset = test["onset_offset"]
    pos, _ = split_tracks(test)
    groups = [idx[onset[idx] >= 0] for idx in pos]
    y = (onset < _H).astype(int)
    sa, sb = _scores(a), _scores(b)

    def auc(index: np.ndarray, scores: list[np.ndarray]) -> float:
        return float(np.mean([roc_auc_score(y[index], s[index]) for s in scores]))

    full = np.concatenate(groups)
    point_a, point_b = auc(full, sa), auc(full, sb)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) == 2:
            diffs.append(auc(idx, sb) - auc(idx, sa))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    gain = point_b - point_a
    reading = ("moves latency" if gain >= 0.02 and lo > 0 else "small" if lo > 0 else "no")
    return {"label": f"{b} - {a}", "when_a": point_a, "when_b": point_b, "gain": gain, "lo": float(lo),
            "hi": float(hi), "reading": reading}


def _detection(test: dict, a: str, b: str, n_boot: int) -> dict:
    out = {}
    for acc, budgets in (("per_window", DEFAULT_BUDGETS), ("per_track", DEFAULT_BUDGETS[1:])):
        est = paired_contrast(test, {a: _scores(a), b: _scores(b)}, {a: -1.0, b: 1.0}, budgets=budgets,
                              accounting=acc, n_boot=n_boot)
        out[acc] = {"budgets": list(est.budgets), "rates": {k: v.tolist() for k, v in est.rates.items()},
                    "gap": est.gap.tolist(), "lo": est.lo.tolist(), "hi": est.hi.tolist(),
                    "verdict": verdict(est, a, b)}
    return out


def cmd_report(args) -> int:
    test = {k: v for k, v in load_dump(_OUT / "base" / "s42" / "onset_test.npz")[0].items() if k != "p_frame"}
    results = {"arms": {a.name: arm_summary(_OUT, a) for a in _ARMS},
               "when": {f"{b}-{a}": _when_contrast(test, a, b, args.n_boot_when)
                        for a, b in (("base", "W"), ("W", "WL"), ("RTS", "RTS_WL"))},
               "detection": {f"{b}-{a}": _detection(test, a, b, args.n_boot)
                             for a, b in (("base", "W"), ("RTS", "RTS_WL"))}}
    (_OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    lines = ["# Waiting audit - results", "", "Rules: PREREG.md (stamped 2026-10-06T07:51:57Z).", "",
             "## Primary: the 'when' (latency) factor", "",
             "| contrast | when (a) | when (b) | gain | 95% CI | reading |", "|---|---|---|---|---|---|"]
    for r in results["when"].values():
        lines.append(f"| {r['label']} | {r['when_a']:.3f} | {r['when_b']:.3f} | {r['gain']:+.3f} | "
                     f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | **{r['reading']}** |")
    for name, r in results["detection"].items():
        for acc in ("per_window", "per_track"):
            d = r[acc]
            lines += ["", f"## Detection {name} `{acc}` - **{d['verdict']}**", "",
                      "| alarms/hr | " + " | ".join(d["rates"]) + " | gap | 95% CI |", "|---|---|---|---|---|"]
            for i, budget in enumerate(d["budgets"]):
                lines.append(f"| {budget:g} | " + " | ".join(f"{100 * d['rates'][k][i]:.1f}" for k in d["rates"])
                             + f" | {100 * d['gap'][i]:+.1f} | [{100 * d['lo'][i]:+.1f}, {100 * d['hi'][i]:+.1f}] |")
    lines += ["", "## Every arm", "", "| arm | det @205 pw | det @205 pt | det @41 pt | AUC | who | when | sep 0-1 s |"
              " 1-2 s | 2-3 s | 3-5 s | 5-10 s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in results["arms"].items():
        pw = {r["budget_per_hour"]: r["detection_rate_mean"] for r in s["detection_per_window"]}
        pt = {r["budget_per_hour"]: r["detection_rate_mean"] for r in s["detection_per_track"]}
        sep = s["separability"]
        lines.append(f"| {name} | {100 * pw[205.0]:.1f} | {100 * pt[205.0]:.1f} | {100 * pt[41.0]:.1f} | "
                     f"{s['streaming']['auc'][0]:.3f} | {s['intent_timing']['who'][0]:.3f} | "
                     f"{s['intent_timing']['when'][0]:.3f} | "
                     + " | ".join(f"{sep[k][0]:.3f}" for k in ("0-1s", "1-2s", "2-3s", "3-5s", "5-10s")) + " |")
    (_OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((_OUT / "report.md").read_text(encoding="utf-8"))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features")
    sub.add_parser("run")
    rep = sub.add_parser("report")
    rep.add_argument("--n-boot", type=int, default=2000)
    rep.add_argument("--n-boot-when", type=int, default=1000)
    args = parser.parse_args(argv)
    _OUT.mkdir(parents=True, exist_ok=True)
    return {"features": cmd_features, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

"""Verdicts and tables for the tree study, from its dumps alone (``outputs/diagnostics/tree_study``).

Every rule here is the one ``PREREG.md`` states. A contrast is judged per false-alarm budget by the paired
pedestrian bootstrap (:func:`pedpredict.eval.paired_bootstrap.paired_contrast`, seeds resampled too) and a
comparison reaches a verdict only at **three or more of the five budgets**:

* ``"<b> better"`` / ``"<a> better"`` — the 95% interval excludes zero in that direction;
* ``"equivalent within 5 pp"`` — the whole interval lies inside +-5 points (the smallest effect of interest);
* ``"inconclusive"`` — anything else.

Window metrics (AUC, F1 at the validation-tuned threshold) and the timing analyses are context, as in the
deep campaign. Nothing here reads a model, a config or the data — only dumps.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score

from pedpredict.eval.detection_curve import DEFAULT_BUDGETS, aggregate_curves, detection_curve
from pedpredict.eval.onset_timing import load_dump
from pedpredict.eval.paired_bootstrap import GapEstimate, paired_contrast
from pedpredict.eval.tree_study import ARMS, ArmSpec, score_key

__all__ = [
    "MARGIN", "N_BOOT", "best_threshold", "window_metrics", "separability", "intent_timing", "verdict",
    "load_arm", "contrast", "build_results", "format_report",
]

MARGIN = 0.05          # smallest detection-rate difference of interest (5 points)
N_BOOT = 2000
_NEEDED = 3            # budgets (of five) a verdict needs
_GRID = np.round(np.arange(1, 1000) / 1000.0, 3)
_SEP_EDGES_S = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)

Arrays = dict[str, np.ndarray]


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def best_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray = _GRID) -> float:
    """F1-optimal threshold on a grid, ties to the lowest, 0.5 when nothing scores — ``_best_f1_threshold``'s
    rule, vectorized so a 999-point grid costs one sort."""
    y, p = np.asarray(y).astype(bool), np.asarray(p, dtype=np.float64)
    pos, every = np.sort(p[y]), np.sort(p)
    tp = len(pos) - np.searchsorted(pos, grid, side="left")
    flagged = len(every) - np.searchsorted(every, grid, side="left")
    f1 = np.where(flagged + len(pos) > 0, 2 * tp / np.maximum(flagged + len(pos), 1), 0.0)
    return 0.5 if f1.max() <= 0 else float(grid[int(np.argmax(f1))])


def window_metrics(test: Arrays, val: Arrays, key: str) -> dict[str, float]:
    """Test AUC, raw F1 (0.5) and F1 at the validation-tuned threshold, on the stored label."""
    y = test["crosses"].astype(int)
    threshold = best_threshold(val["crosses"], val[key])
    return {"auc": _auc(y, test[key]), "f1_raw": float(f1_score(y, test[key] >= 0.5, zero_division=0)),
            "f1_tuned": float(f1_score(y, test[key] >= threshold, zero_division=0)), "threshold": threshold}


def separability(arrays: Arrays, score: np.ndarray, fps: float = 30.0) -> dict[str, float]:
    """AUC of windows whose crossing starts ``lo..hi`` s ahead vs every never-crosser window (paper §Discussion)."""
    onset = np.asarray(arrays["onset_offset"])
    never = np.asarray(arrays["track_crosses"]) == 0
    out = {}
    for lo, hi in zip(_SEP_EDGES_S, [*_SEP_EDGES_S[1:], math.inf], strict=True):
        pos = (onset >= 0) & (onset / fps >= lo) & (onset / fps < hi)
        both = pos | never
        out[f"{lo:g}-{hi:g}s" if math.isfinite(hi) else f">={lo:g}s"] = _auc(pos[both].astype(int), score[both])
    return out


def intent_timing(arrays: Arrays, score: np.ndarray, horizon: int = 32) -> dict[str, float]:
    """*who* (per pedestrian mean score up to the crossing vs never-crossers) and *when* (crossing still
    ahead: within ``horizon`` vs later) — the definitions behind the paper's who/when table."""
    onset, ever = np.asarray(arrays["onset_offset"]), np.asarray(arrays["track_crosses"])
    keep = (onset >= 0) | (ever == 0)
    tracks, inverse = np.unique(np.asarray(arrays["track_id"])[keep], return_inverse=True)
    mean = np.bincount(inverse, weights=score[keep]) / np.bincount(inverse)
    label = np.zeros(len(tracks), dtype=int)
    label[inverse] = ever[keep]
    ahead = onset >= 0
    return {"who": _auc(label, mean), "when": _auc((onset[ahead] < horizon).astype(int), score[ahead])}


def verdict(est: GapEstimate, a: str, b: str) -> str:
    """The pre-registered rule for ``b - a``: a direction at >= 3 budgets, else equivalence, else nothing."""
    sign = est.excludes_zero()
    if (sign == 1).sum() >= _NEEDED:
        return f"{b} better"
    if (sign == -1).sum() >= _NEEDED:
        return f"{a} better"
    if est.within(MARGIN).sum() >= _NEEDED:
        return f"equivalent within {100 * MARGIN:g} pp"
    return "inconclusive"


# --------------------------------------------------------------------------- loading


def load_arm(out_dir: Path, spec: ArmSpec, filename: str = "onset_test.npz") -> list[Arrays]:
    """Every seed's dump of one arm (seed order), possibly empty."""
    dumps = []
    for seed in spec.seeds:
        path = out_dir / spec.name / f"s{seed}" / filename
        if path.exists() and (path.parent / "fit.json").exists():
            dumps.append(load_dump(path)[0])
    return dumps


def _scores(out_dir: Path, name: str) -> list[np.ndarray]:
    spec = _spec(name)
    return [d[score_key(spec)] for d in load_arm(out_dir, spec)]


def _spec(name: str) -> ArmSpec:
    return next(a for a in ARMS if a.name == name)


def contrast(test: Arrays, arms: Mapping[str, Sequence[np.ndarray]], weights: Mapping[str, float],
             accounting: str, n_boot: int = N_BOOT) -> dict[str, object]:
    est = paired_contrast(test, arms, weights, accounting=accounting, n_boot=n_boot)
    return {"budgets": list(est.budgets), "rates": {k: v.tolist() for k, v in est.rates.items()},
            "gap": est.gap.tolist(), "lo": est.lo.tolist(), "hi": est.hi.tolist(),
            "sign": est.excludes_zero().tolist(), "within_margin": est.within(MARGIN).tolist(), "_est": est}


# --------------------------------------------------------------------------- per-arm summaries


def _mean_sd(values: Sequence[float]) -> list[float]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return [math.nan, math.nan]
    return [statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else math.nan]


def arm_summary(out_dir: Path, spec: ArmSpec) -> dict[str, object] | None:
    """Detection curves (both rules), window metrics on both protocols, timing analyses; mean +- sd."""
    tests, vals = load_arm(out_dir, spec), load_arm(out_dir, spec, "onset_val.npz")
    if not tests:
        return None
    key = score_key(spec)
    out: dict[str, object] = {"n_seeds": len(tests), "objective": spec.objective, "tags": list(spec.tags)}
    for acc in ("per_window", "per_track"):
        curves = [detection_curve(t, key, accounting=acc) for t in tests]
        out[f"detection_{acc}"] = [{k: v for k, v in row.items() if k != "detection_rate_values"}
                                   | {"values": row["detection_rate_values"]} for row in aggregate_curves(curves)]
    out["streaming"] = _metric_block([window_metrics(t, v, key) for t, v in zip(tests, vals, strict=True)])
    anc_t, anc_v = load_arm(out_dir, spec, "anchored_test.npz"), load_arm(out_dir, spec, "anchored_val.npz")
    if anc_t and len(anc_t) == len(anc_v):
        out["anchored"] = _metric_block([window_metrics(t, v, key) for t, v in zip(anc_t, anc_v, strict=True)])
    out["separability"] = _metric_block([separability(t, t[key]) for t in tests])
    out["intent_timing"] = _metric_block([intent_timing(t, t[key]) for t in tests])
    return out


def _metric_block(rows: Sequence[Mapping[str, float]]) -> dict[str, list[float]]:
    return {k: _mean_sd([r[k] for r in rows]) for k in rows[0]}


# --------------------------------------------------------------------------- the pre-registered questions


def _question(test: Arrays, out_dir: Path, a: str, b: str, *, extra: Mapping[str, float] | None = None,
             label: str = "", n_boot: int = N_BOOT) -> dict[str, object] | None:
    """``b - a`` (or a custom contrast via ``extra``) under both accountings, with its verdict."""
    weights = dict(extra) if extra else {a: -1.0, b: 1.0}
    arms = {name: _scores(out_dir, name) for name in weights}
    if any(not s for s in arms.values()):
        return None
    out: dict[str, object] = {"contrast": weights, "label": label or f"{b} - {a}"}
    for acc in ("per_window", "per_track"):
        res = contrast(test, arms, weights, acc, n_boot)
        res["verdict"] = verdict(res.pop("_est"), a, b)
        out[acc] = res
    return out


def _hazard_at(horizon: int) -> str:
    """The hazard arm of the horizon sweep at ``horizon`` (H = 32 is HzC: same geometry, same data)."""
    return "HzC" if horizon == 32 else f"Hz{horizon}"


def questions(test: Arrays, out_dir: Path, n_boot: int = N_BOOT) -> dict[str, dict | None]:
    """Every pre-registered contrast. ``Q1`` is the primary question; the rest are secondary."""
    did = {"HzC": -1.0, "Bd32": 1.0, "Hz160": 1.0, "Bd160": -1.0}

    def _q(*args, **kwargs):
        return _question(*args, n_boot=n_boot, **kwargs)

    return {
        "Q1_hazard_vs_binary": _q(test, out_dir, "B", "Hz"),
        "Q1b_hazard_censored_vs_binary": _q(test, out_dir, "B", "HzC"),
        "Q1c_censored_vs_not": _q(test, out_dir, "Hz", "HzC"),
        **{f"Q2_H{h}_hazard_vs_drop": _q(test, out_dir, f"Bd{h}", _hazard_at(h)) for h in (32, 64, 160)},
        **{f"Q2_H{h}_hazard_vs_zero": _q(test, out_dir, f"Bz{h}", _hazard_at(h)) for h in (32, 64, 160)},
        "Q2_growth_160_vs_32": _q(test, out_dir, "H32 gap", "H160 gap", extra=did,
                                  label="(Hz160 - Bd160) - (HzC - Bd32)"),
        "Q3_drop_ego": _q(test, out_dir, "B", "B_no_ego"),
        "Q3b_hazard_vs_binary_without_ego": _q(test, out_dir, "B_no_ego", "Hz_no_ego"),
        "Q5_matched_size_vs_anchored": _q(test, out_dir, "A", "B_matched"),
        **{f"Q4_f{f:g}_hazard_vs_binary": _q(test, out_dir, f"B_f{f:g}", f"Hz_f{f:g}") for f in (0.125, 0.25, 0.5)},
    }


def gap_decomposition(out_dir: Path) -> dict[str, list[float]] | None:
    """The paper's G-split in tree form, seed-paired (A seed s with B seed s), streaming test F1."""
    a_test, a_val = load_arm(out_dir, _spec("A")), load_arm(out_dir, _spec("A"), "onset_val.npz")
    b_test, b_val = load_arm(out_dir, _spec("B")), load_arm(out_dir, _spec("B"), "onset_val.npz")
    if not (a_test and b_test) or len(a_test) != len(b_test):
        return None
    rows = []
    for at, av, bt, bv in zip(a_test, a_val, b_test, b_val, strict=True):
        a, b = window_metrics(at, av, "p_frame"), window_metrics(bt, bv, "p_frame")
        rows.append({"G_total": b["f1_tuned"] - a["f1_raw"], "G_prior": a["f1_tuned"] - a["f1_raw"],
                     "G_hardneg": b["f1_tuned"] - a["f1_tuned"]})
    return _metric_block(rows)


def deep_ensembles(diag_dir: Path, test: Arrays, n_boot: int = N_BOOT) -> dict[str, object] | None:
    """POST HOC (not pre-registered): rank-averaged 3-seed ensembles of the v4 deep arms, R3 - R2 paired."""
    def ensemble(tag: str, key: str) -> np.ndarray | None:
        paths = sorted(diag_dir.glob(f"*_c4_{tag}_s4[234]/onset_test.npz"))
        if len(paths) != 3:
            return None
        return np.mean([rankdata(load_dump(p)[0][key]) / len(test["crosses"]) for p in paths], axis=0)

    arms = {"R2": ensemble("r2s", "p_frame"), "R3": ensemble("r3", "p_readout"), "R3C": ensemble("r3c", "p_readout")}
    if any(v is None for v in arms.values()):
        return None
    out: dict[str, object] = {"note": "post hoc; one ensemble per arm, so only test-set noise enters the interval"}
    for b in ("R3", "R3C"):
        res = paired_contrast(test, {"R2": [arms["R2"]], b: [arms[b]]}, {"R2": -1.0, b: 1.0},
                              n_boot=n_boot, resample_seeds=False)
        out[f"{b} - R2"] = {"rates": {k: v.tolist() for k, v in res.rates.items()}, "gap": res.gap.tolist(),
                            "lo": res.lo.tolist(), "hi": res.hi.tolist(), "verdict": verdict(res, "R2", b)}
    return out


def build_results(out_dir: Path, diag_dir: Path | None = None, n_boot: int = N_BOOT) -> dict[str, object]:
    """Everything the report prints, as plain JSON-able data."""
    reference = next((d for spec in ARMS for d in load_arm(out_dir, spec)[:1]), None)
    if reference is None:
        raise FileNotFoundError(f"no finished tree-study fits under {out_dir}")
    test = {k: reference[k] for k in ("crosses", "track_id", "onset_offset", "future_observed", "track_crosses")}
    results: dict[str, object] = {
        "budgets": list(DEFAULT_BUDGETS), "margin": MARGIN, "n_boot": n_boot,
        "arms": {spec.name: arm_summary(out_dir, spec) for spec in ARMS},
        "questions": questions(test, out_dir, n_boot),
        "gap_decomposition": gap_decomposition(out_dir),
    }
    if diag_dir is not None:
        results["post_hoc_deep_ensembles"] = deep_ensembles(diag_dir, test, n_boot)
    return results


# --------------------------------------------------------------------------- markdown


def _pct(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def _q_table(name: str, q: dict | None) -> list[str]:
    if q is None:
        return [f"### {name}", "", "_not run yet_", ""]
    lines = [f"### {name}: {q['label']}", ""]
    for acc in ("per_window", "per_track"):
        r = q[acc]
        arms = list(r["rates"])
        lines += [f"`{acc}` — **{r['verdict']}**", "",
                  "| alarms/hr | " + " | ".join(arms) + " | contrast | 95% CI |",
                  "|---|" + "---|" * len(arms) + "---|---|"]
        for i, budget in enumerate(r["budgets"]):
            cells = " | ".join(_pct(r["rates"][a][i]) for a in arms)
            lines.append(f"| {budget:g} | {cells} | {100 * r['gap'][i]:+.1f} | "
                         f"[{100 * r['lo'][i]:+.1f}, {100 * r['hi'][i]:+.1f}] |")
        lines.append("")
    return lines


def _arm_table(arms: Mapping[str, dict | None]) -> list[str]:
    lines = ["| arm | seeds | det @205 (pw) | det @41 (pw) | det @205 (pt) | AUC | tuned F1 | who | when |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, s in arms.items():
        if s is None:
            continue
        pw = {r["budget_per_hour"]: r for r in s["detection_per_window"]}
        pt_rows = detection_rows_at(s["detection_per_track"], 205)
        lines.append(
            f"| {name} | {s['n_seeds']} | {_pm(pw[205.0])} | {_pm(pw[41.0])} | {pt_rows} | "
            f"{_pm2(s['streaming']['auc'])} | {_pm2(s['streaming']['f1_tuned'])} | "
            f"{_pm2(s['intent_timing']['who'])} | {_pm2(s['intent_timing']['when'])} |")
    return lines


def detection_rows_at(rows: Sequence[dict], budget: float) -> str:
    match = [r for r in rows if r["budget_per_hour"] == budget]
    return _pm(match[0]) if match else "–"


def _pm(row: Mapping[str, float]) -> str:
    sd = row["detection_rate_sd"]
    return f"{100 * row['detection_rate_mean']:.1f}" + ("" if math.isnan(sd) else f" ± {100 * sd:.1f}")


def _pm2(pair: Sequence[float]) -> str:
    mean, sd = pair
    return f"{mean:.3f}" + ("" if math.isnan(sd) else f" ± {sd:.3f}")


def format_report(results: Mapping[str, object]) -> str:
    """Markdown: verdicts first (primary, then secondary), then the per-arm table and the G-split."""
    lines = ["# Tree study — results", "",
             f"Paired pedestrian bootstrap, {results['n_boot']} resamples, seeds resampled; a verdict needs "
             f"{_NEEDED} of {len(results['budgets'])} budgets; margin {100 * results['margin']:g} pp. "
             "Rules: PREREG.md.", "", "## Pre-registered questions", ""]
    for name, q in results["questions"].items():
        lines += _q_table(name, q)
    lines += ["## Every arm (streaming test; mean ± sd over seeds)", "", *_arm_table(results["arms"]), ""]
    g = results.get("gap_decomposition")
    if g:
        lines += ["## Gap decomposition (tree twins, F1, streaming test)", "",
                  *[f"- {k}: {v[0]:+.3f} ± {v[1]:.3f}" for k, v in g.items()], ""]
    post = results.get("post_hoc_deep_ensembles")
    if post:
        lines += ["## POST HOC — deep 3-seed ensembles (not pre-registered)", "", post["note"], ""]
        for name, r in post.items():
            if name != "note":
                lines.append(f"- {name}: **{r['verdict']}**; gaps " + ", ".join(
                    f"{b:g}: {100 * g:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                    for b, g, lo, hi in zip(results["budgets"], r["gap"], r["lo"], r["hi"], strict=True)))
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, diag_dir: Path | None = None, n_boot: int = N_BOOT) -> tuple[Path, Path]:
    results = build_results(out_dir, diag_dir, n_boot)
    json_path, md_path = out_dir / "results.json", out_dir / "report.md"
    json_path.write_text(json.dumps(results, indent=1, default=float))
    md_path.write_text(format_report(results), encoding="utf-8")
    return json_path, md_path

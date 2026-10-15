"""Generate every number and table body in the paper from the repo's result files.

Nothing in the paper's results is typed by hand. This script reads the result files listed in SOURCES,
writes ``../generated/numbers.tex`` (one ``\\defV{key}{value}`` per number, each with its source) and the
table bodies ``../generated/tab_*.tex``. ``main.tex`` reads a number as ``\\V{key}``; an unknown key is a
compile error, so a renamed or missing number cannot pass silently.

Usage (from paper/figures/):
    <repo>/.venv/Scripts/python.exe make_tables.py          # write generated/
    <repo>/.venv/Scripts/python.exe make_tables.py --list   # also print every key = value
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
DIAG = ROOT / "outputs" / "diagnostics"
OUT = PAPER / "generated"

SOURCES = {
    "tree": DIAG / "tree_study_f64" / "results.json",          # quote this (float64 readout, Amendment 2)
    "tree_reg": DIAG / "tree_study" / "results.json",          # as registered, before the float32 fix
    "bz32": DIAG / "tree_study_f64" / "posthoc" / "bz32_vs_b.json",  # post hoc contrast
    "scene": DIAG / "scene_audit" / "results.json",
    "scene_feat": DIAG / "scene_audit" / "features_summary.json",
    "wait": DIAG / "waiting_audit" / "results.json",
    "v4": DIAG / "v4_report" / "tables" / "results.json",
    "v5_pw": DIAG / "v5_report" / "per_window" / "curves.json",
    "v5_pt": DIAG / "v5_report" / "per_track" / "curves.json",
    "comp": PAPER / "composition_report.json",
    "anc_test": DIAG / "tree_study" / "A" / "s42" / "anchored_test.npz",
    "onset_test": DIAG / "tree_study" / "B" / "s42" / "onset_test.npz",
    "intent_probe": DIAG / "scene_audit" / "posthoc" / "intent_probe.md",
    "fa_conc": DIAG / "scene_audit" / "posthoc" / "fa_concentration.md",
    "fa_who": DIAG / "scene_audit" / "posthoc" / "fa_who.md",
    "fa_irr": DIAG / "scene_audit" / "posthoc" / "fa_irrelevant.md",
    "dscene": DIAG / "deep_scene" / "report.json",
    "dscene_fa": DIAG / "deep_scene" / "posthoc" / "fa_concentration.md",
    "whendist": DIAG / "whowhen_posthoc" / "when_by_distance.json",
    "runs": ROOT / "outputs" / "runs",
}

BUDGETS = [1200, 460, 205, 95, 41]

# Fixed settings quoted in the text (value, source). Not results; kept here so every number has one home.
CONSTANTS = {
    "H": ("32", "configs/data.yaml future_offset 30 + tol 2"),
    "Hsec": ("1.07", "32 frames / 30 fps"),
    "T": ("20", "configs/data.yaml seq_len"),
    "stride": ("3", "configs/data.yaml stride"),
    "fps": ("30", "PIE frame rate"),
    "anc.tte.min": ("30", "configs/data.yaml benchmark_tte_min"),
    "anc.tte.max": ("60", "configs/data.yaml benchmark_tte_max"),
    "band.frames": ("64", "composition_report.json sweep band_frames"),
    "anc.nwin": ("six", "configs/data.yaml: windows end TTE 30..60 frames, step round(20 * (1 - 0.7)) = 6"),
    "anc.obs": ("20", "configs/data.yaml benchmark_obs_len"),
    "decisions": ("36\\,000", "stride 3 at 30 fps = 10 decisions/s per pedestrian-window stream"),
    "budgets": ("1200, 460, 205, 95 and 41", "eval/detection_curve.py budgets"),
    "n.boot": ("2\\,000", "tree_study PREREG.md sec. 6"),
    "verdict.k": ("3", "PREREG.md sec. 6: a direction at >= 3 of 5 budgets"),
    "margin": ("5", "PREREG.md sec. 6: equivalence margin, points"),
    "deep.dim": ("58", "pose_kinematics input: 49 pose + 9 box/ego channels"),
    "deep.pose": ("49", "data/pose.py"),
    "deep.kp": ("23", "COCO-WholeBody body-17 + feet-6"),
    "deep.box": ("9", "data/transforms.py MOTION_STORE_DIM"),
    "deep.dmodel": ("128", "ModelCfg d_model"),
    "deep.params": ("0.66", "v4 pose_kinematics parameter count, M"),
    "deep.clip": ("5", "pose.input_clip"),
    "deep.lr": ("10^{-4}", "v4 recipe"),
    "deep.wd": ("10^{-5}", "v4 recipe"),
    "deep.bs": ("32", "v4 recipe"),
    "deep.epochs": ("30", "v4 recipe, max epochs"),
    "deep.patience": ("15", "v4 recipe, early-stop patience"),
    "deep.seeds": ("3", "seeds 42-44"),
    "tree.seeds": ("5", "seeds 42-46"),
    "tree.feats": ("232", "58 channels x 4 summaries, PREREG.md sec. 4"),
    "tree.iter": ("300", "PREREG.md sec. 4"),
    "tree.leaves": ("31", "PREREG.md sec. 4"),
    "tree.minleaf": ("50", "PREREG.md sec. 4"),
    "tree.lr": ("0.1", "PREREG.md sec. 4"),
    "tree.maxfeat": ("0.5", "PREREG.md sec. 4"),
    "haz.L": ("96", "onset_lookahead"),
    "haz.w": ("4", "onset_bin_width"),
    "haz.K": ("24", "96 / 4"),
    "haz.KH": ("8", "32 / 4"),
    "censored.n": ("7\\,470", "CLAUDE.md: preprocessed_train_censored, 8.5% of train"),
    "censored.pct": ("8.5", "7470 / 88214"),
    "intent.cut": ("0.5", "intent_probe.md threshold"),
    "fa.windows205": ("212", "fa_concentration.md: budget @205/hr = 212 alarm windows"),
    "matched.n": ("4.9", "B_matched size = len(anchored train), thousands"),
    "focal.gamma": ("2", "PREREG.md sec. D"),
    "cb.beta": ("0.9999", "PREREG.md sec. D"),
    "d1.cut": ("0.60", "PREREG.md sec. D1 threshold"),
    "ego.oracle.f1": ("15", "scene PREREG: OBD speed at t+15"),
    "ego.oracle.f2": ("30", "scene PREREG: OBD speed at t+30"),
    "wait.boot": ("1\\,000", "waiting PREREG: resamples of crossing pedestrians"),
    "wait.small": ("0.02", "waiting PREREG H1 threshold"),
}

ENTRIES: list[tuple[str, str, str]] = []  # (key, value, source)


def put(key: str, value: str, source: str) -> None:
    if any(k == key for k, _, _ in ENTRIES):
        raise KeyError(f"duplicate key {key}")
    if not re.fullmatch(r"[A-Za-z0-9.\-]+", key):
        raise ValueError(f"bad key {key!r}")
    ENTRIES.append((key, value, source))


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def a3(x: float) -> str:
    return f"{x:.3f}"


def thousands(n: int) -> str:
    s = f"{n:,}"
    return s.replace(",", "\\,")


def signed(x: float, nd: int = 1) -> str:
    """A signed value for text mode; a rounded zero prints as +0.0, a negative with a proper minus sign."""
    r = round(x, nd)
    if r == 0:
        return f"{0:.{nd}f}"
    return f"+{r:.{nd}f}" if r > 0 else f"$-${-r:.{nd}f}"


QKEYS = {"Q1_hazard_vs_binary": "Q1", "Q1b_hazard_censored_vs_binary": "Q1b", "Q1c_censored_vs_not": "Q1c",
         "Q2_H32_hazard_vs_drop": "Q2drop32", "Q2_H64_hazard_vs_drop": "Q2drop64",
         "Q2_H160_hazard_vs_drop": "Q2drop160", "Q2_H32_hazard_vs_zero": "Q2zero32",
         "Q2_H64_hazard_vs_zero": "Q2zero64", "Q2_H160_hazard_vs_zero": "Q2zero160",
         "Q2_growth_160_vs_32": "Q2growth", "Q3_drop_ego": "Q3", "Q3b_hazard_vs_binary_without_ego": "Q3b",
         "Q5_matched_size_vs_anchored": "Q5"}


SEPKEY = {"0-1s": "0to1", "1-2s": "1to2", "2-3s": "2to3", "3-5s": "3to5", "5-10s": "5to10", ">=10s": "ge10"}


def qkey(name: str) -> str:
    return QKEYS.get(name, re.sub(r"[^A-Za-z0-9.]", "", name))


def load(name: str):
    return json.loads(SOURCES[name].read_text())


def bi(budget: int) -> int:
    return BUDGETS.index(budget)


# --------------------------------------------------------------------------- tree-style arms (tree, scene, wait)


def arm_numbers(prefix: str, arm: dict, src: str) -> None:
    """Every per-arm number in one family: detection (both accountings), AUC, who, when, separability."""
    for acc, tag in (("detection_per_window", "pw"), ("detection_per_track", "pt")):
        for row in arm[acc]:
            b = int(row["budget_per_hour"])
            put(f"{prefix}.det.{tag}.{b}", pct(row["detection_rate_mean"]), src)
            put(f"{prefix}.det.{tag}.{b}.sd", pct(row["detection_rate_sd"]), src)
            put(f"{prefix}.lead.{tag}.{b}", f"{row['mean_lead_s_mean']:.1f}", src)
    put(f"{prefix}.auc", a3(arm["streaming"]["auc"][0]), src)
    put(f"{prefix}.auc.sd", a3(arm["streaming"]["auc"][1]), src)
    put(f"{prefix}.f1raw", a3(arm["streaming"]["f1_raw"][0]), src)
    put(f"{prefix}.f1tuned", a3(arm["streaming"]["f1_tuned"][0]), src)
    put(f"{prefix}.who", a3(arm["intent_timing"]["who"][0]), src)
    put(f"{prefix}.who.sd", a3(arm["intent_timing"]["who"][1]), src)
    put(f"{prefix}.when", a3(arm["intent_timing"]["when"][0]), src)
    put(f"{prefix}.when.sd", a3(arm["intent_timing"]["when"][1]), src)
    put(f"{prefix}.seeds", str(arm["n_seeds"]), src)
    for b, v in arm["separability"].items():
        put(f"{prefix}.sep.{SEPKEY[b]}", a3(v[0]), src)
    if "anchored" in arm:
        for k in ("auc", "f1_raw", "f1_tuned"):
            put(f"{prefix}.anc.{k.replace('_', '')}", a3(arm["anchored"][k][0]), src)
            put(f"{prefix}.anc.{k.replace('_', '')}.sd", a3(arm["anchored"][k][1]), src)


def question_numbers(prefix: str, q: dict, src: str) -> None:
    """Gaps, intervals and verdict of one paired contrast (both accountings)."""
    for acc, tag in (("per_window", "pw"), ("per_track", "pt")):
        if acc not in q:  # e.g. the scene audit's descriptive Q4 rows carry one accounting only
            continue
        r = q[acc]
        put(f"{prefix}.{tag}.verdict", r.get("verdict", "descriptive"), src)
        gaps = [100 * g for g in r["gap"]]
        put(f"{prefix}.{tag}.gapmin", signed(min(gaps)), src)
        put(f"{prefix}.{tag}.gapmax", signed(max(gaps)), src)
        put(f"{prefix}.{tag}.absmin", f"{min(abs(g) for g in gaps):.1f}", src)
        put(f"{prefix}.{tag}.absmax", f"{max(abs(g) for g in gaps):.1f}", src)
        for i, b in enumerate(r["budgets"]):
            b = int(b)
            put(f"{prefix}.{tag}.gap.{b}", signed(100 * r["gap"][i]), src)
            put(f"{prefix}.{tag}.ci.{b}", f"[{signed(100 * r['lo'][i])}, {signed(100 * r['hi'][i])}]", src)
        n_excl = sum(1 for lo, hi in zip(r["lo"], r["hi"], strict=False) if lo > 0 or hi < 0)
        put(f"{prefix}.{tag}.nexcl", str(n_excl), src)
        n_within = sum(1 for lo, hi in zip(r["lo"], r["hi"], strict=False) if lo > -0.05 and hi < 0.05)
        put(f"{prefix}.{tag}.nwithin", str(n_within), src)
        put(f"{prefix}.{tag}.nbudgets", str(len(r["budgets"])), src)


def tree_numbers() -> None:
    t, src = load("tree"), "outputs/diagnostics/tree_study_f64/results.json"
    for name, arm in t["arms"].items():
        if arm is not None:
            arm_numbers(f"tree.{name.replace('_', '')}", arm, src)
    for qname, q in t["questions"].items():
        if q is not None:
            question_numbers(f"treeq.{qkey(qname)}", q, src)
    g = t["gap_decomposition"]
    for k in ("G_total", "G_prior", "G_hardneg"):
        put(f"tree.{k.replace('G_', 'g')}", signed(g[k][0], 3), src)
        put(f"tree.{k.replace('G_', 'g')}.sd", f"{g[k][1]:.3f}", src)
    reg = load("tree_reg")["questions"]["Q1_hazard_vs_binary"]["per_window"]["verdict"]
    put("treeq.Q1.registered.verdict", reg, "outputs/diagnostics/tree_study/results.json (before the fix)")
    # Q2 growth: hazard - drop gaps per horizon (per window), min..max over budgets.
    for h, hz in ((32, "HzC"), (64, "Hz64"), (160, "Hz160")):
        bd = "B" if h == 32 else f"Bd{h}"
        g = [100 * (a - b) for a, b in zip(rate(t, hz), rate(t, bd), strict=False)]
        put(f"tree.hzdrop.{h}.min", signed(min(g)), src)
        put(f"tree.hzdrop.{h}.max", signed(max(g)), src)
    b = load("bz32")
    question_numbers("treeq.Bz32B", b, "outputs/diagnostics/tree_study_f64/posthoc/bz32_vs_b.json (POST HOC)")


def rate(t: dict, arm: str, acc: str = "detection_per_window") -> list[float]:
    return [row["detection_rate_mean"] for row in t["arms"][arm][acc]]


def scene_numbers() -> None:
    s, src = load("scene"), "outputs/diagnostics/scene_audit/results.json"
    for name, arm in s["arms"].items():
        arm_numbers(f"scene.{name.replace('_', '')}", arm, src)
    for qname, q in s["questions"].items():
        question_numbers("sceneq." + qname.replace("_", ""), q, src)
    f, fsrc = load("scene_feat")["test"], "outputs/diagnostics/scene_audit/features_summary.json"
    for k, key in (("regular_light_visible", "light"), ("pedestrian_light_visible", "pedlight"),
                   ("crosswalk_visible", "crosswalk"), ("signalized_pedestrian", "signalised")):
        put(f"scene.cov.{key}", pct(f[k]), fsrc)


def deep_scene_numbers() -> None:
    """The deep scene arm (deep_scene/PREREG.md): the network given the scene audit's R+T+S features, 3 seeds."""
    r, src = load("dscene"), "outputs/diagnostics/deep_scene/report.json"
    for arm in ("base", "scene"):
        s = r["summary"][arm]
        for k in ("who", "when", "auc", "when_within", "when_band"):
            put(f"dscene.{arm}.{k.replace('_', '')}", a3(s[k][0]), src)
            put(f"dscene.{arm}.{k.replace('_', '')}.sd", a3(s[k][1]), src)
        for acc, tag in (("pw", "pw"), ("pt", "pt")):
            for b in BUDGETS:
                if f"{acc}_{b}" in s:
                    put(f"dscene.{arm}.det.{tag}.{b}", f"{s[f'{acc}_{b}'][0]:.1f}", src)
    v = r["verdicts"]
    put("dsceneq.who.gain", v["d_who"], src)
    put("dsceneq.when.gain", v["d_when"], src)
    for k in ("d_who", "d_when"):
        put(f"dsceneq.{k[2:]}.gain.abs", f"{abs(float(v[k])):.3f}", src)
    for k in ("Q1", "Q2"):
        put(f"dsceneq.{k}", v[k], src)
    for k, tag in (("Q3_pw", "pw"), ("Q3_pt", "pt")):
        put(f"dsceneq.Q3.{tag}", v[k].split(" (")[0], src)
    fc, fsrc = SOURCES["dscene_fa"], "outputs/diagnostics/deep_scene/posthoc/fa_concentration.md (POST HOC)"
    for arm, tag in (("deep base", "base"), ("deep \\+ scene", "scene")):
        m = re.search(rf"^{arm}\s+alarms from\s+([0-9.]+) pedestrians.*?top-3 hold\s+([0-9.]+) of", fc.read_text(),
                      re.M)
        if not m:
            raise RuntimeError(f"deep_scene fa_concentration: {arm}")
        put(f"ph.dscene.fa.{tag}.peds", f"{float(m.group(1)):.0f}", fsrc)
        put(f"ph.dscene.fa.{tag}.top3", f"{float(m.group(2)):.0f}", fsrc)


def wait_numbers() -> None:
    w, src = load("wait"), "outputs/diagnostics/waiting_audit/results.json"
    for name, arm in w["arms"].items():
        arm_numbers(f"wait.{name.replace('_', '')}", arm, src)
    for k, v in w["when"].items():
        key = "waitq." + k.replace("_", "").replace("-", "minus")
        put(f"{key}.gain", signed(v["gain"], 3), src)
        put(f"{key}.ci", f"[{signed(v['lo'], 3)}, {signed(v['hi'], 3)}]", src)
        put(f"{key}.reading", v["reading"], src)
    for k, v in w["detection"].items():
        key = "waitq." + k.replace("_", "").replace("-", "minus")
        for acc, tag in (("per_window", "pw"), ("per_track", "pt")):
            put(f"{key}.det.{tag}.verdict", v[acc]["verdict"], src)


# --------------------------------------------------------------------------- deep (v4, v5, c5 runs)


def _ms(vals: list[float]) -> tuple[float, float]:
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)


def deep_numbers() -> None:
    v, src = load("v4"), "outputs/diagnostics/v4_report/tables/results.json"
    names = {"R2 binary": "R2", "R3 pure hazard": "R3", "R1 auxiliary": "R1", "R4 hedge": "R4",
             "R3C censored": "R3C", "R2 anchored-trained": "R2anc", "Model A streaming-trained": "Astr",
             "Model A anchored-trained": "Aanc"}
    for full, short in names.items():
        recs = v["arms"][full]
        p = f"deep.{short}"
        put(f"{p}.seeds", str(len(recs)), src)
        for proto in ("streaming", "anchored"):
            pt = "str" if proto == "streaming" else "anc"
            for k, kk in (("crosses_auc", "auc"), ("crosses_f1", "f1raw"), ("tuned_crosses_f1", "f1tuned")):
                m, s = _ms([r[proto][k] for r in recs])
                put(f"{p}.{pt}.{kk}", a3(m), src)
                put(f"{p}.{pt}.{kk}.sd", a3(s), src)
        for acc, tag in (("per_window", "pw"), ("per_track", "pt")):
            for b in BUDGETS:
                at = [next(x for x in r["detection"][acc] if int(x["budget_per_hour"]) == b) for r in recs]
                m, s = _ms([x["detection_rate"] for x in at])
                put(f"{p}.det.{tag}.{b}", pct(m), src)
                put(f"{p}.det.{tag}.{b}.sd", pct(s), src)
                put(f"{p}.lead.{tag}.{b}", f"{_ms([x['mean_lead_s'] for x in at])[0]:.1f}", src)
                if len(recs) > 1:
                    put(f"{p}.det.{tag}.{b}.min", pct(min(x["detection_rate"] for x in at)), src)
                    put(f"{p}.det.{tag}.{b}.max", pct(max(x["detection_rate"] for x in at)), src)
        for k, kk in (("intent", "who"), ("timing", "when"), ("window", "wauc")):
            m, s = _ms([r["intent_timing"][k] for r in recs])
            put(f"{p}.{kk}", a3(m), src)
            put(f"{p}.{kk}.sd", a3(s), src)
        for b, sk in SEPKEY.items():
            put(f"{p}.sep.{sk}", a3(_ms([r["separability"][b]["auc"] for r in recs])[0]), src)
    for model, short in (("Model B", "B"), ("Model A", "A")):
        rows = v["gap"][model]
        for k in ("G_total", "G_prior", "G_hardneg"):
            m, s = _ms([r[k] for r in rows])
            put(f"deep.gap.{short}.{k.replace('G_', 'g')}", signed(m, 3), src)
            put(f"deep.gap.{short}.{k.replace('G_', 'g')}.sd", f"{s:.3f}", src)
    crit = v["criterion"]["R3 pure hazard | per_window"]
    put("deep.R3R2.verdict", crit["verdict"].lower(), src)
    put("deep.R3R2.nbeyond", str(sum(1 for r in crit["rows"] if r["beyond_sd"] != "neither")), src)

    # Loss-side arms (v5): detection curves and the run dirs' eval logs.
    for acc, key in (("pw", "v5_pw"), ("pt", "v5_pt")):
        c, csrc = load(key), f"outputs/diagnostics/v5_report/per_{'window' if acc == 'pw' else 'track'}/curves.json"
        for full, short in (("R2 focal", "focalstr"), ("R2 class-balanced", "cbstr"), ("anchored focal", "focalanc"),
                            ("anchored class-balanced", "cbanc")):
            for row in c["arms"][full]["aggregate"]:
                b = int(row["budget_per_hour"])
                put(f"deep.{short}.det.{acc}.{b}", pct(row["detection_rate_mean"]), csrc)
                put(f"deep.{short}.det.{acc}.{b}.sd", pct(row["detection_rate_sd"]), csrc)
    for tag, short in (("c5_r2a_focal", "focalanc"), ("c5_r2a_cb", "cbanc"), ("c5_r2s_focal", "focalstr"),
                       ("c5_r2s_cb", "cbstr")):
        aucs, gpri = [], []
        for d in sorted(SOURCES["runs"].glob(f"*_{tag}_s4*")):
            rows = list(csv.DictReader(open(d / "eval_log.csv")))
            r = [x for x in rows if x["protocol"] == "streaming" and x["split"] == "test"][-1]
            aucs.append(float(r["crosses_auc"]))
            gpri.append(float(r["tuned_crosses_f1"]) - float(r["crosses_f1"]))
        rsrc = f"outputs/runs/*_{tag}_s4*/eval_log.csv"
        if len(aucs) != 3:
            raise RuntimeError(f"{tag}: expected 3 runs, found {len(aucs)}")
        put(f"deep.{short}.auc", a3(mean(aucs)), rsrc)
        put(f"deep.{short}.auc.all", " / ".join(a3(a) for a in aucs), rsrc)
        put(f"deep.{short}.auc.max", a3(max(aucs)), rsrc)
        put(f"deep.{short}.gprior", signed(mean(gpri), 3), rsrc)


# --------------------------------------------------------------------------- data, composition, post hoc text


def data_numbers() -> None:
    comp, src = load("comp"), "paper/composition_report.json"
    for row in comp["composition"]:
        s = row["split"]
        tot = row["total"]
        put(f"data.{s}.n", thousands(tot), src)
        put(f"data.{s}.pos", pct(row["positive"] / tot), src)
        put(f"data.{s}.hard", pct(row["hard_temporal"] / tot), src)
        put(f"data.{s}.already", pct(row["already_crossed"] / tot), src)
        put(f"data.{s}.never", pct(row["genuine"] / tot), src)
        put(f"data.{s}.ratio", f"{row['imbalance']:.1f}", src)
        put(f"data.{s}.alreadyn", thousands(row["already_crossed"]), src)
    sw = comp["sweep"][0]
    put("data.band", thousands(sw["band"]), src)
    put("data.band.perpos", f"{sw['band_per_positive']:.2f}", src)
    sys.path.insert(0, str(HERE))
    from make_figures import NUMBERS as FIG  # the composition pass's time-to-onset buckets

    buckets = dict(FIG["time_to_onset"])
    total = sum(buckets.values())
    put("data.hard.beyond5", pct(buckets[">5.0"] / total),
        "paper/figures/make_figures.py NUMBERS time_to_onset (scripts/report_negative_composition.py)")
    anc = np.load(SOURCES["anc_test"], allow_pickle=True)["crosses"]
    put("data.anc.test.n", thousands(len(anc)), "tree_study/A/s42/anchored_test.npz")
    put("data.anc.ratio", f"{(len(anc) - anc.sum()) / anc.sum():.1f}", "tree_study/A/s42/anchored_test.npz")
    on = np.load(SOURCES["onset_test"], allow_pickle=True)
    tc, tid, _off = on["track_crosses"], on["track_id"], on["onset_offset"]
    peds = {}
    for t, c in zip(tid, tc, strict=False):
        peds[str(t)] = max(peds.get(str(t), 0), int(c))
    put("peds.cross", str(sum(peds.values())), "tree_study/B/s42/onset_test.npz")
    put("peds.never", str(len(peds) - sum(peds.values())), "tree_study/B/s42/onset_test.npz")


def _grab(path: Path, pattern: str, group: int = 1) -> str:
    m = re.search(pattern, path.read_text())
    if not m:
        raise RuntimeError(f"pattern not found in {path.name}: {pattern}")
    return m.group(group)


def posthoc_numbers() -> None:
    ip = SOURCES["intent_probe"]
    src = "outputs/diagnostics/scene_audit/posthoc/intent_probe.md (POST HOC)"
    put("ph.human.who", _grab(ip, r"HUMAN 'who' AUC.*?: ([0-9.]+)"), src)
    put("ph.never.hi", _grab(ip, r"intention >= 0.5: (\d+) of"), src)
    n_hi, n_never = int(_grab(ip, r"intention >= 0.5: (\d+) of")), int(_grab(ip, r"intention >= 0.5: \d+ of (\d+)"))
    put("ph.never.hi.pct", f"{100 * n_hi / n_never:.0f}", src)
    put("ph.med.cross", _grab(ip, r"crossers median ([0-9.]+)"), src)
    put("ph.med.never", _grab(ip, r"never-crossers median ([0-9.]+)"), src)
    put("ph.treeB.fa.hi205", _grab(ip, r"tree B s42 @205/hr: (\d+)%"), src)
    put("ph.deepR2.fa.hi205", _grab(ip, r"deep R2 s42 @205/hr: (\d+)%"), src)
    fc, src = SOURCES["fa_conc"], "outputs/diagnostics/scene_audit/posthoc/fa_concentration.md (POST HOC)"
    for arm in ("base", "S", "RTS", "RTS_HE"):
        m = re.search(rf"^{re.escape(arm)}\s+false-alarm windows @205 come from\s+([0-9.]+) pedestrians.*?top-3 "
                      rf"pedestrians hold\s+([0-9.]+) of", fc.read_text(), re.M)
        if not m:
            raise RuntimeError(f"fa_concentration: {arm}")
        put(f"ph.fa.{arm.replace('_', '')}.peds", m.group(1), src)
        put(f"ph.fa.{arm.replace('_', '')}.top3", m.group(2), src)
    fw, src = SOURCES["fa_who"], "outputs/diagnostics/scene_audit/posthoc/fa_who.md (POST HOC)"
    n = int(_grab(fw, r"all (\d+) alarming never-crossers"))
    share = float(_grab(fw, r"share signalized ([0-9.]+)"))
    put("ph.persist.n", str(n), src)
    put("ph.persist.signalised", str(round(share * n)), src)
    put("ph.persist.intent", f"{float(_grab(fw, r'median intention ([0-9.]+)')):.2f}", src)
    fi, src = SOURCES["fa_irr"], "outputs/diagnostics/scene_audit/posthoc/fa_irrelevant.md (POST HOC)"
    put("ph.irr.n", _grab(fi, r"of these, (\d+) physically cross"), src)
    shifts = []
    for arm in ("base", "RTS", "RTS_HE"):
        m = re.search(rf"^\s*{re.escape(arm)}\s*:.*?detection @205/@41 \[([0-9.]+), ([0-9.]+)\] -> \[([0-9.]+), "
                      rf"([0-9.]+)\]", fi.read_text(), re.M)
        shifts += [abs(float(m.group(3)) - float(m.group(1))), abs(float(m.group(4)) - float(m.group(2)))]
    put("ph.irr.maxshift", f"{max(shifts):.1f}", src)


# --------------------------------------------------------------------------- tables


def V(key: str) -> str:
    """The value of a key, for table cells (tables are generated, so cells hold values, not macros)."""
    for k, v, _ in ENTRIES:
        if k == key:
            return v
    raise KeyError(key)


def pm(key: str) -> str:
    return f"${V(key)} \\pm {V(key + '.sd')}$"


def tab_composition() -> str:
    rows = []
    for s in ("train", "val", "test"):
        rows.append(f"    {s} & {V(f'data.{s}.pos')} & {V(f'data.{s}.hard')} & {V(f'data.{s}.already')} & "
                    f"{V(f'data.{s}.never')} \\\\")
    return "\n".join(rows) + "\n"


def tab_matrix() -> str:
    def cell(p: str) -> str:
        return f"{V(p + '.auc')} $\\cdot$ {V(p + '.f1raw')} $\\rightarrow$ {V(p + '.f1tuned')}"

    # The three-task network rows (deep.Aanc / deep.Astr, one seed) were cut for space on 2026-10-14.
    lab = "\\multirow{2}{*}{\\shortstack[l]{neural\\\\net}}"
    rows = [
        (lab, "anchored", "deep.R2anc.anc", "deep.R2anc.str"),
        ("", "streaming", "deep.R2.anc", "deep.R2.str"),
    ]
    out = []
    for lab, proto, anc, st in rows:
        bold = "\\textbf{" if proto == "anchored" else "{"
        out.append(f"    {lab} & {proto} & {cell(anc)} & {bold}{cell(st)}}} \\\\")
    out.append("    \\midrule")

    def tcell(arm: str, proto: str) -> str:
        p = f"tree.{arm}"
        if proto == "anc":
            return f"{V(p + '.anc.auc')} $\\cdot$ {V(p + '.anc.f1raw')} $\\rightarrow$ {V(p + '.anc.f1tuned')}"
        return f"{V(p + '.auc')} $\\cdot$ {V(p + '.f1raw')} $\\rightarrow$ {V(p + '.f1tuned')}"

    out.append("    \\multirow{2}{*}{\\shortstack[l]{boosted\\\\trees}} & anchored & "
               f"{tcell('A', 'anc')} & \\textbf{{{tcell('A', 'str')}}} \\\\")
    out.append(f"    & streaming & {tcell('B', 'anc')} & {tcell('B', 'str')} \\\\")
    return "\n".join(out) + "\n"


def tab_whowhen() -> str:
    rows = [
        ("neural net", "anchored", "deep.R2anc", "deep.R2anc.wauc"),
        ("neural net", "streaming", "deep.R2", "deep.R2.wauc"),
        ("trees", "anchored", "tree.A", "tree.A.auc"),
        ("trees", "streaming", "tree.B", "tree.B.auc"),
        ("trees", "streaming$^\\ast$", "tree.Bmatched", "tree.Bmatched.auc"),
    ]
    out = []
    for fam, proto, p, aucp in rows:
        det = V(f"{p}.det.pw.205")
        out.append(f"    {fam} & {proto} & {V(aucp)} & {pm(p + '.who')} & {pm(p + '.when')} & {det} \\\\")
    return "\n".join(out) + "\n"


def tab_audit() -> str:
    rows = [
        ("pedestrian only", "scene.base"),
        ("$+$ lights", "scene.R"),
        ("$+$ traffic", "scene.T"),
        ("$+$ layout", "scene.S"),
        ("$+$ scene (all three)", "scene.RTS"),
        ("$+$ waiting history", "wait.W"),
        ("$+$ scene, waiting, looking", "wait.RTSWL"),
        ("$+$ intent oracle$^\\dagger$", "scene.H"),
        ("$+$ ego-future oracle$^\\dagger$", "scene.E"),
        ("$+$ scene, both oracles$^\\dagger$", "scene.RTSHE"),
    ]
    out = []
    for lab, p in rows:
        out.append(f"    {lab} & {V(p + '.who')} & {V(p + '.when')} & {V(p + '.auc')} & {V(p + '.det.pw.205')} & "
                   f"{V(p + '.det.pt.205')} & {V(p + '.det.pt.41')} \\\\")
        if p in ("scene.RTS", "wait.RTSWL"):
            out.append("    \\midrule")
    out.append("    \\midrule")
    for lab, p in (("neural net, pedestrian only$^\\S$", "dscene.base"), ("neural net $+$ scene$^\\S$", "dscene.scene")):
        out.append(f"    {lab} & {V(p + '.who')} & {V(p + '.when')} & {V(p + '.auc')} & {V(p + '.det.pw.205')} & "
                   f"{V(p + '.det.pt.205')} & {V(p + '.det.pt.41')} \\\\")
    out.append("    \\midrule")
    out.append(f"    human intention rating$^\\ddagger$ & {V('ph.human.who')} & -- & -- & -- & -- & -- \\\\")
    # (the human-rating row is post hoc; the caption says so)
    return "\n".join(out) + "\n"


def tab_whendist() -> str:
    out = ["    share of negatives (\\%) & " + " & ".join(V(f"ph.wd.share.{c}") for c in WD_COLS[:5]) + " & -- \\\\",
           "    \\midrule"]
    # The anchored tree's row (ph.wd.treeanc.*) is generated but left out of the table for space.
    labels = {"treebase": "trees, pedestrian only", "treescene": "trees $+$ scene",
              "deepbase": "neural net, pedestrian only", "deepscene": "neural net $+$ scene"}
    for tag, lab in labels.items():
        out.append(f"    {lab} & " + " & ".join(V(f"ph.wd.{tag}.{c}") for c in WD_COLS) + " \\\\")
    return "\n".join(out) + "\n"


def tab_objective() -> str:
    rows = [
        ("neural net", "binary", "deep.R2"),
        ("neural net", "hazard", "deep.R3"),
        ("trees", "binary", "tree.B"),
        ("trees", "hazard", "tree.Hz"),
        ("trees", "hazard $+$ censored", "tree.HzC"),
        ("trees", "binary $+$ censored as 0$^\\ast$", "tree.Bz32"),
    ]
    out = []
    for fam, obj, p in rows:
        out.append(f"    {fam} & {obj} & {pm(p + '.det.pw.205')} & {pm(p + '.det.pw.41')} & {pm(p + '.det.pt.205')} & "
                   f"{V(p + '.when')} \\\\")
        if p == "deep.R3":
            out.append("    \\midrule")
    return "\n".join(out) + "\n"


def tab_horizon() -> str:
    out = []
    for _h, lab, drop, zero, haz in ((32, "1.07", "tree.B", "tree.Bz32", "tree.HzC"),
                                    (64, "2.13", "tree.Bd64", "tree.Bz64", "tree.Hz64"),
                                    (160, "5.33", "tree.Bd160", "tree.Bz160", "tree.Hz160")):
        det = " & ".join(V(a + ".det.pw.205") for a in (drop, zero, haz))
        who = " & ".join(V(a + ".who") for a in (drop, zero, haz))
        out.append(f"    {lab}\\,s & {det} & {who} \\\\")
    return "\n".join(out) + "\n"


def _f(key: str) -> float:
    return float(V(key).replace("$-$", "-"))


def derived_numbers() -> None:
    """Numbers computed from other numbers: rounded abstract values, factor gains, spreads, sizes."""
    src = "derived in make_tables.py from the keys named in the code"
    for key, base in (("abs.deep.anc", "deep.R2anc.anc.auc"), ("abs.deep.str", "deep.R2anc.str.auc"),
                      ("abs.tree.anc", "tree.A.anc.auc"), ("abs.tree.str", "tree.A.auc")):
        put(key, f"{_f(base):.2f}", src)
    put("audit.who.gain", signed(_f("scene.RTS.who") - _f("scene.base.who"), 3), src)
    put("audit.when.gain", signed(_f("scene.RTS.when") - _f("scene.base.when"), 3), src)
    deployable = ["scene.R", "scene.T", "scene.S", "scene.RTS", "wait.W", "wait.WL", "wait.RTSWL"]
    every = deployable + ["scene.H", "scene.E", "scene.RTSH", "scene.RTSE", "scene.RTSHE"]
    put("audit.when.maxgain", signed(max(_f(a + ".when") for a in every) - _f("scene.base.when"), 3), src)
    put("audit.when.maxgain.deploy", signed(max(_f(a + ".when") for a in deployable) - _f("scene.base.when"), 3), src)
    for k in ("audit.who.gain", "audit.when.gain", "audit.when.maxgain"):
        put(k + ".abs", f"{abs(_f(k)):.3f}", src)
    put("treeq.Q2zero160.pt.gap.205.abs", f"{abs(_f('treeq.Q2zero160.pt.gap.205')):.1f}", src)
    who_gain = _f("scene.RTS.who") - _f("scene.base.who")
    when_gain = _f("scene.RTS.when") - _f("scene.base.when")
    put("audit.who.ratio", f"{who_gain / when_gain:.1f}", src)
    t = load("tree")
    spread = max(100 * (max(r["values"]) - min(r["values"])) for r in t["arms"]["B"]["detection_per_window"])
    put("tree.B.seedspread", f"{spread:.1f}",
        "tree_study_f64/results.json arms.B per-seed values, max range over budgets")
    fit = json.loads((DIAG / "tree_study_f64" / "A" / "s42" / "fit.json").read_text())
    put("tree.anc.ntrain", thousands(fit["n_rows"]), "tree_study_f64/A/s42/fit.json n_rows")
    ntrain = load("comp")["composition"][0]["total"]
    put("tree.anc.sizeratio", f"{ntrain / fit['n_rows']:.0f}", "88,214 / anchored training windows")


FA_ARMS = {"deep binary (streaming)": "deep.R2", "deep hazard (streaming)": "deep.R3",
           "deep binary (anchored)": "deep.R2anc", "tree binary (anchored)": "tree.A",
           "tree binary (streaming)": "tree.B", "tree hazard (streaming)": "tree.Hz",
           "tree matched size": "tree.Bmatched", "scene: base": "scene.base", "scene: R": "scene.R",
           "scene: T": "scene.T", "scene: S": "scene.S", "scene: RTS": "scene.RTS", "scene: H": "scene.H",
           "scene: E": "scene.E", "scene: RTS_HE": "scene.RTSHE", "waiting: W": "wait.W", "waiting: WL": "wait.WL",
           "waiting: RTS_WL": "wait.RTSWL"}


def factor_numbers() -> None:
    """POST HOC window-AUC decomposition, within-pedestrian and band when scores (whowhen_posthoc/)."""
    path = DIAG / "whowhen_posthoc" / "factor_analysis.json"
    src = "outputs/diagnostics/whowhen_posthoc/factor_analysis.json (POST HOC)"
    fa = json.loads(path.read_text())
    for arm, prefix in FA_ARMS.items():
        r = fa[arm]
        for k in ("auc_hard", "auc_never", "auc_after", "when_band", "when_within"):
            put(f"ph.{prefix}.{k.replace('_', '')}", a3(r[k][0]), src)
        for k in ("w_hard", "w_never", "w_after"):
            put(f"ph.{prefix}.{k.replace('_', '')}", f"{r[k][0]:.3f}", src)
    put("ph.within.n", f"{fa['scene: base']['n_within'][0]:.0f}", src)
    bands = [fa[a]["when_band"][0] for a in FA_ARMS if "anchored" not in a]
    put("ph.band.min", f"{min(bands):.2f}", src)
    put("ph.band.max", f"{max(bands):.2f}", src)
    put("ph.within.gain", signed(fa["scene: RTS"]["when_within"][0] - fa["scene: base"]["when_within"][0], 3), src)
    put("ph.band.gain", signed(fa["scene: RTS"]["when_band"][0] - fa["scene: base"]["when_band"][0], 3), src)
    put("ph.within.gain.abs", f"{fa['scene: RTS']['when_within'][0] - fa['scene: base']['when_within'][0]:.3f}", src)
    put("ph.band.gain.abs", f"{fa['scene: RTS']['when_band'][0] - fa['scene: base']['when_band'][0]:.3f}", src)


WD_ARMS = {"trees, pedestrian only": "treebase", "trees + scene": "treescene",
           "neural net, pedestrian only": "deepbase", "neural net + scene": "deepscene",
           "trees, anchored-trained": "treeanc"}
WD_COLS = ["b1to2", "b2to3", "b3to5", "b5to10", "bge10", "pooled"]


def whendist_numbers() -> None:
    """POST HOC when score split by how far ahead the negative window's crossing starts (whowhen_posthoc/)."""
    w = json.loads(SOURCES["whendist"].read_text())
    src = "outputs/diagnostics/whowhen_posthoc/when_by_distance.json (POST HOC)"
    for col, share in zip(WD_COLS, w["share"]):
        put(f"ph.wd.share.{col}", f"{100 * share:.0f}", src)
    for arm, tag in WD_ARMS.items():
        for col, m in zip(WD_COLS, w["arms"][arm]["mean"], strict=True):
            put(f"ph.wd.{tag}.{col}", a3(m), src)
    streaming = [w["arms"][a]["mean"] for a in WD_ARMS if "anchored" not in a]
    put("ph.wd.near.min", a3(min(r[0] for r in streaming)), src)
    put("ph.wd.near.max", a3(max(r[0] for r in streaming)), src)
    put("ph.wd.far.min", f"{min(r[4] for r in streaming):.2f}", src)
    # gains from the rounded table cells, so a reader subtracting two cells gets the quoted range
    gains = [float(a3(s)) - float(a3(b)) for base, scene in (("trees, pedestrian only", "trees + scene"),
                                                             ("neural net, pedestrian only", "neural net + scene"))
             for b, s in zip(w["arms"][base]["mean"][:5], w["arms"][scene]["mean"][:5], strict=True)]
    put("ph.wd.gain.min", a3(min(gains)), src)
    put("ph.wd.gain.max", a3(max(gains)), src)


def budget_numbers() -> None:
    """What a budget means: share of never-crossers' windows, and hours of never-crosser footage (M-7)."""
    on = np.load(SOURCES["onset_test"], allow_pickle=True)
    tc = np.asarray(on["track_crosses"])
    n_neg_windows = int((tc == 0).sum())
    hours = n_neg_windows / 36000.0
    src = "tree_study/B/s42/onset_test.npz (never-crosser windows / 36,000 decisions per hour)"
    put("budget.negwindows", thousands(n_neg_windows), src)
    put("budget.hours", f"{hours:.2f}", src)
    put("budget.205.share", f"{100 * 205 / 36000:.2f}", "205 alarms / 36,000 decisions per hour")
    put("budget.205.alarms", str(int(205 * hours)), src)
    put("budget.205.pedshare", f"{100 * int(205 * hours) / 495:.0f}", src + "; 495 never-crossers")


def main() -> None:
    for k, (v, s) in CONSTANTS.items():
        put(f"c.{k}", v, s)
    data_numbers()
    tree_numbers()
    scene_numbers()
    wait_numbers()
    deep_numbers()
    deep_scene_numbers()
    posthoc_numbers()
    derived_numbers()
    factor_numbers()
    whendist_numbers()
    budget_numbers()
    OUT.mkdir(exist_ok=True)
    lines = ["% GENERATED by paper/figures/make_tables.py -- do not edit. One \\defV per number, with its source.",
             ""]
    lines += [f"\\defV{{{k}}}{{{v}}}% {s}" for k, v, s in ENTRIES]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf8")
    tables = {"composition": tab_composition, "matrix": tab_matrix, "whowhen": tab_whowhen, "audit": tab_audit,
              "whendist": tab_whendist, "objective": tab_objective, "horizon": tab_horizon}
    for name, fn in tables.items():
        (OUT / f"tab_{name}.tex").write_text("% GENERATED by make_tables.py\n" + fn(), encoding="utf8")
    print(f"wrote {len(ENTRIES)} numbers and {len(tables)} tables to {OUT}")
    if "--list" in sys.argv:
        for k, v, _ in ENTRIES:
            print(f"{k} = {v}")


if __name__ == "__main__":
    main()

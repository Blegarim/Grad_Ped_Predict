#!/usr/bin/env python
"""Every paper table for the pixel-free re-run campaign, from logged artifacts only (docs/RERUN_PLAN_2026-09-26.md).

Inputs, per run: ``outputs/runs/<run>/{train_log.csv, eval_log.csv, resolved_config.yaml}`` and the streaming-test
dump ``outputs/diagnostics/<run>/onset_test.npz``. No model is loaded and no data is read, so this runs on the
laptop and re-running it reproduces every number.

    python scripts/report_campaign.py --out outputs/diagnostics/v4_report/tables

Writes one markdown file per table + ``results.json`` (every number, per seed) + ``README.md`` (provenance).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from pedpredict.eval import detection_curve as dc
from pedpredict.eval.campaign_report import (
    compare_curves,
    intent_timing,
    mean_sd,
    onset_separability,
    smoothed_scores,
)
from pedpredict.eval.onset_timing import load_dump

SEEDS = (42, 43, 44)
#: name -> (run-tag stem, seeds, dump score column). The reported head: p_readout for pure/hedge/censored
#: (onset_report_crosses), p_frame for binary, auxiliary and 3-task arms.
ARMS: dict[str, tuple[str, tuple[int, ...], str]] = {
    "R2 binary": ("c4_r2s", SEEDS, "p_frame"),
    "R3 pure hazard": ("c4_r3", SEEDS, "p_readout"),
    "R1 auxiliary": ("c4_r1", SEEDS, "p_frame"),
    "R4 hedge": ("c4_r4", SEEDS, "p_readout"),
    "R3C censored": ("c4_r3c", SEEDS, "p_readout"),
    "R2 anchored-trained": ("c4_r2a", SEEDS, "p_frame"),
    "Model A streaming-trained": ("c4_mA_str", (42,), "p_frame"),
    "Model A anchored-trained": ("c4_mA_anc", (42,), "p_frame"),
}
GBM = [f"outputs/diagnostics/probe/gbm_s{s}.npz" for s in SEEDS]
EVAL_KEYS = ("crosses_auc", "crosses_f1", "tuned_crosses_f1", "tuned_crosses_precision", "tuned_crosses_recall",
             "tuned_crosses_threshold")


def fmt(values: list[float], digits: int = 3, scale: float = 1.0) -> str:
    m, s = mean_sd([v * scale for v in values])
    if np.isnan(m):
        return "–"
    return f"{m:.{digits}f}" if np.isnan(s) else f"{m:.{digits}f} ± {s:.{digits}f}"


def run_dir(runs: Path, tag: str) -> Path:
    found = sorted(runs.glob(f"*_{tag}"))
    if not found:
        raise SystemExit(f"no run dir for {tag} under {runs}")
    return found[-1]


def eval_row(run: Path, protocol: str) -> dict[str, float]:
    with open(run / "eval_log.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test" and r.get("protocol") == protocol]
    return {k: float(rows[-1][k]) for k in EVAL_KEYS}


def collapse_epochs(rows: list[dict[str, str]], cfg: dict) -> list[int] | None:
    """Epochs that called (nearly) every validation window a crossing — the paper's collapse definition.

    Defined only for streaming-trained runs, whose validation set is ~2.8% crossings, so recall > 0.8 means
    "everything crosses". ``None`` for anchored-trained runs (their ~28%-positive val makes high recall normal).
    The val_loss > 1.0 arm applies to crossing-only runs; a 3-task run's val_loss sums three tasks.
    """
    if cfg["data"]["protocol"] != "streaming":
        return None
    single = list(cfg["train"]["active_tasks"]) == ["crosses"]
    return [int(r["epoch"]) for r in rows
            if float(r["crosses_recall"]) > 0.8 or (single and float(r["val_loss"]) > 1.0)]


def training_row(run: Path) -> dict[str, object]:
    with open(run / "train_log.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    metric = cfg["train"]["selection_metric"]
    sel = [float(r[metric]) for r in rows]
    best = sel.index(min(sel) if metric == "val_loss" else max(sel))
    auc = [float(r["crosses_auc"]) for r in rows]
    out = {"run": run.name, "selection_metric": metric, "epochs": len(rows), "selected_epoch": best + 1,
           "val_auc_selected": auc[best], "val_auc_peak": max(auc), "val_auc_peak_epoch": auc.index(max(auc)) + 1,
           "val_auc_last": auc[-1], "train_loss_first": float(rows[0]["train_loss"]),
           "train_loss_last": float(rows[-1]["train_loss"]), "collapse_epochs": collapse_epochs(rows, cfg)}
    if "onset_readout_p95" in rows[0]:
        spread = [float(r["onset_readout_p95"]) - float(r["onset_readout_p05"]) for r in rows]
        out.update(readout_spread_selected=spread[best], readout_spread_last=spread[-1])
    return out


def collect(runs: Path, dumps: Path) -> dict[str, list[dict]]:
    """Per arm, per seed: eval rows (both protocols), training summary and dump analyses."""
    data: dict[str, list[dict]] = {}
    for name, (stem, seeds, key) in ARMS.items():
        data[name] = []
        for seed in seeds:
            run = run_dir(runs, f"{stem}_s{seed}")
            arrays, _ = load_dump(dumps / run.name / "onset_test.npz")
            score = np.asarray(arrays[key], dtype=float)
            causal = smoothed_scores(arrays, score)
            centered = smoothed_scores(arrays, score, causal=False)
            data[name].append({
                "seed": seed, "run": run.name, "score": key,
                "anchored": eval_row(run, "anchored"), "streaming": eval_row(run, "streaming"),
                "training": training_row(run),
                "detection": {acc: [p.as_dict() for p in dc.detection_curve(arrays, key, accounting=acc)]
                              for acc in ("per_window", "per_track")},
                "separability": onset_separability(arrays, score),
                "smoothing": {"auc_raw": intent_timing(arrays, score)["window"],
                              "auc_causal_k15": intent_timing(arrays, causal)["window"],
                              "auc_centered_k15": intent_timing(arrays, centered)["window"]},
                "intent_timing": intent_timing(arrays, score),
            })
    return data


def gbm_detection() -> dict[str, list[list[dict]]]:
    out: dict[str, list[list[dict]]] = {"per_window": [], "per_track": []}
    for path in GBM:
        if Path(path).exists():
            arrays, _ = load_dump(path)
            for acc in out:
                out[acc].append([p.as_dict() for p in dc.detection_curve(arrays, "p_frame", accounting=acc)])
    return out


# --------------------------------------------------------------------------- tables


def col(seeds: list[dict], proto: str, key: str) -> list[float]:
    """One eval-log column across an arm's seeds, for one test protocol."""
    return [s[proto][key] for s in seeds]


MODELS = {"Model B (crossing only, n=3)": ("R2 anchored-trained", "R2 binary"),
          "Model A (3-task, n=1)": ("Model A anchored-trained", "Model A streaming-trained")}


def table_matrix(data: dict) -> str:
    lines = ["# Cross-protocol matrix (test split; AUC · raw F1 → val-tuned F1; mean ± sd over seeds)", ""]
    for model, (anc, strm) in MODELS.items():
        lines += [f"## {model}", "", "| trained on ↓ / tested on → | anchored | streaming |", "|---|---|---|"]
        for label, arm in (("anchored", anc), ("streaming", strm)):
            cells = [f"{fmt(col(data[arm], p, 'crosses_auc'))} · {fmt(col(data[arm], p, 'crosses_f1'))} → "
                     f"{fmt(col(data[arm], p, 'tuned_crosses_f1'))}" for p in ("anchored", "streaming")]
            lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
        lines.append("")
    return "\n".join(lines)


def gap_rows(anc: list[dict], strm: list[dict]) -> list[dict[str, float]]:
    """G_total = tuned F1 str→str − raw F1 anc→str; G_prior = tuned − raw, anc→str; G_hardneg = the rest."""
    rows = []
    for a, s in zip(anc, strm, strict=True):
        raw_as, tun_as, tun_ss = a["streaming"]["crosses_f1"], a["streaming"]["tuned_crosses_f1"], \
            s["streaming"]["tuned_crosses_f1"]
        rows.append({"seed": a["seed"], "G_total": tun_ss - raw_as, "G_prior": tun_as - raw_as,
                     "G_hardneg": tun_ss - tun_as})
    return rows


def table_gap(data: dict) -> tuple[str, dict]:
    out = {"Model B": gap_rows(data["R2 anchored-trained"], data["R2 binary"]),
           "Model A": gap_rows(data["Model A anchored-trained"], data["Model A streaming-trained"])}
    lines = ["# Gap decomposition (F1 on the streaming test split; G_total = G_prior + G_hardneg)", "",
             "| model | seed | G_total | G_prior | G_hardneg |", "|---|---|---|---|---|"]
    for model, rows in out.items():
        for r in rows:
            lines.append(f"| {model} | {r['seed']} | {r['G_total']:+.3f} | {r['G_prior']:+.3f} | "
                         f"{r['G_hardneg']:+.3f} |")
        lines.append(f"| {model} | mean ± sd | " + " | ".join(fmt([r[k] for r in rows]) for k in
                                                          ("G_total", "G_prior", "G_hardneg")) + " |")
    return "\n".join(lines) + "\n", out


def table_window(data: dict) -> str:
    lines = ["# Window metrics (test split, val-tuned thresholds; mean ± sd over seeds)", ""]
    for proto in ("streaming", "anchored"):
        lines += [f"## Tested on {proto}", "", "| arm | n | score | AUC | raw F1 | tuned F1 | tuned precision | "
                  "tuned recall |", "|---|---|---|---|---|---|---|---|"]
        for arm, seeds in data.items():
            cells = " | ".join(fmt(col(seeds, proto, k)) for k in ("crosses_auc", "crosses_f1", "tuned_crosses_f1",
                                                                   "tuned_crosses_precision", "tuned_crosses_recall"))
            lines.append(f"| {arm} | {len(seeds)} | {seeds[0]['score']} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def _curve_cells(curves: list[list[dict]], i: int) -> str:
    rate = fmt([c[i]["detection_rate"] for c in curves], 1, 100.0)
    lead = fmt([c[i]["mean_lead_s"] for c in curves], 2)
    return f"{rate}% @ {lead} s"


def table_detection(data: dict, gbm: dict, acc: str) -> str:
    arms = {"GBM probe (reference)": gbm[acc]} if gbm[acc] else {}
    arms.update({arm: [s["detection"][acc] for s in seeds] for arm, seeds in data.items()})
    budgets = [p["budget_per_hour"] for p in next(iter(arms.values()))[0]]
    lines = [f"# Detection rate @ mean lead time, `{acc}` accounting (streaming test; mean ± sd over seeds)", "",
             "| alarms/hr | " + " | ".join(f"{a} (n={len(c)})" for a, c in arms.items()) + " |",
             "|---|" + "---|" * len(arms)]
    for i, b in enumerate(budgets):
        lines.append(f"| {b:g} | " + " | ".join(_curve_cells(c, i) for c in arms.values()) + " |")
    return "\n".join(lines) + "\n"


def criterion(data: dict, arm: str, acc: str) -> tuple[str, list[dict]]:
    """Pre-registered rule (``campaign_report.compare_curves``): arm vs R2 at >= 3 of 5 budgets."""
    return compare_curves([s["detection"][acc] for s in data[arm]],
                          [s["detection"][acc] for s in data["R2 binary"]], arm_name=arm, base_name="R2 binary")


def table_criterion(data: dict) -> tuple[str, dict]:
    out, lines = {}, ["# Pre-registered comparison against the binary baseline (R2), 3 seeds each", "",
                      "Rule (SEED_PLAN_2026-09-21): an arm wins if its mean detection rate beats R2's by more than the "
                      "sum of the two sds at >= 3 of the 5 budgets. **Primary: R3 vs R2, `per_window`.** Every other "
                      "row is secondary and was not pre-registered.", ""]
    for arm in ("R3 pure hazard", "R1 auxiliary", "R4 hedge", "R3C censored"):
        for acc in ("per_window", "per_track"):
            verdict, rows = criterion(data, arm, acc)
            out[f"{arm} | {acc}"] = {"verdict": verdict, "rows": rows}
            tag = " (PRIMARY)" if arm == "R3 pure hazard" and acc == "per_window" else ""
            lines += [f"## {arm} vs R2, `{acc}`{tag}: **{verdict}**", "",
                      "| alarms/hr | R2 | arm | gap | sd sum | beyond sd |", "|---|---|---|---|---|---|"]
            lines += [f"| {r['budget']:g} | {100 * r['base']:.1f} ± {100 * r['base_sd']:.1f} | "
                      f"{100 * r['arm']:.1f} ± {100 * r['arm_sd']:.1f} | {100 * r['gap']:+.1f} | "
                      f"{100 * r['sd_sum']:.1f} | {r['beyond_sd']} |" for r in rows]
            lines.append("")
    return "\n".join(lines), out


def collapse_cell(epochs: list[int] | None) -> str:
    if epochs is None:
        return "n/a (anchored val)"
    return str(epochs) if epochs else "none"


def table_training(data: dict) -> str:
    lines = ["# Training summary per run (validation split)", "",
             "| arm | seed | epochs | selected epoch (metric) | val AUC selected | val AUC last | "
             "train loss first→last | collapse epochs | readout spread selected / last |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, seeds in data.items():
        for s in seeds:
            t = s["training"]
            spread = (f"{t['readout_spread_selected']:.2f} / {t['readout_spread_last']:.2f}"
                      if "readout_spread_selected" in t else "–")
            lines.append(f"| {arm} | {s['seed']} | {t['epochs']} | {t['selected_epoch']} ({t['selection_metric']}) | "
                         f"{t['val_auc_selected']:.3f} | {t['val_auc_last']:.3f} | {t['train_loss_first']:.2f}→"
                         f"{t['train_loss_last']:.2f} | {collapse_cell(t['collapse_epochs'])} | {spread} |")
    return "\n".join(lines) + "\n"


def table_analyses(data: dict) -> str:
    buckets = list(next(iter(data.values()))[0]["separability"])
    lines = ["# Separability by time to onset (AUC vs never-crossers; streaming test; mean ± sd)", "",
             "| arm | " + " | ".join(buckets) + " |", "|---|" + "---|" * len(buckets)]
    for arm, seeds in data.items():
        cells = " | ".join(fmt([s["separability"][b]["auc"] for s in seeds]) for b in buckets)
        lines.append(f"| {arm} | {cells} |")
    lines += ["", "# Within-track smoothing (k=15) and the who/when split (streaming test; mean ± sd)", "",
              "Causal = trailing mean (deployable). Centered = looks ahead (offline only; comparable to the paper's "
              "earlier 'k=15 moving average').", "",
              "| arm | window AUC | Δ causal | Δ centered | who crosses (per pedestrian) | "
              "when (crossers only, <32 frames) |", "|---|---|---|---|---|---|"]
    for arm, seeds in data.items():
        raw = [s["smoothing"]["auc_raw"] for s in seeds]
        d_causal = [s["smoothing"]["auc_causal_k15"] - r for s, r in zip(seeds, raw, strict=True)]
        d_center = [s["smoothing"]["auc_centered_k15"] - r for s, r in zip(seeds, raw, strict=True)]
        lines.append(f"| {arm} | {fmt(raw)} | {fmt(d_causal)} | {fmt(d_center)} | "
                     f"{fmt([s['intent_timing']['intent'] for s in seeds])} | "
                     f"{fmt([s['intent_timing']['timing'] for s in seeds])} |")
    return "\n".join(lines) + "\n"


def provenance(out: Path, data: dict) -> str:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    runs = "\n".join(f"- {arm}: " + ", ".join(f"`{s['run']}` ({s['score']})" for s in seeds)
                     for arm, seeds in data.items())
    return (f"# Campaign tables — provenance\n\nGenerated by `scripts/report_campaign.py` at commit `{rev}` "
            f"into `{out}`.\nInputs are logged artifacts only (train/eval logs, config snapshots, streaming-test "
            "dumps).\n\n## Runs\n\n" + runs + "\n\n## Definitions\n\n"
            "- Window metrics: `eval_log.csv` test rows; tuned = threshold swept on val (0.01–0.99 @ 0.01), "
            "applied to test.\n"
            "- Detection: `eval/detection_curve.py`, budgets {1200, 460, 205, 95, 41}/hr, both accountings.\n"
            "- Gap: G_total = tuned F1(str→str) − raw F1(anc→str); G_prior = tuned − raw F1(anc→str); "
            "G_hardneg = rest.\n"
            "- Separability: windows with onset in the bucket vs all windows of never-crossing pedestrians.\n"
            "- Smoothing: mean of 15 windows of a track, causal (trailing) and centered; time order by "
            "future_observed.\n"
            "- Who/when: per-pedestrian mean score vs track_crosses; crossers-only onset < 32 frames vs later.\n"
            "- Selected epoch: best.pth's epoch by the run's own `train.selection_metric`.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--dumps", default="outputs/diagnostics")
    parser.add_argument("--out", default="outputs/diagnostics/v4_report/tables")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(Path(args.runs), Path(args.dumps))
    gbm = gbm_detection()
    gap_md, gap = table_gap(data)
    crit_md, crit = table_criterion(data)
    files = {"matrix.md": table_matrix(data), "gap.md": gap_md, "window_metrics.md": table_window(data),
             "detection_per_window.md": table_detection(data, gbm, "per_window"),
             "detection_per_track.md": table_detection(data, gbm, "per_track"), "criterion.md": crit_md,
             "training.md": table_training(data), "analyses.md": table_analyses(data),
             "README.md": provenance(out, data)}
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"arms": data, "gap": gap, "criterion": crit, "gbm": gbm},
                                                  indent=1, default=float), encoding="utf-8")
    print(crit_md.split("\n## ")[1].split("\n")[0])
    print(f"wrote {len(files) + 1} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

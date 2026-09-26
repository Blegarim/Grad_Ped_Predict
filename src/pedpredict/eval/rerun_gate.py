"""Go/no-go gate for the pixel-free re-run campaign (``docs/RERUN_PLAN_2026-09-26.md`` §2).

The question is whether the pixel-free recipe *genuinely learns* — validation improves and then holds —
rather than peaking in the first few epochs and declining while training loss keeps falling (the
memorization pattern every earlier recipe showed). It is read from the **binary** arm's seeds only: the
gate decides whether the recipe is sound, and must never see the hazard-vs-binary comparison it will later
be used to make.

Five checks, all required:

* **L1 no late decline** — per seed, validation AUC over the last ``tail_epochs`` stays within
  ``tail_tolerance`` of that run's peak.
* **L2 not an early spike** — per seed, the validation-AUC peak is at epoch ``>= min_peak_epoch``.
* **L3 no collapse** — per seed, no epoch predicts "everything crosses" (val recall > ``collapse_recall``)
  or has val loss > ``collapse_val_loss`` (the paper's collapse definition).
* **L4 seeds agree** — detection-rate sample sd at ``spread_budget`` alarms/hour ``<= max_spread_pp``, and
  the validation AUC at each seed's selected epoch spans ``<= max_val_auc_range``.
* **L5 good enough to replace v1** — mean streaming-test AUC ``>= min_test_auc`` and mean detection rate at
  ``level_budget`` alarms/hour ``> min_detection_pct``.

Thresholds were fixed before any fixed-recipe run finished; change them only in a commit dated before the
gate is read.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Check",
    "GateThresholds",
    "curve_checks",
    "read_train_log",
    "seed_checks",
    "selected_epoch_auc",
    "streaming_test_auc",
]


@dataclass(frozen=True)
class GateThresholds:
    tail_epochs: int = 5
    tail_tolerance: float = 0.02
    min_epochs: int = 10
    min_peak_epoch: int = 8
    collapse_recall: float = 0.8
    collapse_val_loss: float = 1.0
    spread_budget: float = 205.0
    max_spread_pp: float = 5.0
    max_val_auc_range: float = 0.02
    min_test_auc: float = 0.78
    level_budget: float = 41.0
    # v3 R2 (image model, shuffled data) measured 15.6098% at 41 alarms/hr, per_window: the recipe being
    # replaced must be beaten, so the bar sits just above it rather than at the rounded 15.6.
    min_detection_pct: float = 15.61


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def read_train_log(path: str | Path) -> list[dict[str, float]]:
    """``train_log.csv`` rows as floats (every column the gate reads is numeric)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(fh)]


def curve_checks(label: str, rows: Sequence[dict[str, float]], th: GateThresholds) -> list[Check]:
    """L1–L3 for one run's per-epoch validation curve."""
    if len(rows) < th.min_epochs:
        return [Check(f"L1-L3 {label}", False, f"only {len(rows)} epochs logged (< {th.min_epochs})")]
    auc = [r["crosses_auc"] for r in rows]
    peak = max(auc)
    peak_epoch = int(rows[auc.index(peak)]["epoch"])
    tail = auc[-th.tail_epochs:]
    collapsed = [int(r["epoch"]) for r in rows
                 if r["crosses_recall"] > th.collapse_recall or r["val_loss"] > th.collapse_val_loss]
    return [
        Check(f"L1 {label}", min(tail) >= peak - th.tail_tolerance,
              f"last {th.tail_epochs} val AUC min {min(tail):.4f} vs peak {peak:.4f} "
              f"(tolerance {th.tail_tolerance})"),
        Check(f"L2 {label}", peak_epoch >= th.min_peak_epoch,
              f"val AUC peak at epoch {peak_epoch} (needs >= {th.min_peak_epoch})"),
        Check(f"L3 {label}", not collapsed,
              f"collapse epochs: {collapsed or 'none'}"),
    ]


def selected_epoch_auc(rows: Sequence[dict[str, float]], metric: str = "crosses_f1") -> float:
    """Validation AUC at the epoch best.pth was taken from (first epoch with the max selection metric)."""
    scores = [r[metric] for r in rows]
    return rows[scores.index(max(scores))]["crosses_auc"]


def streaming_test_auc(eval_log: str | Path) -> float:
    """Crossing AUC of the most recent streaming-protocol test pass in a run's ``eval_log.csv``."""
    with open(eval_log, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test" and r.get("protocol") == "streaming"]
    if not rows:
        raise ValueError(f"{eval_log}: no streaming test row")
    return float(rows[-1]["crosses_auc"])


def seed_checks(
    val_aucs: Sequence[float],
    test_aucs: Sequence[float],
    spread_rates: Sequence[float],
    level_rates: Sequence[float],
    th: GateThresholds,
) -> list[Check]:
    """L4–L5 across seeds. Detection rates are fractions (0–1), reported in percentage points."""
    if len(spread_rates) < 2:
        return [Check("L4-L5", False, f"need >= 2 seeds with dumps, have {len(spread_rates)}")]
    spread_pp = 100.0 * statistics.stdev(spread_rates)
    auc_range = max(val_aucs) - min(val_aucs)
    mean_test_auc = statistics.fmean(test_aucs)
    mean_level_pct = 100.0 * statistics.fmean(level_rates)
    return [
        Check("L4 detection spread", spread_pp <= th.max_spread_pp,
              f"sd {spread_pp:.1f} pp at {th.spread_budget:g}/hr (needs <= {th.max_spread_pp}); "
              f"rates {[round(100 * r, 1) for r in spread_rates]}"),
        Check("L4 val AUC range", auc_range <= th.max_val_auc_range,
              f"range {auc_range:.4f} over {[round(a, 4) for a in val_aucs]} (needs <= {th.max_val_auc_range})"),
        Check("L5 test AUC", mean_test_auc >= th.min_test_auc,
              f"mean streaming test AUC {mean_test_auc:.4f} over {[round(a, 4) for a in test_aucs]} "
              f"(needs >= {th.min_test_auc})"),
        Check("L5 detection level", mean_level_pct > th.min_detection_pct,
              f"mean {mean_level_pct:.1f}% at {th.level_budget:g}/hr (needs > {th.min_detection_pct})"),
    ]

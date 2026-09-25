"""Detection-latency evaluation — the project's primary metric for streaming crossing onset.

A fixed-horizon binary score answers "does a crossing start within 32 frames?" for each window
independently. That is not how a warning system is specified. A deployed system is specified as: *how
many crossings do you catch, how early, at an alarm rate the driver will tolerate?* Two models can tie on
window-F1 and differ several-fold on that question — measured here, they do (see
``outputs/runs/RESULTS_MATRIX.md``).

The protocol, per pedestrian track rather than per window:

* **crossing pedestrians** — a track is detected if any window whose crossing is *genuinely still ahead*
  (``onset_offset >= 0``) scores at or above the threshold. The **lead time** is the largest
  ``onset_offset`` among those windows: the earliest point before the crossing at which the system warned.
  Alarms on a crossing pedestrian *after* they have already stepped out are neither credited nor
  penalised — they are no longer a prediction.

  Taking the *largest offset* rather than the *first row* is deliberate. ``track_id`` is the PIE
  pedestrian id, and PIE splits one pedestrian into several track entries across occlusion gaps, so ~5% of
  tracks reach a dump as several observation segments whose relative order is not recoverable from the
  dump (there is no absolute frame index in it). Within one segment ``onset_offset`` decreases as the
  window slides forward, so "largest offset" and "first row" agree there; across segments only the former
  is well defined. The metric therefore makes no assumption about row order at all.
* **never-crossing pedestrians** — every window is an opportunity for a **false alarm**. Two accounting
  rules are supported because they answer different questions, and the absolute rates differ under them:
  ``per_window`` counts each alarming window (what a naive per-frame alarm rate means) and ``per_track``
  counts each pedestrian who ever triggers one (what a driver actually experiences, since one pedestrian
  alarming ten times in a row is one nuisance event). Both divide by the same exposure, so they are
  directly comparable.
* **the budget** — thresholds are not swept freely. For each false-alarm budget the threshold is the
  lowest one whose realised rate still fits inside the budget, so the constraint is never violated and
  detection is maximised subject to it. Comparing arms at a matched budget is the whole point: an arm
  cannot buy detections with alarms.

Lineage: quickest change detection (Page 1954; Shiryaev, Roberts, Lorden), whose native pair of metrics is
detection delay against average run length to false alarm. Mirrored here as lead time against alarms per
hour. Note that the field's *metric* transfers but its *detector* does not — CUSUM accumulation measures
2-10x worse here, because a crossing approach is a brief excursion rather than the persistent distribution
shift CUSUM assumes.

Input is a dump from :mod:`pedpredict.eval.onset_timing` (``load_dump``), so this module is pure numpy and
runs anywhere — no GPU, no data, no checkpoint.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "DEFAULT_BUDGETS",
    "DecisionClock",
    "DetectionPoint",
    "track_slices",
    "split_tracks",
    "detection_curve",
    "aggregate_curves",
    "format_curve",
]

#: False-alarm budgets (alarms/hour) the project reports at. Roughly halving, spanning "permissive" to
#: "a driver would accept this", so the trend across the budget axis is visible rather than one operating
#: point. The tightest budgets are where the arms separate most.
DEFAULT_BUDGETS: tuple[int, ...] = (1200, 460, 205, 95, 41)

Arrays = dict[str, np.ndarray]


@dataclass(frozen=True)
class DecisionClock:
    """How often the model is asked, which converts an alarm *count* into an alarm *rate*.

    One decision per window: at ``stride`` frames between windows and ``fps`` frames per second, that is
    ``fps / stride`` decisions a second. The repo default (stride 3, 30 fps) gives 36,000 decisions/hour.
    """

    fps: float = 30.0
    stride: int = 3

    @property
    def decisions_per_hour(self) -> float:
        return self.fps / self.stride * 3600.0

    def hours(self, n_windows: int) -> float:
        """Wall-clock exposure represented by ``n_windows`` decisions."""
        return n_windows / self.decisions_per_hour


@dataclass(frozen=True)
class DetectionPoint:
    """One operating point: what the threshold implied by a budget actually buys."""

    budget_per_hour: float
    threshold: float
    accounting: str
    false_alarms: int
    false_alarm_rate: float
    detected: int
    n_positive: int
    detection_rate: float
    mean_lead_s: float
    median_lead_s: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def track_slices(track_id: np.ndarray) -> dict[str, np.ndarray]:
    """Window indices per pedestrian, keyed by ``track_id`` (the PIE pedestrian id).

    A pedestrian may contribute several observation segments; all of them group under the one id, because
    the alarm budget is spent per *pedestrian*, not per segment.
    """
    order: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(np.asarray(track_id)):
        order[str(tid)].append(i)
    return {tid: np.asarray(idx, dtype=np.int64) for tid, idx in order.items()}


def split_tracks(arrays: Arrays) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split pedestrians into (crossing, never-crossing) index arrays.

    A pedestrian counts as crossing if any window has a crossing ahead of it (``onset_offset >= 0``) — the
    same population the lead time is defined over. Row order is never consulted; see the module docstring
    on why it cannot be.
    """
    onset = np.asarray(arrays["onset_offset"])
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    for idx in track_slices(arrays["track_id"]).values():
        (positive if bool((onset[idx] >= 0).any()) else negative).append(idx)
    if not positive or not negative:
        raise ValueError(f"need both crossing and never-crossing tracks, got {len(positive)}/{len(negative)}")
    return positive, negative


def _alarm_statistic(score: np.ndarray, negative: list[np.ndarray], accounting: str) -> np.ndarray:
    """Sorted per-alarm statistic whose tail count *is* the false-alarm count at a threshold.

    ``per_window``: every negative window's score. ``per_track``: each negative track's maximum, since a
    track fires at ``thr`` exactly when its maximum reaches ``thr``. Both are monotone in the threshold,
    so a budget maps to a threshold by a single sorted lookup.
    """
    if accounting == "per_window":
        values = score[np.concatenate(negative)]
    elif accounting == "per_track":
        values = np.asarray([score[idx].max() for idx in negative])
    else:
        raise ValueError(f"accounting must be 'per_window' or 'per_track', got {accounting!r}")
    return np.sort(values)


def _threshold_for(statistic: np.ndarray, allowed: int) -> tuple[float, int]:
    """Lowest threshold whose alarm count fits in ``allowed``, and that count.

    ``statistic`` is ascending, so the count at or above ``statistic[i]`` is ``len - i`` before ties. Ties
    can only push the count up, so walk forward until the budget is genuinely met — never merely assumed.
    """
    n = len(statistic)
    if allowed >= n:
        return float("-inf"), n
    index = n - allowed
    while index < n:
        threshold = float(statistic[index])
        count = int((statistic >= threshold).sum())
        if count <= allowed:
            return threshold, count
        index += 1
    return float(np.nextafter(statistic[-1], np.inf)), 0


def _detections(score: np.ndarray, onset: np.ndarray, positive: list[np.ndarray],
                threshold: float, fps: float) -> tuple[int, list[float]]:
    """Detected pedestrian count, and each detection's lead time (seconds) — its earliest alarm still
    ahead of the crossing, i.e. the largest qualifying ``onset_offset``."""
    leads: list[float] = []
    for idx in positive:
        fired = (onset[idx] >= 0) & (score[idx] >= threshold)
        if fired.any():
            leads.append(float(onset[idx][fired].max()) / fps)
    return len(leads), leads


def detection_curve(
    arrays: Arrays,
    score_key: str = "p_readout",
    *,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    accounting: str = "per_window",
    clock: DecisionClock | None = None,
) -> list[DetectionPoint]:
    """Detection rate and lead time at each false-alarm budget, for one dump and one score.

    ``score_key`` picks the head: ``p_readout`` for an onset model's horizon read-out, ``p_frame`` for the
    binary crossing head (the only score a no-onset baseline dump carries).
    """
    if score_key not in arrays:
        raise KeyError(f"dump has no {score_key!r}; available: {sorted(arrays)}")
    clock = clock or DecisionClock()
    score = np.asarray(arrays[score_key], dtype=np.float64)
    onset = np.asarray(arrays["onset_offset"])
    positive, negative = split_tracks(arrays)
    exposure = clock.hours(sum(len(idx) for idx in negative))
    statistic = _alarm_statistic(score, negative, accounting)

    points: list[DetectionPoint] = []
    for budget in budgets:
        allowed = int(budget * exposure)
        threshold, alarms = _threshold_for(statistic, allowed)
        detected, leads = _detections(score, onset, positive, threshold, clock.fps)
        points.append(
            DetectionPoint(
                budget_per_hour=float(budget), threshold=threshold, accounting=accounting,
                false_alarms=alarms, false_alarm_rate=alarms / exposure,
                detected=detected, n_positive=len(positive), detection_rate=detected / len(positive),
                mean_lead_s=float(np.mean(leads)) if leads else 0.0,
                median_lead_s=float(np.median(leads)) if leads else 0.0,
            )
        )
    return points


def aggregate_curves(curves: list[list[DetectionPoint]]) -> list[dict[str, object]]:
    """Mean and sample sd per budget across seeds — the pre-registered multi-seed reporting form.

    The sd is the sample sd (``n-1``), undefined and reported as ``nan`` for a single seed, which keeps an
    n=1 result visibly n=1 instead of dressing it as a spread of zero.
    """
    if not curves:
        raise ValueError("no curves to aggregate")
    budgets = [p.budget_per_hour for p in curves[0]]
    if any([p.budget_per_hour for p in c] != budgets for c in curves):
        raise ValueError("curves must share the same budgets to aggregate")

    def stats(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else float("nan")

    rows: list[dict[str, object]] = []
    for i, budget in enumerate(budgets):
        rate_mean, rate_sd = stats([c[i].detection_rate for c in curves])
        lead_mean, lead_sd = stats([c[i].mean_lead_s for c in curves])
        rows.append({
            "budget_per_hour": budget, "n_seeds": len(curves),
            "detection_rate_mean": rate_mean, "detection_rate_sd": rate_sd,
            "mean_lead_s_mean": lead_mean, "mean_lead_s_sd": lead_sd,
            "detection_rate_values": [c[i].detection_rate for c in curves],
        })
    return rows


def format_curve(points: list[DetectionPoint], title: str = "Detection curve") -> str:
    """Markdown table for one arm's curve."""
    head = points[0]
    lines = [
        f"## {title}",
        "",
        f"{head.n_positive} crossing pedestrians, alarms counted `{head.accounting}`.",
        "",
        "| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p.budget_per_hour:.0f} | {p.false_alarm_rate:.1f} | {p.threshold:.4f} | "
            f"{p.detection_rate * 100:.1f}% ({p.detected}/{p.n_positive}) | "
            f"{p.mean_lead_s:.2f} s | {p.median_lead_s:.2f} s |"
        )
    return "\n".join(lines) + "\n"

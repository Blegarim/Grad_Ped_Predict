"""Per-dump analyses for the campaign report (``scripts/report_campaign.py``) — each a paper number.

All take a streaming-test prediction dump (``onset_timing.load_dump``) and one score column, and are
order-independent except :func:`smoothed_scores`, which needs time order within a track (see there).

* :func:`onset_separability` — how far ahead crossing is predictable: AUC of windows whose crossing starts
  ``lo..hi`` seconds ahead against windows of pedestrians who never cross (paper §VIII-A).
* :func:`smoothed_scores` — a causal moving average over each track's scores; the AUC change it causes is the
  post-processing effect the paper sets beside the gap between objectives.
* :func:`intent_timing` — splits the deployed question into *who* crosses (per pedestrian) and *when*
  (crossers only: within the horizon vs later), which is where an anchored-trained model fails.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

__all__ = [
    "ALIGN_KEYS",
    "StackedCombiner",
    "assert_aligned",
    "compare_curves",
    "intent_timing",
    "mean_sd",
    "onset_separability",
    "smoothed_scores",
]

#: Per-window fields that must match exactly for two dumps' scores to be combined window by window.
ALIGN_KEYS: tuple[str, ...] = ("track_id", "onset_offset", "future_observed", "crosses")

Arrays = dict[str, np.ndarray]

#: Time-to-onset buckets in seconds (the last is open-ended), the strata the paper reports.
SEPARABILITY_EDGES_S: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample sd (``nan`` for one value — an n=1 number stays visibly n=1)."""
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else float("nan")


def onset_separability(
    arrays: Arrays, score: np.ndarray, *, fps: float = 30.0, edges_s: Sequence[float] = SEPARABILITY_EDGES_S
) -> dict[str, dict[str, float]]:
    """``{bucket: {auc, n}}``: windows with onset in the bucket vs every window of a never-crossing pedestrian."""
    onset_s = np.asarray(arrays["onset_offset"], dtype=float) / fps
    ahead = np.asarray(arrays["onset_offset"]) >= 0
    never = np.asarray(arrays["track_crosses"]) == 0
    bounds = [*zip(edges_s, [*edges_s[1:], math.inf], strict=True)]
    out: dict[str, dict[str, float]] = {}
    for lo, hi in bounds:
        pos = ahead & (onset_s >= lo) & (onset_s < hi)
        both = pos | never
        name = f"{lo:g}-{hi:g}s" if math.isfinite(hi) else f">={lo:g}s"
        out[name] = {"auc": _auc(pos[both].astype(int), score[both]), "n": int(pos.sum())}
    return out


def smoothed_scores(arrays: Arrays, score: np.ndarray, *, k: int = 15, causal: bool = True) -> np.ndarray:
    """Moving average of ``k`` scores within each track.

    ``causal`` (default): the trailing mean of the last ``k`` windows — deployable, it uses no future window.
    Otherwise centered (``k // 2`` windows each side, truncated at the track ends) — an offline upper bound that
    looks ahead, reported so the paper's earlier "k=15 moving average" can be compared like for like.

    Time order within a track is taken from ``future_observed`` (frames left in the track after the window),
    which falls as the window slides forward. PIE splits ~5% of pedestrians across occlusion gaps whose
    relative order is not recoverable; for those the order is approximate.
    """
    tid = np.asarray(arrays["track_id"])
    left = np.asarray(arrays["future_observed"])
    score = np.asarray(score, dtype=float)
    out = np.empty_like(score)
    for track in np.unique(tid):
        idx = np.flatnonzero(tid == track)
        idx = idx[np.argsort(-left[idx], kind="stable")]
        csum = np.cumsum(np.insert(score[idx], 0, 0.0))
        pos = np.arange(len(idx))
        if causal:
            lo, hi = np.maximum(pos + 1 - k, 0), pos + 1
        else:
            lo, hi = np.maximum(pos - k // 2, 0), np.minimum(pos + k // 2 + 1, len(idx))
        out[idx] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def compare_curves(
    arm: list[list[dict]], base: list[list[dict]], *, arm_name: str = "arm", base_name: str = "baseline"
) -> tuple[str, list[dict]]:
    """The pre-registered rule (SEED_PLAN_2026-09-21): an arm wins if its mean detection rate beats the
    baseline's by more than the sum of the two sample sds at >= 3 budgets. Curves are ``DetectionPoint`` dicts,
    one list per seed. Returns the verdict and one row per budget."""
    rows, arm_wins, base_wins = [], 0, 0
    for i in range(len(arm[0])):
        am, asd = mean_sd([c[i]["detection_rate"] for c in arm])
        bm, bsd = mean_sd([c[i]["detection_rate"] for c in base])
        gap, sds = am - bm, asd + bsd
        mark = arm_name if gap > sds else (base_name if -gap > sds else "neither")
        arm_wins += mark == arm_name
        base_wins += mark == base_name
        rows.append({"budget": arm[0][i]["budget_per_hour"], "arm": am, "arm_sd": asd, "base": bm, "base_sd": bsd,
                     "gap": gap, "sd_sum": sds, "beyond_sd": mark})
    verdict = f"{arm_name} WINS" if arm_wins >= 3 else (f"{base_name} WINS" if base_wins >= 3 else "INCONCLUSIVE")
    return verdict, rows


def assert_aligned(a: Arrays, b: Arrays) -> None:
    """Refuse to combine two dumps unless they list the same windows in the same order."""
    for key in ALIGN_KEYS:
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
            raise ValueError(f"dumps are not window-aligned on {key!r}; combining them would mix windows")


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class StackedCombiner:
    """Logistic regression over ``logit p_a``, ``logit p_b`` and their product — fitted on one split (val),
    applied unchanged to another (test). The only fitted combination rule; everything else is fixed."""

    def __init__(self) -> None:
        self._model = LogisticRegression(max_iter=1000)

    @staticmethod
    def _features(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        la, lb = _logit(p_a), _logit(p_b)
        return np.column_stack([la, lb, la * lb])

    def fit(self, p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> StackedCombiner:
        self._model.fit(self._features(p_a, p_b), np.asarray(y).astype(int))
        return self

    def score(self, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._features(p_a, p_b))[:, 1]


def intent_timing(arrays: Arrays, score: np.ndarray, *, horizon: int = 32) -> dict[str, float]:
    """AUCs for the deployed question (stored label), *who* crosses, and *when*.

    who: per pedestrian, mean score over windows not already past the crossing, against ``track_crosses``.
    when: over windows whose crossing is still ahead, onset within ``horizon`` frames against later.
    """
    onset = np.asarray(arrays["onset_offset"])
    ever = np.asarray(arrays["track_crosses"])
    tid = np.asarray(arrays["track_id"])
    score = np.asarray(score, dtype=float)
    keep = (onset >= 0) | (ever == 0)
    tracks, inverse = np.unique(tid[keep], return_inverse=True)
    sums = np.bincount(inverse, weights=score[keep])
    counts = np.bincount(inverse)
    label = np.zeros(len(tracks), dtype=int)
    label[inverse] = ever[keep]
    ahead = onset >= 0
    return {
        "window": _auc(np.asarray(arrays["crosses"]).astype(int), score),
        "intent": _auc(label, sums / counts),
        "timing": _auc((onset[ahead] < horizon).astype(int), score[ahead]),
    }

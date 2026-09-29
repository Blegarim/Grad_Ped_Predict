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
from sklearn.metrics import roc_auc_score

__all__ = ["mean_sd", "intent_timing", "onset_separability", "smoothed_scores"]

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

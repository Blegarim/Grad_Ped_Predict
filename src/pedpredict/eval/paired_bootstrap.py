"""Paired pedestrian bootstrap for detection-rate gaps — how much of a gap the test set can resolve.

The detection curve (:mod:`pedpredict.eval.detection_curve`) is computed on 205 crossing and 495
never-crossing PIE test pedestrians. Resampling those pedestrians alone moves one model's detection rate by
~4 points at 205 alarms/hour and ~9 at 41 (measured 2026-10-04), so a comparison of unpaired means can only
resolve gaps far larger than any objective produces. Scoring two models on the **same** resampled
pedestrians cancels the shared part of that noise when the models' alarms are correlated — strongly for
tree models (seed vs seed: ~2 points), barely for the deep models (6-12 points), which is why this is the
instrument for the tree study and not a rescue for the deep comparison.

One resample draws pedestrians with replacement, crossing and never-crossing separately (their counts are
fixed by the test set). A pedestrian drawn twice counts twice, both in the alarm budget's exposure and in the
detection rate. With every multiplicity at 1 the numbers are exactly :func:`detection_curve`'s — pinned in
``tests/test_paired_bootstrap.py``. Seeds can be resampled too (two-level bootstrap): each arm's rate for a
resample is the mean over seeds drawn with replacement, so training randomness enters the interval.

Pure numpy, no data: input is the standard dump arrays plus one score array per seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pedpredict.eval.detection_curve import DEFAULT_BUDGETS, DecisionClock, split_tracks

__all__ = ["PreparedScore", "prepare_score", "weighted_rates", "GapEstimate", "paired_contrast",
           "paired_gap"]

Arrays = dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedScore:
    """One score array reduced to what any resample needs, per accounting rule.

    ``values`` are the alarm statistic's distinct values, descending; ``group`` maps each alarm unit (a
    never-crosser window, or a never-crosser track under ``per_track``) to its value's index; ``owner``
    maps each unit to its never-crossing track; ``window_track`` counts exposure (windows per never-crosser
    track); ``crosser_max`` is each crossing track's highest score while its crossing is still ahead.
    """

    values: np.ndarray
    group: np.ndarray
    owner: np.ndarray
    window_track: np.ndarray
    crosser_max: np.ndarray
    accounting: str


def prepare_score(arrays: Arrays, score: np.ndarray, accounting: str = "per_window") -> PreparedScore:
    """Precompute ``score``'s alarm statistic and per-crosser maxima over ``arrays``' pedestrians."""
    score = np.asarray(score, dtype=np.float64)
    onset = np.asarray(arrays["onset_offset"])
    positive, negative = split_tracks(arrays)
    crosser_max = np.asarray([score[idx][onset[idx] >= 0].max() for idx in positive])
    window_track = np.asarray([len(idx) for idx in negative], dtype=np.int64)
    if accounting == "per_window":
        stat = score[np.concatenate(negative)]
        owner = np.repeat(np.arange(len(negative)), window_track)
    elif accounting == "per_track":
        stat = np.asarray([score[idx].max() for idx in negative])
        owner = np.arange(len(negative))
    else:
        raise ValueError(f"accounting must be 'per_window' or 'per_track', got {accounting!r}")
    values, inverse = np.unique(-stat, return_inverse=True)   # ascending in -stat = descending in stat
    return PreparedScore(-values, inverse.ravel(), owner, window_track, crosser_max, accounting)


def weighted_rates(
    prep: PreparedScore,
    neg_weight: np.ndarray,
    pos_weight: np.ndarray,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    clock: DecisionClock | None = None,
) -> np.ndarray:
    """Detection rate at each budget when never-crosser track ``j`` counts ``neg_weight[j]`` times and
    crossing track ``i`` counts ``pos_weight[i]`` times. Same threshold rule as the detection curve: the
    lowest threshold whose (weighted) alarm count fits the budget."""
    clock = clock or DecisionClock()
    exposure = clock.hours(float(np.dot(neg_weight, prep.window_track)))
    per_value = np.bincount(prep.group, weights=neg_weight[prep.owner], minlength=len(prep.values))
    alarms_at_or_above = np.cumsum(per_value)        # count with score >= values[j]
    total = alarms_at_or_above[-1]
    n_pos = float(pos_weight.sum())
    rates = np.empty(len(budgets))
    for b, budget in enumerate(budgets):
        allowed = int(budget * exposure)
        if allowed >= total:
            threshold = -np.inf
        else:
            j = int(np.searchsorted(alarms_at_or_above, allowed, side="right")) - 1
            # None fits: just above the top alarm value, as detection_curve does -- a crosser scoring
            # above every never-crosser is still detected at zero false alarms.
            threshold = prep.values[j] if j >= 0 else float(np.nextafter(prep.values[0], np.inf))
        rates[b] = float(pos_weight[prep.crosser_max >= threshold].sum()) / n_pos
    return rates


@dataclass(frozen=True)
class GapEstimate:
    """A linear contrast of seed-averaged detection rates per budget, e.g. ``b - a``: point estimate on the
    full test set and a paired-bootstrap percentile interval."""

    budgets: tuple[float, ...]
    rates: dict[str, np.ndarray]   # per arm: mean over its seeds, full test set
    gap: np.ndarray                # the contrast, full test set
    lo: np.ndarray                 # percentile interval of the resampled contrast
    hi: np.ndarray
    p_not_positive: np.ndarray     # share of resamples with contrast <= 0
    n_boot: int

    def excludes_zero(self) -> np.ndarray:
        """Per budget: +1 if the interval lies above zero, -1 if below, 0 if it covers zero."""
        return np.where(self.lo > 0, 1, np.where(self.hi < 0, -1, 0))

    def within(self, margin: float) -> np.ndarray:
        """Per budget: the whole interval lies inside ``(-margin, +margin)`` (equivalence at that margin)."""
        return (self.lo > -margin) & (self.hi < margin)


def paired_contrast(
    arrays: Arrays,
    arms: Mapping[str, Sequence[np.ndarray]],
    weights: Mapping[str, float],
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    accounting: str = "per_window",
    n_boot: int = 2000,
    level: float = 0.95,
    resample_seeds: bool = True,
    seed: int = 0,
) -> GapEstimate:
    """Paired bootstrap of ``sum_arm weights[arm] * mean_seeds(rate[arm])``.

    Every arm and seed is scored on the same resampled pedestrians, so the shared part of the test-set
    noise cancels. With ``resample_seeds`` each arm's seeds are also drawn with replacement per resample.
    """
    if set(weights) - set(arms) or any(not arms[a] for a in weights):
        raise ValueError(f"every weighted arm needs at least one score array: {sorted(weights)}")
    prep = {a: [prepare_score(arrays, s, accounting) for s in arms[a]] for a in weights}
    first = next(iter(prep.values()))[0]
    n_neg, n_pos = len(first.window_track), len(first.crosser_max)

    def contrast(neg_w: np.ndarray, pos_w: np.ndarray, picks: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
        rates = {a: np.mean([weighted_rates(prep[a][i], neg_w, pos_w, budgets) for i in picks[a]], axis=0)
                 for a in prep}
        return sum(weights[a] * rates[a] for a in prep), rates

    gap, rates = contrast(np.ones(n_neg), np.ones(n_pos), {a: np.arange(len(p)) for a, p in prep.items()})
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(budgets)))
    for r in range(n_boot):
        neg_w = np.bincount(rng.integers(0, n_neg, n_neg), minlength=n_neg).astype(np.float64)
        pos_w = np.bincount(rng.integers(0, n_pos, n_pos), minlength=n_pos).astype(np.float64)
        picks = {a: rng.integers(0, len(p), len(p)) if resample_seeds else np.arange(len(p))
                 for a, p in prep.items()}
        draws[r] = contrast(neg_w, pos_w, picks)[0]
    tail = (1.0 - level) / 2.0
    return GapEstimate(
        budgets=tuple(float(b) for b in budgets), rates=rates, gap=gap,
        lo=np.quantile(draws, tail, axis=0), hi=np.quantile(draws, 1.0 - tail, axis=0),
        p_not_positive=(draws <= 0).mean(axis=0), n_boot=n_boot,
    )


def paired_gap(arrays: Arrays, scores_a: Sequence[np.ndarray], scores_b: Sequence[np.ndarray],
               **kwargs) -> GapEstimate:
    """``mean(b) - mean(a)`` — the two-arm case of :func:`paired_contrast` (arms named ``a`` and ``b``)."""
    return paired_contrast(arrays, {"a": scores_a, "b": scores_b}, {"a": -1.0, "b": 1.0}, **kwargs)

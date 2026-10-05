"""Paired pedestrian bootstrap: at unit weights it IS the detection curve; resampling behaves as stated."""

from __future__ import annotations

import numpy as np
import pytest

from pedpredict.eval.detection_curve import detection_curve, split_tracks
from pedpredict.eval.paired_bootstrap import paired_gap, prepare_score, weighted_rates

_BUDGETS = (1200, 460, 205, 95, 41)


def _synthetic(seed: int = 0, n_cross: int = 30, n_never: int = 60) -> dict[str, np.ndarray]:
    """Tracks of 20-60 windows; crossers count down to an onset, then a few post-onset windows."""
    rng = np.random.default_rng(seed)
    onset, tid = [], []
    for i in range(n_cross):
        n = int(rng.integers(20, 60))
        start = int(rng.integers(0, 300))
        offsets = list(range(start + 3 * (n - 1), start - 1, -3))
        offsets[-3:] = [-1, -1, -1]   # already crossed
        onset += offsets
        tid += [f"c{i}"] * n
    for i in range(n_never):
        n = int(rng.integers(20, 60))
        onset += [-1] * n
        tid += [f"n{i}"] * n
    onset = np.asarray(onset)
    return {"onset_offset": onset, "track_id": np.asarray(tid), "crosses": ((onset >= 0) & (onset < 32)).astype(int)}


def _score(arrays: dict[str, np.ndarray], seed: int, ties: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    near = (arrays["onset_offset"] >= 0) & (arrays["onset_offset"] < 90)
    score = rng.random(len(near)) + 0.6 * near
    return np.round(score, 1) if ties else score


@pytest.mark.parametrize("accounting", ["per_window", "per_track"])
@pytest.mark.parametrize("ties", [False, True])
def test_unit_weights_reproduce_detection_curve(accounting: str, ties: bool) -> None:
    arrays = _synthetic()
    score = _score(arrays, 1, ties=ties)
    expected = [p.detection_rate for p in
                detection_curve({**arrays, "s": score}, "s", budgets=_BUDGETS, accounting=accounting)]
    prep = prepare_score(arrays, score, accounting)
    got = weighted_rates(prep, np.ones(len(prep.window_track)), np.ones(len(prep.crosser_max)), _BUDGETS)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("accounting", ["per_window", "per_track"])
def test_integer_weights_equal_duplicated_pedestrians(accounting: str) -> None:
    """A weight of k is the same as that pedestrian appearing k times as separate tracks."""
    arrays = _synthetic(seed=3)
    score = _score(arrays, 4)
    positive, negative = split_tracks(arrays)
    rng = np.random.default_rng(5)
    pos_w, neg_w = rng.integers(0, 3, len(positive)), rng.integers(0, 3, len(negative))
    pos_w[0] = max(pos_w[0], 1)
    neg_w[0] = max(neg_w[0], 1)
    parts, tids = [], []
    for kind, groups, weights in (("p", positive, pos_w), ("n", negative, neg_w)):
        for j, (idx, w) in enumerate(zip(groups, weights, strict=True)):
            for copy in range(int(w)):
                parts.append(idx)
                tids.append(np.full(len(idx), f"{kind}{j}_{copy}"))
    idx = np.concatenate(parts)
    dup = {"onset_offset": arrays["onset_offset"][idx], "track_id": np.concatenate(tids), "s": score[idx]}
    expected = [p.detection_rate for p in detection_curve(dup, "s", budgets=_BUDGETS, accounting=accounting)]
    got = weighted_rates(prepare_score(arrays, score, accounting), neg_w.astype(float), pos_w.astype(float),
                         _BUDGETS)
    np.testing.assert_allclose(got, expected)


def test_zero_budget_still_detects_crossers_above_every_negative() -> None:
    arrays = _synthetic(seed=6)
    score = np.zeros(len(arrays["onset_offset"]))
    first_crosser = np.flatnonzero(arrays["track_id"] == "c0")
    score[first_crosser] = 2.0    # one crossing track outranks every never-crosser window
    expected = detection_curve({**arrays, "s": score}, "s", budgets=(0,))[0].detection_rate
    prep = prepare_score(arrays, score)
    got = weighted_rates(prep, np.ones(len(prep.window_track)), np.ones(len(prep.crosser_max)), (0,))
    assert got[0] == pytest.approx(expected) and expected > 0


def test_identical_arms_give_a_zero_gap_and_a_degenerate_interval() -> None:
    arrays = _synthetic()
    score = _score(arrays, 1)
    est = paired_gap(arrays, [score], [score], n_boot=50, resample_seeds=False)
    np.testing.assert_allclose(est.gap, 0.0)
    np.testing.assert_allclose(est.lo, 0.0)
    np.testing.assert_allclose(est.hi, 0.0)
    assert (est.excludes_zero() == 0).all()


def test_a_clearly_better_arm_excludes_zero() -> None:
    arrays = _synthetic(n_cross=80, n_never=160)
    weak = [np.random.default_rng(s).random(len(arrays["onset_offset"])) for s in (1, 2, 3)]
    ahead = arrays["onset_offset"] >= 0
    strong = [w + 0.5 * ahead * np.random.default_rng(10 + i).random(len(w)) for i, w in enumerate(weak)]
    est = paired_gap(arrays, weak, strong, n_boot=200, seed=1)
    assert (est.gap > 0).all()
    assert (est.excludes_zero()[1:] == 1).all()


def test_point_estimate_is_the_seed_mean_on_the_full_test_set() -> None:
    arrays = _synthetic()
    scores = [_score(arrays, s) for s in (1, 2)]
    est = paired_gap(arrays, scores[:1], scores, n_boot=10)
    rates = [[p.detection_rate for p in detection_curve({**arrays, "s": s}, "s", budgets=_BUDGETS)] for s in scores]
    np.testing.assert_allclose(est.rates["b"], np.mean(rates, axis=0))
    np.testing.assert_allclose(est.rates["a"], rates[0])


def test_difference_of_gaps_contrast() -> None:
    """(d - c) - (b - a) on the full set equals the arithmetic of the four seed means."""
    from pedpredict.eval.paired_bootstrap import paired_contrast

    arrays = _synthetic()
    arms = {name: [_score(arrays, s)] for name, s in zip("abcd", (1, 2, 3, 4), strict=True)}
    est = paired_contrast(arrays, arms, {"a": 1.0, "b": -1.0, "c": -1.0, "d": 1.0}, n_boot=20)
    np.testing.assert_allclose(est.gap, est.rates["d"] - est.rates["c"] - (est.rates["b"] - est.rates["a"]))
    assert est.within(10.0).all() and est.lo.shape == (5,)

"""Campaign-report analyses: separability by time to onset, causal within-track smoothing, the who/when split."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pedpredict.eval.campaign_report import intent_timing, mean_sd, onset_separability, smoothed_scores


def _arrays(rows: list[tuple[str, int, int, int, float]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """rows: (track_id, onset_offset, future_observed, track_crosses, score); crosses = onset in 0..31."""
    tid, onset, left, ever, score = map(np.asarray, zip(*rows, strict=True))
    arrays = {"track_id": tid.astype(str), "onset_offset": onset.astype(int), "future_observed": left.astype(int),
              "track_crosses": ever.astype(int), "crosses": ((onset >= 0) & (onset < 32)).astype(int)}
    return arrays, score.astype(float)


def test_separability_scores_each_bucket_against_never_crossers() -> None:
    arrays, score = _arrays([
        ("p", 15, 300, 1, 0.9),      # 0.5 s ahead: scored high
        ("q", 150, 300, 1, 0.1),     # 5 s ahead: scored like a never-crosser
        ("n1", -1, 300, 0, 0.2), ("n2", -1, 300, 0, 0.3),
    ])
    out = onset_separability(arrays, score)
    assert out["0-1s"] == {"auc": 1.0, "n": 1}
    assert out["5-10s"]["auc"] == 0.0
    assert math.isnan(out[">=10s"]["auc"]) and out[">=10s"]["n"] == 0


def test_smoothing_is_causal_and_follows_time_not_row_order() -> None:
    # one track, time order = decreasing future_observed; rows deliberately shuffled
    arrays, score = _arrays([("t", -1, 90, 0, 3.0), ("t", -1, 99, 0, 1.0), ("t", -1, 96, 0, 2.0)])
    out = smoothed_scores(arrays, score, k=2)
    # in time order the scores are 1, 2, 3 -> trailing means of 2: 1, 1.5, 2.5
    assert out.tolist() == pytest.approx([2.5, 1.0, 1.5])


def test_centered_smoothing_looks_both_ways() -> None:
    arrays, score = _arrays([("t", -1, 99, 0, 1.0), ("t", -1, 96, 0, 2.0), ("t", -1, 93, 0, 6.0)])
    # k=3 centered: ends truncated -> (1+2)/2, (1+2+6)/3, (2+6)/2
    assert smoothed_scores(arrays, score, k=3, causal=False).tolist() == pytest.approx([1.5, 3.0, 4.0])


def test_smoothing_never_mixes_tracks() -> None:
    arrays, score = _arrays([("a", -1, 9, 0, 1.0), ("b", -1, 9, 0, 5.0)])
    assert smoothed_scores(arrays, score, k=15).tolist() == [1.0, 5.0]


def test_intent_timing_separates_who_from_when() -> None:
    # A model that knows WHO crosses but ranks far-off crossings above imminent ones (the anchored pattern).
    arrays, score = _arrays([
        ("c", 10, 300, 1, 0.6), ("c", 200, 300, 1, 0.9), ("c", -1, 300, 1, 0.0),   # last row: already crossed, dropped
        ("n", -1, 300, 0, 0.1), ("n", -1, 300, 0, 0.2),
    ])
    out = intent_timing(arrays, score)
    assert out["intent"] == 1.0        # the crosser's mean (0.75) is above the never-crosser's (0.15)
    assert out["timing"] == 0.0        # the imminent window scores BELOW the far-off one


def test_mean_sd_keeps_n1_visibly_n1() -> None:
    m, s = mean_sd([0.5])
    assert m == 0.5 and math.isnan(s)
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(math.sqrt(2)))

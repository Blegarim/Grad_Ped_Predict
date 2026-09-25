"""Detection-latency curve — the project's primary metric.

Pins the parts a wrong number would hide in: the false-alarm budget is a hard constraint and is never
exceeded, a threshold is the *lowest* one that fits (so detection is maximised subject to the budget),
lead time is measured from the FIRST alarm that still precedes the crossing, alarms fired after a
pedestrian has already stepped out are neither credited nor penalised, the two accounting rules count
different things over the same exposure, and a dump whose tracks are out of temporal order is rejected
rather than silently mis-timed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pedpredict.eval import detection_curve as dc

_CLOCK = dc.DecisionClock()  # 30 fps / stride 3 -> 36,000 decisions/hour


def _track(tid: str, onsets: list[int], scores: list[float], start: int = 4000) -> dict[str, list]:
    """One track's windows, newest last; ``future_observed`` decreases as the window slides forward."""
    n = len(onsets)
    return {
        "track_id": [tid] * n,
        "onset_offset": list(onsets),
        "future_observed": [start - 3 * i for i in range(n)],
        "track_crosses": [int(any(o >= 0 for o in onsets))] * n,
        "p_readout": list(scores),
    }


def _arrays(*tracks: dict[str, list]) -> dict[str, np.ndarray]:
    keys = tracks[0].keys()
    merged = {k: [v for t in tracks for v in t[k]] for k in keys}
    return {
        "track_id": np.asarray(merged["track_id"], dtype=str),
        "onset_offset": np.asarray(merged["onset_offset"], dtype=np.int64),
        "future_observed": np.asarray(merged["future_observed"], dtype=np.int64),
        "track_crosses": np.asarray(merged["track_crosses"], dtype=np.int64),
        "p_readout": np.asarray(merged["p_readout"], dtype=np.float32),
    }


def _separable(n_neg: int = 200) -> dict[str, np.ndarray]:
    """One crosser silent at 90 frames out, then alarming; many quiet non-crossers."""
    crosser = _track("pos", [90, 60, 30, -1], [0.0, 0.8, 0.9, 0.95])
    negatives = [_track(f"neg{i}", [-1] * 5, [0.01 * (i % 7)] * 5) for i in range(n_neg)]
    return _arrays(crosser, *negatives)


def test_decision_clock_matches_the_documented_rate() -> None:
    assert _CLOCK.decisions_per_hour == pytest.approx(36_000.0)
    assert _CLOCK.hours(36_000) == pytest.approx(1.0)


def test_budget_is_never_exceeded() -> None:
    rng = np.random.default_rng(0)
    tracks = [_track("pos", [60, 30, 10], list(rng.random(3)))]
    tracks += [_track(f"neg{i}", [-1] * 8, list(rng.random(8))) for i in range(300)]
    for point in dc.detection_curve(_arrays(*tracks), budgets=dc.DEFAULT_BUDGETS):
        assert point.false_alarm_rate <= point.budget_per_hour + 1e-9


def test_threshold_is_the_lowest_that_fits_the_budget() -> None:
    """One step lower must break the budget — otherwise detections are being left on the table."""
    arrays = _separable()
    score = np.sort(np.unique(arrays["p_readout"]))
    for point in dc.detection_curve(arrays, budgets=(1200, 205)):
        lower = score[score < point.threshold]
        if not len(lower):
            continue
        negative = np.concatenate(dc.split_tracks(arrays)[1])
        alarms = int((arrays["p_readout"][negative] >= lower[-1]).sum())
        exposure = _CLOCK.hours(len(negative))
        assert alarms / exposure > point.budget_per_hour


def test_lead_time_comes_from_the_first_alarm_still_ahead_of_the_crossing() -> None:
    arrays = _separable()
    point = dc.detection_curve(arrays, budgets=(1200,))[0]
    assert point.detected == 1
    # Windows at onset 90 / 60 / 30 score 0.0 / 0.8 / 0.9. The 90-frame window is below threshold, so the
    # 60-frame one fires first -> 2.0 s. Neither the higher-scoring 30-frame window (1.0 s) nor the
    # highest-scoring window of all (0.95, already crossed) sets the lead.
    assert 0.0 < point.threshold < 0.8
    assert point.mean_lead_s == pytest.approx(60 / 30.0)


def test_alarms_after_the_crossing_do_not_count_as_a_detection() -> None:
    """The last window of a crosser has onset -1 (already stepped out) and the only high score."""
    crosser = _track("pos", [90, 60, -1], [0.0, 0.0, 1.0])
    negatives = [_track(f"neg{i}", [-1] * 5, [0.5] * 5) for i in range(200)]
    point = dc.detection_curve(_arrays(crosser, *negatives), budgets=(1200,))[0]
    assert point.detected == 0
    assert point.mean_lead_s == 0.0


def test_accounting_rules_differ_on_a_repeatedly_alarming_pedestrian() -> None:
    """One noisy non-crosser alarming 10x is 10 window-alarms but a single nuisance pedestrian."""
    crosser = _track("pos", [60, 30], [0.9, 0.9])
    noisy = _track("noisy", [-1] * 10, [0.99] * 10)
    quiet = [_track(f"neg{i}", [-1] * 10, [0.0] * 10) for i in range(50)]
    arrays = _arrays(crosser, noisy, *quiet)
    # 510 negative windows = 0.01417 h of exposure; at 720/hr that is room for 10 window-alarms.
    per_window = dc.detection_curve(arrays, budgets=(720,), accounting="per_window")[0]
    per_track = dc.detection_curve(arrays, budgets=(720,), accounting="per_track")[0]
    assert per_window.false_alarms == 10
    assert per_track.false_alarms == 1
    # Same exposure, so the rates are directly comparable.
    assert per_window.false_alarm_rate == pytest.approx(10 * per_track.false_alarm_rate)


def test_detection_rate_is_monotone_in_the_budget() -> None:
    points = dc.detection_curve(_separable(), budgets=(1200, 460, 205, 95, 41))
    rates = [p.detection_rate for p in points]
    assert rates == sorted(rates, reverse=True)


def test_row_order_does_not_change_any_number() -> None:
    """A pedestrian split across occlusion gaps reaches the dump as several segments whose relative order
    is unrecoverable, so the metric must not depend on row order at all."""
    arrays = _separable()
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(arrays["track_id"]))
    shuffled = {k: v[perm].copy() for k, v in arrays.items()}
    before = dc.detection_curve(arrays, budgets=dc.DEFAULT_BUDGETS)
    after = dc.detection_curve(shuffled, budgets=dc.DEFAULT_BUDGETS)
    assert [p.as_dict() for p in before] == [p.as_dict() for p in after]


def test_lead_time_is_the_earliest_alarm_across_disjoint_segments() -> None:
    """Two segments of one pedestrian, the earlier-warning one placed last in the file."""
    late = _track("pos", [40, 20], [0.9, 0.9], start=200)
    early = _track("pos", [95, 80], [0.9, 0.9], start=900)
    negatives = [_track(f"neg{i}", [-1] * 5, [0.0] * 5) for i in range(200)]
    point = dc.detection_curve(_arrays(late, early, *negatives), budgets=(1200,))[0]
    assert point.detected == 1
    assert point.mean_lead_s == pytest.approx(95 / 30.0)


def test_a_dump_of_one_class_is_rejected() -> None:
    only_negative = _arrays(*[_track(f"neg{i}", [-1] * 4, [0.5] * 4) for i in range(5)])
    with pytest.raises(ValueError, match="crossing and never-crossing"):
        dc.split_tracks(only_negative)


def test_missing_score_key_names_what_is_available() -> None:
    with pytest.raises(KeyError, match="p_frame"):
        dc.detection_curve(_separable(), "p_frame")


def test_aggregate_reports_sample_sd_and_nan_for_one_seed() -> None:
    curves = [dc.detection_curve(_separable(n_neg=n), budgets=(1200,)) for n in (200, 220, 240)]
    rows = dc.aggregate_curves(curves)
    assert rows[0]["n_seeds"] == 3
    assert rows[0]["detection_rate_mean"] == pytest.approx(
        float(np.mean([c[0].detection_rate for c in curves]))
    )
    single = dc.aggregate_curves(curves[:1])
    assert np.isnan(single[0]["detection_rate_sd"])


def test_aggregate_rejects_mismatched_budgets() -> None:
    a = dc.detection_curve(_separable(), budgets=(1200,))
    b = dc.detection_curve(_separable(), budgets=(460,))
    with pytest.raises(ValueError, match="same budgets"):
        dc.aggregate_curves([a, b])


def test_format_curve_reports_counts_and_the_accounting_rule() -> None:
    text = dc.format_curve(dc.detection_curve(_separable(), budgets=(1200,)), title="R3")
    assert "## R3" in text
    assert "per_window" in text
    assert "1/1" in text

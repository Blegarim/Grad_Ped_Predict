"""Negative-composition + horizon-sweep math (S1 onset fields).

Hand-checked expectations on tiny synthetic record sets. Nothing here touches PIE, an LMDB, or a
frame — the whole point of these reports is that they run on a laptop.
"""

from __future__ import annotations

import pytest

from pedpredict.data.onset_stats import (
    NEGATIVE_KINDS,
    classify_window,
    compose,
    format_composition_table,
    format_sweep_table,
    horizon_sweep,
    is_usable,
)


def _rec(onset: int, *, future: int = 1000, track: int | None = None) -> dict[str, int]:
    """One window's S1 annotation; ``track_crosses`` defaults to whether an onset was seen."""
    return {"onset_offset": onset, "future_observed": future,
            "track_crosses": int(onset >= 0) if track is None else track}


@pytest.mark.parametrize(
    ("onset", "track", "expected"),
    [
        (0, 1, "positive"),           # crosses on the very next frame
        (31, 1, "positive"),          # last frame inside H=32
        (32, 1, "hard_temporal"),     # one frame past the horizon — the confusable case
        (900, 1, "hard_temporal"),    # crosses eventually, far away
        (-1, 1, "already_crossed"),   # no future crossing, but the track has one behind it
        (-1, 0, "genuine"),           # never crosses anywhere
    ],
)
def test_classify_window_covers_every_kind(onset: int, track: int, expected: str) -> None:
    assert classify_window(_rec(onset, track=track), 32) == expected


def test_classify_boundary_moves_with_horizon() -> None:
    """The same window flips label as H moves — the sweep's whole premise."""
    record = _rec(40)
    assert classify_window(record, 32) == "hard_temporal"
    assert classify_window(record, 60) == "positive"


def test_is_usable_requires_observed_future_or_a_seen_crossing() -> None:
    assert is_usable(_rec(10, future=5), 150)          # crossing seen -> answer is known
    assert is_usable(_rec(-1, future=150, track=0), 150)
    assert not is_usable(_rec(-1, future=149, track=0), 150)  # H-censored: cannot claim "did not cross"


def test_compose_counts_and_imbalance() -> None:
    records = [_rec(5), _rec(40), _rec(40), _rec(-1, track=1)] + [_rec(-1, track=0)] * 6
    comp = compose("toy", records, 32)
    assert (comp.total, comp.positive) == (10, 1)
    assert (comp.genuine, comp.hard_temporal, comp.already_crossed) == (6, 2, 1)
    assert comp.negatives == 9
    assert comp.imbalance == pytest.approx(9.0)
    assert comp.rate("positive") == pytest.approx(0.1)
    assert sum(comp.rate(k) for k in ("positive", *NEGATIVE_KINDS)) == pytest.approx(1.0)


def test_compose_handles_an_all_negative_split() -> None:
    comp = compose("toy", [_rec(-1, track=0)] * 3, 32)
    assert comp.positive == 0
    assert comp.imbalance != comp.imbalance  # nan, not a ZeroDivisionError


def test_horizon_sweep_grows_positives_and_censors_short_futures() -> None:
    records = [_rec(40, future=200), _rec(100, future=200), _rec(-1, future=50, track=0)]
    at32, at60, at150 = horizon_sweep(records, [32, 60, 150], band_frames=60)

    assert (at32.usable, at32.positive) == (3, 0)
    assert at32.band == 1                      # only the onset=40 window sits in [32, 92)
    assert at60.positive == 1                  # onset=40 is now inside the horizon
    assert at150.positive == 2                 # onset=100 joins it
    assert at150.censored_out == 1             # the future=50 non-crosser is unknowable at H=150
    assert at150.usable == 2


def test_horizon_sweep_band_ratio_and_imbalance() -> None:
    records = [_rec(10), _rec(40), _rec(-1, track=0), _rec(-1, track=0)]
    (point,) = horizon_sweep(records, [32], band_frames=60)
    assert point.positive == 1
    assert point.imbalance == pytest.approx(3.0)
    assert point.band_per_positive == pytest.approx(1.0)


def test_formatters_emit_one_markdown_row_per_entry() -> None:
    records = [_rec(5), _rec(40), _rec(-1, track=0)]
    comp_table = format_composition_table([compose("toy", records, 32)])
    sweep_table = format_sweep_table(horizon_sweep(records, [32, 60]))
    assert comp_table.count("\n") == 2       # header + separator + 1 row
    assert sweep_table.count("\n") == 3      # header + separator + 2 rows
    assert "hard-temporal" in comp_table
    assert "band:pos" in sweep_table

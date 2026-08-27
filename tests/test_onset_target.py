"""Hazard supervision built from the S1 fields — the four cases, and the comparability invariant.

Stage B of the onset-timing method (docs/METHODOLOGY.md prong 2). ``onset_target`` is where the binary
``crosses`` question becomes ``K`` per-bin questions plus a mask, so this is the cheapest place to
establish correctness: pure tensors in, pure tensors out, no model and no LMDB.

The load-bearing test is :func:`test_readout_label_reproduces_stored_crosses` — at the canonical horizon
the readout label must equal the ``crosses`` label the generator already writes, for every window of a
real generated track. If that ever drifts, the hazard model and the four existing baselines stop
answering the same question and every comparison in RESULTS_MATRIX.md silently changes meaning.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pedpredict.config import DataCfg
from pedpredict.data.onset_stats import is_usable
from pedpredict.data.onset_target import OnsetSpec, hazard_targets, readout_targets
from pedpredict.data.pie_sequences import window_track


def _labels(onset: list[int], observed: list[int], ever: list[int]) -> dict[str, torch.Tensor]:
    return {
        "onset_offset": torch.tensor(onset),
        "future_observed": torch.tensor(observed),
        "track_crosses": torch.tensor(ever),
    }


# --------------------------------------------------------------------------- spec


def test_spec_bin_arithmetic() -> None:
    assert OnsetSpec(lookahead=60, bin_width=1, horizon=32).num_bins == 60
    assert OnsetSpec(lookahead=60, bin_width=1, horizon=32).horizon_bins == 32
    assert OnsetSpec(lookahead=60, bin_width=4, horizon=32).num_bins == 15
    assert OnsetSpec(lookahead=60, bin_width=4, horizon=32).horizon_bins == 8


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lookahead": 60, "bin_width": 0, "horizon": 32}, "bin_width must be >= 1"),
        ({"lookahead": 0, "bin_width": 1, "horizon": 32}, "lookahead must be >= 1"),
        ({"lookahead": 60, "bin_width": 1, "horizon": 0}, "horizon must be in"),
        ({"lookahead": 32, "bin_width": 1, "horizon": 60}, "horizon must be in"),
        ({"lookahead": 61, "bin_width": 4, "horizon": 32}, "lookahead=61 is not divisible"),
        ({"lookahead": 60, "bin_width": 7, "horizon": 35}, "lookahead=60 is not divisible"),
        ({"lookahead": 60, "bin_width": 5, "horizon": 32}, "horizon=32 is not divisible"),
    ],
)
def test_spec_rejects_bad_geometry(kwargs, match) -> None:
    """A bin straddling the horizon would leave the reported readout undefined — reject at build."""
    with pytest.raises(ValueError, match=match):
        OnsetSpec(**kwargs)


def test_spec_allows_lookahead_equal_to_horizon() -> None:
    """Legal but methodologically pointless (a crossing at H+5 is still a flat negative) — not the
    schema's job to forbid; ``validate_config`` warns instead."""
    assert OnsetSpec(lookahead=32, bin_width=1, horizon=32).num_bins == 32


# --------------------------------------------------------------------------- the four cases


def test_event_inside_lookahead() -> None:
    """Bins before the event -> 0, the event bin -> 1, everything after -> masked (unobservable)."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([2], [40], [1]), spec)
    assert out.valid.tolist() == [True]
    assert out.event_bin.tolist() == [2]
    assert out.target[0].tolist() == [0, 0, 1, 0, 0, 0]
    assert out.mask[0].tolist() == [1, 1, 1, 0, 0, 0]


def test_event_beyond_lookahead_is_a_full_negative() -> None:
    """The crossing WAS observed, just not within L — so all K bins are honest zeros, none masked."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([9], [40], [1]), spec)
    assert out.valid.tolist() == [True]
    assert out.event_bin.tolist() == [-1]
    assert out.target[0].sum() == 0
    assert out.mask[0].tolist() == [1, 1, 1, 1, 1, 1]


def test_censored_masks_the_unobserved_tail() -> None:
    """The window says 'not yet, for this long' and nothing about the rest — the whole point."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([-1], [4], [0]), spec)
    assert out.valid.tolist() == [True]
    assert out.target[0].sum() == 0
    assert out.mask[0].tolist() == [1, 1, 1, 1, 0, 0]


def test_already_crossed_leaves_the_risk_set() -> None:
    """No future crossing but the track crosses => it happened before. Not a negative, not censored."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([-1], [40], [1]), spec)
    assert out.valid.tolist() == [False]
    assert out.mask[0].sum() == 0
    assert out.target[0].sum() == 0


def test_zero_observed_bins_is_invalid() -> None:
    """Censored before even one whole bin: carries no gradient, so it must not pad the denominator."""
    spec = OnsetSpec(lookahead=8, bin_width=4, horizon=4)
    out = hazard_targets(_labels([-1], [3], [0]), spec)  # 3 frames < one 4-frame bin
    assert out.valid.tolist() == [False]
    assert out.mask.sum() == 0


def test_event_at_bin_zero_masks_one_bin() -> None:
    """Onset on the very next frame: a single supervised bin, target 1."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([0], [40], [1]), spec)
    assert out.event_bin.tolist() == [0]
    assert out.target[0].tolist() == [1, 0, 0, 0, 0, 0]
    assert out.mask[0].tolist() == [1, 0, 0, 0, 0, 0]


def test_event_at_last_bin_is_in_range() -> None:
    """``onset == L - 1`` is the last in-range frame; ``onset == L`` is the first out-of-range one."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    assert hazard_targets(_labels([5], [40], [1]), spec).event_bin.tolist() == [5]
    assert hazard_targets(_labels([6], [40], [1]), spec).event_bin.tolist() == [-1]


# --------------------------------------------------------------------------- bin width


def test_bin_width_groups_frames() -> None:
    """``w=4``: frames 0-3 -> bin 0, 4-7 -> bin 1. The density lever against a collapsed head."""
    spec = OnsetSpec(lookahead=12, bin_width=4, horizon=4)
    out = hazard_targets(_labels([0, 3, 4, 7, 8], [40] * 5, [1] * 5), spec)
    assert out.event_bin.tolist() == [0, 0, 1, 1, 2]


def test_bin_width_censoring_counts_whole_bins_only() -> None:
    """A partially-observed bin is masked: 10 observed frames at ``w=4`` is 2 whole bins, not 2.5."""
    spec = OnsetSpec(lookahead=12, bin_width=4, horizon=4)
    out = hazard_targets(_labels([-1], [10], [0]), spec)
    assert out.mask[0].tolist() == [1, 1, 0]


# --------------------------------------------------------------------------- readout


def test_readout_label_and_validity() -> None:
    """Binary question at H, plus the M4 rule applied at H."""
    spec = OnsetSpec(lookahead=60, bin_width=1, horizon=32)
    label, valid = readout_targets(
        _labels([12, 45, -1, -1, -1], [80, 90, 90, 9, 40], [1, 1, 0, 0, 1]), spec
    )
    assert label.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    #                          seen  late  ruled-out  too-short  already-crossed
    assert valid.tolist() == [True, True, True, False, False]


def test_readout_validity_matches_onset_stats(  ) -> None:
    """Cross-pin against the reporting sibling so the training and reporting rules cannot drift.

    ``is_usable`` is the H-sweep's knowability rule; ``readout_targets`` applies the same rule and then
    additionally drops already-crossed windows (they are not at risk), so agreement is asserted over
    the windows where that extra condition does not bite.
    """
    spec = OnsetSpec(lookahead=60, bin_width=1, horizon=32)
    rows = [
        {"onset_offset": o, "future_observed": f, "track_crosses": c}
        for o in (-1, 0, 12, 31, 32, 45)
        for f in (5, 32, 90)
        for c in (0, 1)
    ]
    at_risk = [r for r in rows if not (r["onset_offset"] < 0 and r["track_crosses"] == 1)]
    _, valid = readout_targets(
        _labels(
            [r["onset_offset"] for r in at_risk],
            [r["future_observed"] for r in at_risk],
            [r["track_crosses"] for r in at_risk],
        ),
        spec,
    )
    assert valid.tolist() == [is_usable(r, spec.horizon) for r in at_risk]


def test_readout_label_reproduces_stored_crosses() -> None:
    """THE comparability invariant, over a real generated track.

    ``window_track`` writes ``crosses = any(crosses[end : end + future_offset + tol])``. The readout
    label at ``H = future_offset + tol`` is ``0 <= onset_offset < H``. These must be the same number
    for every emitted window, or the hazard head's reported probability is answering a different
    question from the one the four baseline runs answered.
    """
    cfg = dataclasses.replace(DataCfg(), seq_len=4, stride=1, future_offset=6, tol=2)
    horizon = cfg.future_offset + cfg.tol
    n = 40
    crosses = [0] * n
    crosses[22:30] = [1] * 8          # one crossing episode partway through the track
    records = window_track(
        images=[f"f{t}.png" for t in range(n)],
        bboxes=[[10.0, 10.0, 60.0, 90.0] for _ in range(n)],
        actions=[0] * n,
        looks=[0] * n,
        crosses=crosses,
        cfg=cfg,
        track_id="ped_1",
        ego_speed=[0.0] * n,
    )
    assert records, "fixture produced no windows — check seq_len/stride/future_offset"

    spec = OnsetSpec(lookahead=horizon * 2, bin_width=1, horizon=horizon)
    label, valid = readout_targets(
        _labels(
            [r["onset_offset"] for r in records],
            [r["future_observed"] for r in records],
            [r["track_crosses"] for r in records],
        ),
        spec,
    )
    stored = torch.tensor([float(r["crosses"]) for r in records])
    assert valid.all(), "generation guarantees a fully-observed future, so every window is knowable"
    assert torch.equal(label, stored)
    assert stored.sum() > 0 and stored.sum() < len(records), "fixture must contain both classes"


# --------------------------------------------------------------------------- contract


def test_shapes_and_dtypes() -> None:
    spec = OnsetSpec(lookahead=12, bin_width=4, horizon=4)
    out = hazard_targets(_labels([2, -1], [40, 8], [1, 0]), spec)
    assert out.target.shape == (2, spec.num_bins)
    assert out.mask.shape == (2, spec.num_bins)
    assert out.target.dtype == torch.float32
    assert out.mask.dtype == torch.float32
    assert out.event_bin.dtype == torch.long
    assert out.valid.dtype == torch.bool


def test_missing_fields_name_the_fix() -> None:
    """Pre-S1 chunks must fail with the backfill instruction, not a bare KeyError."""
    spec = OnsetSpec(lookahead=12, bin_width=1, horizon=4)
    with pytest.raises(KeyError, match="backfill_onset_meta"):
        hazard_targets({"onset_offset": torch.tensor([1])}, spec)
    with pytest.raises(KeyError, match="backfill_onset_meta"):
        readout_targets({"crosses": torch.tensor([1])}, spec)


def test_targets_carry_no_gradient() -> None:
    """Labels are data, never a path back into the model."""
    spec = OnsetSpec(lookahead=6, bin_width=1, horizon=3)
    out = hazard_targets(_labels([2], [40], [1]), spec)
    assert not out.target.requires_grad
    assert not out.mask.requires_grad


# --------------------------------------------------------------------------- M4 refinement


def _track(n: int, cross_at: range | None, cfg: DataCfg):
    """Run the real generator over a synthetic track; return (records, stats)."""
    from pedpredict.data.pie_sequences import WindowStats

    crosses = [0] * n
    if cross_at is not None:
        for i in cross_at:
            crosses[i] = 1
    stats = WindowStats()
    records = window_track(
        images=[f"f{t}.png" for t in range(n)],
        bboxes=[[10.0, 10.0, 60.0, 90.0] for _ in range(n)],
        actions=[0] * n, looks=[0] * n, crosses=crosses, cfg=cfg,
        track_id="ped_1", ego_speed=[0.0] * n, stats=stats,
    )
    return records, stats


def _cfg(**over) -> DataCfg:
    base = {"seq_len": 4, "stride": 1, "future_offset": 6, "tol": 2}
    return dataclasses.replace(DataCfg(), **{**base, **over})


def test_determined_positive_is_recovered_not_binned() -> None:
    """A crossing SEEN inside a truncated remainder is a confirmed positive, not an unknown.

    Track of 20 frames, crossing at 17-19. Horizon is 8, so any window ending after frame 12 has a
    truncated future — including the ones that plainly show the crossing start at 17.
    """
    off, _ = _track(20, range(17, 20), _cfg())
    on, stats = _track(20, range(17, 20), _cfg(emit_determined_positives=True))

    assert len(on) > len(off), "the refined rule must recover windows the blunt filter binned"
    assert stats.determined_positive == len(on) - len(off)
    recovered = on[len(off):]
    assert all(r["crosses"] == 1 for r in recovered), "every recovered window is a confirmed positive"
    assert all(r["onset_offset"] >= 0 for r in recovered), "and its onset was actually observed"


def test_unknowable_windows_are_still_dropped() -> None:
    """A truncated future with NO observed crossing stays binned under either setting.

    That label really would be fabricated — the refinement narrows M4, it does not repeal it.
    """
    off, stats_off = _track(20, None, _cfg())
    on, stats_on = _track(20, None, _cfg(emit_determined_positives=True))
    assert len(on) == len(off)
    assert stats_on.censored == stats_off.censored > 0
    assert stats_on.determined_positive == 0


def test_refinement_never_removes_a_window() -> None:
    """Strictly additive: every window the old rule emitted is still emitted, unchanged."""
    off, _ = _track(30, range(24, 27), _cfg())
    on, _ = _track(30, range(24, 27), _cfg(emit_determined_positives=True))
    assert on[: len(off)] == off


def test_recovered_labels_survive_the_readout_invariant() -> None:
    """The comparability invariant must still hold over the enlarged population.

    Clipping in `_label_window` becomes reachable here, so this is the test that proves the clipped
    `any()` still agrees with the onset-derived label.
    """
    cfg = _cfg(emit_determined_positives=True)
    horizon = cfg.future_offset + cfg.tol
    records, _ = _track(30, range(21, 25), cfg)
    spec = OnsetSpec(lookahead=horizon * 2, bin_width=1, horizon=horizon)
    label, valid = readout_targets(
        _labels(
            [r["onset_offset"] for r in records],
            [r["future_observed"] for r in records],
            [r["track_crosses"] for r in records],
        ),
        spec,
    )
    stored = torch.tensor([float(r["crosses"]) for r in records])
    assert valid.all()
    assert torch.equal(label, stored)


def test_stats_dict_reports_the_recovery() -> None:
    _, stats = _track(20, range(17, 20), _cfg(emit_determined_positives=True))
    assert stats.as_dict()["determined_positive"] == stats.determined_positive

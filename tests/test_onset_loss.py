"""The censored hazard loss — Stage D of docs/METHODOLOGY.md prong 2.

The load-bearing test here is :func:`test_no_gradient_past_the_event_or_censor_point`. Everything the
method claims rests on one property: bins the window never observed must receive **exactly zero**
gradient, not a small one and not a confident zero. That is the difference between saying nothing and
fabricating a label, and it is checkable directly with autograd rather than argued from the algebra.

The rest pins the arithmetic (hand-computed NLL on a three-bin toy), the reduction (per-window sum,
mean over valid windows), the weight knobs that select the three formulations, and the failure modes
that should be loud: a pre-S1 batch, a model/loss bin-width mismatch, a batch with nothing valid in it.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from pedpredict.config import ModelCfg, RootCfg
from pedpredict.data.onset_target import OnsetSpec
from pedpredict.losses.onset import (
    HAZARD_OUTPUT_KEY,
    READOUT_OUTPUT_KEY,
    OnsetHazardLoss,
    build_onset_loss,
    crosses_metric_keys,
)
from pedpredict.models.heads import hazard_to_horizon_logits
from pedpredict.training.metrics import MetricAccumulator

_SPEC = OnsetSpec(lookahead=6, bin_width=1, horizon=3)


def _labels(onset, observed, ever) -> dict[str, torch.Tensor]:
    return {
        "onset_offset": torch.tensor(onset),
        "future_observed": torch.tensor(observed),
        "track_crosses": torch.tensor(ever),
    }


def _outputs(hazard: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        HAZARD_OUTPUT_KEY: hazard,
        READOUT_OUTPUT_KEY: hazard_to_horizon_logits(hazard, _SPEC.horizon_bins),
    }


# --------------------------------------------------------------------------- the masking property


def test_no_gradient_past_the_event_or_censor_point() -> None:
    """THE property the method rests on: unobserved bins get exactly zero gradient.

    Row 0 sees a crossing at bin 2, so bins 3-5 are unobservable — the person has already started.
    Row 1 is censored after 4 frames, so bins 4-5 were never watched. Neither may contribute.
    """
    hazard = torch.zeros(2, _SPEC.num_bins, requires_grad=True)
    loss = OnsetHazardLoss(_SPEC, hazard_weight=1.0, readout_weight=0.0)
    loss(_outputs(hazard), _labels([2, -1], [40, 4], [1, 0])).total.backward()

    grad = hazard.grad
    assert torch.equal(grad[0, 3:], torch.zeros(3)), "bins after the event must be silent"
    assert torch.equal(grad[1, 4:], torch.zeros(2)), "bins after the censor point must be silent"
    assert (grad[0, :3] != 0).all(), "observed bins must carry gradient"
    assert (grad[1, :4] != 0).all()


def test_gradient_signs_push_the_right_way() -> None:
    """Before the event: push the hazard DOWN. On the event bin: push it UP."""
    hazard = torch.zeros(1, _SPEC.num_bins, requires_grad=True)
    OnsetHazardLoss(_SPEC).forward(_outputs(hazard), _labels([2], [40], [1])).total.backward()
    assert (hazard.grad[0, :2] > 0).all(), "pre-event bins: positive grad => gradient descent lowers z"
    assert hazard.grad[0, 2] < 0, "event bin: negative grad => gradient descent raises z"


def test_already_crossed_window_contributes_nothing() -> None:
    """Not at risk of a FIRST crossing: no gradient anywhere, and excluded from the mean."""
    hazard = torch.zeros(1, _SPEC.num_bins, requires_grad=True)
    out = OnsetHazardLoss(_SPEC).forward(_outputs(hazard), _labels([-1], [40], [1]))
    assert float(out.valid_windows) == 0
    assert float(out.hazard) == 0.0
    out.total.backward()
    assert torch.equal(hazard.grad, torch.zeros_like(hazard))


# --------------------------------------------------------------------------- arithmetic


def test_hazard_value_matches_hand_computation() -> None:
    """All logits 0 => every hazard 0.5 => each observed bin contributes exactly ``log 2``."""
    hazard = torch.zeros(2, _SPEC.num_bins)
    out = OnsetHazardLoss(_SPEC).forward(_outputs(hazard), _labels([2, -1], [40, 4], [1, 0]))
    # Row 0: bins 0..2 observed (3 bins). Row 1: bins 0..3 observed (4 bins). Mean over 2 windows.
    expected = (3 * math.log(2) + 4 * math.log(2)) / 2
    assert float(out.hazard) == pytest.approx(expected, rel=1e-6)
    assert float(out.valid_windows) == 2


def test_reduction_is_per_window_not_per_bin() -> None:
    """A deeper-observed window really does weigh more — that is the likelihood, not a bug."""
    hazard = torch.zeros(1, _SPEC.num_bins)
    shallow = OnsetHazardLoss(_SPEC).forward(_outputs(hazard), _labels([-1], [1], [0]))
    deep = OnsetHazardLoss(_SPEC).forward(_outputs(hazard), _labels([-1], [6], [0]))
    assert float(deep.hazard) == pytest.approx(6 * float(shallow.hazard), rel=1e-6)


def test_readout_term_uses_the_binary_horizon_label() -> None:
    """The readout CE is the old binary question, scored on the collapsed hazard."""
    hazard = torch.zeros(2, _SPEC.num_bins)
    loss = OnsetHazardLoss(_SPEC, hazard_weight=0.0, readout_weight=1.0)
    out = loss(_outputs(hazard), _labels([1, 5], [40, 40], [1, 1]))
    # onset 1 < horizon 3 => positive; onset 5 >= 3 => negative. p = 1 - 0.5^3 = 0.875.
    expected = (-math.log(0.875) + -math.log(0.125)) / 2
    assert float(out.readout) == pytest.approx(expected, rel=1e-5)


def test_readout_skips_windows_whose_answer_is_unknowable() -> None:
    """Censored shorter than the horizon: the binary label would be fabricated, so it is excluded."""
    loss = OnsetHazardLoss(_SPEC, hazard_weight=0.0, readout_weight=1.0)
    both = loss(_outputs(torch.zeros(2, _SPEC.num_bins)), _labels([1, -1], [40, 1], [1, 0]))
    only_first = loss(_outputs(torch.zeros(1, _SPEC.num_bins)), _labels([1], [40], [1]))
    assert float(both.readout) == pytest.approx(float(only_first.readout), rel=1e-6)


# --------------------------------------------------------------------------- weights select the arm


def test_weights_combine_linearly() -> None:
    hazard = torch.randn(3, _SPEC.num_bins)
    labels = _labels([2, 5, -1], [40, 40, 4], [1, 1, 0])
    only_h = OnsetHazardLoss(_SPEC, hazard_weight=1.0, readout_weight=0.0)(_outputs(hazard), labels)
    only_r = OnsetHazardLoss(_SPEC, hazard_weight=0.0, readout_weight=1.0)(_outputs(hazard), labels)
    both = OnsetHazardLoss(_SPEC, hazard_weight=0.3, readout_weight=2.0)(_outputs(hazard), labels)
    assert float(both.total) == pytest.approx(
        0.3 * float(only_h.hazard) + 2.0 * float(only_r.readout), rel=1e-5
    )


def test_readout_term_is_skipped_when_its_weight_is_zero() -> None:
    """Zero weight means the term is not computed at all — not computed then multiplied by zero."""
    hazard = torch.zeros(1, _SPEC.num_bins)
    out = OnsetHazardLoss(_SPEC, readout_weight=0.0)(
        {HAZARD_OUTPUT_KEY: hazard}, _labels([2], [40], [1])
    )
    assert float(out.readout) == 0.0


# --------------------------------------------------------------------------- build_onset_loss


def _cfgs(**model_over):
    base = {"onset_head": True, "onset_lookahead": 60, "onset_bin_width": 1, "onset_horizon": 32}
    return dataclasses.replace(ModelCfg(), **{**base, **model_over}), RootCfg().train


def test_build_returns_none_when_head_is_off() -> None:
    assert build_onset_loss(ModelCfg(), RootCfg().train) is None


def test_build_returns_none_when_both_weights_are_zero() -> None:
    model, train = _cfgs()
    train = dataclasses.replace(train, onset_hazard_weight=0.0, onset_readout_weight=0.0)
    assert build_onset_loss(model, train) is None


def test_build_returns_none_when_crosses_is_inactive() -> None:
    """Dropping crosses from active_tasks must kill onset supervision too — it IS crosses supervision."""
    model, train = _cfgs()
    train = dataclasses.replace(train, active_tasks=("actions", "looks"))
    assert build_onset_loss(model, train) is None


def test_build_carries_the_configured_geometry() -> None:
    model, train = _cfgs(onset_bin_width=4)
    loss = build_onset_loss(model, train)
    assert loss.spec.num_bins == 15
    assert loss.spec.horizon_bins == 8


# --------------------------------------------------------------------------- metric routing


def test_metric_routing_default_is_crosses_frame() -> None:
    """Auxiliary arm: the reported number comes from the SAME head as the four baselines."""
    model, _ = _cfgs()
    assert crosses_metric_keys(model) is None
    assert crosses_metric_keys(ModelCfg()) is None


def test_metric_routing_switches_to_the_readout() -> None:
    model, _ = _cfgs(onset_report_crosses=True)
    keys = crosses_metric_keys(model)
    assert keys["crosses"] == READOUT_OUTPUT_KEY
    assert keys["actions"] == "actions" and keys["looks"] == "looks"


def test_metric_routing_ignored_when_head_is_off() -> None:
    off = dataclasses.replace(ModelCfg(), onset_head=False, onset_report_crosses=True)
    assert crosses_metric_keys(off) is None


def test_accumulator_scores_crosses_from_the_readout_when_routed() -> None:
    """End-to-end Stage E: the override actually changes which head the reported number comes from.

    The two heads are given deliberately opposite predictions, so a wrong routing cannot accidentally
    produce the right accuracy.
    """
    model, _ = _cfgs(onset_report_crosses=True)
    hazard = torch.full((4, 60), -8.0)          # readout => P(onset) ~ 0 => predicts class 0
    hazard[:2, :32] = 8.0                       # first two windows => P(onset) ~ 1 => class 1
    outputs = {
        "crosses_frame": torch.tensor([[8.0, -8.0]] * 4),   # crosses_frame says class 0 for ALL four
        HAZARD_OUTPUT_KEY: hazard,
        READOUT_OUTPUT_KEY: hazard_to_horizon_logits(hazard, 32),
    }
    targets = {"crosses": torch.tensor([1, 1, 0, 0])}       # matches the READOUT, not crosses_frame

    routed = MetricAccumulator(tasks=("crosses",), output_keys=crosses_metric_keys(model))
    routed.update(outputs, targets)
    assert routed.compute().per_task["crosses"].accuracy == pytest.approx(1.0)

    default = MetricAccumulator(tasks=("crosses",))         # crosses_frame: gets the two negatives only
    default.update(outputs, targets)
    assert default.compute().per_task["crosses"].accuracy == pytest.approx(0.5)


# --------------------------------------------------------------------------- loud failures


def test_missing_hazard_key_names_the_config_switch() -> None:
    with pytest.raises(KeyError, match="model.onset_head=true"):
        OnsetHazardLoss(_SPEC)({"crosses_frame": torch.zeros(1, 2)}, _labels([1], [40], [1]))


def test_missing_onset_labels_name_the_backfill() -> None:
    with pytest.raises(KeyError, match="backfill_onset_meta"):
        OnsetHazardLoss(_SPEC)(_outputs(torch.zeros(1, _SPEC.num_bins)), {"crosses": torch.tensor([1])})


def test_bin_width_mismatch_between_model_and_loss() -> None:
    """Two configs disagreeing about what a bin is would train silently wrong — fail instead."""
    wrong = torch.zeros(1, 15)
    with pytest.raises(ValueError, match="hazard bins but the loss was built for"):
        OnsetHazardLoss(_SPEC)({HAZARD_OUTPUT_KEY: wrong}, _labels([1], [40], [1]))


def test_batch_with_no_valid_windows_is_finite_and_zero() -> None:
    """An all-already-crossed batch must not divide by zero."""
    hazard = torch.zeros(2, _SPEC.num_bins, requires_grad=True)
    out = OnsetHazardLoss(_SPEC)(_outputs(hazard), _labels([-1, -1], [40, 40], [1, 1]))
    assert torch.isfinite(out.total)
    assert float(out.valid_windows) == 0
    out.total.backward()
    assert torch.isfinite(hazard.grad).all()

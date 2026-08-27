"""The onset hazard head and its horizon readout — Stage C of docs/METHODOLOGY.md prong 2.

Two things need pinning here, and they pull in opposite directions:

* **Off, nothing changed.** ``model.onset_head=false`` is what every existing run was trained under and
  what the golden characterization tests capture. No new output key, no new parameter, so a legacy
  checkpoint still loads ``strict=True``. Any drift here silently invalidates the four baselines.
* **On, the readout is exactly the product.** ``crosses_readout`` must equal ``1 - prod(1 - h_k)`` over
  the horizon bins, computed stably, because that number *is* the comparison against those baselines.
  It is derived in log space, so the arithmetic is worth checking against a naive product rather than
  trusting the algebra.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pedpredict.config import DataCfg, ModelCfg, RootCfg
from pedpredict.config.loader import ConfigError, validate_config
from pedpredict.models.cross_attention import CrossAttentionModule
from pedpredict.models.heads import build_onset_hazard_head, hazard_to_horizon_logits
from pedpredict.models.registry import ModelType, build_model, forward_model

_LOOKAHEAD, _HORIZON, _BINS = 60, 32, 60


def _model_cfg(**over) -> ModelCfg:
    base = {"onset_head": True, "onset_lookahead": _LOOKAHEAD, "onset_bin_width": 1,
            "onset_horizon": _HORIZON}
    return dataclasses.replace(ModelCfg(), **{**base, **over})


# --------------------------------------------------------------------------- readout arithmetic


@pytest.mark.parametrize("horizon_bins", [1, 5, 32])
def test_readout_matches_naive_product(horizon_bins) -> None:
    """``softmax(readout)[:, 1]`` equals ``1 - prod(1 - sigmoid(z_k))`` over the leading bins."""
    torch.manual_seed(0)
    hazard = torch.randn(7, _BINS) * 2.0
    probs = torch.softmax(hazard_to_horizon_logits(hazard, horizon_bins), dim=1)
    naive = 1.0 - torch.prod(1.0 - torch.sigmoid(hazard[:, :horizon_bins]), dim=1)
    torch.testing.assert_close(probs[:, 1], naive, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(7))


def test_readout_ignores_bins_past_the_horizon() -> None:
    """Bins beyond the horizon are trained but never reported — changing them must not move it."""
    torch.manual_seed(1)
    hazard = torch.randn(4, _BINS)
    before = hazard_to_horizon_logits(hazard, _HORIZON)
    hazard[:, _HORIZON:] += 12.0            # slam the unreported tail
    torch.testing.assert_close(hazard_to_horizon_logits(hazard, _HORIZON), before)


def test_readout_known_values() -> None:
    """Hand-checkable: K bins at h=0.5 give ``1 - 0.5^K``; a huge logit saturates to 1."""
    zeros = torch.zeros(1, 4)               # sigmoid(0) = 0.5
    p = torch.softmax(hazard_to_horizon_logits(zeros, 4), dim=1)[0, 1]
    torch.testing.assert_close(p, torch.tensor(1.0 - 0.5**4), rtol=1e-6, atol=1e-7)
    big = torch.full((1, 4), 30.0)
    assert torch.softmax(hazard_to_horizon_logits(big, 4), dim=1)[0, 1] > 0.999999


def test_readout_is_finite_at_both_extremes() -> None:
    """A dead head (all hazards -> 0) is the predicted failure mode; it must not produce nan/inf."""
    for fill in (-60.0, -30.0, 0.0, 30.0, 60.0):
        out = hazard_to_horizon_logits(torch.full((3, _BINS), fill), _HORIZON)
        assert torch.isfinite(out).all(), f"non-finite readout at hazard logit {fill}"
        assert torch.isfinite(torch.softmax(out, dim=1)).all()


def test_readout_gradient_is_finite_when_head_is_dead() -> None:
    """The saturated-low case must still backprop — that is where the collapse lever gets used."""
    hazard = torch.full((2, _BINS), -40.0, requires_grad=True)
    torch.softmax(hazard_to_horizon_logits(hazard, _HORIZON), dim=1)[:, 1].sum().backward()
    assert torch.isfinite(hazard.grad).all()


def test_readout_rejects_horizon_outside_the_head() -> None:
    with pytest.raises(ValueError, match="outside"):
        hazard_to_horizon_logits(torch.zeros(2, 8), 9)
    with pytest.raises(ValueError, match="outside"):
        hazard_to_horizon_logits(torch.zeros(2, 8), 0)


def test_hazard_head_shape() -> None:
    assert build_onset_hazard_head(128, 60)(torch.randn(3, 128)).shape == (3, 60)


# --------------------------------------------------------------------------- gate off = unchanged


def test_gate_off_emits_no_onset_keys() -> None:
    module = CrossAttentionModule.from_config(ModelCfg()).eval()
    with torch.no_grad():
        out = module(torch.randn(2, 20, 128), torch.randn(2, 20, 128))
    assert "crosses_hazard" not in out
    assert "crosses_readout" not in out
    assert set(out) == {"actions", "looks", "crosses_pooled", "crosses_frame", "temporal_weights"}


def test_gate_off_allocates_no_parameters() -> None:
    """Parameter layout must be identical, or existing checkpoints stop loading strict=True."""
    off = CrossAttentionModule.from_config(ModelCfg())
    on = CrossAttentionModule.from_config(_model_cfg())
    assert off.onset_head is None
    assert not [k for k in off.state_dict() if "onset" in k]
    assert [k for k in on.state_dict() if "onset" in k] == ["onset_head.weight", "onset_head.bias"]
    extra = sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters())
    assert extra == 128 * _BINS + _BINS  # exactly one Linear(d_model, K)


def test_gate_off_checkpoint_loads_into_gate_off() -> None:
    """A checkpoint from before the onset head still loads strict into a default-config model."""
    legacy = CrossAttentionModule.from_config(ModelCfg()).state_dict()
    CrossAttentionModule.from_config(ModelCfg()).load_state_dict(legacy, strict=True)


# --------------------------------------------------------------------------- gate on


def test_gate_on_emits_both_onset_keys() -> None:
    module = CrossAttentionModule.from_config(_model_cfg()).eval()
    with torch.no_grad():
        out = module(torch.randn(2, 20, 128), torch.randn(2, 20, 128))
    assert out["crosses_hazard"].shape == (2, _BINS)
    assert out["crosses_readout"].shape == (2, 2)
    # The four supervised keys are still there and still the same shapes.
    assert out["crosses_frame"].shape == (2, 2)
    assert out["actions"].shape == (2, 2)


def test_bin_width_narrows_the_head() -> None:
    module = CrossAttentionModule.from_config(_model_cfg(onset_bin_width=4)).eval()
    with torch.no_grad():
        out = module(torch.randn(2, 20, 128), torch.randn(2, 20, 128))
    assert out["crosses_hazard"].shape == (2, 15)      # 60 / 4
    assert out["crosses_readout"].shape == (2, 2)      # readout still collapses to 2 classes


def test_gradient_reaches_the_onset_head() -> None:
    module = CrossAttentionModule.from_config(_model_cfg())
    out = module(torch.randn(2, 20, 128), torch.randn(2, 20, 128))
    out["crosses_hazard"].sum().backward()
    assert module.onset_head.weight.grad is not None
    assert torch.isfinite(module.onset_head.weight.grad).all()
    assert module.onset_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("model_type", [m.value for m in ModelType])
def test_every_model_type_emits_onset_keys(model_type) -> None:
    """The head is part of the shared output contract, not a privilege of the full model."""
    cfg = RootCfg()
    cfg = dataclasses.replace(cfg, model=_model_cfg(motion_dim=cfg.data.motion_dim))
    model = build_model(cfg, model_type).eval()
    batch, frames = 2, 4
    # forward_model takes the full collate triple and routes per type internally.
    with torch.no_grad():
        out = forward_model(
            model,
            torch.randn(batch, frames, 3, cfg.data.img_height, cfg.data.img_width),
            torch.randn(batch, frames, 3, 224, 224),
            torch.randn(batch, frames, cfg.data.motion_dim),
        )
    assert out["crosses_hazard"].shape == (batch, _BINS)
    assert out["crosses_readout"].shape == (batch, 2)


# --------------------------------------------------------------------------- config validation


def _root(**model_over) -> RootCfg:
    return dataclasses.replace(RootCfg(), model=_model_cfg(**model_over))


def test_validate_accepts_a_sane_onset_config() -> None:
    validate_config(_root())


def test_validate_rejects_lookahead_at_or_below_horizon() -> None:
    """The whole point is a head wider than the reported horizon."""
    with pytest.raises(ConfigError, match="must be GREATER than"):
        validate_config(_root(onset_lookahead=_HORIZON))


def test_validate_rejects_horizon_that_is_not_the_label_horizon() -> None:
    """A readout at a different H answers a different question under the same metric names."""
    with pytest.raises(ConfigError, match="must equal data.future_offset"):
        validate_config(_root(onset_horizon=30))


def test_validate_rejects_indivisible_bin_width() -> None:
    with pytest.raises(ConfigError, match="geometry invalid"):
        validate_config(_root(onset_bin_width=7))


def test_validate_ignores_onset_fields_when_head_is_off() -> None:
    """Nonsense onset values must not break a run that never builds the head."""
    cfg = dataclasses.replace(
        RootCfg(), model=dataclasses.replace(ModelCfg(), onset_head=False, onset_lookahead=1)
    )
    validate_config(cfg)


def test_onset_bin_properties_are_zero_when_off() -> None:
    off = ModelCfg()
    assert off.onset_bins == 0 and off.onset_horizon_bins == 0
    on = _model_cfg(onset_bin_width=4)
    assert on.onset_bins == 15 and on.onset_horizon_bins == 8


def test_data_horizon_matches_the_default_onset_horizon() -> None:
    """The ModelCfg default must track the generator's label horizon, or every default run trips."""
    d = DataCfg()
    assert ModelCfg().onset_horizon == d.future_offset + d.tol

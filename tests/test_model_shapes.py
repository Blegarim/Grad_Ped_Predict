"""Prompt 2.1 — ViT_Hierarchical port tests.

  * GOLDEN PARITY: new ViT loads the OLD post-forward state_dict (strict=True) and reproduces the OLD
    output within tolerance (eval mode -> deterministic fp32 math).
  * B2 (no lazy params): the OLD state_dict loads strict with NO dummy forward; all params (incl. the
    global-stage relative-position table) exist at __init__.
  * GEOMETRY: feature_map_size matches the runtime feature maps and the derived global window.
  * B6 SMOKE: from_config(ModelCfg) forwards to [B, T, d_model].
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pedpredict.config import DataCfg, ModelCfg
from pedpredict.models.geometry import feature_map_size
from pedpredict.models.motion_encoder import MotionEncoder
from pedpredict.models.vit import ViT_Hierarchical

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "vit.pt"
_MOTION_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "motion_encoder.pt"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _FIXTURE.exists():
        pytest.skip(f"missing golden fixture {_FIXTURE} (run tests/_capture/capture_vit_golden.py)")
    return torch.load(_FIXTURE, weights_only=False)


def _build(golden: dict) -> ViT_Hierarchical:
    return ViT_Hierarchical(img_size=golden["img_size"], **golden["vit_kwargs"])


# --------------------------------------------------------------------------- golden parity


def test_golden_vit_parity(golden: dict) -> None:
    model = _build(golden)
    model.load_state_dict(golden["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        y = model(golden["inputs"]["x"])
    tol = golden["meta"]
    torch.testing.assert_close(y, golden["outputs"]["y"], atol=tol["atol"], rtol=tol["rtol"])


# --------------------------------------------------------------------------- B2: eager params / strict load


def test_strict_load_without_dummy_forward(golden: dict) -> None:
    """B2 resolved: the OLD post-forward state_dict loads with NO missing/unexpected keys and NO forward."""
    model = _build(golden)  # never call forward first
    missing, unexpected = model.load_state_dict(golden["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.load_state_dict(golden["state_dict"], strict=True)  # must not raise


def test_global_table_exists_at_init() -> None:
    """The global (last) stage's relative-position table exists immediately after construction (B2)."""
    cfg = ModelCfg()
    model = ViT_Hierarchical.from_config(cfg, img_size=224)
    last = model.stages[-1]["block"][0].attn
    # global window at 224 is 7x7 -> table (2*7-1)^2 = 169 rows, head_nums[-1]=2 cols
    assert tuple(last.relative_position_bias_table.shape) == (169, cfg.head_nums[-1])
    assert tuple(last.relative_position_index.shape) == (49, 49)


# --------------------------------------------------------------------------- geometry


def test_feature_map_size_matches_runtime() -> None:
    """feature_map_size predicts the actual per-stage feature maps captured via forward hooks."""
    model = ViT_Hierarchical.from_config(ModelCfg(), img_size=224).eval()
    seen: list[int] = []

    def _hook(_module, args, _out) -> None:
        seen.append(args[0].shape[-1])  # H==W of the block input

    handles = [blk.register_forward_hook(_hook) for stage in model.stages for blk in stage["block"]]
    try:
        with torch.no_grad():
            model(torch.randn(1, 1, 3, 224, 224))
    finally:
        for h in handles:
            h.remove()

    # first block of each stage should match feature_map_size(224, stage_idx)
    expected_per_stage = [feature_map_size(224, i) for i in range(len(model.stages))]
    assert expected_per_stage == [56, 28, 14, 7]
    # every block input size equals its stage's predicted size
    idx = 0
    for i, stage in enumerate(model.stages):
        for _ in stage["block"]:
            assert seen[idx] == expected_per_stage[i]
            idx += 1


# --------------------------------------------------------------------------- B6 smoke + rebuild


def test_from_config_output_shape() -> None:
    cfg = ModelCfg()
    model = ViT_Hierarchical.from_config(cfg, img_size=224).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 5, cfg.in_channels, 224, 224))
    assert out.shape == (2, 5, cfg.d_model)
    assert torch.isfinite(out).all()


def test_rebuild_position_bias_changes_global_window() -> None:
    """Explicit resolution change rebuilds the global table to the new feature-map size; forward runs."""
    model = ViT_Hierarchical.from_config(ModelCfg(), img_size=224).eval()
    model.rebuild_position_bias(128)
    # 128 -> stem 32 -> 16 -> 8 -> 4; global window 4x4 -> table (2*4-1)^2 = 49 rows
    last = model.stages[-1]["block"][0].attn
    assert tuple(last.relative_position_bias_table.shape) == (49, 2)
    with torch.no_grad():
        out = model(torch.randn(1, 2, 3, 128, 128))
    assert out.shape == (1, 2, 128)


# =========================================================================== Prompt 2.2 — MotionEncoder


@pytest.fixture(scope="module")
def motion_golden() -> dict:
    if not _MOTION_FIXTURE.exists():
        pytest.skip(f"missing golden fixture {_MOTION_FIXTURE} (run tests/_capture/capture_motion_golden.py)")
    return torch.load(_MOTION_FIXTURE, weights_only=False)


def test_golden_motion_parity(motion_golden: dict) -> None:
    """New MotionEncoder loads the OLD state_dict (strict=True) and reproduces the OLD output (eval)."""
    model = MotionEncoder(**motion_golden["motion_kwargs"])
    model.load_state_dict(motion_golden["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        y = model(motion_golden["inputs"]["motion"], motion_golden["inputs"]["tight"])
    tol = motion_golden["meta"]
    torch.testing.assert_close(y, motion_golden["outputs"]["y"], atol=tol["atol"], rtol=tol["rtol"])


def test_strict_load_motion_no_lazy_params(motion_golden: dict) -> None:
    """No B2-style lazy params: the OLD state_dict loads strict with no missing/unexpected keys, no forward."""
    model = MotionEncoder(**motion_golden["motion_kwargs"])  # never call forward first
    missing, unexpected = model.load_state_dict(motion_golden["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.load_state_dict(motion_golden["state_dict"], strict=True)  # must not raise


def test_motion_from_config_output_shape() -> None:
    cfg = ModelCfg()
    model = MotionEncoder.from_config(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 5, cfg.motion_dim), torch.randn(2, 5, 3, 128, 128))
    assert out.shape == (2, 5, cfg.d_model)
    assert torch.isfinite(out).all()


def test_motion_pos_encoding_capacity_guard() -> None:
    """T == capacity forwards; T > capacity raises a clear error instead of an opaque broadcast crash."""
    model = MotionEncoder(max_positions=6).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 6, model.motion_dim), torch.randn(1, 6, 3, 128, 128))
    assert out.shape == (1, 6, model.d_model)
    with pytest.raises(ValueError, match="exceeds positional-encoding capacity"):
        model(torch.randn(1, 7, model.motion_dim), torch.randn(1, 7, 3, 128, 128))


def test_motion_conv_in_channels_matches_datacfg() -> None:
    """Coupling guard (1.2/1.4/2.2): the Conv1d input width equals the 8-dim motion contract."""
    model = MotionEncoder.from_config(ModelCfg())
    assert model.motion_encoder[0].in_channels == DataCfg().motion_dim == ModelCfg().motion_dim

"""Prompt RQ1 / WP2 step 1 — pretrained timm visual-backbone drop-in (docs/BACKBONE_STUDY.md).

Shape + wiring only (no modeling, no pretrained weights): the wrapper honors the
``[B, T, 3, H, W] -> [B, T, d_model]`` visual contract, ``build_visual_backbone`` dispatches on
``model.vit_backbone``, and a ``full`` / ``visual_only`` model builds and forwards end-to-end with a
timm backbone swapped in. All builds use ``vit_pretrained=False`` so the suite stays offline and fast;
the actual pretrained-vs-legacy training comparison is a lab-PC run, not a unit test.
"""

from __future__ import annotations

import pytest
import torch

from pedpredict.config import ModelCfg, RootCfg
from pedpredict.models.ablations import VisualOnlyModel
from pedpredict.models.ensemble import EnsembleModel
from pedpredict.models.registry import build_model
from pedpredict.models.timm_backbone import TimmBackbone, build_visual_backbone
from pedpredict.models.vit import ViT_Hierarchical

timm = pytest.importorskip("timm")

# The two primary picks from the study (small, 224-input, hierarchical). Parametrized so the wrapper
# is exercised across both a windowed (TinyViT) and a spatial-reduction (PVTv2) mechanism.
_CANDIDATES = ["tiny_vit_5m_224", "pvt_v2_b0"]
_IMG = 224
_D_MODEL = 128


def _require_model(name: str) -> None:
    if not timm.is_model(name):
        pytest.skip(f"timm {timm.__version__} has no model {name!r}")


# --------------------------------------------------------------------------- factory dispatch


def test_build_visual_backbone_legacy_is_vit() -> None:
    """Default (``vit_backbone='legacy'``) returns the from-scratch ViT — the golden-pinned path."""
    backbone = build_visual_backbone(ModelCfg(), _IMG)
    assert isinstance(backbone, ViT_Hierarchical)


def test_build_visual_backbone_timm_is_wrapper() -> None:
    """A timm model name routes to the TimmBackbone wrapper."""
    _require_model("tiny_vit_5m_224")
    cfg = ModelCfg(vit_backbone="tiny_vit_5m_224", vit_pretrained=False)
    backbone = build_visual_backbone(cfg, _IMG)
    assert isinstance(backbone, TimmBackbone)


# --------------------------------------------------------------------------- wrapper contract


@pytest.mark.parametrize("name", _CANDIDATES)
def test_timm_backbone_output_shape(name: str) -> None:
    """``[B, T, 3, H, W]`` -> ``[B, T, d_model]``, finite — the visual-stream contract."""
    _require_model(name)
    backbone = TimmBackbone(name, d_model=_D_MODEL, pretrained=False, img_size=_IMG).eval()
    with torch.no_grad():
        out = backbone(torch.randn(1, 2, 3, _IMG, _IMG))
    assert out.shape == (1, 2, _D_MODEL)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", _CANDIDATES)
def test_frame_proj_absorbs_feature_width(name: str) -> None:
    """``frame_proj`` maps the backbone's pooled ``num_features`` down to ``d_model`` (like legacy 36->128)."""
    _require_model(name)
    backbone = TimmBackbone(name, d_model=_D_MODEL, pretrained=False, img_size=_IMG)
    assert backbone.frame_proj.in_features == backbone.net.num_features
    assert backbone.frame_proj.out_features == _D_MODEL


# --------------------------------------------------------------------------- end-to-end wiring


def test_full_model_with_timm_backbone_forwards() -> None:
    """``EnsembleModel`` builds with a timm visual stream and produces the full-model output contract."""
    _require_model("tiny_vit_5m_224")
    cfg = ModelCfg(vit_backbone="tiny_vit_5m_224", vit_pretrained=False)
    model = EnsembleModel.from_config(cfg, img_size=_IMG).eval()
    assert isinstance(model.vit, TimmBackbone)
    b, t = 2, 3
    tight = torch.randn(b, t, cfg.in_channels, 128, 128)
    context = torch.randn(b, t, cfg.in_channels, _IMG, _IMG)
    motions = torch.randn(b, t, cfg.motion_dim)
    with torch.no_grad():
        out = model(tight, context, motions)
    assert {"actions", "looks", "crosses_pooled", "crosses_frame", "temporal_weights"} == set(out)
    assert out["crosses_frame"].shape == (b, cfg.num_classes["crosses"])
    assert torch.isfinite(out["crosses_frame"]).all()


def test_visual_only_with_timm_backbone_forwards() -> None:
    """The ``visual_only`` ablation also swaps in the timm backbone via the same factory."""
    _require_model("pvt_v2_b0")
    cfg = ModelCfg(vit_backbone="pvt_v2_b0", vit_pretrained=False)
    model = VisualOnlyModel.from_config(cfg, img_size=_IMG).eval()
    assert isinstance(model.vit, TimmBackbone)
    with torch.no_grad():
        out = model(torch.randn(2, 3, cfg.in_channels, _IMG, _IMG))
    assert out["crosses_frame"].shape == (2, cfg.num_classes["crosses"])
    assert "temporal_weights" not in out  # structurally full-model-only


def test_build_model_full_with_timm_backbone_from_root() -> None:
    """End-to-end through the typed registry: a RootCfg with a timm backbone builds a full model."""
    _require_model("tiny_vit_5m_224")
    root = RootCfg(model=ModelCfg(vit_backbone="tiny_vit_5m_224", vit_pretrained=False))
    model = build_model(root, "full")
    assert isinstance(model, EnsembleModel)
    assert isinstance(model.vit, TimmBackbone)

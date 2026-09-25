"""Recipe v2 backbone flags (docs/RECIPE_V2_PLAN.md Tasks 4-5): a truly frozen timm backbone, a trainable frame_proj.

Both flags default off, and the off state is pinned here as deliberately as the on state: every existing
run (the four ``pose_full`` baselines, R1, R2) was trained with a frozen-but-train-mode backbone whose
BatchNorm statistics drift, and with ``vit.frame_proj`` frozen at its random init. Those runs must stay
reproducible; v2 is selected by recipe flags, never by an incidental default change.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pedpredict.config import ModelCfg, RootCfg
from pedpredict.config.loader import ConfigError, validate_config
from pedpredict.eval.diagnostics import bn_state
from pedpredict.models.ablations import PoseFullModel
from pedpredict.models.timm_backbone import TimmBackbone
from pedpredict.training.schedule import freeze_vit_backbone

timm = pytest.importorskip("timm")

_NAME = "tiny_vit_5m_224"


def _backbone(keep_eval: bool) -> TimmBackbone:
    torch.manual_seed(0)
    backbone = TimmBackbone(_NAME, d_model=128, pretrained=False, keep_eval=keep_eval)
    for param in backbone.net.parameters():
        param.requires_grad = False
    return backbone


def _stats_changed_by_train_forward(backbone: TimmBackbone) -> bool:
    before = bn_state(backbone.net)
    backbone.train()
    with torch.no_grad():
        backbone(torch.randn(2, 2, 3, 224, 224) * 2 + 1)
    after = bn_state(backbone.net)
    return any(not torch.equal(before[k]["running_mean"], after[k]["running_mean"]) for k in before)


def test_v1_frozen_backbone_still_drifts_in_train_mode() -> None:
    """Pins the v1 behaviour every existing run trained under: requires_grad=False does not stop BN."""
    assert _stats_changed_by_train_forward(_backbone(keep_eval=False))


def test_keep_eval_makes_the_backbone_truly_frozen() -> None:
    backbone = _backbone(keep_eval=True)
    assert not _stats_changed_by_train_forward(backbone)
    assert not backbone.net.training and backbone.frame_proj.training
    x = torch.randn(1, 2, 3, 224, 224)
    with torch.no_grad():
        train_out = backbone.train()(x)
        eval_out = backbone.eval()(x)
    assert torch.equal(train_out, eval_out)


def _pose_full(**model_overrides) -> PoseFullModel:
    cfg = dataclasses.replace(
        ModelCfg(), motion_dim=58, motion_norm="none", vit_pretrained=False, **model_overrides
    )
    return PoseFullModel.from_config(cfg, img_size=224)


def test_model_train_leaves_only_the_feature_extractor_in_eval() -> None:
    model = _pose_full(vit_frozen_eval=True).train()
    assert not model.vit.net.training
    assert model.vit.frame_proj.training and model.kinematics_enc.training and model.cross_attention.training


def _trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_frame_proj_frozen_by_default_matches_the_baseline_counts() -> None:
    """215 frozen tensors / 721,725 trainable — the numbers the research-PC training logs print."""
    model = _pose_full()
    assert freeze_vit_backbone(model) == 215
    assert _trainable(model) == 721_725
    assert not model.vit.frame_proj.weight.requires_grad


def test_train_frame_proj_unfreezes_exactly_the_projection() -> None:
    model = _pose_full()
    assert freeze_vit_backbone(model, train_frame_proj=True) == 213
    assert _trainable(model) == 721_725 + 320 * 128 + 128
    assert model.vit.frame_proj.weight.requires_grad
    assert not any(p.requires_grad for p in model.vit.net.parameters())


def _root(**model_overrides) -> RootCfg:
    return dataclasses.replace(RootCfg(), model=dataclasses.replace(ModelCfg(), **model_overrides))


def test_v2_flags_validate_on_a_frozen_timm_backbone() -> None:
    validate_config(_root(vit_frozen_eval=True, train_frame_proj=True))


@pytest.mark.parametrize(
    "overrides",
    [
        {"vit_frozen_eval": True, "freeze_vit_backbone": False},
        {"vit_frozen_eval": True, "vit_backbone": "legacy"},
        {"train_frame_proj": True, "freeze_vit_backbone": False},
    ],
)
def test_v2_flags_reject_configs_where_they_mean_nothing(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        validate_config(_root(**overrides))

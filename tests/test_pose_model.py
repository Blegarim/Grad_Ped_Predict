"""Pose-arm model surface (docs/POSE_ENCODER.md §9): motion_norm="none", pose_kinematics, pose_full."""

from __future__ import annotations

import pytest
import torch

from pedpredict.config.schema import DataCfg, ModelCfg, PoseCfg, RootCfg
from pedpredict.models.ablations import KinematicsEncoder, KinematicsOnlyModel, PoseFullModel
from pedpredict.models.registry import MODEL_INPUT_SIGNATURE, ModelType, build_model, forward_model

_POSE_DIM = 9 + PoseCfg().feature_dim()  # 58
_B, _T = 2, 3
_SUPERVISED = {"actions", "looks", "crosses_pooled", "crosses_frame"}


def _pose_root() -> RootCfg:
    return RootCfg(
        data=DataCfg(motion_dim=_POSE_DIM),
        model=ModelCfg(motion_dim=_POSE_DIM, motion_norm="none"),
        pose=PoseCfg(enabled=True),
    )


def _pose_model_cfg() -> ModelCfg:
    return ModelCfg(motion_dim=_POSE_DIM, motion_norm="none")


def test_kinematics_encoder_none_norm_shape_and_passthrough() -> None:
    enc = KinematicsEncoder.from_config(_pose_model_cfg()).eval()
    assert not hasattr(enc, "motion_scale")  # buffer only exists under "image" norm
    motions = torch.randn(_B, _T, _POSE_DIM)
    with torch.no_grad():
        out = enc(motions)
    assert out.shape == (_B, _T, 128)
    assert torch.isfinite(out).all()


def test_kinematics_encoder_buffer_guard() -> None:
    """The 9-channel image-norm scale contract still rejects a wide motion_dim; "none" bypasses it."""
    with pytest.raises(ValueError, match="9-channel motion contract"):
        KinematicsEncoder(motion_dim=_POSE_DIM, motion_norm="image")
    with pytest.raises(ValueError, match="motion_norm"):
        KinematicsEncoder(motion_norm="bogus")


def test_kinematics_encoder_golden_modes_untouched() -> None:
    """image/per_sequence still build the buffer and normalize as before (golden-pinned modes)."""
    enc = KinematicsEncoder(motion_dim=8, motion_norm="image")
    assert enc.motion_scale.shape == (1, 1, 8)
    KinematicsEncoder(motion_dim=8, motion_norm="per_sequence")  # must construct


def test_pose_kinematics_builds_and_forwards() -> None:
    model = build_model(_pose_root(), ModelType.POSE_KINEMATICS).eval()
    assert isinstance(model, KinematicsOnlyModel)
    assert model.model_type is ModelType.POSE_KINEMATICS
    assert MODEL_INPUT_SIGNATURE[ModelType.POSE_KINEMATICS] == ("motions",)
    motions = torch.randn(_B, _T, _POSE_DIM)
    with torch.no_grad():
        out = forward_model(model, None, None, motions)  # pixel-free
    assert set(out) == _SUPERVISED
    assert out["crosses_frame"].shape == (_B, 2)


def test_pose_full_builds_and_forwards() -> None:
    root = _pose_root()
    model = build_model(root, ModelType.POSE_FULL).eval()
    assert isinstance(model, PoseFullModel)
    assert MODEL_INPUT_SIGNATURE[ModelType.POSE_FULL] == ("images_context", "motions")
    context = torch.randn(_B, _T, 3, root.data.read_context_height, root.data.read_context_width)
    motions = torch.randn(_B, _T, _POSE_DIM)
    with torch.no_grad():
        out = forward_model(model, None, context, motions)  # no tight crop consumed
    assert set(out) == _SUPERVISED | {"temporal_weights"}
    assert out["temporal_weights"].shape == (_B, _T)
    assert out["crosses_frame"].shape == (_B, 2)

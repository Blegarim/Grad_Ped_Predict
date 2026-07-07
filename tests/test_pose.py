"""Pose feature math (data/pose.py) on synthetic keypoints — docs/POSE_ENCODER.md §9."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pedpredict.config.schema import DataCfg, ModelCfg, PoseCfg, RootCfg
from pedpredict.data.pose import (
    ARM_JOINTS,
    CORE_JOINTS,
    POSE_STORE_JOINTS,
    PoseCache,
    PoseMotionTransform,
    build_pose_features,
    flip_pose,
    kept_joints,
    pose_motion_transform,
    smooth_pose,
)

_T = 6


def _pose_cfg(**kwargs) -> RootCfg:
    pose = PoseCfg(enabled=True, **kwargs)
    dim = 9 + pose.feature_dim()
    return RootCfg(
        data=DataCfg(motion_dim=dim),
        model=ModelCfg(motion_dim=dim, motion_norm="none"),
        pose=pose,
    )


def _synthetic(t: int = _T, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random keypoints inside a moving bbox + the matching raw motion vector."""
    gen = torch.Generator().manual_seed(seed)
    cx = 900.0 + 3.0 * torch.arange(t)
    cy = torch.full((t,), 540.0)
    w, h = torch.full((t,), 40.0), torch.full((t,), 100.0)
    motions = torch.zeros(t, 9)
    motions[:, 0], motions[:, 1], motions[:, 4], motions[:, 5] = cx, cy, w, h
    xy = torch.rand(t, POSE_STORE_JOINTS, 2, generator=gen) - 0.5
    xy = xy * torch.stack([w, h], dim=-1)[:, None, :] + torch.stack([cx, cy], dim=-1)[:, None, :]
    conf = torch.rand(t, POSE_STORE_JOINTS, 1, generator=gen)
    return torch.cat([xy, conf], dim=-1), motions


# --------------------------------------------------------------------------- feature vector


def test_feature_shape_matches_config_dim() -> None:
    pose, motions = _synthetic()
    for include_arms, conf_channel in ((False, True), (True, True), (False, False)):
        feats = build_pose_features(pose, motions, include_arms=include_arms, conf_channel=conf_channel)
        expected = PoseCfg(include_arms=include_arms, conf_channel=conf_channel).feature_dim()
        assert feats.shape == (_T, expected)
    assert len(CORE_JOINTS) == 15 and len(kept_joints(True)) == 15 + len(ARM_JOINTS) == 19


def test_translation_invariance() -> None:
    pose, motions = _synthetic()
    shifted, motions2 = pose.clone(), motions.clone()
    shifted[..., 0] += 123.0
    shifted[..., 1] -= 45.0
    motions2[:, 0] += 123.0
    motions2[:, 1] -= 45.0
    torch.testing.assert_close(build_pose_features(pose, motions), build_pose_features(shifted, motions2))


def test_scale_invariance() -> None:
    pose, motions = _synthetic()
    s = 2.5
    center = motions[:, 0:2]
    scaled, motions2 = pose.clone(), motions.clone()
    scaled[..., :2] = center[:, None, :] + s * (pose[..., :2] - center[:, None, :])
    motions2[:, 4:6] *= s
    torch.testing.assert_close(
        build_pose_features(pose, motions), build_pose_features(scaled, motions2), atol=1e-5, rtol=1e-5
    )


def test_known_skeleton_facing_angles() -> None:
    """Ears level with the nose above them; shoulders/hips R-L pointing image-left -> known (sin, cos)."""
    pose = torch.zeros(1, POSE_STORE_JOINTS, 3)
    pose[..., 2] = 1.0                       # full confidence everywhere
    pose[0, :, 0], pose[0, :, 1] = 500.0, 300.0
    pose[0, 3, :2] = torch.tensor([495.0, 300.0])   # L ear
    pose[0, 4, :2] = torch.tensor([505.0, 300.0])   # R ear
    pose[0, 0, :2] = torch.tensor([500.0, 295.0])   # nose 5px above ear midpoint
    pose[0, 5, :2] = torch.tensor([510.0, 320.0])   # L shoulder (image-right)
    pose[0, 6, :2] = torch.tensor([490.0, 320.0])   # R shoulder (image-left)
    pose[0, 11, :2] = torch.tensor([508.0, 360.0])  # L hip
    pose[0, 12, :2] = torch.tensor([492.0, 360.0])  # R hip
    motions = torch.zeros(1, 9)
    motions[0, 0], motions[0, 1], motions[0, 5] = 500.0, 330.0, 100.0
    feats = build_pose_features(pose, motions)
    head, body = feats[0, 45:47], feats[0, 47:49]
    torch.testing.assert_close(head, torch.tensor([-1.0, 0.0]))  # nose above ears: (sin, cos)=(-1, 0)
    # R-L line points image-left (-x); perp (vy, -vx) = +y (down): (sin, cos)=(1, 0)
    torch.testing.assert_close(body, torch.tensor([1.0, 0.0]))


def test_low_confidence_gates_angles_no_nan() -> None:
    pose, motions = _synthetic()
    pose[:, [0, 3, 4], 2] = 0.0                     # dead nose + ears -> head angle untrusted
    feats = build_pose_features(pose, motions)
    assert torch.isfinite(feats).all()
    torch.testing.assert_close(feats[:, 45:47], torch.zeros(_T, 2))
    degenerate = torch.zeros(_T, POSE_STORE_JOINTS, 3)  # all joints coincident: zero-length vectors
    assert torch.isfinite(build_pose_features(degenerate, motions)).all()


# --------------------------------------------------------------------------- smoothing / flip


def test_smooth_pose_interpolates_missing_joint() -> None:
    pose, _ = _synthetic()
    pose[..., 2] = 1.0
    pose[2, 0, :2] = torch.tensor([9999.0, 9999.0])  # garbage coords on a low-conf frame
    pose[2, 0, 2] = 0.05
    out = smooth_pose(pose, window=1, min_conf=0.3)
    expected = (pose[1, 0, :2] + pose[3, 0, :2]) / 2
    torch.testing.assert_close(out[2, 0, :2], expected)
    assert out[2, 0, 2] == pytest.approx(0.05)       # confidence kept raw
    torch.testing.assert_close(out[1], pose[1])       # confident frames untouched at window=1


def test_smooth_pose_moving_average() -> None:
    pose, _ = _synthetic()
    pose[..., 2] = 1.0
    out = smooth_pose(pose, window=3, min_conf=0.3)
    expected = (pose[1, 5, 0] + pose[2, 5, 0] + pose[3, 5, 0]) / 3
    assert out[2, 5, 0] == pytest.approx(expected.item(), abs=1e-5)


def test_flip_pose_reflects_and_swaps() -> None:
    pose, _ = _synthetic()
    flipped = flip_pose(pose, source_width=1920.0)
    torch.testing.assert_close(flipped[:, 0, 0], 1920.0 - pose[:, 0, 0])   # nose stays nose
    torch.testing.assert_close(flipped[:, 5, 1:], pose[:, 6, 1:])          # L shoulder <- R shoulder
    torch.testing.assert_close(flip_pose(flipped, 1920.0), pose)           # involution


# --------------------------------------------------------------------------- read-time transform


def test_pose_motion_transform_widths_and_norm() -> None:
    pose, motions = _synthetic()
    root = _pose_cfg()
    transform = pose_motion_transform(root)
    assert isinstance(transform, PoseMotionTransform)
    out = transform(pose, motions)
    assert out.shape == (_T, root.data.motion_dim)
    torch.testing.assert_close(out[:, 0], motions[:, 0] / root.data.source_width)   # cx image-normalized
    torch.testing.assert_close(out[:, 8], motions[:, 8] / root.model.ego_speed_scale)
    torch.testing.assert_close(out[:, 9:], build_pose_features(pose, motions))
    assert pose_motion_transform(RootCfg()) is None


# --------------------------------------------------------------------------- cache roundtrip


def test_pose_cache_sequence_roundtrip(tmp_path) -> None:
    arr = np.random.default_rng(0).random((POSE_STORE_JOINTS, 3)).astype(np.float32)
    video_dir = tmp_path / "set01"
    video_dir.mkdir()
    np.savez(video_dir / "video_0001.npz", **{"42_1_1_5": arr, "43_1_1_5": arr + 1.0})
    cache = PoseCache(tmp_path)
    images = [str(tmp_path / "images" / "set01" / "video_0001" / f"{f:05d}.png") for f in (42, 43)]
    seq = cache.sequence(images, "1_1_5")
    assert seq.shape == (2, POSE_STORE_JOINTS, 3)
    torch.testing.assert_close(seq[0], torch.from_numpy(arr))
    with pytest.raises(KeyError, match="stale cache"):
        cache.sequence([images[0]], "9_9_9")
    with pytest.raises(FileNotFoundError, match="extract_pose"):
        cache.sequence([str(tmp_path / "images" / "set99" / "video_0009" / "00001.png")], "1_1_5")

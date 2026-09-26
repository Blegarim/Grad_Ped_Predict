"""Pose feature math (data/pose.py) + LMDB pose plumbing — docs/POSE_ENCODER.md §9."""

from __future__ import annotations

import dataclasses
import json
import pickle

import lmdb
import numpy as np
import pytest
import torch

from pedpredict.config.schema import AugmentCfg, DataCfg, ModelCfg, PoseCfg, RootCfg
from pedpredict.data.augment import SequenceAugmenter
from pedpredict.data.lmdb_dataset import LMDBChunkDataset, read_raw_sample
from pedpredict.data.lmdb_writer import write_dataset_chunks
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


# --------------------------------------------------------------------------- LMDB plumbing (step 3)


def _write_chunk(tmp_path, *, with_pose: bool):
    """Write a 1-record chunk through the real writer (synthetic frames, optional pose cache)."""
    from PIL import Image

    seq_len = 4
    frames_dir = tmp_path / "images" / "set01" / "video_0001"
    frames_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    images, keys = [], {}
    for f in range(seq_len):
        p = frames_dir / f"{f:05d}.png"
        Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8), "RGB").save(p)
        images.append(str(p))
        keys[f"{f}_ped_0"] = rng.random((POSE_STORE_JOINTS, 3)).astype(np.float32)
    pose_cache = None
    if with_pose:
        cache_dir = tmp_path / "pose_cache" / "set01"
        cache_dir.mkdir(parents=True)
        np.savez(cache_dir / "video_0001.npz", **keys)
        pose_cache = PoseCache(tmp_path / "pose_cache")
    record = {
        "images": images,
        "bboxes": [[10.0, 10.0, 60.0, 90.0]] * seq_len,
        "track_id": "ped_0",
        "ego_speed": [1.0] * seq_len,
        "actions": 1,
        "looks": 0,
        "crosses": 1,
    }
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=64 * 1024 * 1024)
    paths = write_dataset_chunks([record], tmp_path / "lmdb", cfg, num_workers=0, pose_cache=pose_cache)
    return paths[0], keys, seq_len


def test_lmdb_pose_roundtrip_and_read_path(tmp_path) -> None:
    """Writer stores raw pose in the meta; the dataset read path emits the built [T, 58] motions."""
    chunk, keys, seq_len = _write_chunk(tmp_path, with_pose=True)
    env = lmdb.open(str(chunk), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            meta = pickle.loads(txn.get(b"0_meta"))
            assert meta["pose"].shape == (seq_len, POSE_STORE_JOINTS, 3)
            torch.testing.assert_close(meta["pose"][0], torch.from_numpy(keys["0_ped_0"]))
            assert read_raw_sample(txn, "0").pose is not None  # offline-augment source keeps pose
    finally:
        env.close()

    root = _pose_cfg()
    ds = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))
    sample = ds[0]
    assert sample["motions"].shape == (seq_len, root.data.motion_dim)
    # A pose-less read of the same chunk stays on the 9-dim contract (additive meta key ignored).
    plain = LMDBChunkDataset.from_config(chunk, DataCfg())
    assert plain[0]["motions"].shape == (seq_len, DataCfg().motion_dim)


def test_pose_read_of_poseless_chunk_fails_loudly(tmp_path) -> None:
    chunk, _, _ = _write_chunk(tmp_path, with_pose=False)
    root = _pose_cfg()
    ds = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))
    with pytest.raises(ValueError, match="no pose meta"):
        ds[0]


def test_augment_flip_transforms_pose(tmp_path) -> None:
    """horizontal_flip keeps pose and motions geometrically coupled (reflect + L/R swap)."""
    chunk, _, _ = _write_chunk(tmp_path, with_pose=True)
    env = lmdb.open(str(chunk), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            sample = read_raw_sample(txn, "0")
    finally:
        env.close()
    cfg = DataCfg()
    flipped = SequenceAugmenter(AugmentCfg(), cfg.source_width).horizontal_flip(sample)
    torch.testing.assert_close(flipped.pose[:, 0, 0], cfg.source_width - sample.pose[:, 0, 0])
    torch.testing.assert_close(flipped.pose[:, 5, 1], sample.pose[:, 6, 1])  # L <- R shoulder y


# --------------------------------------------------------------------------- input standardization (2026-09-25)


def _stats_cfg(mean, std, clip: float = 5.0) -> RootCfg:
    root = _pose_cfg()
    pose = dataclasses.replace(root.pose, input_mean=tuple(mean), input_std=tuple(std), input_clip=clip)
    return dataclasses.replace(root, pose=pose)


def test_input_standardization_off_by_default_is_byte_identical() -> None:
    root = _pose_cfg()
    pose, motions = torch.rand(_T, POSE_STORE_JOINTS, 3) * 500, torch.rand(_T, 9) * 50
    raw = PoseMotionTransform(root)(pose, motions)
    assert PoseMotionTransform(root).mean is None
    torch.testing.assert_close(PoseMotionTransform(root)(pose, motions), raw)


def test_input_standardization_applies_and_clamps() -> None:
    root = _pose_cfg()
    width = root.data.motion_dim
    pose, motions = torch.rand(_T, POSE_STORE_JOINTS, 3) * 500, torch.rand(_T, 9) * 50
    raw = PoseMotionTransform(root)(pose, motions)
    mean, std = torch.rand(width), torch.rand(width) * 0.01 + 1e-3   # tiny std -> some entries clamp
    out = PoseMotionTransform(_stats_cfg(mean.tolist(), std.tolist(), clip=2.0))(pose, motions)
    torch.testing.assert_close(out, ((raw - mean) / std).clamp(-2.0, 2.0))
    assert out.abs().max() <= 2.0


def test_input_stats_json_fills_config_and_survives_the_resolved_snapshot(tmp_path) -> None:
    """The numbers travel in resolved_config.yaml, so eval never needs (or re-reads) the JSON."""
    from pedpredict.config import dump_config, load_config, load_resolved_config

    width = _pose_cfg().data.motion_dim
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"mean": [0.5] * width, "std": [2.0] * width}), encoding="utf-8")
    bundle = ["pose.enabled=true", "model.motion_norm=none", f"data.motion_dim={width}", f"model.motion_dim={width}"]
    root = load_config("configs", overrides=[*bundle, f"pose.input_stats={stats.as_posix()}"])
    assert root.pose.input_mean == (0.5,) * width and root.pose.input_std == (2.0,) * width

    snapshot = dump_config(root, tmp_path / "run")
    stats.unlink()                                             # eval must not need the file
    again = load_resolved_config(snapshot, validate=True)
    assert again.pose.input_mean == root.pose.input_mean and again.pose.input_std == root.pose.input_std


def test_input_stats_validation() -> None:
    from pedpredict.config import ConfigError
    from pedpredict.config.loader import validate_config

    width = _pose_cfg().data.motion_dim
    with pytest.raises(ConfigError, match="one entry per read-path channel"):
        validate_config(_stats_cfg([0.0] * (width - 1), [1.0] * (width - 1)))
    with pytest.raises(ConfigError, match="must be > 0"):
        validate_config(_stats_cfg([0.0] * width, [1.0] * (width - 1) + [0.0]))
    with pytest.raises(ConfigError, match="input_clip"):
        validate_config(_stats_cfg([0.0] * width, [1.0] * width, clip=0.0))
    no_pose = dataclasses.replace(RootCfg(), pose=PoseCfg(input_mean=(0.0,) * 9, input_std=(1.0,) * 9))
    with pytest.raises(ConfigError, match="require pose.enabled"):
        validate_config(no_pose)
    with pytest.raises(ConfigError, match="not found"):
        from pedpredict.config import load_config
        load_config("configs", overrides=["pose.enabled=true", "model.motion_norm=none", f"data.motion_dim={width}",
                                          f"model.motion_dim={width}", "pose.input_stats=does/not/exist.json"])


def test_compute_input_stats_matches_numpy(tmp_path) -> None:
    from pedpredict.data.input_stats import compute_input_stats, raw_transform

    chunk, _, _ = _write_chunk(tmp_path, with_pose=True)
    root = _stats_cfg([9.0] * _pose_cfg().data.motion_dim, [9.0] * _pose_cfg().data.motion_dim)  # must be stripped
    stats = compute_input_stats([chunk.parent], root)

    env = lmdb.open(str(chunk), readonly=True, lock=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"0_meta"))
    env.close()
    frames = raw_transform(root)(torch.as_tensor(meta["pose"]), torch.as_tensor(meta["motions"])).double()
    np.testing.assert_allclose(stats["mean"], frames.mean(0).numpy(), rtol=1e-6, atol=1e-9)
    expected_std = frames.std(0, unbiased=False).clamp_min(1e-6).numpy()
    np.testing.assert_allclose(stats["std"], expected_std, rtol=1e-5, atol=1e-7)
    assert stats["n_windows"] == 1 and stats["width"] == root.data.motion_dim


# --------------------------------------------------------------------------- pixel-free read (data.visual_input=none)


def _pixel_free(root: RootCfg) -> RootCfg:
    return dataclasses.replace(root, data=dataclasses.replace(root.data, visual_input="none"),
                               eval=dataclasses.replace(root.eval, model_type="pose_kinematics"))


def test_pixel_free_read_skips_images_and_keeps_motions(tmp_path) -> None:
    chunk, _, seq_len = _write_chunk(tmp_path, with_pose=True)
    root = _pose_cfg()
    image_path = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))[0]
    pf = _pixel_free(root)
    sample = LMDBChunkDataset.from_config(chunk, pf.data, pose_transform=pose_motion_transform(pf))[0]
    assert sample["images_tight"].shape == (seq_len, 0) and sample["images_context"].shape == (seq_len, 0)
    torch.testing.assert_close(sample["motions"], image_path["motions"])
    assert sample["crosses"] == image_path["crosses"]


@pytest.mark.parametrize("aug_seed", range(6))
def test_pixel_free_augmentation_matches_image_path(tmp_path, aug_seed: int) -> None:
    """Same (seed, idx) -> same flip + noise on motions AND pose, with or without pixels."""
    from pedpredict.data.augment import RuntimeAugmentor

    chunk, _, _ = _write_chunk(tmp_path, with_pose=True)
    root = _pose_cfg()
    aug_cfg = AugmentCfg(p_flip=0.5, p_color=0.5, p_noise=0.5, p_erase=0.5, motion_noise_std=5.0)
    aug = RuntimeAugmentor(aug_cfg, root.data.source_width)
    img = LMDBChunkDataset.from_config(chunk, root.data, augmentor=aug, aug_seed=aug_seed,
                                       pose_transform=pose_motion_transform(root))[0]
    pf = _pixel_free(root)
    free = LMDBChunkDataset.from_config(chunk, pf.data, augmentor=aug, aug_seed=aug_seed,
                                        pose_transform=pose_motion_transform(pf))[0]
    torch.testing.assert_close(free["motions"], img["motions"])


def test_visual_input_none_rejects_models_that_read_pixels() -> None:
    from pedpredict.config import ConfigError
    from pedpredict.config.loader import validate_config

    root = _pose_cfg()
    bad = dataclasses.replace(root, data=dataclasses.replace(root.data, visual_input="none"),
                              eval=dataclasses.replace(root.eval, model_type="pose_full"))
    with pytest.raises(ConfigError, match="pixel|kinematics"):
        validate_config(bad)
    validate_config(_pixel_free(root))


def test_objects_pickled_by_the_previous_version_still_read(tmp_path) -> None:
    """A live run's DataLoader workers re-import the package from disk on every spawn: instances built by
    the pre-update code (no _decode_images, no mean/std) must keep reading exactly as before."""
    chunk, _, _ = _write_chunk(tmp_path, with_pose=True)
    root = _pose_cfg()
    ds = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))
    expected = ds[0]["motions"]
    del ds._decode_images
    del ds._pose_transform.mean, ds._pose_transform.std
    old = pickle.loads(pickle.dumps(ds))
    sample = old[0]
    torch.testing.assert_close(sample["motions"], expected)
    assert sample["images_context"].ndim == 4          # still decoding images

"""Deep scene arm: scene encoding, the sidecar lookup, the read-path append, config + eval inheritance, the builder."""

from __future__ import annotations

import dataclasses
import json
import math
import pickle

import lmdb
import numpy as np
import pytest
import torch

from pedpredict.config import ConfigError
from pedpredict.config.loader import dump_config, merge_eval_config, validate_config
from pedpredict.config.schema import DataCfg, EvalCfg, ModelCfg, PoseCfg, RootCfg
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.pose import POSE_STORE_JOINTS, pose_motion_transform
from pedpredict.data.scene_lookup import SceneLookup, encode_scene

_T, _W = 4, 1920.0


def _root(scene_path: str = "", scene_dim: int = 0) -> RootCfg:
    pose = PoseCfg(enabled=True)
    dim = 9 + pose.feature_dim() + scene_dim
    return RootCfg(
        data=DataCfg(motion_dim=dim, visual_input="none", scene_features=scene_path, scene_dim=scene_dim),
        model=ModelCfg(motion_dim=dim, motion_norm="none"), pose=pose, eval=EvalCfg(model_type="pose_kinematics"),
    )


def _write_sidecar(path, rows: list[tuple[str, int, int, float, float]], features: np.ndarray) -> None:
    tid, fo, oo, cx, cy = (list(c) for c in zip(*rows, strict=True))
    np.savez_compressed(path, track_id=np.asarray(tid), future_observed=np.asarray(fo, dtype=np.int64),
                        onset_offset=np.asarray(oo, dtype=np.int64), cx=np.asarray(cx), cy=np.asarray(cy),
                        features=features.astype(np.float32), spec=np.asarray(json.dumps({"source_width": _W})))


def _meta(track_id: str, future_observed: int, onset_offset: int, cx: float) -> dict:
    motions = torch.zeros(_T, 9)
    motions[:, 0], motions[:, 1], motions[:, 4], motions[:, 5] = cx, 540.0, 40.0, 100.0
    pose = torch.rand(_T, POSE_STORE_JOINTS, 3, generator=torch.Generator().manual_seed(1))
    pose[..., 0] = pose[..., 0] * 40 + cx - 20
    pose[..., 1] = pose[..., 1] * 100 + 490
    return {"motions": motions, "actions": 0, "looks": 0, "crosses": 0, "track_id": track_id, "pose": pose,
            "onset_offset": onset_offset, "future_observed": future_observed, "track_crosses": 0}


def _write_meta_chunk(path, metas: list[dict]) -> None:
    """A metadata-only chunk (what a pixel-free read consumes) through plain lmdb puts."""
    env = lmdb.open(str(path), map_size=16 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for j, meta in enumerate(metas):
            txn.put(f"{j}_meta".encode(), pickle.dumps(meta))
    env.close()


# --------------------------------------------------------------------------- encoding


def test_encode_scene_fits_on_train_rows_and_encodes_types() -> None:
    raw = {"t_n_vehicles": np.array([1.0, 3.0, 100.0, np.nan]),     # rows 2-3 are not fit rows
           "s_cw_dist": np.array([0.1, np.nan, 0.3, 0.2]),
           "r_state": np.array([0.0, 1.0, 3.0, np.nan]),
           "s_num_lanes": np.array([2.0, 2.0, 2.0, 2.0])}
    fit = np.array([True, True, False, False])
    block, spec = encode_scene(raw, list(raw), fit_rows=fit, clip=5.0)
    assert spec["names"] == ["t_n_vehicles", "s_cw_dist", "s_cw_dist_missing", "r_state=0", "r_state=1",
                             "r_state=2", "r_state=3", "s_num_lanes"]
    assert spec["mean"]["t_n_vehicles"] == 2.0 and spec["std"]["t_n_vehicles"] == 1.0   # from rows 0-1 only
    np.testing.assert_allclose(block[:, 0], [-1.0, 1.0, 5.0, 0.0])                     # clipped; NaN -> 0
    assert "t_n_vehicles_missing" not in spec["names"]          # never missing in the fit rows -> no flag
    np.testing.assert_allclose(block[:, 2], [0, 1, 0, 0])       # missing flag
    assert block[1, 1] == 0.0                                   # missing value -> 0
    np.testing.assert_allclose(block[:, 3:7], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    np.testing.assert_allclose(block[:, 7], 0.0)                # constant feature stays 0, never inf


# --------------------------------------------------------------------------- lookup


def test_lookup_resolves_collisions_and_flipped_copies(tmp_path) -> None:
    rows = [("1_1_1", 40, 10, 300.0, 500.0), ("1_1_1", 40, 10, 1500.0, 520.0), ("1_1_2", 40, -1, 800.0, 600.0)]
    path = tmp_path / "s.npz"
    _write_sidecar(path, rows, np.arange(6, dtype=np.float32).reshape(3, 2))
    lut = SceneLookup(path)
    assert lut.width == 2
    assert lut.row("1_1_1", 40, 10, 1500.0, 520.0) == 1
    assert lut.row("1_1_1", 40, 10, 300.0, 500.0) == 0
    assert lut.row("1_1_1", 40, 10, _W - 300.0, 500.0) == 0      # a flipped copy reads its source's row
    assert lut.row("1_1_2", 40, -1, 800.01, 600.0) == 2           # noise-sized offsets are fine
    torch.testing.assert_close(lut.vector("1_1_2", 40, -1, 800.0, 600.0), torch.tensor([4.0, 5.0]))
    with pytest.raises(KeyError, match="no scene row"):
        lut.row("9_9_9", 40, 10, 300.0, 500.0)
    with pytest.raises(KeyError, match="matches no row"):
        lut.row("1_1_2", 40, -1, 900.0, 600.0)


# --------------------------------------------------------------------------- read path


def test_read_path_appends_the_windows_scene_to_every_frame(tmp_path) -> None:
    sidecar = tmp_path / "s.npz"
    _write_sidecar(sidecar, [("1_1_1", 40, 10, 300.0, 540.0), ("1_1_2", 30, -1, 800.0, 540.0)],
                   np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    chunk = tmp_path / "chunk_0.lmdb"
    flipped = _meta("1_1_1", 40, 10, _W - 300.0)                  # an offline flipped copy of window 0
    _write_meta_chunk(chunk, [_meta("1_1_1", 40, 10, 300.0), _meta("1_1_2", 30, -1, 800.0), flipped])
    root = _root(str(sidecar), 3)
    validate_config(root)
    plain_root = _root()
    scene = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))
    plain = LMDBChunkDataset.from_config(chunk, plain_root.data, pose_transform=pose_motion_transform(plain_root))
    for i, expected in enumerate(([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0])):
        motions = scene[i]["motions"]
        assert motions.shape == (_T, root.data.motion_dim)
        torch.testing.assert_close(motions[:, :-3], plain[i]["motions"])        # the pose block is untouched
        torch.testing.assert_close(motions[:, -3:], torch.tensor(expected).expand(_T, 3))


def test_read_path_fails_loudly_on_a_window_the_sidecar_lacks(tmp_path) -> None:
    sidecar = tmp_path / "s.npz"
    _write_sidecar(sidecar, [("1_1_1", 40, 10, 300.0, 540.0)], np.ones((1, 2)))
    chunk = tmp_path / "chunk_0.lmdb"
    _write_meta_chunk(chunk, [_meta("2_2_2", 40, 10, 300.0)])
    root = _root(str(sidecar), 2)
    ds = LMDBChunkDataset.from_config(chunk, root.data, pose_transform=pose_motion_transform(root))
    with pytest.raises(KeyError, match="no scene row"):
        ds[0]


def test_scene_needs_the_pose_read_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="pose_transform"):
        LMDBChunkDataset.from_config(tmp_path, dataclasses.replace(DataCfg(), scene_features="x.npz"))


# --------------------------------------------------------------------------- config + eval inheritance


def test_config_validation_of_the_scene_fields() -> None:
    validate_config(_root("x.npz", 5))
    validate_config(_root())
    bad_width = _root("x.npz", 5)
    with pytest.raises(ConfigError, match="scene_dim"):
        validate_config(dataclasses.replace(bad_width, data=dataclasses.replace(bad_width.data, motion_dim=58),
                                            model=dataclasses.replace(bad_width.model, motion_dim=58)))
    no_path = _root()
    with pytest.raises(ConfigError, match="needs data.scene_features"):
        validate_config(dataclasses.replace(no_path, data=dataclasses.replace(no_path.data, scene_dim=3)))
    no_dim = _root()
    with pytest.raises(ConfigError, match="positive"):
        validate_config(dataclasses.replace(no_dim, data=dataclasses.replace(no_dim.data, scene_features="x.npz")))
    pixels = _root("x.npz", 5)
    with pytest.raises(ConfigError, match="pixel-free"):
        validate_config(dataclasses.replace(pixels, eval=dataclasses.replace(pixels.eval, model_type="pose_full"),
                                            data=dataclasses.replace(pixels.data, visual_input="images")))


def test_eval_inherits_the_scene_input_from_the_checkpoint(tmp_path) -> None:
    trained = _root("outputs/features/scene/scene_rts.npz", 43)
    path = dump_config(trained, tmp_path)
    merged = merge_eval_config(_root(), path)
    assert merged.data.scene_features == trained.data.scene_features
    assert merged.data.scene_dim == 43 and merged.data.motion_dim == trained.data.motion_dim
    validate_config(merged)


# --------------------------------------------------------------------------- builder


def _write_exports(d, *, collide: bool) -> None:
    from pedpredict.eval.onset_timing import save_dump

    rng = np.random.default_rng(0)
    for split, n, prefix in (("train", 6, "1"), ("val", 3, "4"), ("test", 4, "3")):
        x = np.zeros((n, _T, 58), dtype=np.float32)
        x[:, -1, 0] = (100.0 + 200.0 * np.arange(n)) / _W
        x[:, -1, 1] = 0.5
        tid = np.asarray([f"{prefix}_1_{i}" for i in range(n)])
        if collide and split == "train":
            tid[1] = tid[0]                              # one pedestrian split into two segments
        np.savez(d / f"{split}.npz", X=x, track_id=tid, future_observed=np.full(n, 40),
                 onset_offset=np.full(n, -1))
        scene = {"t_n_vehicles": rng.integers(0, 5, n).astype(np.float32),
                 "s_cw_dist": np.where(np.arange(n) % 2, np.nan, 0.2).astype(np.float32),
                 "r_state": rng.integers(0, 4, n).astype(np.float32)}
        save_dump(d / f"scene_{split}.npz", scene, {"split": split})


def test_build_writes_a_sidecar_the_lookup_reads(tmp_path, monkeypatch) -> None:
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "build_scene_sidecar", pathlib.Path(__file__).parents[1] / "scripts" / "build_scene_sidecar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("pedpredict.eval.scene_features.SCENE_GROUPS",
                        {"X": ("t_n_vehicles", "s_cw_dist", "r_state")})
    _write_exports(tmp_path, collide=True)
    out = tmp_path / "side.npz"
    assert mod.main(["build", "--exports", str(tmp_path), "--groups", "X", "--out", str(out)]) == 0
    lut = SceneLookup(out)
    assert lut.width == 1 + 2 + 4                       # numeric, numeric + missing flag, 4-way one-hot
    assert lut.row("1_1_0", 40, -1, 300.0, 540.0) == 1  # the collided key resolves by box
    assert lut.row("3_1_3", 40, -1, 700.0, 540.0) == 6 + 3 + 3
    assert lut.spec["rows"] == {"train": 6, "val": 3, "test": 4} and lut.spec["shared_keys"] == 1
    assert not math.isnan(float(lut.features.sum()))

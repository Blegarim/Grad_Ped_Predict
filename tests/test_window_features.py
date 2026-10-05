"""Window-feature export: the arrays are the read path's output, in dump order, with honest optional keys."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch

from pedpredict.config.schema import DataCfg, ModelCfg, PoseCfg, RootCfg
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.pose import POSE_STORE_JOINTS, pose_motion_transform
from pedpredict.eval.window_features import channel_names, check_against_dump, export_windows

_T = 5
_MAP = 8 * 1024 * 1024   # small: Windows pre-allocates the LMDB file at map_size


def _cfg(**pose_kwargs) -> RootCfg:
    pose = PoseCfg(enabled=True, **pose_kwargs)
    dim = 9 + pose.feature_dim()
    return RootCfg(data=dataclasses.replace(DataCfg(motion_dim=dim), visual_input="none"),
                   model=ModelCfg(motion_dim=dim, motion_norm="none"), pose=pose)


def _meta(rng: np.random.Generator, i: int, *, onset: bool, tte: bool) -> dict:
    motions = torch.zeros(_T, 9)
    motions[:, 0] = torch.as_tensor(900.0 + 3.0 * np.arange(_T) + i)
    motions[:, 1], motions[:, 4], motions[:, 5] = 540.0, 40.0, 100.0
    motions[1:, 2] = 3.0
    motions[:, 8] = float(i % 7)
    pose = torch.as_tensor(rng.random((_T, POSE_STORE_JOINTS, 3)), dtype=torch.float32)
    pose[..., :2] = pose[..., :2] * 80.0 + 860.0
    meta = {"motions": motions, "pose": pose, "actions": i % 2, "looks": 0, "crosses": -1 if i == 3 else i % 3 == 0,
            "track_id": f"1_1_{i // 3}"}
    if onset:
        meta.update(onset_offset=i - 4, future_observed=40 + i, track_crosses=int(i >= 2))
    if tte:
        meta["tte"] = 30 + i
    return meta


def _chunk(path: Path, metas: list[dict]) -> str:
    env = lmdb.open(str(path), map_size=_MAP)
    with env.begin(write=True) as txn:
        for j, meta in enumerate(metas):
            txn.put(f"{j}_meta".encode(), pickle.dumps(meta))
    env.close()
    return str(path)


def _chunks(tmp_path: Path, *, onset: bool = True, tte: bool = False) -> list[str]:
    rng = np.random.default_rng(0)
    metas = [_meta(rng, i, onset=onset, tte=tte) for i in range(9)]
    return [_chunk(tmp_path / "chunk_000000.lmdb", metas[:4]), _chunk(tmp_path / "chunk_000004.lmdb", metas[4:])]


def test_export_is_the_read_path_in_order(tmp_path) -> None:
    cfg = _cfg()
    paths = _chunks(tmp_path)
    out = export_windows(cfg, paths, batch_size=3)
    assert out["X"].shape == (9, _T, 58) and out["X"].dtype == np.float32
    transform = pose_motion_transform(cfg)
    expected = []
    for p in paths:
        ds = LMDBChunkDataset.from_config(p, cfg.data, pose_transform=transform)
        expected += [ds[i]["motions"].numpy() for i in range(len(ds))]
        ds.close()
    np.testing.assert_array_equal(out["X"], np.stack(expected))
    np.testing.assert_array_equal(out["chunk"], [0] * 4 + [1] * 5)
    assert out["crosses"].tolist() == [1, 0, 0, 0, 0, 0, 1, 0, 0]   # the stored -1 is clipped, as dumps do
    assert out["onset_offset"].tolist() == list(range(-4, 5))
    assert "tte" not in out


def test_anchored_style_chunks_carry_tte_and_no_onset_fields(tmp_path) -> None:
    out = export_windows(_cfg(), _chunks(tmp_path, onset=False, tte=True), batch_size=4)
    assert out["tte"].tolist() == list(range(30, 39))
    assert not {"onset_offset", "future_observed", "track_crosses"} & set(out)


def test_mixed_vintage_dirs_fail_loudly(tmp_path) -> None:
    rng = np.random.default_rng(0)
    a = _chunk(tmp_path / "a.lmdb", [_meta(rng, i, onset=True, tte=False) for i in range(3)])
    b = _chunk(tmp_path / "b.lmdb", [_meta(rng, i, onset=False, tte=False) for i in range(3)])
    with pytest.raises(ValueError, match="mixed-vintage"):
        export_windows(_cfg(), [a, b], batch_size=8)


def test_refuses_standardized_or_image_reads(tmp_path) -> None:
    paths = _chunks(tmp_path)
    cfg = _cfg()
    standardized = dataclasses.replace(cfg, pose=dataclasses.replace(cfg.pose, input_mean=(0.0,) * 58,
                                                                    input_std=(1.0,) * 58))
    with pytest.raises(ValueError, match="RAW"):
        export_windows(standardized, paths)
    with pytest.raises(ValueError, match="visual_input"):
        export_windows(dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, visual_input="images")), paths)


def test_check_against_dump_reports_each_disagreement(tmp_path) -> None:
    out = export_windows(_cfg(), _chunks(tmp_path))
    dump = {k: out[k].copy() for k in ("crosses", "track_id", "onset_offset", "future_observed", "track_crosses")}
    assert check_against_dump(out, dump) == []
    dump["onset_offset"][2] += 1
    dump["track_id"][0] = "x"
    assert check_against_dump(out, dump) == ["track_id: 1 rows differ", "onset_offset: 1 rows differ"]
    assert check_against_dump(out, {"crosses": np.zeros(3)})[0].startswith("length")


def test_channel_names_follow_the_feature_block_order() -> None:
    names = channel_names(PoseCfg(enabled=True))
    assert len(names) == 58 and names[8] == "ego" and names[9:11] == ("nose_x", "nose_y")
    assert names[39] == "nose_conf" and names[-4:] == ("head_sin", "head_cos", "body_sin", "body_cos")
    assert len(channel_names(PoseCfg(enabled=True, include_arms=True))) == 70

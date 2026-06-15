"""Runtime LMDB dataset + collate tests (v2 read contract).

Rebuilds the exact LMDB the legacy capture used (same frames -> same deterministic writer geometry),
reads it with :class:`LMDBChunkDataset` + :func:`build_collate`, and diffs against the golden fixture
— with the motion expectation DERIVED from the golden per the v2 contract (A4 frame-0 zeros + M9 ego,
sliced back to ``motion_dim`` at read). Pixel outputs stay exact golden parity.

Covers: per-item parity, batched-tuple parity, ``seq_ids`` order, the store-wide/slice-narrow motion
contract, ``track_id`` passthrough, the v1-meta loud failure, the B7 collate guard, and worker-safety
under ``num_workers>0`` (the per-process env + picklable dataset/collate).
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import lmdb
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from pedpredict.config import DataCfg
from pedpredict.data import LMDBChunkDataset, build_collate
from pedpredict.data.collate import collate_sequences
from pedpredict.data.lmdb_writer import write_dataset_chunks

_FIXTURE = Path(__file__).parent / "fixtures" / "golden" / "lmdb_dataset_cases.pt"
_SMALL_MAP = 64 * 1024 * 1024
_IMG_ATOL = 1e-6


def _load_fixture():
    if not _FIXTURE.exists():
        pytest.skip(f"missing golden fixture {_FIXTURE} (run tests/_capture/capture_lmdb_dataset_golden.py)")
    return torch.load(_FIXTURE, weights_only=False)


def _ego_for(s: int, t_frames: int) -> list[float]:
    return [float(10 * s + t) for t in range(t_frames)]


def _rebuild_lmdb(fx, tmp_path: Path) -> tuple[Path, DataCfg]:
    """Re-materialize the captured frames -> v2 records -> LMDB via the new writer."""
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = []
    for s, (frames, bboxes, labels) in enumerate(
        zip(fx["inputs"]["frames"], fx["inputs"]["bboxes"], fx["inputs"]["labels"], strict=True)
    ):
        paths = []
        for t, arr in enumerate(frames):
            p = frames_dir / f"s{s}_f{t}.png"
            Image.fromarray(arr.numpy()).save(p)
            paths.append(str(p))
        records.append({
            "images": paths,
            "bboxes": bboxes,
            "track_id": f"ped_{s}",
            "ego_speed": _ego_for(s, len(paths)),
            **labels,
        })
    chunk_paths = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    assert len(chunk_paths) == 1
    return chunk_paths[0], cfg


def _v2_motions(golden: torch.Tensor, motion_dim: int, ego: list[float] | None = None) -> torch.Tensor:
    """Golden legacy 8-ch motions -> v2 expectation (frame-0 deltas zeroed, +ego), sliced to motion_dim."""
    out = golden.clone()
    out[0, 2:4] = 0.0
    out[0, 6:8] = 0.0
    if ego is not None:
        out = torch.cat([out, torch.tensor(ego, dtype=torch.float32).unsqueeze(1)], dim=1)
    return out[:, :motion_dim]


def test_dataset_per_item_parity(tmp_path) -> None:
    fx = _load_fixture()
    chunk, cfg = _rebuild_lmdb(fx, tmp_path)
    ds = LMDBChunkDataset.from_config(chunk, cfg)

    assert ds.seq_ids == fx["seq_ids"]
    assert len(ds) == len(fx["per_item"])
    for i, golden in enumerate(fx["per_item"]):
        item = ds[i]
        torch.testing.assert_close(item["images_tight"], golden["images_tight"], rtol=0, atol=_IMG_ATOL)
        torch.testing.assert_close(item["images_context"], golden["images_context"], rtol=0, atol=_IMG_ATOL)
        torch.testing.assert_close(
            item["motions"], _v2_motions(golden["motions"], cfg.motion_dim), rtol=0, atol=0
        )
        assert item["motions"].shape[-1] == cfg.motion_dim   # ego sliced away at the default 8
        assert item["track_id"] == f"ped_{int(ds.seq_ids[i])}"
        assert "tte" not in item                             # standard (non-benchmark) chunk
        for k in ("actions", "looks", "crosses"):
            torch.testing.assert_close(item[k], golden[k], rtol=0, atol=0)


def test_dataset_motion_dim_9_exposes_ego(tmp_path) -> None:
    """The same chunk read with motion_dim=9 yields the stored ego channel (store wide, slice narrow)."""
    fx = _load_fixture()
    chunk, _ = _rebuild_lmdb(fx, tmp_path)
    cfg9 = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP, motion_dim=9)
    ds = LMDBChunkDataset.from_config(chunk, cfg9)
    for i in range(len(ds)):
        item = ds[i]
        s = int(ds.seq_ids[i])
        assert item["motions"].shape[-1] == 9
        assert item["motions"][:, 8].tolist() == _ego_for(s, item["motions"].shape[0])


def test_dataset_rejects_narrow_stored_motion(tmp_path) -> None:
    """A chunk whose stored motions are narrower than motion_dim fails loudly (stale build)."""
    path = tmp_path / "stale.lmdb"
    env = lmdb.open(str(path), map_size=_SMALL_MAP)
    meta = {
        "motions": torch.zeros(2, 8),  # v1 width
        "actions": torch.tensor(0), "looks": torch.tensor(0), "crosses": torch.tensor(0),
        "track_id": "ped_0",
    }
    with env.begin(write=True) as txn:
        txn.put(b"0_meta", pickle.dumps(meta))
    env.close()
    ds = LMDBChunkDataset.from_config(path, dataclasses.replace(DataCfg(), motion_dim=9))
    with pytest.raises(ValueError, match="rebuild required"):
        ds[0]


def test_dataset_rejects_v1_meta(tmp_path) -> None:
    """v1 meta (no track_id) is a pre-rebuild chunk — loud error, never silent defaults."""
    path = tmp_path / "v1.lmdb"
    env = lmdb.open(str(path), map_size=_SMALL_MAP)
    meta = {
        "motions": torch.zeros(2, 8),
        "actions": torch.tensor(0), "looks": torch.tensor(0), "crosses": torch.tensor(0),
    }
    with env.begin(write=True) as txn:
        txn.put(b"0_meta", pickle.dumps(meta))
    env.close()
    ds = LMDBChunkDataset.from_config(path, DataCfg())
    with pytest.raises(ValueError, match="v1 meta"):
        ds[0]


def test_collate_batch_parity(tmp_path) -> None:
    fx = _load_fixture()
    chunk, cfg = _rebuild_lmdb(fx, tmp_path)
    ds = LMDBChunkDataset.from_config(chunk, cfg)
    collate = build_collate(cfg)

    it, ic, mo, lab = collate([ds[i] for i in range(len(ds))])
    g = fx["batch"]
    torch.testing.assert_close(it, g["images_tight"], rtol=0, atol=_IMG_ATOL)
    torch.testing.assert_close(ic, g["images_context"], rtol=0, atol=_IMG_ATOL)
    exp_motions = torch.stack(
        [_v2_motions(item["motions"], cfg.motion_dim) for item in fx["per_item"]], dim=0
    )
    torch.testing.assert_close(mo, exp_motions, rtol=0, atol=0)
    assert mo.shape[-1] == cfg.motion_dim          # dataset slices to motion_dim; collate guard agrees
    for k in ("actions", "looks", "crosses"):
        torch.testing.assert_close(lab[k], g["labels"][k], rtol=0, atol=0)


def test_collate_guard_rejects_wide_motion() -> None:
    """B7 guard: an unsliced batch with >motion_dim channels fails loudly instead of silent-slicing."""
    item = {
        "images_tight": torch.zeros(2, 3, 4, 4),
        "images_context": torch.zeros(2, 3, 4, 4),
        "motions": torch.zeros(2, 9),              # full stored width, not sliced
        "actions": torch.tensor(0), "looks": torch.tensor(0), "crosses": torch.tensor(0),
    }
    with pytest.raises(ValueError, match="motion_dim"):
        collate_sequences([item], max_seq_len=20, motion_dim=8)


def test_worker_safety_matches_single_process(tmp_path) -> None:
    """num_workers>0: per-process env (pid-keyed) + picklable dataset/collate reproduce the batch."""
    fx = _load_fixture()
    chunk, cfg = _rebuild_lmdb(fx, tmp_path)
    ds = LMDBChunkDataset.from_config(chunk, cfg)

    loader = DataLoader(
        ds, batch_size=len(ds), shuffle=False, num_workers=2, collate_fn=build_collate(cfg)
    )
    it, ic, mo, lab = next(iter(loader))
    g = fx["batch"]
    torch.testing.assert_close(it, g["images_tight"], rtol=0, atol=_IMG_ATOL)
    torch.testing.assert_close(ic, g["images_context"], rtol=0, atol=_IMG_ATOL)
    exp_motions = torch.stack(
        [_v2_motions(item["motions"], cfg.motion_dim) for item in fx["per_item"]], dim=0
    )
    torch.testing.assert_close(mo, exp_motions, rtol=0, atol=0)
    for k in ("actions", "looks", "crosses"):
        torch.testing.assert_close(lab[k], g["labels"][k], rtol=0, atol=0)

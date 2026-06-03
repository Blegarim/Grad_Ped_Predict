"""Prompt 1.5 — runtime LMDB dataset + collate parity (vs OLD ``lmdb_dataset`` + ``collate_fn``).

Rebuilds the exact LMDB the capture used (same frames -> same deterministic 1.2 writer -> byte-identical
chunk), reads it with the new :class:`LMDBChunkDataset` + :func:`build_collate`, and diffs against the
golden fixture captured from the OLD code (``tests/_capture/capture_lmdb_dataset_golden.py``).

Covers: per-item dict parity, batched-tuple parity, ``seq_ids`` order, the B7 collate guard, and
worker-safety under ``num_workers>0`` (the per-process env + picklable dataset/collate).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

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


def _rebuild_lmdb(fx, tmp_path: Path) -> tuple[Path, DataCfg]:
    """Re-materialize the captured frames -> records -> identical LMDB via the new writer."""
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
        records.append({"images": paths, "bboxes": bboxes, **labels})
    chunk_paths = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    assert len(chunk_paths) == 1
    return chunk_paths[0], cfg


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
        torch.testing.assert_close(item["motions"], golden["motions"], rtol=0, atol=0)
        for k in ("actions", "looks", "crosses"):
            torch.testing.assert_close(item[k], golden[k], rtol=0, atol=0)


def test_collate_batch_parity(tmp_path) -> None:
    fx = _load_fixture()
    chunk, cfg = _rebuild_lmdb(fx, tmp_path)
    ds = LMDBChunkDataset.from_config(chunk, cfg)
    collate = build_collate(cfg)

    it, ic, mo, lab = collate([ds[i] for i in range(len(ds))])
    g = fx["batch"]
    torch.testing.assert_close(it, g["images_tight"], rtol=0, atol=_IMG_ATOL)
    torch.testing.assert_close(ic, g["images_context"], rtol=0, atol=_IMG_ATOL)
    torch.testing.assert_close(mo, g["motions"], rtol=0, atol=0)
    assert mo.shape[-1] == cfg.motion_dim          # B7: writer emits exactly motion_dim (slice deleted)
    for k in ("actions", "looks", "crosses"):
        torch.testing.assert_close(lab[k], g["labels"][k], rtol=0, atol=0)


def test_collate_guard_rejects_wide_motion() -> None:
    """B7 guard: a stale LMDB with >motion_dim channels fails loudly instead of silent-slicing."""
    item = {
        "images_tight": torch.zeros(2, 3, 4, 4),
        "images_context": torch.zeros(2, 3, 4, 4),
        "motions": torch.zeros(2, 9),              # legacy 9-col motion
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
    torch.testing.assert_close(mo, g["motions"], rtol=0, atol=0)
    for k in ("actions", "looks", "crosses"):
        torch.testing.assert_close(lab[k], g["labels"][k], rtol=0, atol=0)

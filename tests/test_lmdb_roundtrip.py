"""LMDB write/read roundtrip (the v2 schema contract, exercised end-to-end).

Writes a tiny chunk with the real writer, reopens it with raw ``lmdb``, and asserts the key/value
contract consumed by ``lmdb_dataset``: per-frame tight/context JPEG blobs that decode to the right
uint8 shapes, and a ``_meta`` pickle holding exactly ``{motions[T,9], actions, looks, crosses,
track_id}`` (+ ``tte`` for M5 benchmark records; ``bboxes`` deliberately dropped vs legacy). Also
pins ``compute_map_size`` to the legacy formula.
"""

from __future__ import annotations

import dataclasses
import pickle

import lmdb
import numpy as np
import torch
from PIL import Image
from torchvision.io import decode_jpeg

from pedpredict.config import DataCfg
from pedpredict.data.lmdb_writer import compute_map_size, encode_jpeg_bytes, write_dataset_chunks

_SEQ_LEN = 4
_N = 3
# Tiny explicit map_size: Windows pre-allocates the file, so the 4 GiB floor would reserve 4 GiB/chunk.
_SMALL_MAP = 64 * 1024 * 1024


def _make_record(dirpath, idx: int, *, tte: int | None = None):
    """One v2 record: ``_SEQ_LEN`` deterministic PNG frames + bboxes + track_id/ego + labels."""
    paths = []
    for t in range(_SEQ_LEN):
        yy, xx = np.mgrid[0:200, 0:200]
        arr = np.stack([(xx + t) % 256, (yy + idx) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        p = dirpath / f"s{idx}_f{t}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    bboxes = [[10.0 + t, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(_SEQ_LEN)]
    rec = {
        "images": paths,
        "bboxes": bboxes,
        "track_id": f"ped_{idx}",
        "ego_speed": [float(10 * idx + t) for t in range(_SEQ_LEN)],
        "actions": 1,
        "looks": 0,
        "crosses": 1,
    }
    if tte is not None:
        rec["tte"] = tte
    return rec


def _decode(buf: bytes) -> torch.Tensor:
    return decode_jpeg(torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).copy()))


def test_lmdb_roundtrip(tmp_path) -> None:
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP)  # context_scale 3.0 -> 384px
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = [_make_record(frames_dir, i) for i in range(_N)]

    paths = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    assert len(paths) == 1                       # _N < chunk_size -> single chunk
    assert paths[0].name == "chunk_000000.lmdb"

    env = lmdb.open(str(paths[0]), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            for j in range(_N):
                for t in range(_SEQ_LEN):
                    tight = _decode(txn.get(f"{j}_{t}_tight".encode()))
                    context = _decode(txn.get(f"{j}_{t}_context".encode()))
                    assert tight.shape == (3, 128, 128) and tight.dtype == torch.uint8
                    assert context.shape == (3, 384, 384) and context.dtype == torch.uint8

                meta = pickle.loads(txn.get(f"{j}_meta".encode()))
                assert set(meta) == {"motions", "actions", "looks", "crosses", "track_id"}  # no 'bboxes'
                assert meta["motions"].shape == (_SEQ_LEN, 9)  # MOTION_STORE_DIM (8 bbox + ego)
                assert meta["motions"].dtype == torch.float32
                # frame-0 deltas are true zeros (A4); ego channel carries the record's speeds (M9)
                assert meta["motions"][0, 2:4].tolist() == [0.0, 0.0]
                assert meta["motions"][0, 6:8].tolist() == [0.0, 0.0]
                assert meta["motions"][:, 8].tolist() == [float(10 * j + t) for t in range(_SEQ_LEN)]
                assert meta["track_id"] == f"ped_{j}"
                for key in ("actions", "looks", "crosses"):
                    assert meta[key].dtype == torch.long and meta[key].ndim == 0

            # exactly N*(SEQ_LEN tight + SEQ_LEN context + 1 meta) entries, nothing extra
            assert txn.stat()["entries"] == _N * (2 * _SEQ_LEN + 1)
    finally:
        env.close()


def test_benchmark_record_roundtrips_tte(tmp_path) -> None:
    """M5 benchmark records carry ``tte`` through the writer into the meta pickle."""
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = [_make_record(frames_dir, 0, tte=45)]
    paths = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    env = lmdb.open(str(paths[0]), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            meta = pickle.loads(txn.get(b"0_meta"))
            assert meta["tte"] == 45
            assert set(meta) == {"motions", "actions", "looks", "crosses", "track_id", "tte"}
    finally:
        env.close()


def test_write_dataset_chunks_splits_on_chunk_size(tmp_path) -> None:
    cfg = dataclasses.replace(DataCfg(), chunk_size=2, lmdb_map_size_bytes=_SMALL_MAP)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = [_make_record(frames_dir, i) for i in range(_N)]  # 3 records, chunk_size 2 -> 2 chunks
    paths = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    assert [p.name for p in paths] == ["chunk_000000.lmdb", "chunk_000002.lmdb"]

    # second chunk holds the single remainder sample under per-chunk index j=0
    env = lmdb.open(str(paths[1]), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            assert txn.get(b"0_meta") is not None
            assert txn.get(b"1_meta") is None
            assert txn.stat()["entries"] == 2 * _SEQ_LEN + 1
    finally:
        env.close()


def test_encode_jpeg_bytes_roundtrip() -> None:
    img = torch.rand(3, 32, 24)  # [0,1] CHW
    decoded = _decode(encode_jpeg_bytes(img, quality=90))
    assert decoded.shape == (3, 32, 24) and decoded.dtype == torch.uint8


def test_compute_map_size_matches_legacy_formula() -> None:
    # The heuristic path (lmdb_map_size_bytes=None) still pins the OLD preprocess formula exactly.
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=None)
    for n in (100, 100_000):  # floor branch, then estimate branch
        expected = max(int(n * 2 * (512 * 512 * 3) * 0.25 * 5 * 1.5), 4 * 1024**3)
        assert compute_map_size(n, cfg) == expected


def test_compute_map_size_default_is_explicit_4gib() -> None:
    # C3: the default is an explicit 4 GiB (Windows pre-allocates the file at map_size).
    assert compute_map_size(999_999, DataCfg()) == 4 * 1024**3


def test_compute_map_size_override() -> None:
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=123_456)
    assert compute_map_size(999_999, cfg) == 123_456

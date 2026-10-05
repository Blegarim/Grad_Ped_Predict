"""S1 onset fields survive the whole offline->runtime path.

Stage A of the onset-timing method (docs/METHODOLOGY.md prong 2). The three fields
(``onset_offset`` / ``future_observed`` / ``track_crosses``) were computed by ``pie_sequences`` from the
start but dropped by ``pack_meta``, so the trainer never saw them. These tests pin the repaired path:

* the writer packs them and the two readers hand them back (``read_raw_sample`` for the offline
  augment round-trip, ``LMDBChunkDataset`` for the runtime path, which tensorises them);
* the collate lifts them into ``labels`` so they reach the loss through the existing Trainer plumbing;
* pre-S1 records still write, read and collate cleanly — the keys are additive, not required.
"""

from __future__ import annotations

import dataclasses
import pickle

import lmdb
import numpy as np
import pytest
import torch
from PIL import Image

from pedpredict.config import DataCfg
from pedpredict.data.collate import collate_sequences
from pedpredict.data.lmdb_dataset import LMDBChunkDataset, read_raw_sample
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.data.pie_sequences import ONSET_FIELDS

_SEQ_LEN = 4
# Windows pre-allocates map_size, so the 4 GiB floor would reserve 4 GiB per chunk on a near-full disk.
# These chunks hold 2 samples of 4 tiny frames — a few hundred KB.
_SMALL_MAP = 4 * 1024 * 1024


def _cfg() -> DataCfg:
    """Tiny, single-process writer config. ``preprocess_num_workers=0`` matters: the default 8 spawns
    8 Windows worker processes *per chunk* (~10 s each) to encode a handful of 120x120 PNGs."""
    return dataclasses.replace(
        DataCfg(), lmdb_map_size_bytes=_SMALL_MAP, chunk_size=2, preprocess_num_workers=0
    )


def _read_items(chunk_path, cfg, count):
    """Read the first ``count`` items, then CLOSE the dataset's mmap (Windows keeps mapped files locked)."""
    ds = LMDBChunkDataset.from_config(chunk_path, cfg)
    try:
        return [ds[i] for i in range(count)]
    finally:
        ds.close()


def _make_record(dirpath, idx: int, *, onset: dict[str, int] | None = None) -> dict:
    """One window: ``_SEQ_LEN`` deterministic PNG frames + bboxes + labels, optionally S1-annotated."""
    paths = []
    for t in range(_SEQ_LEN):
        yy, xx = np.mgrid[0:120, 0:120]
        arr = np.stack([(xx + t) % 256, (yy + idx) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        p = dirpath / f"s{idx}_f{t}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    rec = {
        "images": paths,
        "bboxes": [[10.0 + t, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(_SEQ_LEN)],
        "track_id": f"ped_{idx}",
        "ego_speed": [float(10 * idx + t) for t in range(_SEQ_LEN)],
        "actions": 1,
        "looks": 0,
        "crosses": idx % 2,
    }
    if onset is not None:
        rec.update(onset)
    return rec


#: Four windows covering the four supervision cases the hazard loss distinguishes. Only the *values*
#: matter here (these tests pin that the ints survive the round trip); the semantics are exercised in
#: tests/test_onset_target.py.
_ONSETS = [
    {"onset_offset": 12, "future_observed": 80, "track_crosses": 1},   # event seen inside the horizon
    {"onset_offset": 45, "future_observed": 90, "track_crosses": 1},   # hard-temporal: crosses later
    {"onset_offset": -1, "future_observed": 9, "track_crosses": 1},    # already crossed (before the window)
    {"onset_offset": -1, "future_observed": 90, "track_crosses": 0},   # genuine negative
]


def _build(tmp_path, records, name="chunks"):
    out = tmp_path / name
    return write_dataset_chunks(records, out, _cfg()), out


@pytest.fixture
def annotated(tmp_path):
    """Four S1-annotated records written to a 2-chunk LMDB dir."""
    frames = tmp_path / "frames"
    frames.mkdir()
    records = [_make_record(frames, i, onset=_ONSETS[i]) for i in range(4)]
    chunks, out_dir = _build(tmp_path, records)
    return records, chunks, out_dir


# --------------------------------------------------------------------------- write / read


def test_writer_packs_onset_keys(annotated) -> None:
    """``pack_meta`` carries all three fields as plain ints, verbatim from the record."""
    records, chunks, _ = annotated
    env = lmdb.open(str(chunks[0]), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            meta = pickle.loads(txn.get(b"0_meta"))
    finally:
        env.close()
    for field in ONSET_FIELDS:
        assert meta[field] == records[0][field]
        assert isinstance(meta[field], int)


def test_read_raw_sample_roundtrips_onset(annotated) -> None:
    """The offline-augment reader hands the fields back unchanged (so re-augment preserves them)."""
    records, chunks, _ = annotated
    env = lmdb.open(str(chunks[0]), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            sample = read_raw_sample(txn, "1")
    finally:
        env.close()
    for field in ONSET_FIELDS:
        assert getattr(sample, field) == records[1][field]


def test_runtime_dataset_emits_onset_tensors(annotated) -> None:
    """``LMDBChunkDataset`` emits 0-dim long tensors, ready for ``torch.stack`` in the collate."""
    records, _, out_dir = annotated
    item = _read_items(out_dir / "chunk_000000.lmdb", _cfg(), 1)[0]
    for field in ONSET_FIELDS:
        assert isinstance(item[field], torch.Tensor)
        assert item[field].dtype == torch.long
        assert item[field].ndim == 0
        assert int(item[field]) == records[0][field]


def test_collate_lifts_onset_into_labels(annotated) -> None:
    """The fields ride in ``labels``, batched, alongside the three CE tasks."""
    cfg = _cfg()
    _, _, out_dir = annotated
    batch = _read_items(out_dir / "chunk_000000.lmdb", cfg, 2)
    *_, labels = collate_sequences(batch, max_seq_len=cfg.max_seq_len, motion_dim=cfg.motion_dim)
    assert set(ONSET_FIELDS) <= set(labels)
    assert labels["onset_offset"].tolist() == [_ONSETS[0]["onset_offset"], _ONSETS[1]["onset_offset"]]
    assert labels["onset_offset"].shape == (2,)
    assert {"actions", "looks", "crosses"} <= set(labels)  # the CE tasks are untouched


def test_collate_rejects_mixed_vintage_batch(annotated) -> None:
    """A batch where only some samples carry the keys is a build error, not a silent drop."""
    cfg = _cfg()
    _, _, out_dir = annotated
    first, second = _read_items(out_dir / "chunk_000000.lmdb", cfg, 2)
    stale = {k: v for k, v in second.items() if k not in ONSET_FIELDS}
    with pytest.raises(ValueError, match="mixes S1-annotated and pre-S1"):
        collate_sequences([first, stale], max_seq_len=cfg.max_seq_len, motion_dim=cfg.motion_dim)


# --------------------------------------------------------------------------- back-compat


def test_pre_s1_records_still_write_and_read(tmp_path) -> None:
    """Records without the fields produce chunks without the keys — additive, never required."""
    cfg = _cfg()
    frames = tmp_path / "frames"
    frames.mkdir()
    records = [_make_record(frames, i) for i in range(2)]  # no onset kwargs
    _, out_dir = _build(tmp_path, records)
    item, second = _read_items(out_dir / "chunk_000000.lmdb", cfg, 2)
    assert not (set(ONSET_FIELDS) & set(item))
    *_, labels = collate_sequences([item, second], max_seq_len=cfg.max_seq_len, motion_dim=cfg.motion_dim)
    assert not (set(ONSET_FIELDS) & set(labels))
    assert set(labels) == {"actions", "looks", "crosses"}

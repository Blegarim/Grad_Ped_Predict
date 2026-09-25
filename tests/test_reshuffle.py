"""Shuffle-rewrite of the training chunks: content is preserved exactly, block structure is not."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import lmdb
import pytest

from pedpredict.config.schema import DataCfg
from pedpredict.data.reshuffle import (
    chunk_positive_rates,
    enumerate_dir,
    enumerate_dirs,
    plan_shuffle,
    write_shuffled,
)

# Small map_size matters on Windows, where LMDB PRE-ALLOCATES the file at map_size (C3).
_CFG = dataclasses.replace(DataCfg(), chunk_size=4, lmdb_map_size_bytes=8 * 1024 * 1024)
_FRAMES = 2


def _write_chunk(path: Path, samples: list[tuple[str, int]]) -> None:
    """One fixture chunk: ``{j}_meta`` + ``{j}_{t}_{tight,context}`` with recognisable blob bytes."""
    env = lmdb.open(str(path), map_size=_CFG.lmdb_map_size_bytes)
    with env.begin(write=True) as txn:
        for j, (tag, crosses) in enumerate(samples):
            meta = {"crosses": crosses, "track_id": tag, "actions": 0, "looks": 0}
            txn.put(f"{j}_meta".encode(), pickle.dumps(meta))
            for t in range(_FRAMES):
                txn.put(f"{j}_{t}_tight".encode(), f"tight:{tag}:{t}".encode())
                txn.put(f"{j}_{t}_context".encode(), f"ctx:{tag}:{t}".encode())
    env.close()


def _build_sources(root: Path) -> list[Path]:
    """Mimics the real layout: an all-negative base dir and an all-positive augmented dir."""
    base, aug = root / "preprocessed_train", root / "preprocessed_train_aug"
    base.mkdir()
    aug.mkdir()
    _write_chunk(base / "chunk_000000.lmdb", [(f"b{i}", 0) for i in range(4)])
    _write_chunk(base / "chunk_000004.lmdb", [(f"b{i}", 0) for i in range(4, 8)])
    _write_chunk(aug / "chunk_000000.lmdb", [(f"a{i}", 1) for i in range(4)])
    return [base, aug]


def _dump(directory: Path) -> dict[str, dict[str, bytes]]:
    """``track_id -> {key suffix: blob}`` for every sample, so content can be compared order-free."""
    out: dict[str, dict[str, bytes]] = {}
    for chunk in sorted(p for p in directory.iterdir() if p.name.startswith("chunk_")):
        env = lmdb.open(str(chunk), readonly=True, lock=False)
        with env.begin() as txn:
            metas = {k.decode().split("_")[0]: pickle.loads(v)
                     for k, v in txn.cursor() if k.decode().endswith("_meta")}
            for seq_id, meta in metas.items():
                blobs = {k.decode()[len(seq_id) + 1 :]: v
                         for k, v in txn.cursor() if k.decode().startswith(f"{seq_id}_")}
                out[meta["track_id"]] = blobs
        env.close()
    return out


def test_enumerate_reads_every_sample_and_its_label(tmp_path: Path) -> None:
    refs = enumerate_dirs(_build_sources(tmp_path))
    assert len(refs) == 12
    assert sum(r.crosses for r in refs) == 4


def test_plan_shuffle_is_seeded_and_permutes(tmp_path: Path) -> None:
    refs = enumerate_dirs(_build_sources(tmp_path))
    a, b = plan_shuffle(refs, 42), plan_shuffle(refs, 42)
    assert a == b, "same seed must give the same permutation"
    assert plan_shuffle(refs, 43) != a
    assert sorted(a, key=repr) == sorted(refs, key=repr), "a permutation adds and drops nothing"


def test_shuffling_flattens_the_block_structure(tmp_path: Path) -> None:
    """The source ordering swings 0.0 -> 1.0 between chunks; the shuffled one must not."""
    refs = enumerate_dirs(_build_sources(tmp_path))
    before = chunk_positive_rates(refs, _CFG.chunk_size)
    assert max(before) - min(before) == pytest.approx(1.0), "fixture must reproduce the real pathology"
    after = chunk_positive_rates(plan_shuffle(refs, 42), _CFG.chunk_size)
    assert max(after) - min(after) < max(before) - min(before)


def test_rewrite_preserves_every_sample_byte_for_byte(tmp_path: Path) -> None:
    sources = _build_sources(tmp_path)
    refs = plan_shuffle(enumerate_dirs(sources), 42)
    out = tmp_path / "preprocessed_train_shuffled"
    written = write_shuffled(refs, out, _CFG)

    assert len(written) == 3, "12 samples at chunk_size 4"
    original: dict[str, dict[str, bytes]] = {}
    for src in sources:
        original.update(_dump(src))
    assert _dump(out) == original, "blobs and metas must survive the rewrite unchanged"


def test_rewritten_chunks_use_contiguous_in_chunk_ids(tmp_path: Path) -> None:
    """Downstream readers derive the sample index from the key prefix, so ids must restart per chunk."""
    sources = _build_sources(tmp_path)
    out = tmp_path / "preprocessed_train_shuffled"
    write_shuffled(plan_shuffle(enumerate_dirs(sources), 42), out, _CFG)
    for chunk in sorted(p for p in out.iterdir() if p.name.startswith("chunk_")):
        ids = sorted(int(r.seq_id) for r in enumerate_dir(chunk.parent) if r.chunk == chunk)
        assert ids == list(range(len(ids)))

"""Chunked LMDB serialization of processed sequences.

Ports OLD ``scripts/preprocess_data_lmdb.py::save_dataset_in_chunks_lmdb`` — but ONLY the
serialization/chunking concern. All crop/motion math lives in :mod:`pedpredict.data.transforms`
(band-aid B5). The writer JPEG-encodes the (already un-normalized ``[0, 1]``) crops and pickles the
per-sample metadata into per-chunk LMDB environments.

LMDB key/value contract v2 (consumed by ``lmdb_dataset``) — keys reset **per chunk**, where
``j`` is the sample index within the chunk and ``t`` the frame index:

====================  ==========================  =====================================================
key (utf-8)           value                       decodes to
====================  ==========================  =====================================================
``f"{j}_{t}_tight"``   JPEG bytes                  uint8 ``[3, img_height, img_width]``
``f"{j}_{t}_context"`` JPEG bytes                  uint8 ``[3, H*scale, W*scale]``
``f"{j}_meta"``        ``pickle`` dict             ``{motions[T,9], actions, looks, crosses,
                                                   track_id, (tte)}``
====================  ==========================  =====================================================

v2 meta (hole audit): ``motions`` is the full ``MOTION_STORE_DIM`` (9) vector — 8 bbox channels with
true-zero frame-0 deltas + ego-speed (A4/M9); ``track_id`` is the PIE pedestrian id (M6); ``tte`` is
present only in M5 benchmark-protocol chunks. ``bboxes`` stays dropped (motions encode the geometry).
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

import lmdb
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import encode_jpeg
from tqdm.auto import tqdm

from pedpredict.config.schema import DataCfg
from pedpredict.data.pie_sequences import SequenceRecord
from pedpredict.data.transforms import CropSequenceDataset, ProcessedSample

__all__ = [
    "compute_map_size",
    "encode_jpeg_bytes",
    "pack_meta",
    "write_sample",
    "write_dataset_to_lmdb",
    "write_chunk",
    "write_dataset_chunks_from",
    "write_dataset_chunks",
]

# map_size heuristic factors (OLD preprocess_data_lmdb.py lines 52-54), each documented:
_CROPS_PER_SAMPLE = 2          # tight + context
_BYTES_PER_CROP = 512 * 512 * 3  # assumed worst-case crop pixels*channels (deliberate over-estimate)
_JPEG_RATIO = 0.25             # assumed JPEG size / raw size
_SAMPLE_FUDGE = 5              # empirical per-sample safety multiplier (NOT seq_len)


def compute_map_size(num_samples: int, cfg: DataCfg) -> int:
    """LMDB ``map_size`` in bytes for a chunk of ``num_samples`` sequences.

    Reproduces the legacy heuristic with default ``cfg`` values
    (``max(int(N * 2 * 512*512*3 * 0.25 * 5 * 1.5), 4 GiB)``). ``cfg.lmdb_map_size_bytes`` forces an
    exact value. NOTE: LMDB sparse-allocates on Linux but **pre-allocates the file on Windows**, so
    an over-estimate reserves real disk there — tune via the ``lmdb_map_size_*`` config fields.
    """
    if cfg.lmdb_map_size_bytes is not None:
        return int(cfg.lmdb_map_size_bytes)
    est = num_samples * _CROPS_PER_SAMPLE * _BYTES_PER_CROP * _JPEG_RATIO * _SAMPLE_FUDGE
    floor = int(cfg.lmdb_map_size_floor_gib * 1024**3)
    return max(int(est * cfg.lmdb_map_size_safety), floor)


def encode_jpeg_bytes(img01: Tensor, quality: int) -> bytes:
    """``[0, 1]`` CHW float -> uint8 -> torchvision ``encode_jpeg`` -> raw bytes (OLD lines 70-72)."""
    img_uint8 = (img01 * 255.0).clamp(0, 255).to(torch.uint8).contiguous()
    return encode_jpeg(img_uint8, quality=quality).numpy().tobytes()


def pack_meta(sample: ProcessedSample) -> bytes:
    """Pickle the per-sample v2 metadata (see the module-docstring contract table)."""
    meta = {
        "motions": sample.motions,
        "actions": sample.actions,
        "looks": sample.looks,
        "crosses": sample.crosses,
        "track_id": sample.track_id,
    }
    if sample.tte is not None:  # M5 benchmark-protocol chunks only
        meta["tte"] = sample.tte
    return pickle.dumps(meta)


def write_sample(txn: lmdb.Transaction, j: int, sample: ProcessedSample, cfg: DataCfg) -> None:
    """Put all tight/context JPEG blobs + the meta pickle for one sample under per-chunk index ``j``."""
    for t in range(sample.images_tight.shape[0]):
        txn.put(f"{j}_{t}_tight".encode(), encode_jpeg_bytes(sample.images_tight[t], cfg.jpeg_quality))
    for t in range(sample.images_context.shape[0]):
        txn.put(f"{j}_{t}_context".encode(), encode_jpeg_bytes(sample.images_context[t], cfg.jpeg_quality))
    txn.put(f"{j}_meta".encode(), pack_meta(sample))


def _passthrough_collate(batch: list[ProcessedSample]) -> ProcessedSample:
    """Undo the ``batch_size=1`` wrapping (replaces the legacy ``unbatch`` hack). Module-level so it
    pickles to DataLoader workers on Windows ``spawn``."""
    return batch[0]


def write_dataset_to_lmdb(
    dataset: Dataset,
    lmdb_path: str | Path,
    cfg: DataCfg,
    *,
    num_workers: int,
    prefetch_factor: int,
) -> Path:
    """Encode every ``ProcessedSample`` of ``dataset`` (deterministic order) into one chunk LMDB.

    The single low-level write path: a ``DataLoader(batch_size=1)`` drives crop/encode in workers, the
    main process puts each sample's blobs. Source-agnostic — ``dataset`` may be a plain
    :class:`CropSequenceDataset` (1.2) or an augmenting dataset (1.4); the bytes written are identical.
    """
    loader_kwargs: dict = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "collate_fn": _passthrough_collate,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    path = Path(lmdb_path)
    env = lmdb.open(str(path), map_size=compute_map_size(len(dataset), cfg))  # type: ignore[arg-type]
    try:
        with env.begin(write=True) as txn:
            progress = tqdm(loader, total=len(dataset), desc=path.name, unit="seq")  # type: ignore[arg-type]
            for j, sample in enumerate(progress):
                write_sample(txn, j, sample, cfg)
        env.sync()
    finally:
        env.close()
    return path


def write_chunk(
    records: list[SequenceRecord],
    lmdb_path: str | Path,
    cfg: DataCfg,
    *,
    num_workers: int,
    prefetch_factor: int,
) -> Path:
    """Crop+encode ``records`` (deterministic order) into a single chunk LMDB at ``lmdb_path``."""
    return write_dataset_to_lmdb(
        CropSequenceDataset(records, cfg), lmdb_path, cfg, num_workers=num_workers, prefetch_factor=prefetch_factor
    )


def write_dataset_chunks_from(
    dataset: Dataset,
    out_dir: str | Path,
    cfg: DataCfg,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> list[Path]:
    """Split any ``Dataset[ProcessedSample]`` into ``cfg.chunk_size`` chunks via ``Subset``.

    The generalization of :func:`write_dataset_chunks` to a pre-built dataset (used by 1.4 augmentation,
    whose sample count != record count). Per-chunk keys reset to ``j=0`` because each chunk gets its own
    ``Subset`` + LMDB. Worker/prefetch counts default to ``cfg.preprocess_*``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if end_idx is None:
        end_idx = len(dataset)  # type: ignore[arg-type]
    nw = cfg.preprocess_num_workers if num_workers is None else num_workers
    pf = cfg.preprocess_prefetch_factor if prefetch_factor is None else prefetch_factor

    paths: list[Path] = []
    for i in range(start_idx, end_idx, cfg.chunk_size):
        sub = Subset(dataset, list(range(i, min(i + cfg.chunk_size, end_idx))))
        paths.append(write_dataset_to_lmdb(sub, out / f"chunk_{i:06d}.lmdb", cfg, num_workers=nw, prefetch_factor=pf))
        gc.collect()
    return paths


def write_dataset_chunks(
    records: list[SequenceRecord],
    out_dir: str | Path,
    cfg: DataCfg,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> list[Path]:
    """Split ``records`` into ``cfg.chunk_size`` chunks -> one ``chunk_{i:06d}.lmdb`` each.

    Worker/prefetch counts default to ``cfg.preprocess_*`` (offline parallelism only — output is
    identical to a serial write). Returns the written chunk paths in order.
    """
    end = len(records) if end_idx is None else end_idx
    return write_dataset_chunks_from(
        CropSequenceDataset(records, cfg),
        out_dir,
        cfg,
        start_idx=start_idx,
        end_idx=end,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

"""Runtime (on-the-fly) train-time augmentation — RuntimeAugmentor + LMDBChunkDataset hook + wiring.

Covers: the augmentor is deterministic per seed and does NOT perturb the global torch RNG (num_workers=0
safety); the dataset applies aug BEFORE ImageNet norm (a pure flip equals flipping the normalized baseline,
proving the pre-norm→aug→norm ordering); and ChunkPrefetcher enables aug on the train loader only, gated by
augment.runtime.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import torch
from PIL import Image

from pedpredict.config import AugmentCfg, DataCfg, RootCfg
from pedpredict.data.augment import _FLIP_NEGATE_IDX, _FLIP_REFLECT_IDX, RuntimeAugmentor
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.data.transforms import ProcessedSample
from pedpredict.training.chunk_loader import ChunkPrefetcher

_SMALL_MAP = 64 * 1024 * 1024
_ALL_ON = AugmentCfg(p_flip=1.0, p_color=1.0, p_noise=1.0, p_erase=1.0)
_FLIP_ONLY = AugmentCfg(p_flip=1.0, p_color=0.0, p_noise=0.0, p_erase=0.0)


def _proc_sample(t: int = 4) -> ProcessedSample:
    torch.manual_seed(0)
    return ProcessedSample(
        images_tight=torch.rand(t, 3, 8, 8),
        images_context=torch.rand(t, 3, 12, 12),
        motions=(torch.rand(t, 8) * 100.0),
        actions=torch.tensor(1), looks=torch.tensor(0), crosses=torch.tensor(1),
    )


# --------------------------------------------------------------------------- RuntimeAugmentor unit


def test_augmentor_deterministic_per_seed_and_varies() -> None:
    aug = RuntimeAugmentor(_ALL_ON, DataCfg().source_width)
    a = aug(_proc_sample(), random.Random(42))
    b = aug(_proc_sample(), random.Random(42))
    torch.testing.assert_close(a.images_tight, b.images_tight, rtol=0, atol=0)  # same seed -> identical
    c = aug(_proc_sample(), random.Random(7))
    assert not torch.allclose(a.images_tight, c.images_tight)                   # different seed -> differs


def test_augmentor_does_not_perturb_global_torch_rng() -> None:
    """color/noise seed torch locally and restore state, so aug never consumes the training RNG stream."""
    aug = RuntimeAugmentor(_ALL_ON, DataCfg().source_width)
    s = _proc_sample()                     # build the sample BEFORE measuring (its own RNG use is not aug's)
    torch.manual_seed(0)
    before = torch.rand(4)
    torch.manual_seed(0)
    aug(s, random.Random(1))
    after = torch.rand(4)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


# --------------------------------------------------------------------------- dataset integration


def _png_record(dirpath, n_frames: int = 4) -> dict:
    paths = []
    for t in range(n_frames):
        yy, xx = np.mgrid[0:120, 0:120]
        arr = np.stack([(xx + t) % 256, (yy + t) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        p = dirpath / f"f{t}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    bboxes = [[10.0 + t, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(n_frames)]
    return {
        "images": paths, "bboxes": bboxes, "track_id": "ped_rt",
        "ego_speed": [float(t) for t in range(n_frames)],
        "actions": 1, "looks": 1, "crosses": 1,
    }


def _tiny_chunk(tmp_path, n: int = 2) -> tuple[str, DataCfg]:
    frames = tmp_path / "frames"
    frames.mkdir()
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP)
    records = [_png_record(frames) for _ in range(n)]
    chunks = write_dataset_chunks(records, tmp_path / "lmdb", cfg, num_workers=0)
    return str(chunks[0]), cfg


def test_dataset_flip_equals_flip_of_normalized_baseline(tmp_path) -> None:
    """Pure flip (color/noise/erase off) must equal flipping the non-aug (normalized) output — the
    pre-norm→aug→norm ordering is correct iff this holds (flip commutes with per-channel norm)."""
    chunk, cfg = _tiny_chunk(tmp_path)
    plain = LMDBChunkDataset.from_config(chunk, cfg)
    aug = LMDBChunkDataset.from_config(
        chunk, cfg, augmentor=RuntimeAugmentor(_FLIP_ONLY, cfg.source_width), aug_seed=123
    )
    for i in range(len(plain)):
        p, a = plain[i], aug[i]
        torch.testing.assert_close(a["images_tight"], torch.flip(p["images_tight"], dims=[3]), rtol=0, atol=0)
        torch.testing.assert_close(a["images_context"], torch.flip(p["images_context"], dims=[3]), rtol=0, atol=0)
        torch.testing.assert_close(a["motions"][:, _FLIP_NEGATE_IDX], -p["motions"][:, _FLIP_NEGATE_IDX])
        torch.testing.assert_close(
            a["motions"][:, _FLIP_REFLECT_IDX], cfg.source_width - p["motions"][:, _FLIP_REFLECT_IDX]
        )


def test_dataset_aug_is_normalized_and_deterministic(tmp_path) -> None:
    chunk, cfg = _tiny_chunk(tmp_path)
    aug = LMDBChunkDataset.from_config(
        chunk, cfg, augmentor=RuntimeAugmentor(_ALL_ON, cfg.source_width), aug_seed=5
    )
    first = aug[0]
    assert first["images_tight"].min() < 0.0            # ImageNet norm applied after aug (has negatives)
    assert first["images_tight"].shape[0] == first["motions"].shape[0]
    torch.testing.assert_close(aug[0]["images_tight"], first["images_tight"], rtol=0, atol=0)  # per-idx det.


def test_default_read_path_unchanged(tmp_path) -> None:
    """No augmentor (the default / val path) leaves the dataset's aug fields inert."""
    chunk, cfg = _tiny_chunk(tmp_path)
    ds = LMDBChunkDataset.from_config(chunk, cfg)
    assert ds._augmentor is None and ds._normalize is None


# --------------------------------------------------------------------------- ChunkPrefetcher wiring


def _root_cfg(runtime: bool) -> RootCfg:
    cfg = RootCfg()
    return dataclasses.replace(
        cfg,
        augment=dataclasses.replace(cfg.augment, runtime=runtime),
        train=dataclasses.replace(cfg.train, use_weighted_sampler=False),
    )


def test_prefetcher_augments_train_only(tmp_path) -> None:
    chunk, _ = _tiny_chunk(tmp_path)
    pf = ChunkPrefetcher(_root_cfg(runtime=True), [chunk], [chunk], pin_memory=False)
    train = pf._build_train_loader(chunk, epoch=0)
    val = pf._build_val_loader(chunk)
    assert train.dataset._augmentor is not None
    assert val.dataset._augmentor is None


def test_prefetcher_runtime_off_no_aug(tmp_path) -> None:
    chunk, _ = _tiny_chunk(tmp_path)
    pf = ChunkPrefetcher(_root_cfg(runtime=False), [chunk], [chunk], pin_memory=False)
    assert pf._build_train_loader(chunk, epoch=0).dataset._augmentor is None

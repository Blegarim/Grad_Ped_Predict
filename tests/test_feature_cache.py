"""Recipe-v2 feature cache (docs/RECIPE_V2_PLAN.md Tasks 6-7).

The cache replaces the context-crop JPEG decode, so it is only correct if it is invisible: cached features
must equal what the eval-mode backbone computes from the same crops, the dataset must return the same
labels / motions / pose either way, the train-time flip must stay consistent between features and
motions + pose (the M9 silent-corruption rule), and any stale or foreign cache must fail loudly.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pedpredict.config import AugmentCfg, DataCfg, ModelCfg, PathsCfg, RootCfg, TrainCfg
from pedpredict.config.loader import ConfigError, validate_config
from pedpredict.data import feature_cache as fc
from pedpredict.data.augment import RuntimeAugmentor
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.models.ablations import PoseFullModel
from pedpredict.models.timm_backbone import TimmBackbone
from pedpredict.training.chunk_loader import ChunkPrefetcher

pytest.importorskip("timm")

_SEQ, _N = 4, 3
_CPU = torch.device("cpu")
_NO_AUG = {"p_flip": 0.0, "p_color": 0.0, "p_noise": 0.0, "p_erase": 0.0}


@pytest.fixture(scope="module")
def chunk(tmp_path_factory) -> Path:
    """One tiny LMDB chunk (3 windows x 4 frames) written by the real writer."""
    tmp = tmp_path_factory.mktemp("fc")
    frames = tmp / "frames"
    frames.mkdir()
    records = []
    for idx in range(_N):
        paths = []
        for t in range(_SEQ):
            yy, xx = np.mgrid[0:200, 0:200]
            arr = np.stack([(xx * (idx + 1) + t) % 256, (yy + 40 * idx) % 256, (xx + yy) % 256], -1).astype(np.uint8)
            Image.fromarray(arr).save(frames / f"s{idx}_f{t}.png")
            paths.append(str(frames / f"s{idx}_f{t}.png"))
        records.append({
            "images": paths, "bboxes": [[10.0 + t, 12.0, 70.0 + 2 * t, 110.0] for t in range(_SEQ)],
            "track_id": f"ped_{idx}", "ego_speed": [0.0] * _SEQ, "actions": 1, "looks": 0, "crosses": idx % 2,
        })
    data = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=64 * 1024 * 1024)
    (path,) = write_dataset_chunks(records, tmp / "preprocessed_train", data, num_workers=0)
    return Path(path)


def _cfg(tmp: Path, **augment) -> RootCfg:
    return dataclasses.replace(
        RootCfg(),
        paths=dataclasses.replace(PathsCfg(), feature_cache_dir=str(tmp / "cache")),
        augment=dataclasses.replace(AugmentCfg(), cache_color_variants=1, **augment),
        train=dataclasses.replace(TrainCfg(), use_amp=False, num_workers=0, batch_size=2, use_weighted_sampler=False),
    )


@pytest.fixture(scope="module")
def net() -> torch.nn.Module:
    torch.manual_seed(0)
    return TimmBackbone("tiny_vit_5m_224", pretrained=False).net.eval()


@pytest.fixture(scope="module")
def built(chunk, net, tmp_path_factory) -> tuple[RootCfg, Path]:
    cfg = _cfg(tmp_path_factory.mktemp("cache_root"))
    out, did_build = fc.build_chunk_cache(chunk, cfg, net, augmented=True, device=_CPU, batch_windows=2)
    assert did_build
    return cfg, out


def _image_context(chunk: Path, idx: int) -> torch.Tensor:
    ds = LMDBChunkDataset.from_config(chunk, DataCfg())
    try:
        return ds[idx]["images_context"]          # [T, 3, 224, 224], ImageNet-normalised
    finally:
        ds.close()


def test_layout_order_and_resume(built, chunk, net) -> None:
    cfg, out = built
    features = fc.open_chunk_features(dataclasses.replace(cfg, data=dataclasses.replace(
        cfg.data, visual_input="cached_features")), chunk)
    assert features.variants == ["clean", "flip", "color0", "color0_flip"]
    assert np.load(out / fc.FEATURES_FILE, mmap_mode="r").shape == (_N, 4, _SEQ, 320)
    ds = LMDBChunkDataset.from_config(chunk, DataCfg())
    assert features.meta["seq_ids"] == ds.seq_ids
    ds.close()
    assert fc.build_chunk_cache(chunk, cfg, net, augmented=True, device=_CPU)[1] is False   # up to date


def test_clean_and_flip_equal_the_backbone_on_the_image_path(built, chunk, net) -> None:
    _, out = built
    features = fc.ChunkFeatures(out)
    for idx in range(_N):
        context = _image_context(chunk, idx)
        with torch.no_grad():
            clean, flipped = net(context), net(context.flip(-1))
        assert torch.allclose(features.read(idx, "clean"), clean, atol=5e-3)
        assert torch.allclose(features.read(idx, "flip"), flipped, atol=5e-3)


def test_verify_chunk_cache_reports_parity(built, chunk, net) -> None:
    cfg, _ = built
    stats = fc.verify_chunk_cache(chunk, cfg, net, device=_CPU, windows=2)
    assert stats.max < 5e-3 and stats.p999 <= stats.max and stats.mean <= stats.p999
    assert stats.count > 0


def test_color_variants_are_real_and_deterministic(built, chunk, net, tmp_path) -> None:
    cfg, out = built
    features = fc.ChunkFeatures(out)
    assert not torch.allclose(features.read(0, "color0"), features.read(0, "clean"), atol=1e-2)
    assert not torch.allclose(features.read(0, "color0_flip"), features.read(0, "color0"), atol=1e-2)
    again = dataclasses.replace(cfg, paths=dataclasses.replace(cfg.paths, feature_cache_dir=str(tmp_path)))
    out2, _ = fc.build_chunk_cache(chunk, again, net, augmented=True, device=_CPU, batch_windows=3)
    assert np.array_equal(np.load(out / fc.FEATURES_FILE), np.load(out2 / fc.FEATURES_FILE))


def test_stale_or_foreign_caches_fail_loudly(built, chunk, tmp_path) -> None:
    cfg, out = built
    assert fc.open_chunk_features(cfg, chunk) is None                         # image path: no cache
    cached = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, visual_input="cached_features"))
    foreign = dataclasses.replace(cached, model=dataclasses.replace(ModelCfg(), vit_backbone="x"))
    with pytest.raises(ValueError, match="backbone"):
        fc.open_chunk_features(foreign, chunk)
    with pytest.raises(FileNotFoundError):
        fc.ChunkFeatures(tmp_path / "nothing")
    torch.manual_seed(1)
    other = TimmBackbone("tiny_vit_5m_224", pretrained=False).net
    with pytest.raises(ValueError, match="weights"):
        fc.verify_cache_weights(other, fc.ChunkFeatures(out))
    with pytest.raises(ValueError, match="window order"):
        fc.ChunkFeatures(out).check_seq_ids(["a", "b", "c"])


def test_dataset_cached_mode_matches_the_image_path(built, chunk) -> None:
    _, out = built
    image_ds = LMDBChunkDataset.from_config(chunk, DataCfg())
    cached_ds = LMDBChunkDataset.from_config(chunk, DataCfg(), features=fc.ChunkFeatures(out))
    try:
        for idx in range(_N):
            a, b = image_ds[idx], cached_ds[idx]
            assert torch.equal(a["motions"], b["motions"]) and a["crosses"] == b["crosses"]
            assert a["track_id"] == b["track_id"]
            assert b["images_tight"].shape == (_SEQ, 0)
            assert torch.equal(b["images_context"], fc.ChunkFeatures(out).read(idx, "clean"))
    finally:
        image_ds.close()
        cached_ds.close()


def test_train_time_flip_matches_the_image_augmentor(built, chunk) -> None:
    _, out = built
    aug_cfg = dataclasses.replace(AugmentCfg(), **{**_NO_AUG, "p_flip": 1.0})
    image_ds = LMDBChunkDataset.from_config(
        chunk, DataCfg(), augmentor=RuntimeAugmentor(aug_cfg, 1920), aug_seed=7
    )
    cached_ds = LMDBChunkDataset.from_config(
        chunk, DataCfg(), features=fc.ChunkFeatures(out),
        feature_augmentor=fc.CachedFeatureAugmentor(aug_cfg, 1920), aug_seed=7,
    )
    try:
        for idx in range(_N):
            a, b = image_ds[idx], cached_ds[idx]
            assert torch.allclose(a["motions"], b["motions"])                 # dx negated, cx reflected
            assert torch.equal(b["images_context"], fc.ChunkFeatures(out).read(idx, "flip"))
    finally:
        image_ds.close()
        cached_ds.close()


def test_augmentor_variant_names() -> None:
    always = dataclasses.replace(AugmentCfg(), **{**_NO_AUG, "p_flip": 1.0, "p_color": 1.0})
    aug = fc.CachedFeatureAugmentor(always, 1920)
    choice = aug.choose(random.Random(0), color_variants=2)
    assert choice.flip and choice.variant in {"color0_flip", "color1_flip"} and choice.noise_seed is None
    assert aug.choose(random.Random(0), color_variants=0).variant == "flip"   # no color variants cached


def test_pose_full_on_features_equals_pose_full_on_images() -> None:
    cfg = dataclasses.replace(ModelCfg(), motion_dim=58, motion_norm="none", vit_pretrained=False, vit_frozen_eval=True)
    torch.manual_seed(0)
    model = PoseFullModel.from_config(cfg, img_size=224).eval()
    images, motions = torch.randn(2, 3, 3, 224, 224), torch.randn(2, 3, 58)
    with torch.no_grad():
        feats = model.vit.net(images.flatten(0, 1)).view(2, 3, -1)
        a, b = model(images, motions), model(feats, motions)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-5), key


def test_prefetcher_serves_features_in_cached_mode(built, chunk) -> None:
    cfg, _ = built
    cached = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, visual_input="cached_features"),
        augment=dataclasses.replace(cfg.augment, runtime=True),
    )
    prefetcher = ChunkPrefetcher(cached, [str(chunk)], [str(chunk)], pin_memory=False)
    for loader in (prefetcher._build_train_loader(str(chunk), epoch=0), prefetcher._build_val_loader(str(chunk))):
        tight, context, motions, labels = next(iter(loader))
        assert tight.shape == (2, _SEQ, 0) and context.shape == (2, _SEQ, 320)
        loader.dataset.close()


def test_cached_mode_validation() -> None:
    base = dataclasses.replace(RootCfg(), data=dataclasses.replace(DataCfg(), visual_input="cached_features"))
    with pytest.raises(ConfigError, match="vit_frozen_eval"):
        validate_config(base)
    with pytest.raises(ConfigError, match="visual_input must be"):
        validate_config(dataclasses.replace(RootCfg(), data=dataclasses.replace(DataCfg(), visual_input="pixels")))

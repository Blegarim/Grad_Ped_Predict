"""Cached frozen-backbone features — recipe v2 (docs/RECIPE_V2_PLAN.md Tasks 6-7).

With ``model.vit_frozen_eval`` the timm backbone is a fixed function of the context crop, so its pooled
features can be computed once and read back instead of decoding 20 context JPEGs per window per epoch —
the cost that bounds training speed on the research PC (the GPU sat ~80% idle). ``frame_proj`` is NOT part
of the cache, so it can still train (``model.train_frame_proj``).

**Layout.** ``<paths.feature_cache_dir>/<LMDB dir name>/<chunk stem>/`` holds ``features.npy`` (float16
``[N, V, T, F]``, rows in the chunk's LMDB cursor order == ``LMDBChunkDataset.seq_ids``) and ``meta.json``
(written last — its presence marks a complete cache). ``V`` variants: ``clean`` everywhere; train dirs add
``flip`` and ``K = augment.cache_color_variants`` color-jitter variants, each also flipped.

**Augmentation in cached mode** (:class:`CachedFeatureAugmentor`) makes the same per-sample draws as
``RuntimeAugmentor``: flip picks a flipped variant and flips motions + pose with the shared rule
(:func:`pedpredict.data.augment.flip_motions_pose`); color picks one of the pre-built jitter variants
(jitter drawn once per window at build time, not per epoch); motion noise is applied at read time; frame
erase has no cached equivalent and is skipped.

**Integrity.** Readers check the chunk's seq-id order, the backbone name, the crop size and the ImageNet
norm; :func:`verify_cache_weights` checks a fingerprint of the backbone's weights *and BN buffers*, so a
cache built from different weights fails loudly instead of feeding silently wrong features.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

from pedpredict.config.schema import AugmentCfg, DataCfg, RootCfg
from pedpredict.data.augment import _isolated_torch_seed, flip_motions_pose
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.transforms import imagenet_normalize, resize_to_tensor
from pedpredict.paths import resolve_paths
from pedpredict.utils.amp import autocast_ctx

__all__ = [
    "CACHE_VERSION",
    "FEATURES_FILE",
    "META_FILE",
    "ChunkFeatures",
    "ParityStats",
    "FeatureChoice",
    "CachedFeatureAugmentor",
    "variant_names",
    "chunk_cache_dir",
    "open_chunk_features",
    "weights_fingerprint",
    "verify_cache_weights",
    "verify_model_cache",
    "verify_chunk_cache",
    "build_chunk_cache",
]

CACHE_VERSION = 1
FEATURES_FILE = "features.npy"
META_FILE = "meta.json"


def variant_names(color_variants: int, *, augmented: bool) -> list[str]:
    """``["clean"]`` for eval dirs; train dirs add ``flip`` and ``color{k}`` / ``color{k}_flip``."""
    if not augmented:
        return ["clean"]
    names = ["clean", "flip"]
    for k in range(color_variants):
        names += [f"color{k}", f"color{k}_flip"]
    return names


def chunk_cache_dir(cache_root: str | Path, chunk_path: str | Path) -> Path:
    """``<cache_root>/<LMDB dir name>/<chunk stem>`` — LMDB dir names are unique across splits/protocols."""
    chunk = Path(chunk_path)
    stem = chunk.name[: -len(".lmdb")] if chunk.name.endswith(".lmdb") else chunk.name
    return Path(cache_root) / chunk.parent.name / stem


def weights_fingerprint(net: nn.Module) -> str:
    """Hash of every floating tensor in ``net.state_dict()`` — weights AND BN running statistics, the two
    things an eval-mode backbone's output depends on."""
    digest = hashlib.sha1()
    for name, tensor in sorted(net.state_dict().items()):
        if tensor.is_floating_point():
            digest.update(name.encode())
            digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- read side


class ChunkFeatures:
    """Read side for one chunk's cache. Picklable: the memory map opens lazily in each process."""

    def __init__(self, cache_dir: str | Path, *, expect: dict[str, object] | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no complete feature cache at {self.cache_dir} (missing {META_FILE}) — build it with "
                f"scripts/build_feature_cache.py"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in (expect or {}).items():
            if self.meta.get(key) != value:
                raise ValueError(
                    f"feature cache {self.cache_dir}: {key}={self.meta.get(key)!r} but the config expects "
                    f"{value!r} — rebuild the cache (scripts/build_feature_cache.py --overwrite)"
                )
        self.variants: list[str] = list(self.meta["variants"])
        self._index = {name: i for i, name in enumerate(self.variants)}
        self._array: np.ndarray | None = None
        self._pid: int | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array"], state["_pid"] = None, None
        return state

    @property
    def color_variants(self) -> int:
        return sum(1 for v in self.variants if v.startswith("color") and not v.endswith("_flip"))

    def check_seq_ids(self, seq_ids: Sequence[str]) -> None:
        if list(seq_ids) != self.meta["seq_ids"]:
            raise ValueError(f"feature cache {self.cache_dir} was built for a different window order than the chunk")

    def read(self, idx: int, variant: str = "clean") -> Tensor:
        """``[T, F]`` float32 features for window ``idx``."""
        if variant not in self._index:
            raise KeyError(f"feature cache {self.cache_dir} has no variant {variant!r} (has {self.variants})")
        pid = os.getpid()
        if self._array is None or self._pid != pid:
            self._array, self._pid = np.load(self.cache_dir / FEATURES_FILE, mmap_mode="r"), pid
        return torch.from_numpy(np.asarray(self._array[idx, self._index[variant]], dtype=np.float32))


def open_chunk_features(cfg: RootCfg, chunk_path: str | Path) -> ChunkFeatures | None:
    """The chunk's cache when ``data.visual_input=cached_features``, else ``None`` (the image path)."""
    if cfg.data.visual_input != "cached_features":
        return None
    return ChunkFeatures(
        chunk_cache_dir(resolve_paths(cfg.paths).feature_cache_dir, chunk_path),
        expect={
            "version": CACHE_VERSION,
            "backbone": cfg.model.vit_backbone,
            "context_size": [cfg.data.read_context_height, cfg.data.read_context_width],
            "norm_mean": list(cfg.data.norm_mean),
            "norm_std": list(cfg.data.norm_std),
        },
    )


def verify_cache_weights(net: nn.Module, features: ChunkFeatures) -> None:
    """Fail loudly when the cache was built from different backbone weights or BN statistics."""
    expected, actual = features.meta.get("weights_fingerprint"), weights_fingerprint(net)
    if expected != actual:
        raise ValueError(
            f"feature cache {features.cache_dir} was built from backbone weights {expected}, but this model's "
            f"backbone is {actual} — different pretrained weights? Rebuild the cache."
        )


def verify_model_cache(cfg: RootCfg, model: nn.Module, chunk_paths: Sequence[str | Path]) -> None:
    """Cached mode: check the model's backbone against the first chunk's cache. No-op on the image path.

    One chunk is enough — a cache dir is built in one pass by one backbone — and the per-chunk structural
    checks (window order, crop size, norm) still run when each dataset opens.
    """
    if cfg.data.visual_input != "cached_features" or not chunk_paths:
        return
    backbone = getattr(model, "vit", None)
    if backbone is None or not hasattr(backbone, "net"):
        raise TypeError("cached features need a timm visual backbone at model.vit")
    features = open_chunk_features(cfg, chunk_paths[0])
    assert features is not None
    verify_cache_weights(backbone.net, features)


@dataclass(frozen=True)
class FeatureChoice:
    variant: str
    flip: bool
    noise_seed: int | None


class CachedFeatureAugmentor:
    """Train-time augmentation for cached features: ``RuntimeAugmentor``'s draws mapped onto cached variants."""

    def __init__(self, cfg: AugmentCfg, source_width: int) -> None:
        self.cfg = cfg
        self.source_width = source_width

    def choose(self, rng: random.Random, color_variants: int) -> FeatureChoice:
        flip = rng.random() < self.cfg.p_flip
        use_color = rng.random() < self.cfg.p_color and color_variants > 0
        color = rng.randrange(color_variants) if use_color else None
        noise_seed = rng.randrange(2**31) if rng.random() < self.cfg.p_noise else None
        base = "clean" if color is None else f"color{color}"
        if flip:
            variant = "flip" if base == "clean" else f"{base}_flip"
        else:
            variant = base
        return FeatureChoice(variant=variant, flip=flip, noise_seed=noise_seed)

    def apply(self, choice: FeatureChoice, motions: Tensor, pose: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """The non-feature half of the choice: flip motions + pose, then motion noise."""
        if choice.flip:
            motions, pose = flip_motions_pose(motions, pose, self.source_width)
        if choice.noise_seed is not None:
            with _isolated_torch_seed(choice.noise_seed):
                motions = motions + torch.randn_like(motions) * self.cfg.motion_noise_std
        return motions, pose


# --------------------------------------------------------------------------- build side


class _ContextCrops(Dataset):
    """Decode-only reader for the builder: ``(idx, [T, 3, H, W] context crops in [0, 1])``."""

    def __init__(self, lmdb_path: str | Path, cfg: DataCfg) -> None:
        self.lmdb_path = str(lmdb_path)
        index = LMDBChunkDataset.from_config(lmdb_path, cfg)      # same cursor order as training reads
        self.seq_ids = list(index.seq_ids)
        index.close()
        self._resize = resize_to_tensor((cfg.read_context_height, cfg.read_context_width))
        self._env: lmdb.Environment | None = None
        self._pid: int | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_env"], state["_pid"] = None, None
        return state

    def __len__(self) -> int:
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> tuple[int, Tensor]:
        if self._env is None or self._pid != os.getpid():
            self._env, self._pid = lmdb.open(self.lmdb_path, readonly=True, lock=False), os.getpid()
        seq_id = self.seq_ids[idx]
        with self._env.begin(write=False) as txn:
            frames = pickle.loads(txn.get(f"{seq_id}_meta".encode()))["motions"].shape[0]
            crops = []
            for k in range(frames):
                buf = txn.get(f"{seq_id}_{k}_context".encode())
                if buf is None:
                    raise ValueError(f"{self.lmdb_path}: window {seq_id!r} is missing context frame {k}")
                crops.append(self._resize(Image.open(io.BytesIO(buf)).convert("RGB")))
        return idx, torch.stack(crops)


def _jitter_seed(base: int, chunk_path: str | Path, seq_id: str, k: int) -> int:
    chunk = Path(chunk_path)
    return (zlib.crc32(f"{chunk.parent.name}/{chunk.name}/{seq_id}/{k}".encode()) + base * 1_000_003) % 2**31


def _variant_frames(frames: Tensor, names: list[str], jitter: ColorJitter, seeds: list[list[int]]) -> Tensor:
    """``[B, T, 3, H, W]`` -> ``[B, V, T, 3, H, W]``; ``seeds[b][k]`` drives window ``b``'s color variant ``k``."""
    jittered: dict[int, Tensor] = {}
    out = []
    for name in names:
        base, flipped = name.removesuffix("_flip"), name.endswith("_flip") or name == "flip"
        if base in ("clean", "flip"):
            x = frames
        else:
            k = int(base[len("color"):])
            if k not in jittered:
                windows = []
                for b in range(frames.shape[0]):
                    with _isolated_torch_seed(seeds[b][k]):
                        windows.append(torch.stack([jitter(f) for f in frames[b]]))
                jittered[k] = torch.stack(windows)
            x = jittered[k]
        out.append(x.flip(-1) if flipped else x)
    return torch.stack(out, dim=1)


def _encode(net: nn.Module, variants: Tensor, normalize, *, use_amp: bool, sub_batch: int) -> np.ndarray:
    b, v, t = variants.shape[:3]
    flat = normalize(variants.flatten(0, 2))
    with torch.no_grad(), autocast_ctx(use_amp, flat.device.type):
        feats = torch.cat([net(chunk).float() for chunk in flat.split(sub_batch)])
    return feats.view(b, v, t, -1).cpu().numpy().astype(np.float16)


def _existing_cache_ok(out_dir: Path, names: list[str], fingerprint: str, seq_ids: list[str]) -> bool:
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wanted = {"variants": names, "weights_fingerprint": fingerprint, "seq_ids": seq_ids}
    if all(meta.get(key) == value for key, value in wanted.items()):
        return True
    raise ValueError(f"{out_dir} holds a cache built differently (variants/weights/windows) — pass overwrite=True")


def build_chunk_cache(
    chunk_path: str | Path,
    cfg: RootCfg,
    net: nn.Module,
    *,
    augmented: bool,
    device: torch.device,
    batch_windows: int = 16,
    num_workers: int = 0,
    sub_batch: int = 512,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Build one chunk's cache with the eval-mode ``net``. Returns ``(cache dir, built)`` — ``built`` is
    False when a matching complete cache already existed. ``augmented`` adds the flip + color variants."""
    net = net.eval().to(device)
    names = variant_names(cfg.augment.cache_color_variants, augmented=augmented)
    dataset = _ContextCrops(chunk_path, cfg.data)
    fingerprint = weights_fingerprint(net)
    out_dir = chunk_cache_dir(resolve_paths(cfg.paths).feature_cache_dir, chunk_path)
    if not overwrite and _existing_cache_ok(out_dir, names, fingerprint, dataset.seq_ids):
        return out_dir, False
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META_FILE).unlink(missing_ok=True)
    a = cfg.augment
    jitter = ColorJitter(a.color_brightness, a.color_contrast, a.color_saturation, a.color_hue)
    normalize, use_amp = imagenet_normalize(cfg.data), device.type == "cuda" and cfg.train.use_amp
    loader = DataLoader(dataset, batch_size=batch_windows, num_workers=num_workers, shuffle=False)
    partial = out_dir / (FEATURES_FILE + ".partial")
    store = None
    for idx, frames in loader:
        seeds = [[_jitter_seed(a.seed, chunk_path, dataset.seq_ids[i], k) for k in range(a.cache_color_variants)]
                 for i in idx.tolist()]
        feats = _encode(net, _variant_frames(frames.to(device), names, jitter, seeds), normalize,
                        use_amp=use_amp, sub_batch=sub_batch)
        if store is None:
            shape = (len(dataset), *feats.shape[1:])
            store = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=shape)
        store[idx.numpy()] = feats
    if store is None:
        raise ValueError(f"{chunk_path}: empty chunk")
    shape = store.shape
    store.flush()
    del store
    os.replace(partial, out_dir / FEATURES_FILE)
    _write_meta(out_dir, cfg, chunk_path, dataset.seq_ids, names, shape, fingerprint, use_amp)
    return out_dir, True


@dataclass(frozen=True)
class ParityStats:
    """Spread of ``|cache - recompute|`` over every compared feature value.

    ``max`` is the worst single value out of ``count`` (64 windows x T frames x F features is ~4e5), so it
    is a tail statistic with no stability guarantee under AMP: measured 2026-09-17, it moved 5.4x between
    splits (6.3e-3 train, 3.4e-2 train_benchmark) while ``mean`` moved 11% (7.7e-4 vs 8.5e-4) and the
    inputs themselves matched exactly. Gate on ``p999``; report ``max`` as a diagnostic.
    """

    max: float
    p999: float
    mean: float
    count: int

    def __str__(self) -> str:
        return (f"p99.9 = {self.p999:.2e}, max = {self.max:.2e}, mean = {self.mean:.2e} "
                f"over {self.count} values")


def verify_chunk_cache(
    chunk_path: str | Path, cfg: RootCfg, net: nn.Module, *, device: torch.device, windows: int = 32
) -> ParityStats:
    """Difference between the cached ``clean`` features and the eval-mode backbone run on the image path's
    own context crops, over the chunk's first ``windows`` windows (real-data parity).

    The recompute mirrors the build's own autocast setting (``meta["autocast"]``): the cache stores what an
    AMP forward produced, so an fp32 recompute measures AMP-vs-fp32 drift (~1.9e-2 on TinyViT-5M) rather
    than whether the cache is right — that asymmetry failed G_parity on the first v2 cache build. Residual
    after matching it is fp16 storage plus AMP reduction order (the build batches 512 frames, this batches
    one window); see :class:`ParityStats` for which statistic to gate on.
    """
    features = ChunkFeatures(chunk_cache_dir(resolve_paths(cfg.paths).feature_cache_dir, chunk_path))
    dataset = LMDBChunkDataset.from_config(chunk_path, cfg.data)
    net = net.eval().to(device)
    use_amp = bool(features.meta.get("autocast", False)) and device.type == "cuda"
    diffs: list[np.ndarray] = []
    try:
        features.check_seq_ids(dataset.seq_ids)
        with torch.no_grad(), autocast_ctx(use_amp, device.type):
            for idx in range(min(windows, len(dataset))):
                recomputed = net(dataset[idx]["images_context"].to(device)).float().cpu()
                diffs.append((recomputed - features.read(idx, "clean")).abs().numpy().ravel())
    finally:
        dataset.close()
    d = np.concatenate(diffs) if diffs else np.zeros(1, dtype=np.float32)
    return ParityStats(float(d.max()), float(np.percentile(d, 99.9)), float(d.mean()), int(d.size))


def _write_meta(
    out_dir: Path, cfg: RootCfg, chunk_path, seq_ids, names, shape, fingerprint: str, use_amp: bool
) -> None:
    import timm  # local: heavy, and only the builder needs its version string

    a = cfg.augment
    meta = {
        "version": CACHE_VERSION,
        "lmdb_path": str(chunk_path),
        "seq_ids": list(seq_ids),
        "variants": names,
        "shape": list(shape),
        "dtype": "float16",
        "num_features": int(shape[-1]),
        "backbone": cfg.model.vit_backbone,
        "pretrained": cfg.model.vit_pretrained,
        "weights_fingerprint": fingerprint,
        "timm_version": timm.__version__,
        "context_size": [cfg.data.read_context_height, cfg.data.read_context_width],
        "norm_mean": list(cfg.data.norm_mean),
        "norm_std": list(cfg.data.norm_std),
        "color_jitter": {"seed": a.seed, "brightness": a.color_brightness, "contrast": a.color_contrast,
                         "saturation": a.color_saturation, "hue": a.color_hue},
        "autocast": use_amp,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta), encoding="utf-8")

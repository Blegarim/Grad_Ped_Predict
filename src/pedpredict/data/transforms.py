"""Crop geometry + motion features + normalization.

Owns the per-frame *math* of the LMDB pipeline, ported from OLD
``scripts/PIE_sequence_Dataset_1.py::PIESequenceDataset._process_sequence``. The companion
``lmdb_writer.py`` owns only serialization/chunking — this split is band-aid B5 (one fused
``preprocess`` script becomes geometry + writer).

Behavior is preserved exactly vs the legacy ``_process_sequence`` (verified by
``tests/test_transforms.py`` against ``tests/fixtures/golden/lmdb_process_record.pt``), with two
deliberate, contract-aligned changes flagged in docs/archive/MIGRATION.md:

* **TurboJPEG dropped** — frames are decoded with PIL only (:func:`load_rgb`). The legacy hardcoded
  ``C:\\libjpeg-turbo64`` DLL path is removed (a path/B5 smell); decode output is otherwise identical.
* **ImageNet normalize is NOT applied at write time.** Stored crops are plain ``[0, 1]`` → JPEG;
  :func:`imagenet_normalize` is *defined* here but consumed at read time by ``lmdb_dataset`` (1.5).

The 8-dim motion feature is pinned and documented in :func:`compute_motion` (upstream half of B7):
``(cx, cy, dx, dy, w, h, dw, dh)`` from the **int-truncated** bbox. The writer emits exactly
``motion_dim`` channels so the legacy ``motions[..., :8]`` slice (1.5) becomes a no-op to delete.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from pedpredict.config.schema import DataCfg
from pedpredict.data.pie_sequences import SequenceRecord

__all__ = [
    "load_rgb",
    "crop_tight",
    "crop_context",
    "compute_motion",
    "resize_to_tensor",
    "build_write_transforms",
    "build_read_transforms",
    "imagenet_normalize",
    "ProcessedSample",
    "process_record",
    "CropSequenceDataset",
]

_BoxInt = tuple[int, int, int, int]


def load_rgb(path: str | Path) -> Image.Image:
    """Open ``path`` as RGB via PIL, with the legacy missing-file fallback (zero-pad / ``.jpg``).

    Mirrors OLD ``_process_sequence`` lines 47-60: if the exact path is absent, try the ``.jpg``
    suffix and, for purely-numeric stems, a 6-digit zero-padded name (PIE frame naming). TurboJPEG
    is intentionally not used (see module docstring).
    """
    p = Path(path)
    if not p.exists():
        candidates = [p.with_suffix(".jpg")]
        if p.stem.isdigit():
            padded = p.stem.zfill(6)
            candidates += [p.with_name(padded + p.suffix), p.with_name(padded + ".jpg")]
        for alt in candidates:
            if alt.exists():
                p = alt
                break
        else:
            raise FileNotFoundError(f"Image not found: {path} or any of {candidates}")
    return Image.open(p).convert("RGB")


def crop_tight(img: Image.Image, bbox: list[float] | tuple[float, ...]) -> tuple[Image.Image, _BoxInt]:
    """Int-truncate ``bbox`` and PIL-crop the tight pedestrian box. Returns ``(crop, int_box)``.

    The int box is reused for the context crop and the motion feature, so all three share one
    coordinate definition (OLD computed ``x1..y2 = map(int, bbox)`` once, line 72).
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def crop_context(img: Image.Image, box_int: _BoxInt, scale: float) -> Image.Image:
    """Crop a ``scale``×-enlarged box around the int-bbox center, clamped to the image bounds.

    Verbatim geometry from OLD lines 74-82 (float box coords passed straight to ``PIL.Image.crop``).
    """
    x1, y1, x2, y2 = box_int
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w2, h2 = w * scale, h * scale
    x1c, y1c = cx - w2 / 2, cy - h2 / 2
    x2c, y2c = cx + w2 / 2, cy + h2 / 2
    x1c, y1c = max(0, x1c), max(0, y1c)
    x2c, y2c = min(img.width, x2c), min(img.height, y2c)
    return img.crop((x1c, y1c, x2c, y2c))


def compute_motion(boxes_int: list[_BoxInt] | tuple[_BoxInt, ...]) -> Tensor:
    """Build the ``[T, 8]`` motion feature ``(cx, cy, dx, dy, w, h, dw, dh)`` from int boxes.

    Channel definition (preserved EXACTLY from OLD lines 95-108 — including the frame-0 asymmetry):

    ===  ====  ===================================================  ==================================
    idx  name  value                                                notes
    ===  ====  ===================================================  ==================================
    0    cx    ``(x1 + x2) / 2``                                    absolute center x
    1    cy    ``(y1 + y2) / 2``                                    absolute center y
    2    dx    ``cx_t - cx_{t-1}``;  t=0 -> ``cx_1 - cx_0``         flip-negated channel in 1.4
    3    dy    ``cy_t - cy_{t-1}``;  t=0 -> ``cy_1 - cy_0``         first delta duplicated
    4    w     ``x2 - x1``
    5    h     ``y2 - y1``
    6    dw    ``w_t - w_{t-1}``;  **t=0 -> ``w_0`` (raw width)**   legacy quirk: NOT a delta at t=0
    7    dh    ``h_t - h_{t-1}``;  **t=0 -> ``h_0`` (raw height)**  legacy quirk: NOT a delta at t=0
    ===  ====  ===================================================  ==================================

    The dw/dh frame-0 quirk (raw size, not a delta) is almost certainly an unintended legacy bug;
    Phase A preserves it because the trained weights depend on it (Phase-B fix candidate).
    """
    centers, widths, heights = [], [], []
    for x1, y1, x2, y2 in boxes_int:
        widths.append(x2 - x1)
        heights.append(y2 - y1)
        centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
    centers_t = torch.tensor(centers, dtype=torch.float32)
    widths_t = torch.tensor(widths, dtype=torch.float32)
    heights_t = torch.tensor(heights, dtype=torch.float32)
    if centers_t.shape[0] < 2:  # OLD would index dt[0:1] on an empty delta; fail loudly instead
        raise ValueError(f"compute_motion needs T >= 2 frames; got T={centers_t.shape[0]}")
    dt = centers_t[1:] - centers_t[:-1]
    dt = torch.cat([dt[0:1], dt], dim=0)
    dw = torch.cat([widths_t[0:1], widths_t[1:] - widths_t[:-1]], dim=0)
    dh = torch.cat([heights_t[0:1], heights_t[1:] - heights_t[:-1]], dim=0)
    return torch.cat(
        [centers_t, dt, widths_t.unsqueeze(1), heights_t.unsqueeze(1), dw.unsqueeze(1), dh.unsqueeze(1)],
        dim=1,
    )


def resize_to_tensor(size: tuple[int, int]) -> Callable[[Image.Image], Tensor]:
    """``Resize(size)`` (on the PIL image) -> ``ToTensor`` -> ``[0, 1]`` float. No normalize."""
    return transforms.Compose([transforms.Resize(size), transforms.ToTensor()])


def build_write_transforms(cfg: DataCfg) -> tuple[Callable, Callable]:
    """Deterministic write-time transforms ``(tight, context)`` sized from ``cfg`` (OLD ``img_resize``)."""
    tight = resize_to_tensor((cfg.img_height, cfg.img_width))
    ctx_h, ctx_w = int(cfg.img_height * cfg.context_scale), int(cfg.img_width * cfg.context_scale)
    return tight, resize_to_tensor((ctx_h, ctx_w))


def imagenet_normalize(cfg: DataCfg) -> transforms.Normalize:
    """ImageNet ``Normalize`` for READ-time use (lmdb_dataset, 1.5) — not applied by the writer."""
    return transforms.Normalize(mean=list(cfg.norm_mean), std=list(cfg.norm_std))


def build_read_transforms(cfg: DataCfg) -> tuple[Callable, Callable]:
    """Read-time transforms ``(tight, context)``: ``Resize -> ToTensor -> ImageNet Normalize``.

    Mirrors OLD ``train.py:355-366`` (the only place the runtime dataset's transforms were defined).
    Stored crops are un-normalized ``[0, 1]`` JPEG (1.2 contract); normalize is applied here, at read
    time, by ``lmdb_dataset``. Tight resizes to ``(img_height, img_width)``; context resizes to
    ``(read_context_height, read_context_width)`` — the model input (224), NOT the larger write size.
    """
    norm = imagenet_normalize(cfg)
    tight = transforms.Compose(
        [transforms.Resize((cfg.img_height, cfg.img_width)), transforms.ToTensor(), norm]
    )
    context = transforms.Compose(
        [transforms.Resize((cfg.read_context_height, cfg.read_context_width)), transforms.ToTensor(), norm]
    )
    return tight, context


@dataclass(slots=True)
class ProcessedSample:
    """One windowed sequence after cropping/resize — the writer's serialization input."""

    images_tight: Tensor    # [T, 3, img_height, img_width]            float [0, 1]
    images_context: Tensor  # [T, 3, H*scale, W*scale]                 float [0, 1]
    motions: Tensor         # [T, 8]  (cx, cy, dx, dy, w, h, dw, dh)   float32
    actions: Tensor         # 0-dim long
    looks: Tensor           # 0-dim long
    crosses: Tensor         # 0-dim long


def process_record(
    rec: SequenceRecord,
    cfg: DataCfg,
    transform_tight: Callable | None = None,
    transform_context: Callable | None = None,
) -> ProcessedSample:
    """Turn one :class:`SequenceRecord` into a :class:`ProcessedSample` (crop + resize + motion).

    Replaces OLD ``PIESequenceDataset._process_sequence``. Transforms default to
    :func:`build_write_transforms` (config-driven); callers may inject augmentation transforms
    without changing the geometry/motion math.
    """
    if transform_tight is None or transform_context is None:
        bt, bc = build_write_transforms(cfg)
        transform_tight = bt if transform_tight is None else transform_tight
        transform_context = bc if transform_context is None else transform_context

    tights, contexts, boxes = [], [], []
    for img_path, bbox in zip(rec["images"], rec["bboxes"], strict=True):
        img = load_rgb(img_path)
        tight, box = crop_tight(img, bbox)
        tights.append(transform_tight(tight))
        contexts.append(transform_context(crop_context(img, box, cfg.context_scale)))
        boxes.append(box)

    return ProcessedSample(
        images_tight=torch.stack(tights, dim=0),
        images_context=torch.stack(contexts, dim=0),
        motions=compute_motion(boxes),
        actions=torch.tensor(rec["actions"], dtype=torch.long),
        looks=torch.tensor(rec["looks"], dtype=torch.long),
        crosses=torch.tensor(rec["crosses"], dtype=torch.long),
    )


class CropSequenceDataset(Dataset):
    """Thin ``Dataset`` over records calling :func:`process_record` (enables the writer's workers).

    Build-time transforms are constructed once (not per ``__getitem__``) so DataLoader workers reuse
    them. Output order is deterministic (no shuffle) — parity with the legacy serial writer holds.
    """

    def __init__(
        self,
        records: list[SequenceRecord],
        cfg: DataCfg,
        *,
        transform_tight: Callable | None = None,
        transform_context: Callable | None = None,
    ) -> None:
        bt, bc = build_write_transforms(cfg)
        self.records = records
        self.cfg = cfg
        self.transform_tight = transform_tight or bt
        self.transform_context = transform_context or bc

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ProcessedSample:
        return process_record(self.records[index], self.cfg, self.transform_tight, self.transform_context)

"""Crop geometry + motion features + normalization (v2 motion contract).

Owns the per-frame *math* of the LMDB pipeline. The companion ``lmdb_writer.py`` owns only
serialization/chunking (B5 split). Crop geometry and JPEG handling are unchanged from the legacy
port; the **motion feature is the v2 contract** (hole audit A4 + M9, deliberate behavior changes):

* **Frame-0 deltas are true zeros.** The legacy quirk (``dx``/``dy`` duplicated the first delta;
  ``dw``/``dh`` held the *raw* initial size, which per-sequence normalization turned into a t=0
  spike that deleted both channels) is fixed.
* **Ego-vehicle speed is the 9th channel.** The writer ALWAYS stores the full
  ``MOTION_STORE_DIM`` (9) vector; runtime consumers slice to ``data.motion_dim``
  ("store wide, slice narrow") so with/without-ego are two configs over one dataset.
* **No normalization is baked in** — motions are stored in raw pixel / km/h units; the
  normalization choice is a *runtime* model flag (``model.motion_norm``, A4 ablation).

Other write/read conventions are unchanged:

* Frames are decoded with PIL only (:func:`load_rgb`).
* **ImageNet normalize is NOT applied at write time.** Stored crops are plain ``[0, 1]`` → JPEG;
  :func:`imagenet_normalize` is *defined* here but consumed at read time by ``lmdb_dataset``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from pedpredict.config.schema import MOTION_STORE_DIM, DataCfg
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


def compute_motion(
    boxes_int: list[_BoxInt] | tuple[_BoxInt, ...],
    ego_speed: Sequence[float] | None = None,
) -> Tensor:
    """Build the ``[T, 9]`` v2 motion feature ``(cx, cy, dx, dy, w, h, dw, dh, ego)`` from int boxes.

    Channel definition (v2 contract — hole audit A4 + M9; values are RAW pixel / km/h units, the
    normalization choice lives at runtime in ``MotionEncoder``):

    ===  ====  ===================================================  ==================================
    idx  name  value                                                notes
    ===  ====  ===================================================  ==================================
    0    cx    ``(x1 + x2) / 2``                                    absolute center x; flip-REFLECTED
    1    cy    ``(y1 + y2) / 2``                                    absolute center y
    2    dx    ``cx_t - cx_{t-1}``;  **t=0 -> 0**                   flip-NEGATED channel
    3    dy    ``cy_t - cy_{t-1}``;  **t=0 -> 0**
    4    w     ``x2 - x1``
    5    h     ``y2 - y1``
    6    dw    ``w_t - w_{t-1}``;  **t=0 -> 0**                     legacy raw-size quirk REMOVED (A4)
    7    dh    ``h_t - h_{t-1}``;  **t=0 -> 0**
    8    ego   OBD ego-vehicle speed (km/h)                         M9; ``None`` -> zeros (video infer)
    ===  ====  ===================================================  ==================================

    Always returns ``MOTION_STORE_DIM`` (9) channels — runtime consumers slice to ``motion_dim``.
    ``ego_speed=None`` fills the ego channel with zeros (the raw-video inference path has no OBD);
    callers that then feed a ``motion_dim=9`` model are knowingly feeding a dead channel.
    """
    centers, widths, heights = [], [], []
    for x1, y1, x2, y2 in boxes_int:
        widths.append(x2 - x1)
        heights.append(y2 - y1)
        centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
    centers_t = torch.tensor(centers, dtype=torch.float32)
    widths_t = torch.tensor(widths, dtype=torch.float32)
    heights_t = torch.tensor(heights, dtype=torch.float32)
    t = centers_t.shape[0]
    if t < 2:
        raise ValueError(f"compute_motion needs T >= 2 frames; got T={t}")
    dt = torch.cat([torch.zeros(1, 2), centers_t[1:] - centers_t[:-1]], dim=0)
    dw = torch.cat([torch.zeros(1), widths_t[1:] - widths_t[:-1]], dim=0)
    dh = torch.cat([torch.zeros(1), heights_t[1:] - heights_t[:-1]], dim=0)
    if ego_speed is None:
        ego_t = torch.zeros(t, dtype=torch.float32)
    else:
        if len(ego_speed) != t:
            raise ValueError(f"compute_motion: ego_speed has {len(ego_speed)} frames, expected {t}")
        ego_t = torch.tensor([float(v) for v in ego_speed], dtype=torch.float32)
    out = torch.cat(
        [
            centers_t,
            dt,
            widths_t.unsqueeze(1),
            heights_t.unsqueeze(1),
            dw.unsqueeze(1),
            dh.unsqueeze(1),
            ego_t.unsqueeze(1),
        ],
        dim=1,
    )
    assert out.shape[1] == MOTION_STORE_DIM
    return out


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

    images_tight: Tensor    # [T, 3, img_height, img_width]                 float [0, 1]
    images_context: Tensor  # [T, 3, H*scale, W*scale]                      float [0, 1]
    motions: Tensor         # [T, 9]  (cx, cy, dx, dy, w, h, dw, dh, ego)   float32, raw units
    actions: Tensor         # 0-dim long
    looks: Tensor           # 0-dim long
    crosses: Tensor         # 0-dim long
    track_id: str = ""      # PIE pedestrian id (M6) — eval-side track aggregation
    tte: int | None = None  # M5 benchmark windows only: frames to crossing_point


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
        motions=compute_motion(boxes, ego_speed=rec["ego_speed"]),
        actions=torch.tensor(rec["actions"], dtype=torch.long),
        looks=torch.tensor(rec["looks"], dtype=torch.long),
        crosses=torch.tensor(rec["crosses"], dtype=torch.long),
        track_id=rec["track_id"],
        tte=rec.get("tte"),
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

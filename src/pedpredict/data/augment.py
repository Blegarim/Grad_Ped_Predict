"""Offline minority-class augmentation, porting OLD ``scripts/augment_sequences.py``.

⚠️ The OLD ``SequenceAugmenter`` indexed ``seq['images_tight'] / 'images_context' / 'motions'`` —
pre-cropped *tensor* sequences. But the real pipeline pkl is **path-based** (``{images, bboxes,
actions, looks, crosses}``, see :mod:`pedpredict.data.pie_sequences`) and the writer crops from paths,
so the legacy augmenter could never have run on it (dead/broken code — a B5 fragmentation smell). This
port re-homes the **transform math, unchanged** onto :class:`~pedpredict.data.transforms.ProcessedSample`
(the writer's crop output) and applies it at write time.

Faithful semantics preserved from OLD:
  * ``horizontal_flip`` mirrors the width axis of both crops **and** negates motion channel
    :data:`_FLIP_NEGATE_IDX` (= ``dx``, idx 2 in ``compute_motion``). The flip↔channel coupling — which
    the schematic flags as silently corrupting if mismatched — is made explicit here and guarded by
    ``tests/test_augment.py``. ⚠️ Legacy quirks ride along (Phase B): absolute ``cx`` (idx 0) is NOT
    reflected; motion noise hits absolute channels too.
  * Each minority record yields its **original** plus several **single-transform** copies (OLD ``__call__``
    appends a fresh copy per selected transform — it does NOT compose transforms). One :class:`AugItem`
    therefore carries exactly one transform (or ``None`` for the identity copy).
  * Negatives are NOT re-emitted (decided 1.4): the aug LMDB holds only minority records + their copies,
    unioned at train time with ``preprocessed_train`` (which already holds the negatives).

All probabilities / multipliers / params come from :class:`~pedpredict.config.schema.AugmentCfg`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from pedpredict.config.schema import AugmentCfg, DataCfg
from pedpredict.data.lmdb_writer import write_dataset_chunks_from
from pedpredict.data.pie_sequences import SequenceRecord, load_sequences
from pedpredict.data.transforms import ProcessedSample, build_write_transforms, process_record

__all__ = [
    "TransformName",
    "AugItem",
    "SequenceAugmenter",
    "plan_oversample",
    "summarize_plan",
    "AugmentedCropSequenceDataset",
    "augment_sequence_file",
]

#: Motion channel negated by a horizontal flip — ``dx`` in ``compute_motion``'s
#: ``(cx, cy, dx, dy, w, h, dw, dh)``. LOCKED (docs/archive/MIGRATION.md 1.2/1.4); a mismatch silently corrupts data.
_FLIP_NEGATE_IDX: int = 2


class TransformName(str, Enum):
    """The four augmentations OLD ``SequenceAugmenter`` could apply (one per augmented copy)."""

    FLIP = "flip"
    COLOR = "color"
    NOISE = "noise"
    ERASE = "erase"


@dataclass(slots=True, frozen=True)
class AugItem:
    """One output sample: a source record + the single transform to apply (``None`` = identity copy)."""

    record_index: int
    transform: TransformName | None
    seed: int


# --------------------------------------------------------------------------- tensor transforms


class SequenceAugmenter:
    """Faithful port of OLD ``SequenceAugmenter``'s four transforms, operating on ``ProcessedSample``.

    Each method returns a NEW ``ProcessedSample`` (touched tensors cloned), never mutating the input —
    the value-level equivalent of the legacy ``copy.deepcopy``. ``color_augment`` / ``motion_noise``
    consume the global ``torch`` RNG and ``random_erase_frames`` consumes the passed ``random.Random``;
    :meth:`apply` seeds them deterministically per :class:`AugItem`.
    """

    def __init__(self, cfg: AugmentCfg) -> None:
        self.cfg = cfg
        self._jitter = ColorJitter(
            brightness=cfg.color_brightness,
            contrast=cfg.color_contrast,
            saturation=cfg.color_saturation,
            hue=cfg.color_hue,
        )

    def horizontal_flip(self, s: ProcessedSample) -> ProcessedSample:
        """Mirror the width axis of both crops; negate motion ``dx`` (idx :data:`_FLIP_NEGATE_IDX`)."""
        motions = s.motions.clone()
        motions[:, _FLIP_NEGATE_IDX] *= -1
        return replace(
            s,
            images_tight=torch.flip(s.images_tight, dims=[3]),
            images_context=torch.flip(s.images_context, dims=[3]),
            motions=motions,
        )

    def color_augment(self, s: ProcessedSample) -> ProcessedSample:
        """Per-frame ``ColorJitter`` on tight then context (OLD interleaved RNG-consumption order)."""
        tight = s.images_tight.clone()
        context = s.images_context.clone()
        for t in range(tight.shape[0]):
            tight[t] = self._jitter(tight[t])
            context[t] = self._jitter(context[t])
        return replace(s, images_tight=tight, images_context=context)

    def motion_noise(self, s: ProcessedSample) -> ProcessedSample:
        """Add ``N(0, motion_noise_std)`` to ALL 8 motion channels (faithful to OLD)."""
        return replace(s, motions=s.motions + torch.randn_like(s.motions) * self.cfg.motion_noise_std)

    def random_erase_frames(self, s: ProcessedSample, rng: random.Random) -> ProcessedSample:
        """Replace ``erase_n_frames`` frames with the mean of their neighbors (in-place read order)."""
        n = self.cfg.erase_n_frames
        num_frames = s.images_tight.shape[0]
        if num_frames < n:
            return s
        tight = s.images_tight.clone()
        context = s.images_context.clone()
        for f in rng.sample(range(num_frames), n):
            prev_f, next_f = max(0, f - 1), min(num_frames - 1, f + 1)
            tight[f] = (tight[prev_f] + tight[next_f]) / 2
            context[f] = (context[prev_f] + context[next_f]) / 2
        return replace(s, images_tight=tight, images_context=context)

    def apply(self, s: ProcessedSample, transform: TransformName, seed: int) -> ProcessedSample:
        """Dispatch one transform, seeding its RNG from ``seed`` (deterministic, worker-independent)."""
        if transform is TransformName.FLIP:
            return self.horizontal_flip(s)
        if transform is TransformName.COLOR:
            torch.manual_seed(seed)
            return self.color_augment(s)
        if transform is TransformName.NOISE:
            torch.manual_seed(seed)
            return self.motion_noise(s)
        if transform is TransformName.ERASE:
            return self.random_erase_frames(s, random.Random(seed))
        raise ValueError(f"unknown transform {transform!r}")


# --------------------------------------------------------------------------- oversampling plan

_NAMES: tuple[TransformName, ...] = (TransformName.FLIP, TransformName.COLOR, TransformName.NOISE, TransformName.ERASE)


def _probs(cfg: AugmentCfg) -> dict[TransformName, float]:
    return {
        TransformName.FLIP: cfg.p_flip,
        TransformName.COLOR: cfg.p_color,
        TransformName.NOISE: cfg.p_noise,
        TransformName.ERASE: cfg.p_erase,
    }


def _expand(subset: list[int], multiplier: int, cfg: AugmentCfg, rng: random.Random) -> list[AugItem]:
    """Cycle ``subset``, emitting ``original + selected single-transform copies`` until ``len*multiplier``.

    Mirrors OLD ``expand_subset`` + ``SequenceAugmenter.__call__``: each call appends the identity copy,
    draws ``randint(n_augs_min, n_augs_max)`` transform names, and keeps each that passes its prob gate.
    The result is truncated to the exact target count.
    """
    if not subset:
        return []
    target = len(subset) * multiplier
    probs = _probs(cfg)
    out: list[AugItem] = []
    idx = 0
    while len(out) < target:
        rec = subset[idx % len(subset)]
        out.append(AugItem(rec, None, rng.randrange(2**31)))  # identity copy (OLD augmented[0])
        for name in rng.sample(_NAMES, rng.randint(cfg.n_augs_min, cfg.n_augs_max)):
            if rng.random() < probs[name]:
                out.append(AugItem(rec, name, rng.randrange(2**31)))
        idx += 1
    return out[:target]


def plan_oversample(records: Sequence[SequenceRecord], cfg: AugmentCfg) -> list[AugItem]:
    """Deterministic (seeded by ``cfg.seed``) augmentation plan for the minority records.

    Crosses=1 records are expanded ×``crosses_multiplier`` and looks=1 ×``looks_multiplier`` — a record
    that is positive for both appears in both subsets (OLD double-counts it). Negatives are excluded.
    """
    rng = random.Random(cfg.seed)
    cross_idx = [i for i, r in enumerate(records) if r["crosses"] == 1]
    look_idx = [i for i, r in enumerate(records) if r["looks"] == 1]
    return _expand(cross_idx, cfg.crosses_multiplier, cfg, rng) + _expand(look_idx, cfg.looks_multiplier, cfg, rng)


def summarize_plan(records: Sequence[SequenceRecord], items: Sequence[AugItem]) -> dict[str, int]:
    """Counts for logging: total output samples, identity copies, and per-task positive source records."""
    return {
        "total": len(items),
        "identity": sum(1 for it in items if it.transform is None),
        "augmented": sum(1 for it in items if it.transform is not None),
        "crosses_pos_sources": sum(1 for r in records if r["crosses"] == 1),
        "looks_pos_sources": sum(1 for r in records if r["looks"] == 1),
    }


# --------------------------------------------------------------------------- write-time integration


class AugmentedCropSequenceDataset(Dataset):
    """``Dataset[ProcessedSample]`` yielding one (possibly augmented) sample per :class:`AugItem`.

    ``__getitem__`` crops the source record (:func:`process_record`) then applies the item's transform.
    Build-time transforms are constructed once; the augmenter is stateless beyond its ``ColorJitter``, so
    the dataset is picklable for Windows-``spawn`` DataLoader workers.
    """

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        items: Sequence[AugItem],
        cfg: DataCfg,
        aug_cfg: AugmentCfg,
        *,
        transform_tight=None,
        transform_context=None,
    ) -> None:
        bt, bc = build_write_transforms(cfg)
        self.records = records
        self.items = items
        self.cfg = cfg
        self.augmenter = SequenceAugmenter(aug_cfg)
        self.transform_tight = transform_tight or bt
        self.transform_context = transform_context or bc

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ProcessedSample:
        item = self.items[index]
        sample = process_record(self.records[item.record_index], self.cfg, self.transform_tight, self.transform_context)
        if item.transform is None:
            return sample
        return self.augmenter.apply(sample, item.transform, item.seed)


def augment_sequence_file(in_path: str | Path, out_dir: str | Path, cfg: DataCfg, aug_cfg: AugmentCfg) -> list[Path]:
    """Load a sequence pkl, plan the minority oversampling, and write the augmented LMDB chunks.

    Returns the written ``chunk_*.lmdb`` paths. The output is an ordinary LMDB consumable by 1.5 exactly
    like ``preprocessed_train`` — it just contains minority records + their augmented copies.
    """
    records = load_sequences(in_path)
    items = plan_oversample(records, aug_cfg)
    dataset = AugmentedCropSequenceDataset(records, items, cfg, aug_cfg)
    return write_dataset_chunks_from(dataset, out_dir, cfg)

"""Offline minority-class augmentation, porting OLD ``scripts/augment_sequences.py``.

⚠️ The OLD ``SequenceAugmenter`` indexed ``seq['images_tight'] / 'images_context' / 'motions'`` —
pre-cropped *tensor* sequences. But the real pipeline pkl is **path-based** (``{images, bboxes,
actions, looks, crosses}``, see :mod:`pedpredict.data.pie_sequences`) and the writer crops from paths,
so the legacy augmenter could never have run on it (dead/broken code — a B5 fragmentation smell). This
port re-homes the **transform math, unchanged** onto :class:`~pedpredict.data.transforms.ProcessedSample`
(the writer's crop output) and applies it at write time.

**Source = the built base-train LMDB, not the PIE frames.** The crops augmentation transforms are
already stored in ``preprocessed_train`` (un-normalized JPEG), and ``build_lmdb_incremental.py`` deletes
the original frames as it goes. So the source is :class:`LMDBSampleSource` (the inverse of the writer,
via :func:`~pedpredict.data.lmdb_dataset.read_raw_sample`) — uniform with the rest of the pipeline, where
every stage consumes the prior stage's LMDB artifact. Any indexable ``source[i] -> ProcessedSample``
works (:class:`~pedpredict.data.transforms.CropSequenceDataset` still augments on-disk frames in tests),
since the transform math is source-agnostic.

Semantics (v2 — hole audit A4 applied):
  * ``horizontal_flip`` mirrors the width axis of both crops, negates motion channel
    :data:`_FLIP_NEGATE_IDX` (= ``dx``, idx 2 in ``compute_motion``) **and reflects absolute ``cx``
    (idx 0) about the source-frame width** (A4 fix — the legacy version left ``cx`` unreflected,
    silently corrupting absolute geometry for flipped copies). ``cy``/sizes/``ego`` are flip-invariant.
    The flip↔channel coupling is guarded by ``tests/test_augment.py``.
    ⚠️ Remaining known quirk (C4, WP1 redesign): motion noise hits absolute channels too.
  * Each minority record yields its **original** plus several **single-transform** copies (OLD ``__call__``
    appends a fresh copy per selected transform — it does NOT compose transforms). One :class:`AugItem`
    therefore carries exactly one transform (or ``None`` for the identity copy).
  * Negatives are NOT re-emitted (decided 1.4): the aug LMDB holds only minority records + their copies,
    unioned at train time with ``preprocessed_train`` (which already holds the negatives).

All probabilities / multipliers / params come from :class:`~pedpredict.config.schema.AugmentCfg`.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

import lmdb
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from pedpredict.config.schema import AugmentCfg, DataCfg
from pedpredict.data.lmdb_dataset import read_raw_sample
from pedpredict.data.lmdb_writer import write_dataset_chunks_from
from pedpredict.data.pie_sequences import SequenceRecord
from pedpredict.data.pose import flip_pose
from pedpredict.data.sampler import scan_chunk_labels
from pedpredict.data.stats import iter_chunk_lmdbs
from pedpredict.data.transforms import ProcessedSample

__all__ = [
    "TransformName",
    "AugItem",
    "SequenceAugmenter",
    "RuntimeAugmentor",
    "plan_oversample",
    "summarize_plan",
    "SampleSource",
    "LMDBSampleSource",
    "AugmentedSampleDataset",
    "augment_lmdb_dir",
]

#: Motion channels touched by a horizontal flip, indexing ``compute_motion``'s
#: ``(cx, cy, dx, dy, w, h, dw, dh, ego)``. LOCKED to the channel table; a mismatch silently corrupts data.
_FLIP_NEGATE_IDX: int = 2    # dx  -> -dx
_FLIP_REFLECT_IDX: int = 0   # cx  -> source_width - cx (A4)


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

    def __init__(self, cfg: AugmentCfg, source_width: int) -> None:
        self.cfg = cfg
        self.source_width = source_width  # PIE frame width in px (DataCfg.source_width) — cx reflection
        self._jitter = ColorJitter(
            brightness=cfg.color_brightness,
            contrast=cfg.color_contrast,
            saturation=cfg.color_saturation,
            hue=cfg.color_hue,
        )

    def horizontal_flip(self, s: ProcessedSample) -> ProcessedSample:
        """Mirror the width axis of both crops; negate ``dx``, reflect ``cx`` about the frame width,
        and mirror the raw pose keypoints (reflect x + swap left/right joints) when present."""
        motions = s.motions.clone()
        motions[:, _FLIP_NEGATE_IDX] *= -1
        motions[:, _FLIP_REFLECT_IDX] = self.source_width - motions[:, _FLIP_REFLECT_IDX]
        return replace(
            s,
            images_tight=torch.flip(s.images_tight, dims=[3]),
            images_context=torch.flip(s.images_context, dims=[3]),
            motions=motions,
            pose=flip_pose(s.pose, self.source_width) if s.pose is not None else None,
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
        """Add ``N(0, motion_noise_std)`` to ALL stored motion channels (C4: de-facto identity at
        0.02 px; redesign rides the WP1 lever ablation)."""
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


@contextmanager
def _isolated_torch_seed(seed: int) -> Iterator[None]:
    """Seed the global torch RNG for one op, then restore prior state — keeps the color/noise draws
    deterministic per sample without perturbing the training RNG (matters at ``num_workers=0``)."""
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


class RuntimeAugmentor:
    """On-the-fly train-time augmentation (S-scarcity): a fresh random composition of the four
    :class:`SequenceAugmenter` transforms per sample, on un-normalized ``[0, 1]`` crops (before ImageNet
    norm — color jitter is undefined on normalized tensors). Ratio-preserving (every sample, both
    classes), so it is the data-scarcity regularizer, not the offline minority-oversampling lever. All
    randomness flows from the passed per-sample ``rng`` (seeded by the dataset from run seed + epoch +
    index), so the same run reproduces and each epoch varies."""

    def __init__(self, cfg: AugmentCfg, source_width: int) -> None:
        self._aug = SequenceAugmenter(cfg, source_width)
        self._probs = _probs(cfg)

    def __call__(self, sample: ProcessedSample, rng: random.Random) -> ProcessedSample:
        if rng.random() < self._probs[TransformName.FLIP]:
            sample = self._aug.horizontal_flip(sample)
        if rng.random() < self._probs[TransformName.COLOR]:
            with _isolated_torch_seed(rng.randrange(2**31)):
                sample = self._aug.color_augment(sample)
        if rng.random() < self._probs[TransformName.NOISE]:
            with _isolated_torch_seed(rng.randrange(2**31)):
                sample = self._aug.motion_noise(sample)
        if rng.random() < self._probs[TransformName.ERASE]:
            sample = self._aug.random_erase_frames(sample, rng)
        return sample


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


# --------------------------------------------------------------------------- sample sources


class SampleSource(Protocol):
    """Any indexable random-access source of :class:`ProcessedSample` (image-based or LMDB-based)."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ProcessedSample: ...


class LMDBSampleSource:
    """Picklable, worker-safe view over base-train LMDB chunks yielding un-normalized
    :class:`ProcessedSample` s by global index — the augmentation source once the incremental build has
    deleted the original PIE frames (augment reads its crops straight from ``preprocessed_train``).

    Mirrors :class:`~pedpredict.data.lmdb_dataset.LMDBChunkDataset`'s pid-keyed per-process env handling
    so it survives the writer's ``spawn`` DataLoader workers. ONE metadata pass (one
    :func:`~pedpredict.data.sampler.scan_chunk_labels` per chunk, in the MAIN process) builds both the
    flat global index and the minority-plan labels — no extra scanner (B3).
    """

    def __init__(self, chunk_paths: Sequence[str | Path]) -> None:
        self.chunk_paths = [str(p) for p in chunk_paths]
        if not self.chunk_paths:
            raise FileNotFoundError("LMDBSampleSource: no base train LMDB chunks to augment from")
        self._index: list[tuple[int, str]] = []        # (chunk ordinal, seq_id) in scan order
        self._labels: list[dict[str, int]] = []         # {actions, looks, crosses} aligned to _index
        for ci, path in enumerate(self.chunk_paths):
            scan = scan_chunk_labels(path)
            for seq_id, (actions, looks, crosses) in zip(scan.seq_ids, scan.label_rows, strict=True):
                self._index.append((ci, seq_id))
                self._labels.append({"actions": actions, "looks": looks, "crosses": crosses})
        self._envs: dict[int, lmdb.Environment] = {}
        self._pid: int | None = None

    @property
    def label_records(self) -> list[dict[str, int]]:
        """Per-sample ``{actions, looks, crosses}`` (scan order) — the input to :func:`plan_oversample`."""
        return self._labels

    def __len__(self) -> int:
        return len(self._index)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_envs"] = {}   # live handles must not cross the worker pickle boundary
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        self._close_envs()

    def _close_envs(self) -> None:
        for env in self._envs.values():
            env.close()
        self._envs = {}

    def _get_env(self, ci: int) -> lmdb.Environment:
        pid = os.getpid()
        if self._pid != pid:                 # new worker — drop any inherited handles, reopen lazily
            self._close_envs()
            self._pid = pid
        env = self._envs.get(ci)
        if env is None:
            env = lmdb.open(self.chunk_paths[ci], readonly=True, lock=False)
            self._envs[ci] = env
        return env

    def __getitem__(self, index: int) -> ProcessedSample:
        ci, seq_id = self._index[index]
        with self._get_env(ci).begin(write=False) as txn:
            return read_raw_sample(txn, seq_id, self.chunk_paths[ci])


# --------------------------------------------------------------------------- write-time integration


class AugmentedSampleDataset(Dataset):
    """``Dataset[ProcessedSample]`` applying each :class:`AugItem`'s transform to a sample from ``source``.

    Source-agnostic (``source[i] -> ProcessedSample``) so the transform math is shared: pass an
    :class:`LMDBSampleSource` to augment the built train LMDB (the default), or a
    :class:`~pedpredict.data.transforms.CropSequenceDataset` to augment on-disk frames. The augmenter is
    stateless beyond its ``ColorJitter``, so the dataset is picklable for Windows-``spawn`` workers.
    """

    def __init__(
        self, source: SampleSource, items: Sequence[AugItem], aug_cfg: AugmentCfg, source_width: int
    ) -> None:
        self.source = source
        self.items = items
        self.augmenter = SequenceAugmenter(aug_cfg, source_width)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ProcessedSample:
        item = self.items[index]
        sample = self.source[item.record_index]
        if item.transform is None:
            return sample
        return self.augmenter.apply(sample, item.transform, item.seed)


def augment_lmdb_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    data_cfg: DataCfg,
    aug_cfg: AugmentCfg,
    *,
    num_workers: int | None = None,
) -> list[Path]:
    """Augment the minority classes of a built base-train LMDB into ``out_dir`` (the LMDB twin of
    :func:`~pedpredict.data.balance.balance_sequence_file`).

    Reads source crops straight from the ``chunk_*.lmdb`` under ``in_dir`` (no PIE frames needed — the
    incremental build deletes them), plans the oversampling from the stored labels, applies the
    single-transform copies, and writes ``preprocessed_train_aug``-style chunks (track_id preserved, so
    the runtime :class:`LMDBChunkDataset` accepts them). Returns the written chunk paths.
    """
    chunk_paths = iter_chunk_lmdbs(Path(in_dir))
    if not chunk_paths:
        raise FileNotFoundError(f"augment_lmdb_dir: no chunk_*.lmdb under {in_dir}")
    source = LMDBSampleSource(chunk_paths)
    items = plan_oversample(source.label_records, aug_cfg)
    print(f"[augment] {Path(in_dir).name} -> {Path(out_dir).name} | "
          f"plan: {summarize_plan(source.label_records, items)}")
    dataset = AugmentedSampleDataset(source, items, aug_cfg, data_cfg.source_width)
    return write_dataset_chunks_from(dataset, out_dir, data_cfg, num_workers=num_workers)

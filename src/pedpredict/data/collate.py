"""Batch collation for the runtime LMDB dataset (Prompt 1.5).

Ports OLD ``scripts/train_utils.py::collate_fn``. Stacks the per-sequence dicts from
:class:`~pedpredict.data.lmdb_dataset.LMDBChunkDataset` into the model's input tuple
``(images_tight, images_context, motions, labels)``.

B7 closure: the magic ``MAX_SEQ_LEN`` constant becomes :attr:`DataCfg.max_seq_len`, and the legacy
``motions[..., :8]`` slice is **deleted** — the writer (1.2) emits exactly ``motion_dim`` channels, so
the slice is a no-op; a cheap guard replaces it to fail loudly on a stale wider-motion LMDB.

Sequence-length policy: windows are fixed at ``seq_len`` frames by construction (generation drops short
tracks), so the ``[:max_seq_len]`` cap is a defensive truncation that is a no-op while
``seq_len <= max_seq_len``. There is no padding path — stacking requires a common ``T``.

``build_collate`` returns a ``functools.partial`` (picklable under Windows ``spawn``) so the collate can
ride along to DataLoader workers / the chunk prefetcher (4.2) without a non-picklable closure.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from torch import Tensor

from pedpredict.config.schema import DataCfg

__all__ = ["collate_sequences", "build_collate"]

_LABEL_KEYS = ("actions", "looks", "crosses")


def collate_sequences(
    batch: list[dict[str, Tensor]], *, max_seq_len: int, motion_dim: int
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Stack a list of per-sequence dicts into ``(images_tight, images_context, motions, labels)``."""
    images_tight = torch.stack([item["images_tight"][:max_seq_len] for item in batch], dim=0)
    images_context = torch.stack([item["images_context"][:max_seq_len] for item in batch], dim=0)
    motions = torch.stack([item["motions"][:max_seq_len] for item in batch], dim=0)
    if motions.shape[-1] != motion_dim:  # B7 guard: writer must emit exactly motion_dim channels
        raise ValueError(
            f"collate_sequences: motions has {motions.shape[-1]} channels, expected motion_dim="
            f"{motion_dim}. Re-run the LMDB writer (1.2) — the legacy [..., :8] slice was removed."
        )
    labels = {k: torch.stack([item[k] for item in batch], dim=0) for k in _LABEL_KEYS}
    return images_tight, images_context, motions, labels


def build_collate(cfg: DataCfg) -> Callable[[list[dict[str, Tensor]]], tuple]:
    """Bind ``max_seq_len`` / ``motion_dim`` from config into a picklable collate callable."""
    return functools.partial(collate_sequences, max_seq_len=cfg.max_seq_len, motion_dim=cfg.motion_dim)

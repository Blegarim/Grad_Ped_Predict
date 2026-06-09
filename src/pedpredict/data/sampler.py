"""Online imbalance lever + the single LMDB metadata scanner.

Unifies the **duplicated LMDB scan** in OLD ``train.py:34-123`` (band-aid B3). Two legacy functions
scanned the same ``_meta`` pickles for overlapping purposes:

* ``compute_class_weights_from_lmdb`` (train.py:34-72) — *global* inverse-frequency CE class weights.
* ``build_sampler_weights`` + ``_inverse_class_weights`` (train.py:74-123) — *per-chunk* per-sample
  ``WeightedRandomSampler`` weights with per-task powers (``crosses^1.5 · actions^0.3 · looks^0.7``).

The dedup target is the **scan**, not the math: :func:`scan_chunk_labels` is the ONE cursor pass, and
:class:`LabelScanCache` caches it per chunk (replacing the inline ``weight_cache`` dict at train.py:426).
The two inverse-frequency *formulas* legitimately differ (loss: ``t/(2·max(c,1))`` over fixed 2 classes;
sampler: ``t/(len(counts)·c)`` over observed classes) and are preserved verbatim — :func:`class_weights_ce`
and :func:`sample_weights` are byte-for-byte equivalents of the legacy bodies.

Imbalance policy (B3 — one policy, not three accidents; see CLAUDE.md / docs/archive/MIGRATION.md):

* **Lever 1** offline balance (``data/balance.py``) — OPT-IN, OFF by default.
* **Lever 2** online sampler (this module) — ON by default; per-chunk weights.
* **Lever 3** loss class weights (``losses/multitask.py``, Prompt 3.1) — ON by default; uses GLOBAL
  frequencies from :meth:`LabelScanCache.aggregate_counts`. Prompt 3.1 imports :func:`class_weights_ce`
  from here rather than re-scanning (the whole point of the dedup).

Behavior preserved vs OLD, with one flagged simplification: the scan applies a single canonical crosses
clamp (:func:`pedpredict.data.balance.clamp_cross`, ``==1 → 1 else 0``) instead of the two legacy clamps
(loss ``max(0,min(1,·))``; sampler ``<0 → 0``). All three coincide on in-contract data (writer guarantees
``crosses ∈ {0,1}``); they diverge only on the never-occurring value ``2`` (where the legacy sampler would
spuriously create a 3rd class via ``n_classes=len(counts)``). Zero effect on real data — parity stays exact.
"""

from __future__ import annotations

import pickle
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import lmdb
import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler

from pedpredict.config.schema import TrainCfg
from pedpredict.data.balance import clamp_cross

__all__ = [
    "TASKS",
    "TaskCounts",
    "ChunkLabelScan",
    "scan_chunk_labels",
    "LabelScanCache",
    "class_weights_ce",
    "sample_weights",
    "build_weighted_sampler",
]

#: Parity-critical task order (matches OLD ``train.py`` and the output contract).
TASKS: tuple[str, ...] = ("actions", "looks", "crosses")

#: Observed class -> count for one task (Counter-like; absent classes simply omitted).
TaskCounts = dict[int, int]


@dataclass(frozen=True, slots=True)
class ChunkLabelScan:
    """Result of ONE metadata pass over one LMDB chunk — feeds both imbalance levers."""

    lmdb_path: str
    seq_ids: list[str]                          # LMDB cursor (lexicographic) order — matches LMDBChunkDataset
    label_rows: list[tuple[int, int, int]]      # (actions, looks, crosses) clamped, in seq_ids order
    counts: dict[str, TaskCounts]               # per-task observed-class -> count

    @property
    def n(self) -> int:
        return len(self.label_rows)


# --------------------------------------------------------------------------- the one scanner


def scan_chunk_labels(lmdb_path: str | Path, seq_ids: Sequence[str] | None = None) -> ChunkLabelScan:
    """Single ``_meta`` cursor pass over one LMDB chunk (replaces both legacy scan loops, B3).

    ``seq_ids`` defaults to LMDB cursor order (identical to :class:`LMDBChunkDataset.seq_ids`); pass the
    dataset's ids explicitly to guarantee per-sample weight alignment. ``crosses`` is clamped with
    :func:`clamp_cross`; ``actions``/``looks`` are read as-is (already binary by the data contract).
    """
    path = str(lmdb_path)
    env = lmdb.open(path, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn:
            if seq_ids is None:
                ids = [
                    key.decode().split("_")[0]
                    for key, _ in txn.cursor()
                    if key.decode().endswith("_meta")
                ]
            else:
                ids = list(seq_ids)

            label_rows: list[tuple[int, int, int]] = []
            counts: dict[str, Counter] = {task: Counter() for task in TASKS}
            for seq_id in ids:
                meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
                actions = int(meta["actions"])
                looks = int(meta["looks"])
                crosses = clamp_cross(int(meta["crosses"]))
                label_rows.append((actions, looks, crosses))
                counts["actions"][actions] += 1
                counts["looks"][looks] += 1
                counts["crosses"][crosses] += 1
    finally:
        env.close()

    return ChunkLabelScan(
        lmdb_path=path,
        seq_ids=ids,
        label_rows=label_rows,
        counts={task: dict(counts[task]) for task in TASKS},
    )


@dataclass
class LabelScanCache:
    """Per-chunk :class:`ChunkLabelScan` cache keyed by ``lmdb_path`` (replaces train.py:426 weight_cache).

    One scan per chunk serves both levers: :meth:`get` for per-chunk sampler weights, and
    :meth:`aggregate_counts` for the GLOBAL class frequencies the loss lever (3.1) consumes.
    """

    _store: dict[str, ChunkLabelScan] = field(default_factory=dict)

    def get(self, lmdb_path: str | Path, seq_ids: Sequence[str] | None = None) -> ChunkLabelScan:
        """Return the cached scan for ``lmdb_path``, scanning once on a miss."""
        key = str(lmdb_path)
        scan = self._store.get(key)
        if scan is None:
            scan = scan_chunk_labels(key, seq_ids)
            self._store[key] = scan
        return scan

    def aggregate_counts(self, lmdb_paths: Sequence[str | Path]) -> dict[str, TaskCounts]:
        """Sum per-chunk counts across ``lmdb_paths`` into GLOBAL per-task frequencies (for loss weights)."""
        totals: dict[str, Counter] = {task: Counter() for task in TASKS}
        for path in lmdb_paths:
            scan = self.get(path)
            for task in TASKS:
                totals[task].update(scan.counts[task])
        return {task: dict(totals[task]) for task in TASKS}

    def clear(self) -> None:
        self._store.clear()


# --------------------------------------------------------------------------- loss lever


def class_weights_ce(counts: dict[str, TaskCounts], *, device: torch.device | str | None = None
                     ) -> dict[str, Tensor]:
    """Inverse-frequency CE class weights — EXACT ``compute_class_weights_from_lmdb`` (train.py:60-72).

    Per task: ``tensor([total/(2·max(c0,1)), total/(2·max(c1,1))])`` where ``total = c0 + c1``; an empty
    task falls back to ``[1.0, 1.0]``. Pass GLOBAL counts (``LabelScanCache.aggregate_counts``), not a
    single chunk's. Consumed by ``losses/multitask.py``.
    """
    weights: dict[str, Tensor] = {}
    for task in TASKS:
        task_counts = counts.get(task, {})
        c0 = int(task_counts.get(0, 0))
        c1 = int(task_counts.get(1, 0))
        total = c0 + c1
        if total > 0:
            weights[task] = torch.tensor(
                [total / (2 * max(c0, 1)), total / (2 * max(c1, 1))],
                dtype=torch.float32, device=device,
            )
        else:
            weights[task] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    return weights


# --------------------------------------------------------------------------- online sampler lever


def _inverse_class_weights(counts: TaskCounts) -> dict[int, float]:
    """Per-class inverse weight ``total/(n_classes·v)``, ``0.0`` if ``v==0`` — EXACT (train.py:74-83).

    ``n_classes = len(counts)`` counts only OBSERVED classes (Counter semantics), so a single-class chunk
    uses ``n_classes=1`` — the legacy quirk is intentionally preserved.
    """
    total = sum(counts.values())
    n_classes = len(counts)
    return {label: (0.0 if v == 0 else total / (n_classes * v)) for label, v in counts.items()}


def sample_weights(scan: ChunkLabelScan, powers: dict[str, float], min_weight: float) -> list[float]:
    """Per-sample ``WeightedRandomSampler`` weights for one chunk — EXACT ``build_sampler_weights`` body.

    For each sample: ``∏_task max(min_weight, invw[task][label]) ** powers[task]``, where a power of ``0``
    skips its task (the legacy ``if pow > 0`` guard). Returned list aligns with ``scan.seq_ids`` order.
    """
    cross_pow = powers["crosses"]
    action_pow = powers["actions"]
    look_pow = powers["looks"]
    action_w = _inverse_class_weights(scan.counts["actions"])
    look_w = _inverse_class_weights(scan.counts["looks"])
    cross_w = _inverse_class_weights(scan.counts["crosses"])

    weights: list[float] = []
    for actions, looks, crosses in scan.label_rows:
        weight = max(min_weight, cross_w.get(crosses, min_weight)) ** cross_pow
        if action_pow > 0:
            weight *= max(min_weight, action_w.get(actions, min_weight)) ** action_pow
        if look_pow > 0:
            weight *= max(min_weight, look_w.get(looks, min_weight)) ** look_pow
        weights.append(weight)
    return weights


def build_weighted_sampler(scan: ChunkLabelScan, cfg: TrainCfg) -> WeightedRandomSampler:
    """Build the online ``WeightedRandomSampler`` for one chunk from ``TrainCfg`` (replaces train.py:439).

    Powers and ``min_weight`` come from config; ``replacement=True`` and ``num_samples=len(weights)``
    reproduce the legacy sampler exactly.
    """
    weights = sample_weights(scan, cfg.sampler_powers, cfg.sampler_min_weight)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

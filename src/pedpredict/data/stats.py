"""Dataset label statistics / drift check.

The data layer's verification tool: aggregate per-split positive-class rates and diff them against the
documented canonical table so drift fails loudly (CI-friendly). It adds **no new scanner** — it reuses the
single 1.6 metadata pass (:func:`pedpredict.data.sampler.scan_chunk_labels` via
:class:`~pedpredict.data.sampler.LabelScanCache`), satisfying "don't add a third scanner" (B3).

Scope (the canonical table = *generated-sequence* distribution): a split's stats come from its **base**
LMDB dir(s) — ``train`` from ``lmdb_train[0]`` only. ``preprocessed_train_aug`` intentionally redistributes
classes (the imbalance lever, policy 1.3/1.4), so it is excluded by default and bypasses the drift gate
when opted in. Because the base-train LMDB is a deterministic 1:1 image of ``sequences_train.pkl`` (writer
applies no filter), these counts reproduce the 1.1 fixture exactly — drift is EXACT integer equality.

Changes vs OLD ``label_count.py`` (flagged, see docs/archive/MIGRATION.md): per-chunk rows -> per-split aggregate; the
``crosses[irrelevant]`` (-1) column is dropped (crosses are clamped to {0,1} at 1.1); the legacy ``label_counts``
Counter helper is deleted in favour of the 1.6 scanner.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pedpredict.data.sampler import TASKS, LabelScanCache, TaskCounts
from pedpredict.paths import ResolvedPaths

__all__ = [
    "SplitStats",
    "iter_chunk_lmdbs",
    "split_lmdb_dirs",
    "compute_split_stats",
    "compute_dataset_stats",
    "format_table",
    "write_stats_csv",
    "load_reference",
    "check_drift",
]

_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class SplitStats:
    """Aggregate label counts for one split (positive rate per task = the drift-checked quantity)."""

    split: str
    n: int
    counts: dict[str, TaskCounts]            # task -> {0: ..., 1: ...} (clamped, from the 1.6 scan)

    @property
    def pos(self) -> dict[str, int]:
        """Positive-class (label==1) count per task."""
        return {task: int(self.counts.get(task, {}).get(1, 0)) for task in TASKS}

    @property
    def pos_rate(self) -> dict[str, float]:
        """Positive-class fraction per task (``0.0`` when the split is empty)."""
        return {task: (c / self.n if self.n else 0.0) for task, c in self.pos.items()}


def iter_chunk_lmdbs(lmdb_dir: Path) -> list[Path]:
    """Sorted ``chunk_*.lmdb`` envs in one split dir (lexicographic order == write order)."""
    return sorted(Path(lmdb_dir).glob("chunk_*.lmdb"))


def split_lmdb_dirs(paths: ResolvedPaths, *, include_aug: bool = False) -> dict[str, tuple[Path, ...]]:
    """Canonical split -> base LMDB dir(s). ``train`` is ``lmdb_train[0]`` unless ``include_aug``."""
    train = paths.lmdb_train if include_aug else paths.lmdb_train[:1]
    return {"train": tuple(train), "val": (paths.lmdb_val,), "test": (paths.lmdb_test,)}


def compute_split_stats(split: str, dirs: Sequence[Path],
                        cache: LabelScanCache | None = None) -> SplitStats:
    """Aggregate one split's per-task counts over every chunk in ``dirs`` (raises if none found)."""
    cache = cache or LabelScanCache()
    chunks = [p for d in dirs for p in iter_chunk_lmdbs(d)]
    if not chunks:
        raise FileNotFoundError(f"[{split}] no chunk_*.lmdb under {[str(d) for d in dirs]}")
    counts = cache.aggregate_counts([str(p) for p in chunks])
    n = sum(counts.get(TASKS[0], {}).values())   # every sample increments each task once -> total == N
    return SplitStats(split=split, n=n, counts=counts)


def compute_dataset_stats(paths: ResolvedPaths, *, include_aug: bool = False,
                          splits: Sequence[str] = _SPLITS,
                          skip_missing: bool = False) -> list[SplitStats]:
    """Stats for each requested split. With ``skip_missing`` a split lacking chunks is omitted, not raised."""
    cache = LabelScanCache()
    dirs = split_lmdb_dirs(paths, include_aug=include_aug)
    out: list[SplitStats] = []
    for split in splits:
        try:
            out.append(compute_split_stats(split, dirs[split], cache))
        except FileNotFoundError:
            if not skip_missing:
                raise
    return out


def format_table(stats: Sequence[SplitStats]) -> str:
    """Render the CLAUDE.md-form markdown table (``Split | N | actions=1 | looks=1 | crosses=1``)."""
    header = "| Split | N | actions=1 | looks=1 | crosses=1 |"
    sep = "|---|---|---|---|---|"
    rows = [
        f"| {s.split} | {s.n} | "
        + " | ".join(f"{s.pos_rate[t] * 100:.1f}%" for t in TASKS)
        + " |"
        for s in stats
    ]
    return "\n".join([header, sep, *rows])


def write_stats_csv(stats: Sequence[SplitStats], csv_path: Path) -> None:
    """Write per-split counts + rates to ``csv_path`` (parent created if absent)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "N", *(f"{t}_pos" for t in TASKS), *(f"{t}_pos_rate" for t in TASKS)]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for s in stats:
            writer.writerow(
                [s.split, s.n, *(s.pos[t] for t in TASKS), *(round(s.pos_rate[t], 6) for t in TASKS)]
            )


def load_reference(fixture: Path) -> dict[str, dict[str, int]]:
    """Load the canonical per-split counts from the 1.1 golden (``N`` + per-task positive counts)."""
    with open(fixture, encoding="utf-8") as handle:
        return json.load(handle)["splits"]


def check_drift(stats: Sequence[SplitStats], reference: dict[str, dict[str, int]]) -> list[str]:
    """Return EXACT-mismatch messages vs the reference table (empty == no drift). Compares N + the 3
    positive counts; an unreferenced split is reported as such."""
    messages: list[str] = []
    for s in stats:
        ref = reference.get(s.split)
        if ref is None:
            messages.append(f"[{s.split}] no reference entry to diff against")
            continue
        if s.n != ref["N"]:
            messages.append(f"[{s.split}] N drift: got {s.n}, expected {ref['N']}")
        for task in TASKS:
            got, exp = s.pos[task], int(ref[task])
            if got != exp:
                messages.append(f"[{s.split}] {task}=1 drift: got {got}, expected {exp}")
    return messages

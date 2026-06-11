"""Effective training-distribution instrument (hole M1).

The imbalance levers (offline augmentation x online sampler x loss class weights) compose into a
training distribution that nothing in the repo measured — toggling levers was flying blind. This
module computes, per task, the positive fraction an epoch of sampler draws is EXPECTED to contain,
next to the base rate actually stored in the chunks, and writes the result into every run dir as
``train_distribution.json`` (hooked in ``build_trainer``; standalone via ``scripts/report_distribution.py``).

No simulation noise: ``WeightedRandomSampler(replacement=True)`` draws sample ``i`` with probability
``w_i / Σw``, so the expected positive fraction of one chunk's draws is exactly ``Σ(w_i·y_i) / Σw_i``.
Chunks contribute in proportion to their length (the sampler's per-chunk ``num_samples``). With the
sampler off, the effective rate equals the base rate (plain shuffle). Weights reuse the canonical
:func:`~pedpredict.data.sampler.sample_weights` math via the shared :class:`LabelScanCache`, so the
instrument can never drift from what training actually does.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pedpredict.config.schema import TrainCfg
from pedpredict.data.sampler import (
    TASKS,
    LabelScanCache,
    class_weights_ce,
    sample_weights,
)

__all__ = ["DISTRIBUTION_FILENAME", "effective_distribution", "write_distribution_report"]

#: Canonical per-run artifact name (lands next to resolved_config.yaml / train_log.csv).
DISTRIBUTION_FILENAME = "train_distribution.json"


def effective_distribution(
    train_lmdb_paths: Sequence[str | Path],
    cfg: TrainCfg,
    *,
    scan_cache: LabelScanCache | None = None,
) -> dict[str, object]:
    """Expected per-task positive rate of one epoch of draws vs. the base rate in the chunks.

    Returns a json-safe dict: ``base_rate`` (stored label frequencies over all chunks),
    ``effective_rate`` (expected positive fraction of sampler draws under ``cfg``), the lever
    settings that produced it, and the inverse-frequency CE ``class_weights`` lever 3 would apply
    (reported even when ``use_class_weights=false``, so the run dir records the whole stack).
    """
    cache = scan_cache if scan_cache is not None else LabelScanCache()
    n_total = 0
    base_pos = dict.fromkeys(TASKS, 0)            # Σ positives stored, per task
    drawn_pos = dict.fromkeys(TASKS, 0.0)         # Σ expected positives drawn, per task

    for path in train_lmdb_paths:
        scan = cache.get(str(path))
        n = scan.n
        if n == 0:
            continue
        n_total += n
        if cfg.use_weighted_sampler:
            weights = sample_weights(scan, cfg.sampler_powers, cfg.sampler_min_weight)
            w_sum = sum(weights)
        else:
            weights, w_sum = None, float(n)
        for row_idx, labels in enumerate(scan.label_rows):
            w = weights[row_idx] if weights is not None else 1.0
            for task_idx, task in enumerate(TASKS):
                y = labels[task_idx]
                base_pos[task] += y
                # n draws per chunk, each positive with probability Σ(w·y)/Σw
                drawn_pos[task] += n * (w * y) / w_sum if w_sum > 0 else 0.0

    if n_total == 0:
        raise ValueError("effective_distribution: no samples found in the given train chunks.")

    counts = cache.aggregate_counts([str(p) for p in train_lmdb_paths])
    return {
        "n_samples": n_total,
        "n_chunks": len(train_lmdb_paths),
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "sampler_powers": dict(cfg.sampler_powers),
        "use_class_weights": cfg.use_class_weights,
        "class_weights": {t: [round(float(v), 4) for v in w] for t, w in class_weights_ce(counts).items()},
        "base_rate": {t: round(base_pos[t] / n_total, 6) for t in TASKS},
        "effective_rate": {t: round(drawn_pos[t] / n_total, 6) for t in TASKS},
    }


def write_distribution_report(
    run_dir: str | Path,
    train_lmdb_paths: Sequence[str | Path],
    cfg: TrainCfg,
    *,
    scan_cache: LabelScanCache | None = None,
) -> Path:
    """Compute :func:`effective_distribution` and write ``<run_dir>/train_distribution.json``."""
    report = effective_distribution(train_lmdb_paths, cfg, scan_cache=scan_cache)
    path = Path(run_dir) / DISTRIBUTION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    eff = report["effective_rate"]
    base = report["base_rate"]
    print(
        "[train_distribution] expected positive rate of sampler draws vs stored rate — "
        + ", ".join(f"{t}: {eff[t]:.1%} (base {base[t]:.1%})" for t in TASKS)  # type: ignore[index]
    )
    return path

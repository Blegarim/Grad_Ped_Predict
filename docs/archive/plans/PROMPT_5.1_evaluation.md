# Sub-plan — Prompt 5.1: Evaluation pipeline (`eval/evaluate.py` + `scripts/evaluate.py`)

> Phase-A, behavior-preserving port of `OLD/Undergrad_thesis_project/test.py`.
> **Deliverable: a detailed sub-plan** (skeletons + signatures), not final production code.
> Status of dependencies: **all green** — 1.5 (dataset/collate), 2.4 (registry), 3.2 (metrics),
> 4.2/4.3/4.5 (chunk loader, checkpoints, logging/run-dir), 0.2 (config) are implemented.

---

## 0. Scope & guiding principle

`OLD/test.py` is a 619-line god-script that conflates **five** concerns:

1. metric computation (sklearn calls, AUC, threshold sweep) — `evaluate()`, `find_optimal_thresholds()`;
2. **efficiency** measurement (FLOPs / latency / FPS) — `compute_flops()`, `inference_latency()`;
3. **legacy-weight loading** fix-ups — `_init_global_rel_pos_from_ckpt()`, `_infer_window_hw()`;
4. data loading (per-chunk `DataLoader` loop) + model build (stringly `get_model`/`model_forward`);
5. artifact writing (ad-hoc `training_log/test_log_*.csv`, `plots/temporal_weights.npz`, `shutil.copy2`
   model-type suffix).

The rebuild keeps **only concern #1's orchestration and #4–#5** in `eval/evaluate.py`, and delegates the
*math* of #1 to the already-golden `MetricAccumulator` (3.2). The other concerns are routed to their
owning modules:

| OLD `test.py` piece | Rebuild home | Why |
|---|---|---|
| `evaluate()` sklearn block | **reuse** `MetricAccumulator.compute()` (3.2) | B1 — no second metric path |
| `find_optimal_thresholds()` | **reuse** `MetricAccumulator.optimal_threshold_metrics(EvalCfg)` (3.2) | already ported |
| `compute_flops()`, `inference_latency()` | **5.2** `eval/benchmark.py` | efficiency is its own prompt |
| `_init_global_rel_pos_from_ckpt`, `_infer_window_hw` | **Prompt 9** `load_legacy_model_weights()` | B2 fixed → new ckpts need no fix-up |
| `get_model`, `model_forward` | **reuse** `registry.build_model` / `forward_model` (2.4) | B10 |
| per-chunk `DataLoader` loop | **reuse** `LMDBChunkDataset.from_config` + `build_collate` (1.5) | B7/B11 |
| `test_log_*.csv`, `shutil.copy2` suffix | **reuse** `RunDir.eval_logger` + `index.csv` (4.5) | B11 |
| `round_metric` | **reuse** `logging.round_row` | dedup |

This keeps `evaluate.py` a thin orchestrator: *build → iterate → accumulate → compute → write*.

---

## 1. Target files & public API

### 1a. `src/pedpredict/eval/evaluate.py`

```python
"""Test-set evaluation (Prompt 5.1) — ports OLD test.py's evaluate()/main() metric path.

Reuses the SHARED MetricAccumulator (3.2) — no second sklearn path (B1); the typed registry
build_model/forward_model (2.4) — no stringly dispatch (B10); LMDBChunkDataset + build_collate (1.5);
and the RunDir / eval_log.csv / index.csv conventions (4.5) — no flat training_log writes (B11).
Efficiency (FLOPs/latency/FPS) lives in 5.2 benchmark.py; legacy-weight fix-ups in Prompt 9.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pedpredict.config.schema import DataCfg, EvalCfg, RootCfg
from pedpredict.data.collate import build_collate
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.losses.multitask import TASKS
from pedpredict.models.registry import ModelType, build_model, forward_model
from pedpredict.training.metrics import METRIC_COLUMNS, MetricAccumulator, MetricResult
from pedpredict.utils.amp import autocast_ctx, resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.logging import RunDir, build_index_row, append_index_row, init_run, round_row
from pedpredict.utils.memory import free_cuda

__all__ = [
    "EVAL_LOG_COLUMNS",
    "EvalArtifacts",
    "EvalReport",
    "evaluate_model",
    "run_evaluation",
    "load_eval_weights",
    "save_predictions_npz",
    "save_temporal_weights_npz",
]

# Per-task threshold-sweep columns (eval-only enrichment; 3.2 optimal_threshold_metrics).
_OPT_SUFFIXES: tuple[str, ...] = ("threshold", "acc", "f1", "precision", "recall")
# Efficiency columns (filled by 5.2 benchmark when run, else blank). OQ2: composed HERE.
_EFFICIENCY_COLUMNS: tuple[str, ...] = ("params", "flops_per_frame", "latency_ms_per_frame", "fps", "peak_vram_mb")

#: WIDE eval-log schema (OQ2): context + shared METRIC_COLUMNS (default-0.5) + opt_* + efficiency.
EVAL_LOG_COLUMNS: tuple[str, ...] = (
    ("timestamp", "checkpoint", "model_type", "split", "n_samples")
    + METRIC_COLUMNS
    + tuple(f"opt_{t}_{s}" for t in TASKS for s in _OPT_SUFFIXES)
    + tuple(f"opt_overall_acc",)  # mean-of-per-task acc at optimal thresholds (test.py:495-497)
    + _EFFICIENCY_COLUMNS
)


@dataclass(frozen=True)
class EvalArtifacts:
    """Pure in-memory result of one evaluation pass (no I/O)."""

    metrics: MetricResult                       # shared MetricAccumulator.compute() — default 0.5 thresh
    optimal: dict[str, float]                    # optimal_threshold_metrics(EvalCfg) — opt thresholds + metrics
    n_samples: int
    predictions: dict[str, np.ndarray] | None    # {task_true, task_prob0, task_prob1, task_pred} flat arrays
    temporal_weights: np.ndarray | None          # [N, T] softmax weights — FULL model only (else None)


@dataclass(frozen=True)
class EvalReport:
    """`EvalArtifacts` + the on-disk artifact paths (returned by `run_evaluation`)."""

    artifacts: EvalArtifacts
    run_dir: Path
    eval_log_path: Path
    predictions_path: Path | None
    temporal_weights_path: Path | None


# --------------------------------------------------------------------------- core pass (no I/O)

def evaluate_model(
    model: nn.Module,
    loaders: Iterable[DataLoader],
    device: torch.device,
    cfg: EvalCfg,
    *,
    use_amp: bool = False,
    collect_predictions: bool = False,
    collect_temporal_weights: bool = False,
    progress: bool = False,
) -> EvalArtifacts:
    """Run model over every loader; return metrics (+ optional per-sample preds / temporal weights).

    Single shared `MetricAccumulator` over ALL chunks (== OLD aggregated `all_*_global` metrics).
    `crosses` is scored on `crosses_frame` once (B4, via TASK_OUTPUT_KEY inside the accumulator).
    `temporal_weights` is collected from `outputs["temporal_weights"]` only when present (full model).
    """
    ...


# --------------------------------------------------------------------------- orchestration (I/O)

def run_evaluation(
    cfg: RootCfg,
    *,
    checkpoint: str | Path,
    device: torch.device | None = None,
    split: str = "test",
    save_predictions: bool = False,
    save_temporal_weights: bool = False,
    efficiency: dict[str, float] | None = None,
    strict: bool = True,
) -> EvalReport:
    """Build model (cfg.eval.model_type) → load `checkpoint` → eval over `split` chunks → write artifacts.

    Run-dir (OQ7): reuse the checkpoint's `outputs/runs/{run_id}/` when the path is resolvable, else a
    fresh `init_run(cfg, kind="eval")` dir. Writes `eval_log.csv` (EVAL_LOG_COLUMNS), appends one
    `index.csv` row (`kind="eval"`), and optionally `plots/predictions.npz` + `plots/temporal_weights.npz`.
    `efficiency` (from 5.2) is folded into the CSV row; absent → blanks.
    """
    ...


# --------------------------------------------------------------------------- helpers

def load_eval_weights(
    model: nn.Module, checkpoint: str | Path, *, device: torch.device, strict: bool = True
) -> None:
    """Load model weights for eval (model-only; no optimizer/scheduler needed).

    Accepts a rebuilt `CheckpointManager` payload (extracts `model_state_dict`, strict load — B2) OR a
    bare `state_dict`. Legacy (pre-B2) raw weights that need `relative_position_bias` fix-up are routed
    to `Prompt 9`'s `load_legacy_model_weights()` (raises a clear pointer here if `strict` fails on them).
    """
    ...


def _eval_chunk_loaders(cfg: RootCfg, chunk_paths: list[str]) -> Iterator[DataLoader]:
    """Yield a stable-order, no-sampler `DataLoader` per chunk (OLD test.py:393-405); gc between (B11)."""
    ...


def save_predictions_npz(path: Path, predictions: dict[str, np.ndarray]) -> Path:
    """Save per-sample {true, prob0, prob1, pred} per task (replaces OLD predictions CSV; NPZ for viz 6.2)."""
    ...


def save_temporal_weights_npz(
    path: Path, temporal_weights: np.ndarray, predictions: dict[str, np.ndarray] | None
) -> Path:
    """Save `[N, T]` temporal weights (+ `{task}_true`) for the attention viz phase (OLD test.py:595-608)."""
    ...
```

### 1b. `scripts/evaluate.py` (thin CLI, mirrors `scripts/train.py`)

```python
"""Evaluation entry point (Prompt 5.1).

    python scripts/evaluate.py --set eval.model_type=full \
        --checkpoint outputs/runs/<run_id>/checkpoints/best.pth
    python scripts/evaluate.py --set eval.model_type=motion_only --checkpoint <path> --save-predictions
"""
from __future__ import annotations

import multiprocessing as mp
import sys

from pedpredict.config import build_argparser, load_config
from pedpredict.eval.evaluate import run_evaluation
from pedpredict.utils.device import get_device


def main(argv=None) -> int:
    parser = build_argparser()                      # reuse --config-dir / --set dotted overrides
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-temporal-weights", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="also run 5.2 efficiency metrics")
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)
    device = get_device()

    efficiency = None
    if args.benchmark:                              # optional 5.2 hook; absent until 5.2 lands
        from pedpredict.eval.benchmark import measure_efficiency
        efficiency = measure_efficiency(cfg, device=device)

    report = run_evaluation(
        cfg, checkpoint=args.checkpoint, device=device, split=args.split,
        save_predictions=args.save_predictions,
        save_temporal_weights=args.save_temporal_weights,
        efficiency=efficiency,
    )
    print(f"Eval complete → {report.eval_log_path}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
```

---

## 2. Step-by-step port procedure (referencing OLD `test.py`)

1. **Core loop** (OLD `evaluate()` lines 31–104) → `evaluate_model()`:
   - Move batch to device + `labels[...].long()` + `clamp(crosses, 0, 1)` — the 1.6/`remap_cross_labels`
     contract (identical to `Trainer._move_batch`, trainer.py:230-238; replicated inline so eval has no
     `training` dependency).
   - `with autocast_ctx(use_amp): outputs = forward_model(model, *batch[:3])` (replaces
     `model_forward(model, model_type, ...)`; `model_type` is intrinsic to the model — 2.4/B10).
   - `acc.update(outputs, labels)` — **deletes** OLD lines 51–72 (manual per-head softmax/argmax/accuracy
     accumulation) **and** lines 74–104 (the sklearn block); `MetricAccumulator` owns all of it (B1).
   - When `collect_temporal_weights` and `"temporal_weights" in outputs`: append
     `outputs["temporal_weights"].float().cpu()` (OLD lines 48–49). Full-model-only (2.4 decision).
   - Use `torch.inference_mode()` (Trainer.validate parity) instead of `torch.no_grad()`.
2. **Aggregation across chunks** (OLD `main()` lines 382–470): the single accumulator already aggregates
   globally — the OLD per-chunk concat bookkeeping (`all_*_global`) **vanishes**. One `compute()` at the
   end yields the global `MetricResult`; `optimal_threshold_metrics(cfg.eval)` yields the sweep (OLD lines
   472–497). No re-implementation.
3. **Per-sample predictions** (OLD lines 555–593): instead of a second accumulation, extract directly from
   the accumulator via a new public `MetricAccumulator.task_arrays(task)` (see §3) → build the flat
   `predictions` dict → `save_predictions_npz` (NPZ, not CSV — friendlier for viz 6.2; the OLD wide CSV
   columns are reproducible from it).
4. **Temporal weights** (OLD lines 595–608): `np.concatenate` the collected chunks →
   `save_temporal_weights_npz` under `run_dir/plots/` (B11; OLD wrote `plots/` at repo root).
5. **Model build + load** (OLD lines 356–366): `build_model(cfg, cfg.eval.model_type)` then
   `load_eval_weights(...)` (strict for rebuilt ckpts; legacy fix-up deferred to Prompt 9). Drops
   `_init_global_rel_pos_from_ckpt`/`_infer_window_hw` (B2 dead).
6. **Data** (OLD lines 372–405): `gather_lmdb_chunks([resolve_paths(cfg.paths).lmdb_test])` →
   `_eval_chunk_loaders` building `LMDBChunkDataset.from_config(path, cfg.data)` + `build_collate(cfg.data)`,
   `shuffle=False`, `cfg.eval.batch_size/num_workers`, `pin_memory=(device=='cuda')`, gc between (OLD 432-435).
7. **CSV / index** (OLD lines 335–350, 530–545, 610–616): one **aggregate** row to
   `RunDir.eval_logger(EVAL_LOG_COLUMNS)` (rounded via `round_row`); append an `index.csv` row with
   `kind="eval"` carrying the headline `crosses_f1/…` (replaces the `shutil.copy2` model-suffix hack — the
   `model_type` column + run-id encode it; B11).
8. **Console summary** (OLD lines 499–553): a small `_print_summary(metrics, optimal)` helper — cosmetic,
   kept for parity of the printed table.
9. **Efficiency**: `run_evaluation` accepts an optional `efficiency` dict from 5.2 and folds it into the
   row; `evaluate.py` itself never imports `fvcore` (concern relocated).

---

## 3. One small, justified cross-prompt edit to `training/metrics.py` (3.2)

`evaluate_model` needs per-sample `(y_true, y_pred, y_prob)` to build the predictions NPZ. The accumulator
**already stores** these (`_probs`/`_targets`) — re-accumulating in eval would duplicate state and risk
divergence (against the 3.2 "single store" principle). So **promote the existing private
`_task_arrays(task)` to public `task_arrays(task)`** (one-line rename + `__all__`/docstring; the method body
is unchanged and already golden-tested). No behavior change; it just exposes what eval needs from the
canonical store. Flag in MIGRATION.md under "Metrics decisions (3.2)".

---

## 4. Band-aids removed (and how)

| Band-aid | How 5.1 removes it |
|---|---|
| **B1** (eval/metric duplication) | `evaluate()`'s sklearn block + threshold sweep deleted; `MetricAccumulator` (3.2) is the *only* metric path, shared with training-validation. FLOPs/latency removed (→5.2). Ad-hoc CSV writer → `RunDir`/`CsvLogger` (4.5). |
| **B10** (stringly dispatch) | `get_model(str)`/`model_forward(model, str, …)` → `build_model(cfg, ModelType)` + `forward_model(model, …)`; `model_type` is intrinsic to the module (no separately-threaded string; typos raise via `ModelType.coerce`). |
| **B8** (scattered `.float()`) | The lone upcast is `to_float_logits` inside `MetricAccumulator.update`; eval adds no `.float()` except the documented `temporal_weights.float().cpu()` collection. |
| **B11** (artifact sprawl) | All outputs under the per-run dir (`eval_log.csv`, `plots/predictions.npz`, `plots/temporal_weights.npz`); cross-run `index.csv` row instead of `shutil.copy2` suffix files; no `training_log/` writes. |
| **B4** (dead crosses head) | crosses scored on `crosses_frame` only (TASK_OUTPUT_KEY); `crosses_pooled`/`temporal_weights` never scored — inherited from 3.2, asserted by `test_crosses_scored_on_frame_not_pooled`. |
| **B2** (load side) | `load_eval_weights` strict-loads rebuilt ckpts (all ViT params eager); `_init_global_rel_pos_from_ckpt` dropped. |

---

## 5. Golden fixtures & test list (behavior preservation)

New file `tests/test_eval.py`; capture script `tests/_capture/capture_eval_golden.py` →
`tests/fixtures/golden/eval_cases.pt`.

**Golden capture (from OLD `test.py`):** run OLD `evaluate()` over a tiny synthetic 2-chunk loader using
the **existing golden ensemble weights** (`tests/fixtures/golden/ensemble.pt`) on CPU, AMP off; store
`{metrics_dict, temporal_weights, y_true/y_pred/y_prob per task, optimal_thresholds}`. (The metric *values*
are already golden via `metrics_cases.pt`; this fixture pins the **eval-level orchestration** — cross-chunk
aggregation, temporal-weight collection, predictions extraction, threshold-sweep wiring.)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_evaluate_matches_legacy_oracle` | `evaluate_model` metrics == OLD `evaluate()` (atol 1e-6) over the golden ensemble + synthetic chunks. |
| 2 | `test_temporal_weights_collected_full_model` | full model → `[N, T]` array; equals OLD-collected weights. |
| 3 | `test_temporal_weights_none_for_ablations` | `motion_only`/`visual_only`/`vanilla_concat` → `temporal_weights is None` (2.4 contract). |
| 4 | `test_crosses_scored_on_frame_not_pooled` | perturbing `crosses_pooled`/`temporal_weights` does not change metrics (B4). |
| 5 | `test_predictions_npz_shape_and_contents` | NPZ keys + shapes; `pred == argmax(prob)`; round-trips OLD wide-CSV columns. |
| 6 | `test_eval_log_schema` | written `eval_log.csv` header == `EVAL_LOG_COLUMNS`; one aggregate row; floats rounded. |
| 7 | `test_index_row_kind_eval` | `run_evaluation` appends an `index.csv` row with `kind="eval"` + headline `crosses_f1`. |
| 8 | `test_runs_for_all_four_model_types` | `evaluate_model` runs a forward + `compute()` for every `ModelType` on a dummy batch (B10). |
| 9 | `test_load_eval_weights_strict_roundtrip` | a rebuilt-format ckpt strict-loads into a fresh model (no forward; B2). |
| 10 | `test_aggregation_equals_single_pass` | metrics over 2 chunks == metrics over their concatenation (aggregation correctness). |
| 11 | **SMOKE** `test_run_evaluation_smoke_tiny_chunk` | end-to-end on a tiny real LMDB chunk (`tmp_path`): finite metrics, `eval_log.csv` + NPZs exist, no leaked child processes. |

**Smoke-test construction** (test 11): reuse the LMDB-roundtrip test's tiny-chunk writer
(`tests/test_lmdb_roundtrip.py`) to write a ~6-sample chunk to `tmp_path`; point `cfg.paths.lmdb_test`
at it via `dataclasses.replace`; `model = build_model(RootCfg())` (random init — values irrelevant for the
smoke); `run_evaluation(cfg, checkpoint=<saved tmp state_dict>)`; assert artifacts + `set(mp.active_children())`
unchanged (mirrors `test_trainer.py::test_fit_smoke_one_tiny_chunk`). For the **pure** `evaluate_model`
tests (1–10), pass an in-memory `list[list[Batch]]` as `loaders` (a list is a valid `Iterable[DataLoader]`
for iteration), exactly like `_ListChunkProvider` in `test_trainer.py`.

---

## 6. Output artifacts (written contract)

```
outputs/runs/{run_id}/                  # reused ckpt run-dir, else {ts}_{model_type}_eval (OQ7)
  eval_log.csv                          # 1 aggregate row, header == EVAL_LOG_COLUMNS (OQ2 WIDE)
  plots/
    predictions.npz                     # optional (--save-predictions): per-task true/prob0/prob1/pred
    temporal_weights.npz                # optional (--save-temporal-weights): [N,T] + {task}_true (full only)
outputs/runs/index.csv                  # + 1 row, kind="eval", crosses_f1-led headline
```

`predictions.npz` keys: `{task}_true`, `{task}_prob_0`, `{task}_prob_1`, `{task}_pred` for each task
(int/float arrays, length N). `temporal_weights.npz` keys: `temporal_weights [N,T]` + `{task}_true`
(consumed by 6.2 attention overlay). Both supersede OLD's `predictions_*.csv` / root `plots/temporal_weights.npz`.

---

## 7. Risks & open questions (confirm before coding)

1. **Per-chunk CSV rows.** OLD wrote one row *per chunk* + an aggregate block. **Proposed:** the canonical
   `eval_log.csv` holds a **single aggregate row** (the number the thesis reports); per-chunk rows become an
   opt-in (`EvalCfg.log_per_chunk`, default off) if useful for debugging. Confirm we don't need per-chunk
   rows by default.
2. **Efficiency columns now or later.** `EVAL_LOG_COLUMNS` includes the 5 efficiency columns per OQ2, filled
   only when `--benchmark` runs (else blank). Confirm including-but-blank is acceptable (vs deferring the
   columns until 5.2). Keeps the schema stable across runs.
3. **Legacy (ported) weight loading for cutover parity (9.1).** `load_eval_weights` natively handles the
   rebuilt `CheckpointManager` format. Loading *OLD* `.pth` weights for the parity gate needs
   `_init_global_rel_pos_from_ckpt` — deferred to `Prompt 9 load_legacy_model_weights()`. Confirm 5.1 need
   not port that fix-up now (i.e., parity-vs-OLD is run under Prompt 9, not here).
4. **Run-dir reuse heuristic (OQ7).** Detect the ckpt's run-dir by walking parents for an `outputs/runs/{id}`
   ancestor; fall back to `init_run(kind="eval")`. Confirm the detection rule (parent-of-`checkpoints/`).
5. **Promoting `MetricAccumulator._task_arrays` → public.** Small edit to the 3.2 module (rename only).
   Confirm this is preferred over re-accumulating predictions in eval (it is, for single-source-of-truth).
6. **`val` split evaluation.** `--split val` is offered (reuses `lmdb_val`) for sanity; confirm it's wanted
   or restrict to `test`.
7. **AMP at eval.** OLD set `use_amp = (device=='cuda')`. Proposed: `resolve_amp(cfg.train.use_amp, device)`
   for parity. Numerically eval is AMP-tolerant (metrics upcast in `update`); confirm gating source
   (`train.use_amp` vs a new `eval.use_amp`).

---

## 8. Coupling notes (keep the contract singular)

- Honors the **output-contract** siblings (2.3/2.4/2.5/3.1/3.2): crosses→`crosses_frame`,
  `crosses_pooled` unsupervised, `temporal_weights` full-model-only.
- Reuses **4.5** logging primitives verbatim (`RunDir`, `EVAL_LOG_FILENAME`, `eval_logger`,
  `build_index_row(kind="eval")`, `round_row`) — no new logging machinery.
- The **5.2** seam is a single optional `efficiency: dict` argument; `evaluate.py` never imports `fvcore`.
- The **6.2** seam is the two NPZ artifacts (predictions + temporal weights) written under `run_dir/plots/`.
```

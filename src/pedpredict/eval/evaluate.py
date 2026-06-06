"""Test-set evaluation (Prompt 5.1) — ports OLD ``test.py`` (``evaluate()`` + ``main()`` metric path).

A thin orchestrator: *build model -> iterate chunks -> accumulate -> compute -> write artifacts*. The
*math* is delegated to already-golden components, so this module adds none of its own:

* metrics: the SHARED :class:`~pedpredict.training.metrics.MetricAccumulator` (3.2) — no second sklearn
  path (B1); also owns the eval-only F1-optimal threshold sweep (ports OLD ``find_optimal_thresholds``).
* model: the typed :func:`~pedpredict.models.registry.build_model` / ``forward_model`` (2.4) — no stringly
  ``get_model``/``model_forward`` dispatch, ``model_type`` is intrinsic to the module (B10).
* data: :class:`~pedpredict.data.lmdb_dataset.LMDBChunkDataset` + ``build_collate`` (1.5).
* logging: the per-run :class:`~pedpredict.utils.logging.RunDir` + ``eval_log.csv`` + cross-run ``index.csv``
  (4.5) — no flat ``training_log/`` writes, no ``shutil.copy2`` model-suffix files (B11).

Deliberately OUT of scope (relocated to their owning prompts): efficiency metrics — FLOPs / latency / FPS —
live in 5.2 ``eval/benchmark.py`` (folded into the CSV row here via the optional ``efficiency`` arg);
legacy ``relative_position_bias`` checkpoint fix-ups (OLD ``_init_global_rel_pos_from_ckpt``) are dead (B2
fixed in 2.1) and any true OLD-weight migration is Prompt 9's ``load_legacy_model_weights``.

Output contract honored (B4, coupled with 2.3/3.1/3.2): ``crosses`` is scored on ``crosses_frame`` only;
``crosses_pooled`` is never scored; ``temporal_weights`` is collected for the viz phase (6.2) and is
structurally full-model-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pedpredict.config.schema import EvalCfg, RootCfg
from pedpredict.data.collate import build_collate
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.losses.multitask import TASKS
from pedpredict.models.registry import ModelType, build_model, forward_model
from pedpredict.paths import resolve_paths
from pedpredict.training.chunk_loader import gather_lmdb_chunks
from pedpredict.training.metrics import METRIC_COLUMNS, MetricAccumulator, MetricResult
from pedpredict.utils.amp import autocast_ctx, resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.logging import (
    RunDir,
    append_index_row,
    build_index_row,
    init_run,
    round_row,
)
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

#: Collate tuple (1.5): ``(images_tight, images_context, motions, labels)``.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]

#: Per-task threshold-sweep columns (eval-only enrichment; ``optimal_threshold_metrics``, 3.2).
_OPT_SUFFIXES: tuple[str, ...] = ("threshold", "acc", "f1", "precision", "recall")
#: Efficiency columns — filled by 5.2 ``benchmark`` when run, else blank (OQ2: composed HERE alongside 5.1).
_EFFICIENCY_COLUMNS: tuple[str, ...] = (
    "params",
    "flops_per_frame",
    "latency_ms_per_frame",
    "fps",
    "peak_vram_mb",
)

#: WIDE eval-log schema (OQ2): context + shared METRIC_COLUMNS (default 0.5) + opt_* sweep + efficiency.
EVAL_LOG_COLUMNS: tuple[str, ...] = (
    ("timestamp", "checkpoint", "model_type", "split", "n_samples")
    + METRIC_COLUMNS
    + tuple(f"opt_{task}_{suffix}" for task in TASKS for suffix in _OPT_SUFFIXES)
    + ("opt_overall_acc",)
    + _EFFICIENCY_COLUMNS
)


@dataclass(frozen=True)
class EvalArtifacts:
    """Pure in-memory result of one evaluation pass (no I/O)."""

    metrics: MetricResult                          # MetricAccumulator.compute() — default-0.5-threshold
    optimal: dict[str, float]                       # optimal_threshold_metrics(EvalCfg) — thresholds + metrics
    n_samples: int
    predictions: dict[str, np.ndarray] | None        # populated if EITHER collect flag set (else None)
    temporal_weights: np.ndarray | None              # [N, T] softmax weights — FULL model only (else None)


@dataclass(frozen=True)
class EvalReport:
    """:class:`EvalArtifacts` + the on-disk artifact paths (returned by :func:`run_evaluation`)."""

    artifacts: EvalArtifacts
    run_dir: Path
    eval_log_path: Path
    predictions_path: Path | None
    temporal_weights_path: Path | None


# --------------------------------------------------------------------------- core pass (no I/O)


def _prepare_batch(
    batch: Batch, device: torch.device, pin: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Move a collate tuple to ``device``; long + clamp crosses labels (OLD ``remap_cross_labels``, 1.6)."""
    images_tight, images_context, motions, labels = batch
    images_tight = images_tight.to(device, non_blocking=pin)
    images_context = images_context.to(device, non_blocking=pin)
    motions = motions.to(device, non_blocking=pin)
    labels = {k: v.to(device, non_blocking=pin).long() for k, v in labels.items()}
    labels["crosses"] = torch.clamp(labels["crosses"], 0, 1)
    return images_tight, images_context, motions, labels


def _extract_predictions(acc: MetricAccumulator) -> dict[str, np.ndarray]:
    """Per-sample ``{task}_true / _prob_0 / _prob_1 / _pred`` from the accumulator's canonical store."""
    preds: dict[str, np.ndarray] = {}
    for task in TASKS:
        y_true, y_pred, y_prob = acc.task_arrays(task)
        preds[f"{task}_true"] = y_true
        preds[f"{task}_prob_0"] = y_prob[:, 0]
        preds[f"{task}_prob_1"] = y_prob[:, 1]
        preds[f"{task}_pred"] = y_pred
    return preds


def evaluate_model(
    model: nn.Module,
    loaders: Iterable[DataLoader],
    device: torch.device,
    cfg: EvalCfg,
    *,
    use_amp: bool = False,
    collect_predictions: bool = False,
    collect_temporal_weights: bool = False,
) -> EvalArtifacts:
    """Run ``model`` over every loader; return metrics (+ optional per-sample preds / temporal weights).

    A single shared :class:`MetricAccumulator` aggregates over ALL chunks (== OLD's global ``all_*_global``
    concatenation). ``crosses`` is scored on ``crosses_frame`` once (B4). ``temporal_weights`` is collected
    from ``outputs["temporal_weights"]`` only when present (full model). ``predictions`` is populated when
    either collection flag is set (the temporal-weight NPZ needs the per-task ground truth).
    """
    model.eval()
    acc = MetricAccumulator()
    tw_chunks: list[torch.Tensor] = []
    pin = device.type == "cuda"
    with torch.inference_mode():
        for loader in loaders:
            for batch in loader:
                images_tight, images_context, motions, labels = _prepare_batch(batch, device, pin)
                with autocast_ctx(use_amp):
                    outputs = forward_model(model, images_tight, images_context, motions)
                acc.update(outputs, labels)
                if collect_temporal_weights and outputs.get("temporal_weights") is not None:
                    tw_chunks.append(outputs["temporal_weights"].float().cpu())
    predictions = (
        _extract_predictions(acc) if (collect_predictions or collect_temporal_weights) else None
    )
    temporal_weights = torch.cat(tw_chunks).numpy() if tw_chunks else None
    return EvalArtifacts(
        metrics=acc.compute(),
        optimal=acc.optimal_threshold_metrics(cfg),
        n_samples=acc.n_samples,
        predictions=predictions,
        temporal_weights=temporal_weights,
    )


# --------------------------------------------------------------------------- artifact writers


def save_predictions_npz(path: Path, predictions: dict[str, np.ndarray]) -> Path:
    """Save per-sample ``{task}_true / _prob_0 / _prob_1 / _pred`` (replaces OLD predictions CSV; for 6.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)
    return path


def save_temporal_weights_npz(
    path: Path, temporal_weights: np.ndarray, predictions: dict[str, np.ndarray] | None
) -> Path:
    """Save ``[N, T]`` temporal weights (+ each ``{task}_true`` when available) for the attention viz (6.2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = (
        {f"{task}_true": predictions[f"{task}_true"] for task in TASKS}
        if predictions is not None
        else {}
    )
    np.savez_compressed(path, temporal_weights=temporal_weights, **labels)
    return path


# --------------------------------------------------------------------------- model loading


def load_eval_weights(
    model: nn.Module, checkpoint: str | Path, *, device: torch.device, strict: bool = True
) -> None:
    """Load model weights for eval (model-only — no optimizer/scheduler needed).

    Accepts a rebuilt :class:`~pedpredict.training.callbacks.CheckpointManager` payload (extracts
    ``model_state_dict``) OR a bare ``state_dict``; loads ``strict=True`` (B2: all ViT params are eager).
    Legacy pre-B2 raw weights that need a ``relative_position_bias`` fix-up are out of scope — route them
    through Prompt 9's ``load_legacy_model_weights`` (a clear pointer is raised if strict load fails).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("pedpredict_ckpt_version") is not None:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt  # bare state_dict (e.g. ModelStateCheckpointer or a hand-saved dict)
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"load_eval_weights: strict load of {checkpoint} failed. If these are legacy "
            "(pre-rebuild) weights, migrate them via Prompt 9's load_legacy_model_weights()."
        ) from exc


# --------------------------------------------------------------------------- data loaders


def _eval_chunk_loaders(cfg: RootCfg, chunk_paths: list[str], device: torch.device) -> Iterator[DataLoader]:
    """Yield a stable-order, no-sampler ``DataLoader`` per chunk (OLD ``test.py:393-405``); gc between (B11)."""
    collate = build_collate(cfg.data)
    pin = device.type == "cuda"
    for path in chunk_paths:
        dataset = LMDBChunkDataset.from_config(path, cfg.data)
        loader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.eval.num_workers,
            pin_memory=pin,
            collate_fn=collate,
        )
        yield loader
        del dataset, loader
        free_cuda(device)


def _split_chunk_paths(cfg: RootCfg, split: str) -> list[str]:
    """Resolve the LMDB chunk paths for ``split`` ('test' | 'val') from ``paths.yaml``."""
    resolved = resolve_paths(cfg.paths)
    folder = resolved.lmdb_test if split == "test" else resolved.lmdb_val
    return gather_lmdb_chunks([folder])


# --------------------------------------------------------------------------- run-dir resolution (OQ7)


def _resolve_run_dir(cfg: RootCfg, checkpoint: str | Path) -> RunDir:
    """Reuse the checkpoint's ``outputs/runs/{run_id}/`` when resolvable, else a fresh eval run-dir (OQ7)."""
    runs_root = resolve_paths(cfg.paths).runs_dir
    ckpt = Path(checkpoint).resolve()
    try:
        run_id = ckpt.relative_to(runs_root).parts[0]
    except ValueError:
        return init_run(cfg, kind="eval")
    return RunDir(run_id=run_id, path=runs_root / run_id)


# --------------------------------------------------------------------------- orchestration (I/O)


def _build_eval_row(
    artifacts: EvalArtifacts,
    *,
    checkpoint: str | Path,
    model_type: str,
    split: str,
    efficiency: dict[str, float] | None,
) -> dict[str, object]:
    """Assemble one ``EVAL_LOG_COLUMNS`` row (context + metrics + threshold sweep + efficiency)."""
    row: dict[str, object] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "split": split,
        "n_samples": artifacts.n_samples,
    }
    row.update(artifacts.metrics.as_flat_dict())
    for task in TASKS:
        for suffix in _OPT_SUFFIXES:
            row[f"opt_{task}_{suffix}"] = artifacts.optimal[f"{task}_{suffix}"]
    row["opt_overall_acc"] = artifacts.optimal["overall_acc"]
    eff = efficiency or {}
    for col in _EFFICIENCY_COLUMNS:
        row[col] = eff.get(col, "")
    return row


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
    """Build model (``cfg.eval.model_type``) -> load ``checkpoint`` -> eval over ``split`` -> write artifacts.

    Writes ``eval_log.csv`` (``EVAL_LOG_COLUMNS``) and appends one ``index.csv`` row (``kind='eval'``) under
    the run-dir (reused from the checkpoint, else a fresh ``..._eval`` dir; OQ7). Optionally saves
    ``plots/predictions.npz`` + ``plots/temporal_weights.npz`` for viz (6.2). ``efficiency`` (from 5.2) is
    folded into the CSV row; absent -> blanks.
    """
    device = device if device is not None else get_device()
    enable_perf_flags(device)
    model_type = cfg.eval.model_type
    model = build_model(cfg, model_type).to(device)
    load_eval_weights(model, checkpoint, device=device, strict=strict)

    is_full = ModelType.coerce(model_type) is ModelType.FULL
    artifacts = evaluate_model(
        model,
        _eval_chunk_loaders(cfg, _split_chunk_paths(cfg, split), device),
        device,
        cfg.eval,
        use_amp=resolve_amp(cfg.train.use_amp, device),
        collect_predictions=save_predictions,
        collect_temporal_weights=save_temporal_weights and is_full,
    )

    run = _resolve_run_dir(cfg, checkpoint)
    row = _build_eval_row(
        artifacts, checkpoint=checkpoint, model_type=model_type, split=split, efficiency=efficiency
    )
    with run.eval_logger(EVAL_LOG_COLUMNS) as logger:
        logger.log(round_row(row))
    _append_eval_index(cfg, run, artifacts, model_type=model_type, checkpoint=checkpoint)

    preds_path = (
        save_predictions_npz(run.plots_dir / "predictions.npz", artifacts.predictions)
        if save_predictions and artifacts.predictions is not None
        else None
    )
    tw_path = (
        save_temporal_weights_npz(
            run.plots_dir / "temporal_weights.npz", artifacts.temporal_weights, artifacts.predictions
        )
        if save_temporal_weights and artifacts.temporal_weights is not None
        else None
    )
    return EvalReport(
        artifacts=artifacts,
        run_dir=run.path,
        eval_log_path=run.eval_log_path,
        predictions_path=preds_path,
        temporal_weights_path=tw_path,
    )


def _append_eval_index(
    cfg: RootCfg,
    run: RunDir,
    artifacts: EvalArtifacts,
    *,
    model_type: str,
    checkpoint: str | Path,
) -> None:
    """Append this eval's headline row to ``runs_dir/index.csv`` (``kind='eval'``, ``crosses_f1``-led)."""
    row = build_index_row(
        run,
        model_type=model_type,
        kind="eval",
        epochs_run=0,
        best_epoch=0,
        best_val_loss=float("nan"),
        headline=artifacts.metrics.as_flat_dict(),
        best_ckpt=checkpoint,
    )
    append_index_row(resolve_paths(cfg.paths).runs_dir, row)

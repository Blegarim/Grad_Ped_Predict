"""Multi-phase training schedule (Prompt 4.4, B1).

Replaces OLD ``train_two_phase.py`` god-script.  The three phases (balanced-subset warmup ->
full augmented fine-tune -> decouple classifiers) are now expressed as a list of ``PhaseCfg``
objects in ``schedule.yaml`` / ``ScheduleCfg``.  ``run_phase_schedule`` orchestrates them by
calling ``Trainer.reset_for_phase`` + ``Trainer.fit`` in a loop, with checkpoint reload and
backbone freezing handled between phases.

Intentional deviations from OLD script (documented in MIGRATION.md):
  D1  Scheduler steps on val_loss (OLD: train loss)   — more principled, matches main Trainer.
  D2  EarlyStopping on val_loss   (OLD: 1-macro_f1)  — consistent with single-phase path.
  D3  MultiTaskLoss CE weights    (OLD: FocalLoss)    — inherits from Trainer (3.1).
  D4  per-phase best_val_loss reset; Phase N+1 always loads Phase N best.pth explicitly.

Band-aids resolved here:
  B1  ``train_two_phase.py`` god-script -> config-driven phase list + thin orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn

from pedpredict.config.schema import RootCfg, ScheduleCfg
from pedpredict.training.callbacks import CheckpointManager
from pedpredict.training.trainer import TRAIN_LOG_COLUMNS, EpochResult, Trainer
from pedpredict.utils.logging import CsvLogger, RunDir, append_index_row, build_index_row

if TYPE_CHECKING:
    from pedpredict.training.trainer import ChunkProvider

__all__ = [
    "PhaseResult",
    "freeze_backbone",
    "unfreeze_all",
    "run_phase_schedule",
]

# Names whose presence anywhere in a parameter name marks it as a TRAINABLE classifier param.
# Exact port of OLD train_two_phase.py:freeze_backbone() lines 122-125.
_TRAINABLE_SUBSTRINGS: frozenset[str] = frozenset(
    {"classifier", "crosses_frame_head", "pool_mlp"}
)


class PhaseResult(NamedTuple):
    """Outcome of one completed training phase."""

    phase_name: str
    epoch_results: list[EpochResult]    # one entry per epoch actually run (may be < max_epochs)
    best_ckpt: Path | None              # best.pth path for this phase; None if val never improved


# --------------------------------------------------------------------------- freeze helpers


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all params except classifier / crosses_frame_head / pool_mlp.

    Exact port of OLD ``train_two_phase.py:freeze_backbone()`` (lines 122-125).  In the new
    ``EnsembleModel`` the trainable partition is:
      * ``cross_attention.pool_mlp.*``
      * ``cross_attention.classifier.*``
      * ``cross_attention.crosses_frame_head.*``
    Everything else (ViT, MotionEncoder, CrossAttention, LayerNorms) is frozen.
    """
    for name, param in model.named_parameters():
        if not any(k in name for k in _TRAINABLE_SUBSTRINGS):
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Re-enable ``requires_grad=True`` for every parameter (e.g. to restore after a phase)."""
    for param in model.parameters():
        param.requires_grad = True


# --------------------------------------------------------------------------- schedule runner


def run_phase_schedule(
    cfg: RootCfg,
    trainer: Trainer,
    schedule: ScheduleCfg,
    chunk_builders: dict[str, Callable[[], ChunkProvider]],
    *,
    run_dir: Path | None = None,
) -> list[PhaseResult]:
    """Execute ``schedule.phases`` sequentially on a shared ``trainer``.

    Args:
        cfg:            Full resolved config (used for any future per-phase config lookups).
        trainer:        A wired ``Trainer`` whose model will be mutated in-place across phases.
                        The trainer's initial chunk provider will be closed at the start of
                        Phase 0 (``Trainer.fit`` always closes chunks in its ``finally`` block).
        schedule:       The phase list to execute.
        chunk_builders: Mapping ``data_source`` -> zero-arg factory returning a fresh
                        ``ChunkProvider``.  Required keys match ``phase.data_source`` for every
                        phase.  Each factory is called once per phase; providers are owned and
                        closed by ``Trainer.fit``.
        run_dir:        If set, per-phase checkpoints go under
                        ``<run_dir>/phase_{i}_{name}/checkpoints/`` and the phase log CSV goes
                        to ``<run_dir>/phase_{i}_{name}_log.csv``.

    Returns:
        One ``PhaseResult`` per phase, in order.
    """
    results: list[PhaseResult] = []
    best_ckpt: Path | None = None
    trainer.write_index_on_fit = False  # one aggregated index row for the whole schedule, written below

    for i, phase in enumerate(schedule.phases):
        # ---------------------------------------------------------------- 1. reload best ckpt
        if phase.reload_best and best_ckpt is not None:
            state = torch.load(best_ckpt, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(state["model_state_dict"], strict=True)

        # ---------------------------------------------------------------- 2. build fresh chunks
        if phase.data_source not in chunk_builders:
            raise KeyError(
                f"Phase '{phase.name}' requires data_source={phase.data_source!r} but "
                f"chunk_builders only has: {sorted(chunk_builders)}"
            )
        new_chunks = chunk_builders[phase.data_source]()

        # ---------------------------------------------------------------- 3. per-phase wiring
        if run_dir is not None:
            phase_dir = run_dir / f"phase_{i}_{phase.name}"
            ckpt_dir = phase_dir / "checkpoints"
            trainer.checkpointer = CheckpointManager(
                ckpt_dir,
                run_id=f"phase_{i}_{phase.name}",
                model_type=cfg.eval.model_type,
            )
            trainer.logger = CsvLogger(
                phase_dir.parent / f"phase_{i}_{phase.name}_log.csv",
                TRAIN_LOG_COLUMNS,
            )
        else:
            trainer.checkpointer = CheckpointManager(None, run_id="", model_type="")

        # ---------------------------------------------------------------- 4. phase transition
        trainer.reset_for_phase(phase, new_chunks)

        # ---------------------------------------------------------------- 5. run this phase
        epoch_results = trainer.fit(max_epochs=phase.max_epochs)

        # ---------------------------------------------------------------- 6. collect outcome
        phase_ckptr = trainer.checkpointer
        phase_best = phase_ckptr.best_path() if isinstance(phase_ckptr, CheckpointManager) else None
        best_ckpt = phase_best  # next phase reloads from this
        results.append(PhaseResult(phase.name, epoch_results, phase_best))

    _write_schedule_index(cfg, results, run_dir)
    return results


def _write_schedule_index(
    cfg: RootCfg, results: list[PhaseResult], run_dir: Path | None
) -> None:
    """Append ONE cross-run index row for the whole schedule (the final phase = the deliverable model).

    Headline = the final phase's best epoch (its ``best.pth`` is what downstream eval loads);
    ``epochs_run`` = total epochs across all phases. No-op without a run dir / epochs.
    """
    if run_dir is None or not results:
        return
    final = results[-1]
    if not final.epoch_results:
        return
    best = min(final.epoch_results, key=lambda r: r.val_loss)
    total_epochs = sum(len(pr.epoch_results) for pr in results)
    run = RunDir(run_id=run_dir.name, path=run_dir)
    row = build_index_row(
        run,
        model_type=cfg.eval.model_type,
        tag="schedule",
        kind="schedule",
        epochs_run=total_epochs,
        best_epoch=best.epoch + 1,
        best_val_loss=best.val_loss,
        headline=best.metrics.as_flat_dict(),
        best_ckpt=final.best_ckpt,
    )
    append_index_row(Path(cfg.paths.runs_dir), row)

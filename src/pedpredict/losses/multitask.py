"""Unified multitask loss + the loss-side imbalance lever (Prompt 3.1).

Consolidates the loss logic smeared through OLD ``train.py`` into ONE ``nn.Module``:

* ``train.py:341-345`` — ``criterion = {task: CrossEntropyLoss(weight=class_weights[task])}``.
* ``train.py:145-153`` — the per-head accumulation in ``train_one_chunk`` (``total += loss_weight[task] *
  CE``), with ``crosses`` reading ``outputs["crosses_frame"]``.
* ``train.py:209-219`` — the validation twin (identical per-task math; only its sum-over-samples
  *accumulation* differs, and that is logging — it stays in the Trainer, Prompt 4.1).

Resolves **B3** (the loss lever of the imbalance policy), **part of B1** (extracting loss math from the
635-line god-script), part of **B4** (the supervised crosses routing made an explicit contract), and part
of **B8** (the scattered ``logits.float()`` casts → one ``to_float_logits`` call).

Imbalance policy (single source of truth — CLAUDE.md / MIGRATION.md):

* **Lever 1** offline balance (``data/balance.py``) — OPT-IN, OFF by default.
* **Lever 2** online sampler (``data/sampler.py``) — ON by default.
* **Lever 3** loss class weights (THIS module) — ON by default. Class weights are computed ONCE by the
  Trainer from Prompt 1.6's single LMDB scan (``data.sampler.class_weights_ce`` over
  ``LabelScanCache.aggregate_counts``) and passed in — this module NEVER re-scans (the whole point of
  the dedup).

Crosses-label clamping is NOT done here: the Trainer applies the single canonical clamp
(``data.balance.clamp_cross``, Prompt 1.6) to labels just before calling this loss, mirroring the OLD
in-loop ``remap_cross_labels`` position. This module assumes already-binary ``{0, 1}`` targets and stays
pure (no data cleaning).
"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor, nn

from pedpredict.config.schema import TrainCfg
from pedpredict.utils.amp import to_float_logits

__all__ = [
    "TASKS",
    "TASK_OUTPUT_KEY",
    "MultiTaskLossOutput",
    "MultiTaskLoss",
    "build_multitask_loss",
]

#: Supervised tasks, in the parity-critical legacy order (OLD train.py:145).
TASKS: tuple[str, ...] = ("actions", "looks", "crosses")

#: EXPLICIT output contract (B4 / Prompt 2.3): which output-dict key supervises each task.
#: ``crosses -> "crosses_frame"`` (logsumexp-pooled over time). ``"crosses_pooled"`` is deliberately
#: absent — it is emitted as a live-but-UNSUPERVISED auxiliary head and must never be routed here.
TASK_OUTPUT_KEY: dict[str, str] = {
    "actions": "actions",
    "looks": "looks",
    "crosses": "crosses_frame",
}


class MultiTaskLossOutput(NamedTuple):
    """Return bundle: live ``total`` for backward + detached per-task scalars for CSV logging."""

    total: Tensor                    # live (carries grad): sum_t loss_weight[t] * CE_t
    per_task: dict[str, Tensor]      # detached raw mean-CE per task (cross-run comparable)
    weighted: dict[str, Tensor]      # detached loss_weight[t] * CE_t per task


class MultiTaskLoss(nn.Module):
    """Per-task weighted cross-entropy with class-weight + per-task scalar imbalance handling.

    Mirrors OLD ``train_one_chunk`` loss math exactly: for each task, ``CrossEntropyLoss`` (with the
    inverse-frequency class ``weight``) over the contract-routed logits, scaled by ``loss_weight[task]``,
    summed across tasks. ``crosses`` is supervised on ``crosses_frame`` per :data:`TASK_OUTPUT_KEY`.
    """

    def __init__(
        self,
        class_weights: dict[str, Tensor],
        loss_weight: dict[str, float],
        *,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        # nn.ModuleDict of CrossEntropyLoss mirrors OLD ``criterion`` (train.py:341-345). The per-task
        # ``weight`` lives as a registered buffer, so ``loss.to(device)`` moves it — preserving OLD's
        # "weights on device" behavior without manual ``.to()`` at every call site.
        self.criteria = nn.ModuleDict({
            task: nn.CrossEntropyLoss(weight=class_weights[task], reduction=reduction)
            for task in TASKS
        })
        # Plain floats (not parameters); ``.get(task, 1.0)`` matches OLD's defaulting (train.py:153).
        self.loss_weight: dict[str, float] = {task: float(loss_weight.get(task, 1.0)) for task in TASKS}

    def forward(
        self,
        outputs: dict[str, Tensor],
        labels: dict[str, Tensor],
    ) -> MultiTaskLossOutput:
        """Compute the weighted multitask loss.

        ``outputs`` must contain the supervised contract keys (``actions``, ``looks``, ``crosses_frame``);
        any extra keys (``crosses_pooled``, ``temporal_weights``) are ignored. ``labels`` provides binary
        ``{0, 1}`` targets per task (clamped upstream by the Trainer).
        """
        floated = to_float_logits(outputs)   # B8: single upcast site (no-op outside autocast)
        per_task: dict[str, Tensor] = {}
        weighted: dict[str, Tensor] = {}
        total: Tensor | None = None
        for task in TASKS:
            key = TASK_OUTPUT_KEY[task]
            if key not in floated:
                raise KeyError(
                    f"MultiTaskLoss: task '{task}' requires output key '{key}', "
                    f"missing from outputs (have: {sorted(outputs)})."
                )
            logits = floated[key]
            targets = labels[task]
            head_loss = self.criteria[task](logits, targets)
            contribution = self.loss_weight[task] * head_loss
            total = contribution if total is None else total + contribution
            per_task[task] = head_loss.detach()
            weighted[task] = contribution.detach()
        assert total is not None  # TASKS is non-empty
        return MultiTaskLossOutput(total=total, per_task=per_task, weighted=weighted)


def build_multitask_loss(
    cfg: TrainCfg,
    class_weights: dict[str, Tensor],
    *,
    reduction: str = "mean",
) -> MultiTaskLoss:
    """Wire a :class:`MultiTaskLoss` from ``TrainCfg.loss_weight`` + precomputed class weights.

    ``class_weights`` is produced ONCE by the Trainer via
    ``class_weights_ce(LabelScanCache.aggregate_counts(train_lmdbs), device=...)`` (Prompt 1.6) — this
    factory does not scan. Move the loss to the model device with ``loss.to(device)`` after building.
    """
    return MultiTaskLoss(class_weights, cfg.loss_weight, reduction=reduction)

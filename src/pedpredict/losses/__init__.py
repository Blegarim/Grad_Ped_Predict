"""Multitask loss with unified imbalance policy (Prompt 3.1)."""

from __future__ import annotations

from pedpredict.losses.multitask import (
    TASK_OUTPUT_KEY,
    TASKS,
    MultiTaskLoss,
    MultiTaskLossOutput,
    build_multitask_loss,
)

__all__ = [
    "TASKS",
    "TASK_OUTPUT_KEY",
    "MultiTaskLoss",
    "MultiTaskLossOutput",
    "build_multitask_loss",
]

"""Training: trainer, chunk_loader, callbacks, metrics (P4)."""

from __future__ import annotations

from pedpredict.training.callbacks import EarlyStopping
from pedpredict.training.metrics import (
    METRIC_COLUMNS,
    MetricAccumulator,
    MetricResult,
    TaskMetrics,
)
from pedpredict.training.trainer import (
    TRAIN_LOG_COLUMNS,
    Checkpointer,
    ChunkProvider,
    EpochResult,
    ModelStateCheckpointer,
    Trainer,
    build_trainer,
)

__all__ = [
    "METRIC_COLUMNS",
    "TRAIN_LOG_COLUMNS",
    "Checkpointer",
    "ChunkProvider",
    "EarlyStopping",
    "EpochResult",
    "MetricAccumulator",
    "MetricResult",
    "ModelStateCheckpointer",
    "TaskMetrics",
    "Trainer",
    "build_trainer",
]

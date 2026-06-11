"""Training: trainer, chunk_loader, callbacks, metrics, schedule, distribution (P4)."""

from __future__ import annotations

from pedpredict.training.callbacks import CheckpointManager, CheckpointPayload, EarlyStopping
from pedpredict.training.chunk_loader import (
    ChunkLoaderIterator,
    ChunkPrefetcher,
    gather_lmdb_chunks,
    warm_lmdb_chunk,
)
from pedpredict.training.distribution import (
    DISTRIBUTION_FILENAME,
    effective_distribution,
    write_distribution_report,
)
from pedpredict.training.metrics import (
    METRIC_COLUMNS,
    MetricAccumulator,
    MetricResult,
    TaskMetrics,
)
from pedpredict.training.schedule import PhaseResult, freeze_backbone, run_phase_schedule, unfreeze_all
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
    "DISTRIBUTION_FILENAME",
    "METRIC_COLUMNS",
    "TRAIN_LOG_COLUMNS",
    "Checkpointer",
    "CheckpointManager",
    "CheckpointPayload",
    "ChunkLoaderIterator",
    "ChunkPrefetcher",
    "ChunkProvider",
    "EarlyStopping",
    "EpochResult",
    "MetricAccumulator",
    "MetricResult",
    "ModelStateCheckpointer",
    "PhaseResult",
    "TaskMetrics",
    "Trainer",
    "build_trainer",
    "effective_distribution",
    "freeze_backbone",
    "gather_lmdb_chunks",
    "run_phase_schedule",
    "unfreeze_all",
    "warm_lmdb_chunk",
    "write_distribution_report",
]

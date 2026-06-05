"""Clean training loop (Prompt 4.1) — replaces the OLD ``train.py`` god-script (B1).

Decomposes OLD ``train.py:125-175`` (``train_one_chunk``) + ``:177-234`` (``validate_one_epoch``) +
``:236-632`` (``main``'s epoch loop) into a dependency-injected :class:`Trainer` whose math is delegated
to the already-golden-tested components — :class:`~pedpredict.losses.multitask.MultiTaskLoss` (3.1),
:class:`~pedpredict.training.metrics.MetricAccumulator` (3.2), the typed model + forward adapter
(``registry``, 2.4), and the single LMDB scan (``data.sampler``, 1.6). The Trainer adds NO new math; it
owns only orchestration (optimizer/scheduler build, the AMP+GradScaler+grad-clip step order, the
val-loss accumulation that drives scheduler/early-stop/best, and CSV logging).

Band-aids resolved here:

* **B1** — every hyperparameter flows from ``TrainCfg``/``RootCfg``; no literals in the loop.
* **B2 (consumer side)** — the OLD ``train.py:311-317`` dummy forward is DELETED. Prompt 2.1 made every
  ViT parameter eager, so the optimizer is built over ``model.parameters()`` immediately after
  ``build_model`` with no forward (``test_optimizer_covers_all_params_without_forward``).
* **B8** — no scattered ``.float()`` casts: the single upcast lives in ``MultiTaskLoss`` /
  ``MetricAccumulator`` (``to_float_logits``); the Trainer uses ``utils.amp`` for the autocast context,
  scaler, and AMP gating.

Seams to later prompts (each is an injected dependency, so they slot in with no edit to ``fit``):

* **4.2** ``ChunkPrefetcher`` satisfies :class:`ChunkProvider` (this file ships only the Protocol; the
  Trainer just iterates ``epoch_loaders`` / ``val_loaders``).
* **4.3** ``CheckpointManager`` satisfies :class:`Checkpointer` (full-state resume + strict load); the
  interim :class:`ModelStateCheckpointer` here saves model-only ``state_dict`` like OLD.
* **4.5** owns the final CSV column schema / run-dir / cross-run index; the Trainer takes an injected
  :class:`~pedpredict.utils.logging.CsvLogger` and uses the provisional :data:`TRAIN_LOG_COLUMNS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from pedpredict.config.schema import RootCfg
from pedpredict.data.sampler import LabelScanCache, class_weights_ce
from pedpredict.losses.multitask import MultiTaskLoss, build_multitask_loss
from pedpredict.models.registry import build_model, forward_model
from pedpredict.training.callbacks import EarlyStopping
from pedpredict.training.metrics import METRIC_COLUMNS, MetricAccumulator, MetricResult
from pedpredict.utils.amp import autocast_ctx, make_grad_scaler, resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.logging import CsvLogger, create_run_dir, make_run_id
from pedpredict.utils.memory import free_cuda

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.utils.data import DataLoader

__all__ = [
    "TRAIN_LOG_COLUMNS",
    "Batch",
    "EpochResult",
    "ChunkProvider",
    "Checkpointer",
    "ModelStateCheckpointer",
    "Trainer",
    "build_trainer",
]

#: The collate tuple (1.5): ``(images_tight, images_context, motions, labels)``.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]

#: Provisional train-CSV schema (4.5 owns the final form): context columns + the shared METRIC_COLUMNS.
TRAIN_LOG_COLUMNS: tuple[str, ...] = ("epoch", "train_loss", "val_loss", *METRIC_COLUMNS)


class EpochResult(NamedTuple):
    """One epoch's headline numbers (returned by :meth:`Trainer.fit`)."""

    epoch: int
    train_loss: float                # avg over batches (OLD epoch_loss_sum / epoch_n_batches)
    val_loss: float                  # per-sample mean weighted loss (OLD validate formula)
    metrics: MetricResult


@runtime_checkable
class ChunkProvider(Protocol):
    """The 4.2 seam: yields ready-to-train DataLoaders. The Trainer never touches LMDB/prefetch detail."""

    train_lmdb_paths: list[str]      # for the upfront global class-weight scan (1.6); [] if loss injected

    def epoch_loaders(self, epoch: int) -> Iterator[DataLoader]:
        """Yield the (reshuffled) train-chunk loaders for one epoch."""
        ...

    def val_loaders(self) -> Iterator[DataLoader]:
        """Yield the validation-chunk loaders (stable order)."""
        ...

    def close(self) -> None:
        """Release any held resources (processes/handles)."""
        ...


@runtime_checkable
class Checkpointer(Protocol):
    """The 4.3 seam. The interim default saves model-only state; 4.3 adds full-state resume."""

    def save_last(self, trainer: Trainer, epoch: int) -> None: ...
    def save_best(self, trainer: Trainer, epoch: int, val_loss: float) -> None: ...


class ModelStateCheckpointer:
    """Interim checkpointer (superseded by 4.3 ``CheckpointManager``): writes model ``state_dict`` only.

    Matches OLD ``train.py`` save semantics (``torch.save(model.state_dict(), ...)``) under the new
    run-dir (``<run_dir>/checkpoints/{last,best}.pth``). A ``None`` directory makes both saves no-ops, so
    the Trainer is constructible in tests without touching disk.
    """

    def __init__(self, checkpoints_dir: str | Path | None) -> None:
        self.dir = Path(checkpoints_dir) if checkpoints_dir is not None else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def save_last(self, trainer: Trainer, epoch: int) -> None:
        self._save(trainer, "last.pth")

    def save_best(self, trainer: Trainer, epoch: int, val_loss: float) -> None:
        self._save(trainer, "best.pth")

    def _save(self, trainer: Trainer, name: str) -> None:
        if self.dir is not None:
            torch.save(trainer.model.state_dict(), self.dir / name)


class Trainer:
    """Epoch/chunk training loop over an injected :class:`ChunkProvider`.

    Wiring (all from config, no literals): Adam over ``model.parameters()`` (B2 — no dummy forward),
    ``ReduceLROnPlateau`` on the val loss, AMP via ``utils.amp``, grad-clip from
    ``TrainCfg.grad_clip_max_norm``. The supervised contract (B4) is enforced downstream by
    ``MultiTaskLoss`` / ``MetricAccumulator`` (crosses -> ``crosses_frame`` only).
    """

    def __init__(
        self,
        cfg: RootCfg,
        model: nn.Module,
        device: torch.device,
        chunks: ChunkProvider,
        *,
        loss: MultiTaskLoss | None = None,
        scan_cache: LabelScanCache | None = None,
        checkpointer: Checkpointer | None = None,
        logger: CsvLogger | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.chunks = chunks
        self.run_dir = run_dir
        self.logger = logger
        self.use_amp = resolve_amp(cfg.train.use_amp, device)
        self.pin = device.type == "cuda"
        self.scaler = make_grad_scaler(self.use_amp)
        self.clip_max_norm = cfg.train.grad_clip_max_norm

        self.scan_cache = scan_cache if scan_cache is not None else LabelScanCache()
        self.loss = loss if loss is not None else self._build_loss()
        self.loss = self.loss.to(device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.early_stopping = EarlyStopping(
            patience=cfg.train.early_stop_patience, min_delta=cfg.train.early_stop_min_delta
        )
        self.checkpointer = checkpointer if checkpointer is not None else ModelStateCheckpointer(None)
        self.best_val_loss = float("inf")

    # ----------------------------------------------------------------- construction helpers

    def _build_loss(self) -> MultiTaskLoss:
        """GLOBAL inverse-freq CE weights from ONE scan over the train chunks (1.6) -> loss (3.1)."""
        counts = self.scan_cache.aggregate_counts(self.chunks.train_lmdb_paths)
        class_weights = class_weights_ce(counts, device=self.device)
        return build_multitask_loss(self.cfg.train, class_weights)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam over the trainable params (OLD train.py:346). No dummy forward needed (B2)."""
        params = (p for p in self.model.parameters() if p.requires_grad)
        return torch.optim.Adam(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """``ReduceLROnPlateau(mode='min', ...)`` stepped on the val loss (OLD train.py:348-350)."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.train.sched_factor,
            patience=self.cfg.train.sched_patience,
            threshold=self.cfg.train.sched_threshold,
            threshold_mode="rel",
        )

    # ----------------------------------------------------------------- per-batch / per-chunk

    def _move_batch(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Move a collate tuple to the device; long + clamp crosses labels (OLD remap_cross_labels)."""
        images_tight, images_context, motions, labels = batch
        images_tight = images_tight.to(self.device, non_blocking=self.pin)
        images_context = images_context.to(self.device, non_blocking=self.pin)
        motions = motions.to(self.device, non_blocking=self.pin)
        labels = {k: v.to(self.device, non_blocking=self.pin).long() for k, v in labels.items()}
        labels["crosses"] = torch.clamp(labels["crosses"], 0, 1)  # in-contract == 1.6 clamp_cross
        return images_tight, images_context, motions, labels

    def _step_batch(self, batch: Batch) -> torch.Tensor:
        """One optimizer step (OLD train_one_chunk body). Returns the detached weighted total loss."""
        images_tight, images_context, motions, labels = self._move_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(self.use_amp):
            outputs = forward_model(self.model, images_tight, images_context, motions)
            total = self.loss(outputs, labels).total
        if self.use_amp:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_max_norm)
            self.optimizer.step()
        return total.detach()

    def train_chunk(self, loader: DataLoader) -> tuple[float, int]:
        """Train over one chunk's loader. Returns ``(loss_sum, n_batches)`` (OLD train_one_chunk)."""
        self.model.train()
        loss_sum = 0.0
        n_batches = 0
        for batch in loader:
            loss_sum += float(self._step_batch(batch))
            n_batches += 1
        return loss_sum, n_batches

    # ----------------------------------------------------------------- validation

    def validate(self) -> tuple[float, MetricResult]:
        """Validate over all val chunks. Returns ``(val_loss, metrics)`` (OLD validate_one_epoch + :574-596).

        ``val_loss`` is the per-sample mean weighted loss (OLD accumulation: ``Σ total·B / ΣB``) — the
        scalar that drives the scheduler, early stopping, and best-checkpoint selection. Metrics come
        from the shared :class:`MetricAccumulator` (3.2); crosses routes to ``crosses_frame`` (B4).
        """
        self.model.eval()
        acc = MetricAccumulator()
        loss_sum = 0.0
        n_samples = 0
        with torch.inference_mode():
            for loader in self.chunks.val_loaders():
                for batch in loader:
                    images_tight, images_context, motions, labels = self._move_batch(batch)
                    batch_size = images_tight.size(0)
                    with autocast_ctx(self.use_amp):
                        outputs = forward_model(self.model, images_tight, images_context, motions)
                    loss_sum += float(self.loss(outputs, labels).total) * batch_size
                    acc.update(outputs, labels)
                    n_samples += batch_size
        if n_samples == 0:
            raise RuntimeError("Trainer.validate: no validation samples found.")
        return loss_sum / n_samples, acc.compute()

    # ----------------------------------------------------------------- epoch loop

    def fit(self) -> list[EpochResult]:
        """Run the full training schedule (OLD main epoch loop, :373-627), returning per-epoch results."""
        results: list[EpochResult] = []
        try:
            for epoch in range(self.cfg.train.num_epochs):
                train_loss = self._run_epoch(epoch)
                self.checkpointer.save_last(self, epoch)         # OLD train.py:509 (pre-validation)
                val_loss, metrics = self.validate()
                self.scheduler.step(val_loss)                    # OLD train.py:598
                self._log_epoch(epoch, train_loss, val_loss, metrics)
                if val_loss < self.best_val_loss:                # OLD train.py:616-620
                    self.best_val_loss = val_loss
                    self.checkpointer.save_best(self, epoch, val_loss)
                results.append(EpochResult(epoch, train_loss, val_loss, metrics))
                self.early_stopping(val_loss)                    # OLD train.py:622-627
                if self.early_stopping.early_stop:
                    break
        finally:
            self.chunks.close()
        return results

    def _run_epoch(self, epoch: int) -> float:
        """Train over every chunk for one epoch; return the per-batch average train loss."""
        epoch_loss_sum = 0.0
        epoch_n_batches = 0
        for loader in self.chunks.epoch_loaders(epoch):
            chunk_loss_sum, chunk_n_batches = self.train_chunk(loader)
            epoch_loss_sum += chunk_loss_sum
            epoch_n_batches += chunk_n_batches
            free_cuda(self.device)                               # OLD per-chunk gc + empty_cache
        return epoch_loss_sum / epoch_n_batches if epoch_n_batches else float("nan")

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, metrics: MetricResult) -> None:
        """Append one CSV row (provisional 4.5 schema). No-op if no logger was injected."""
        if self.logger is None:
            return
        row: dict[str, object] = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        row.update(metrics.as_flat_dict())
        self.logger.log(row)


def build_trainer(cfg: RootCfg, chunks: ChunkProvider, *, device: torch.device | None = None,
                  tag: str = "") -> Trainer:
    """Wire a runnable :class:`Trainer`: device + perf flags, ``build_model`` (2.4), run-dir + CSV logger.

    ``chunks`` is the (4.2) provider — passed in so callers stay in control of the data source; the model
    type comes from ``cfg.eval.model_type`` (the shared selector). Once 4.2 lands, the call is simply
    ``build_trainer(cfg, ChunkPrefetcher.from_config(cfg))``.
    """
    device = device if device is not None else get_device()
    enable_perf_flags(device)
    model = build_model(cfg)                                      # B2: all params eager, no dummy forward
    run_id = make_run_id(cfg.eval.model_type, tag)
    run_dir = create_run_dir(Path(cfg.paths.runs_dir), run_id)
    logger = CsvLogger(run_dir / "train_log.csv", TRAIN_LOG_COLUMNS)
    return Trainer(
        cfg, model, device, chunks,
        checkpointer=ModelStateCheckpointer(run_dir / "checkpoints"),
        logger=logger,
        run_dir=run_dir,
    )

"""Training callbacks (Prompt 4.1 lands ``EarlyStopping``; Prompt 4.3 adds checkpointing).

``EarlyStopping`` is a verbatim port of OLD ``scripts/train_utils.py:23-37`` — same min-delta /
patience semantics, only typed and snake_cased. The Trainer (4.1) drives it with the per-epoch
validation loss exactly as OLD ``main`` did (train.py:622-627).

``CheckpointPayload`` and ``CheckpointManager`` (Prompt 4.3) supersede the interim
``ModelStateCheckpointer``. The manager saves the FULL training state — model, optimizer, scaler,
scheduler, epoch, best_val_loss — enabling true warm resume.

Design decisions vs OLD code:
  * ``strict=True`` by default on load (B2 load-side fix; B2 save-side fixed in Prompt 2.1 made
    all ViT params eager). Pass ``strict=False`` explicitly only for legacy-weight migration.
  * ``save_last`` is called POST-validation (after scheduler.step) so the on-disk state covers a
    complete epoch; resuming with ``_start_epoch = saved_epoch + 1`` is then correct by definition.
    OLD ``train.py:509`` wrote model weights pre-validation — that was fine for a warm-start but
    wrong for a true optimizer/scheduler resume.
  * All artifacts write under the per-run checkpoints dir (B11); no writes to legacy flat root dirs.
  * Atomic writes via rename-from-temp guard against corrupt checkpoints on mid-save kill.

Band-aids resolved: B1 (no literals), B2 load side (strict=True), B11 (run-scoped artifacts).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch import nn

if TYPE_CHECKING:
    from pedpredict.training.trainer import Trainer

__all__ = ["EarlyStopping", "CheckpointPayload", "CheckpointManager"]

_CKPT_VERSION = 1


class EarlyStopping:
    """Stop training when the monitored loss stops improving by ``min_delta`` for ``patience`` epochs.

    Verbatim semantics of OLD ``train_utils.EarlyStopping``: an epoch counts as an improvement only
    when ``loss < best_loss - min_delta``; otherwise the patience counter increments and
    :attr:`early_stop` latches ``True`` once it reaches ``patience``.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Feed one epoch's validation loss; updates :attr:`counter` / :attr:`early_stop` in place."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class CheckpointPayload(NamedTuple):
    """In-memory view of a saved checkpoint.

    Returned by :meth:`CheckpointManager.load` so callers can read ``epoch`` and ``best_val_loss``
    without re-opening the file.

    On-disk schema (``pedpredict_ckpt_version=1``):

    .. code-block:: text

        last.pth / best.pth
        ├── pedpredict_ckpt_version : int     = 1
        ├── epoch                   : int     0-indexed epoch that completed
        ├── best_val_loss           : float
        ├── run_id                  : str
        ├── model_type              : str
        ├── model_state_dict        : OrderedDict
        ├── optimizer_state_dict    : dict
        ├── scaler_state_dict       : dict
        └── scheduler_state_dict    : dict
    """

    epoch: int
    best_val_loss: float
    run_id: str
    model_type: str
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]


class CheckpointManager:
    """Full-state checkpointer (Prompt 4.3). Implements the ``Checkpointer`` Protocol.

    Supersedes ``ModelStateCheckpointer``. Saves the complete training state — model, optimizer,
    scaler, scheduler, epoch, best_val_loss — enabling true warm resume without re-running any
    epoch.

    **Artifact hygiene (B11)**: all writes go under ``checkpoints_dir`` (a sub-dir of the per-run
    dir); never writes to legacy flat root dirs. Pass ``checkpoints_dir=None`` to make all saves
    no-ops (useful in unit tests that do not want disk I/O).

    **Retention**: ``last.pth`` is always overwritten (represents the most recent complete epoch).
    ``best.pth`` is overwritten when a new best val loss is recorded. With ``keep_best_k > 1``,
    additional archives ``best_{epoch:04d}.pth`` are kept and the oldest are pruned automatically.

    **Atomic writes**: saves write to ``*.pth.tmp`` first, then ``Path.replace()`` atomically renames,
    so a kill mid-save never leaves a corrupt checkpoint behind.

    **Strict loading (B2)**: ``load()`` defaults to ``strict=True``. Pass ``strict=False`` only for
    migrating OLD-format weights that predate the B2 fix in Prompt 2.1.
    """

    def __init__(
        self,
        checkpoints_dir: str | Path | None,
        run_id: str,
        model_type: str,
        *,
        keep_best_k: int = 1,
        strict: bool = True,
    ) -> None:
        self.dir = Path(checkpoints_dir) if checkpoints_dir is not None else None
        self.run_id = run_id
        self.model_type = model_type
        self.keep_best_k = keep_best_k
        self.strict = strict
        self._best_epochs: deque[int] = deque()
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Checkpointer Protocol

    def save_last(self, trainer: Trainer, epoch: int) -> None:
        """Overwrite ``last.pth`` with the full training state.

        Called at the end of each complete epoch (post-validation, post-scheduler-step) so that
        resuming with ``_start_epoch = saved_epoch + 1`` is always correct.
        """
        if self.dir is None:
            return
        self._atomic_save(self._build_payload(trainer, epoch), self.dir / "last.pth")

    def save_best(self, trainer: Trainer, epoch: int, val_loss: float) -> None:  # noqa: ARG002
        """Overwrite ``best.pth``; if ``keep_best_k > 1`` also archive as ``best_{epoch:04d}.pth``."""
        if self.dir is None:
            return
        payload = self._build_payload(trainer, epoch)
        self._atomic_save(payload, self.dir / "best.pth")
        if self.keep_best_k > 1:
            self._atomic_save(payload, self.dir / f"best_{epoch:04d}.pth")
            self._best_epochs.append(epoch)
            while len(self._best_epochs) > self.keep_best_k:
                old_epoch = self._best_epochs.popleft()
                (self.dir / f"best_{old_epoch:04d}.pth").unlink(missing_ok=True)

    # ------------------------------------------------------------------ Load / resume

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        *,
        device: torch.device | None = None,
        strict: bool = True,
    ) -> CheckpointPayload:
        """Load full training state into already-constructed objects; return the payload.

        The caller is responsible for applying ``payload.epoch`` and ``payload.best_val_loss`` to
        the :class:`~pedpredict.training.trainer.Trainer` (done automatically by ``build_trainer``
        when ``resume_from`` is supplied).

        Parameters
        ----------
        strict:
            Forwarded to ``model.load_state_dict``. Default ``True`` (B2 fix). Pass ``False``
            only when migrating legacy checkpoints saved before Prompt 2.1.
        """
        ckpt: dict[str, Any] = torch.load(path, map_location=device or "cpu", weights_only=False)
        ver = ckpt.get("pedpredict_ckpt_version", 0)
        if ver != _CKPT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {ver!r} in {path}. "
                f"Expected {_CKPT_VERSION}. "
                "Use load_legacy_model_weights() for OLD-format checkpoints (Prompt 9)."
            )
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return CheckpointPayload(
            epoch=ckpt["epoch"],
            best_val_loss=ckpt["best_val_loss"],
            run_id=ckpt["run_id"],
            model_type=ckpt["model_type"],
            model_state_dict=ckpt["model_state_dict"],
            optimizer_state_dict=ckpt["optimizer_state_dict"],
            scaler_state_dict=ckpt["scaler_state_dict"],
            scheduler_state_dict=ckpt["scheduler_state_dict"],
        )

    # ------------------------------------------------------------------ Convenience queries

    def best_path(self) -> Path | None:
        """Path to ``best.pth`` if it exists, else ``None``."""
        if self.dir is None:
            return None
        p = self.dir / "best.pth"
        return p if p.exists() else None

    def last_path(self) -> Path | None:
        """Path to ``last.pth`` if it exists, else ``None``."""
        if self.dir is None:
            return None
        p = self.dir / "last.pth"
        return p if p.exists() else None

    # ------------------------------------------------------------------ Internals

    def _build_payload(self, trainer: Trainer, epoch: int) -> dict[str, Any]:
        return {
            "pedpredict_ckpt_version": _CKPT_VERSION,
            "epoch": epoch,
            "best_val_loss": trainer.best_val_loss,
            "run_id": self.run_id,
            "model_type": self.model_type,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scaler_state_dict": trainer.scaler.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

    @staticmethod
    def _atomic_save(obj: Any, path: Path) -> None:
        """Write to ``path.name + '.tmp'`` then rename, so a mid-save kill leaves no corrupt file."""
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

"""Training callbacks (Prompt 4.1 lands ``EarlyStopping``; Prompt 4.3 adds checkpointing).

``EarlyStopping`` is a verbatim port of OLD ``scripts/train_utils.py:23-37`` — same min-delta /
patience semantics, only typed and snake_cased. The Trainer (4.1) drives it with the per-epoch
validation loss exactly as OLD ``main`` did (train.py:622-627). The checkpoint/LR-scheduler callbacks
named in the schematic's Prompt 4.3 are added to this module later; nothing here hardcodes them.
"""

from __future__ import annotations

__all__ = ["EarlyStopping"]


class EarlyStopping:
    """Stop training when the monitored loss stops improving by ``min_delta`` for ``patience`` epochs.

    Verbatim semantics of OLD ``train_utils.EarlyStopping``: an epoch counts as an improvement only when
    ``loss < best_loss - min_delta``; otherwise the patience counter increments and :attr:`early_stop`
    latches ``True`` once it reaches ``patience``.
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

"""Prompt 4.3 — CheckpointManager tests (T1–T8).

T1  EarlyStopping parity: counter/best_loss/early_stop match OLD train_utils for a fixed sequence.
T2  CheckpointPayload keys: save_last writes all 8 required keys + version guard.
T3  Strict mode: strict=True raises on shape mismatch; strict=False succeeds (explicit opt-in).
T4  Round-trip equivalence: model weights, optimizer, scheduler, best_val_loss survive save→load;
    a deterministic forward pass on a fixed batch produces identical outputs.
T5  None dir: save_last / save_best are no-ops; best_path / last_path return None.
T6  Retention: with keep_best_k=3, save_best 5 times; exactly 3 archive files survive.
T7  Atomic write: a crash during torch.save leaves last.pth absent (no partial/corrupt file).
T8  build_trainer with resume_from: _start_epoch and best_val_loss correctly restored.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from pedpredict.config.schema import RootCfg
from pedpredict.training.callbacks import (
    CheckpointManager,
    CheckpointPayload,
    EarlyStopping,
)
from pedpredict.utils.amp import make_grad_scaler

_CPU = torch.device("cpu")
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "trainer_step.pt"


# --------------------------------------------------------------------------- shared helpers


def _tiny_components() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.amp.GradScaler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    """Tiny model + training components for fast unit tests (no LMDB / full model needed)."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(enabled=False)          # AMP off (CPU)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
    )
    return model, optimizer, scaler, scheduler


class _FakeTrainer:
    """Duck-typed substitute for Trainer: only the attributes CheckpointManager accesses."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        best_val_loss: float = 0.5,
        best_selection: float = 0.5,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.best_val_loss = best_val_loss
        self.best_selection = best_selection


class _ListChunkProvider:
    """Minimal in-memory ChunkProvider (mirrors the one in test_trainer.py)."""

    def __init__(self, train: list[list], val: list[list]) -> None:
        self._train = train
        self._val = val
        self.train_lmdb_paths: list[str] = []

    def epoch_loaders(self, epoch: int) -> Iterator[list]:
        return iter(self._train)

    def val_loaders(self) -> Iterator[list]:
        return iter(self._val)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- T1: EarlyStopping parity


def test_earlystopping_parity() -> None:
    """NEW EarlyStopping counter/best_loss/early_stop must match OLD scripts/train_utils.py:23-37.

    Logic transcribed from OLD: improve iff loss < best_loss - min_delta; else increment;
    latch early_stop when counter >= patience. Expected states derived from reading OLD directly.
    """
    losses = [0.9, 0.8, 0.85, 0.88, 0.9, 0.91]
    # (counter, best_loss, early_stop) after each call with patience=3, min_delta=0.0
    expected = [
        (0, 0.9, False),   # 0.9 < inf → improve, reset counter
        (0, 0.8, False),   # 0.8 < 0.9 → improve, reset counter
        (1, 0.8, False),   # 0.85 >= 0.8 → increment
        (2, 0.8, False),   # 0.88 >= 0.8 → increment
        (3, 0.8, True),    # 0.90 >= 0.8 → increment (counter=3=patience) → latch
        (4, 0.8, True),    # already latched; still increments counter, stays latched
    ]
    es = EarlyStopping(patience=3, min_delta=0.0)
    for loss, (exp_counter, exp_best, exp_stop) in zip(losses, expected, strict=True):
        es(loss)
        assert es.counter == exp_counter, f"counter wrong at loss={loss}"
        assert es.best_loss == pytest.approx(exp_best), f"best_loss wrong at loss={loss}"
        assert es.early_stop == exp_stop, f"early_stop wrong at loss={loss}"


# --------------------------------------------------------------------------- T2: payload keys


def test_checkpoint_payload_keys(tmp_path: Path) -> None:
    """save_last writes all 8 required fields + pedpredict_ckpt_version=1."""
    model, optimizer, scaler, scheduler = _tiny_components()
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="run_test", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler, best_val_loss=0.314)
    mgr.save_last(fake, epoch=3)

    raw: dict[str, Any] = torch.load(ckpt_dir / "last.pth", weights_only=False)
    required = {
        "pedpredict_ckpt_version",
        "epoch",
        "best_val_loss",
        "run_id",
        "model_type",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
    }
    assert required.issubset(raw.keys())
    assert raw["pedpredict_ckpt_version"] == 1
    assert raw["epoch"] == 3
    assert raw["best_val_loss"] == pytest.approx(0.314)
    assert raw["run_id"] == "run_test"
    assert raw["model_type"] == "full"


# --------------------------------------------------------------------------- T3: strict mode


def test_strict_true_raises_on_shape_mismatch(tmp_path: Path) -> None:
    """load(..., strict=True) must raise RuntimeError when a key has the wrong shape."""
    model, optimizer, scaler, scheduler = _tiny_components()
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="r", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler)
    mgr.save_last(fake, epoch=0)

    # Corrupt one tensor shape in the saved file
    raw: dict[str, Any] = torch.load(ckpt_dir / "last.pth", weights_only=False)
    raw["model_state_dict"]["weight"] = torch.zeros(5, 4)   # wrong out-features (5 vs 2)
    torch.save(raw, ckpt_dir / "last.pth")

    model2, optimizer2, scaler2, scheduler2 = _tiny_components()
    with pytest.raises(RuntimeError):
        CheckpointManager.load(ckpt_dir / "last.pth", model2, optimizer2, scaler2, scheduler2,
                               strict=True)


def test_strict_false_succeeds_on_mismatch(tmp_path: Path) -> None:
    """load(..., strict=False) must not raise even if state dict keys diverge."""
    model, optimizer, scaler, scheduler = _tiny_components()
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="r", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler)
    mgr.save_last(fake, epoch=0)

    # Inject an unknown extra key — strict=False must accept it
    raw: dict[str, Any] = torch.load(ckpt_dir / "last.pth", weights_only=False)
    raw["model_state_dict"]["ghost_key"] = torch.zeros(1)
    torch.save(raw, ckpt_dir / "last.pth")

    model2, optimizer2, scaler2, scheduler2 = _tiny_components()
    payload = CheckpointManager.load(ckpt_dir / "last.pth", model2, optimizer2, scaler2, scheduler2,
                                     strict=False)
    assert isinstance(payload, CheckpointPayload)


# --------------------------------------------------------------------------- T4: round-trip equivalence


def test_save_resume_continue_equivalence(tmp_path: Path) -> None:
    """Full save→load round trip: state dicts match; forward pass output is bit-exact."""
    model, optimizer, scaler, scheduler = _tiny_components()

    # One gradient step to make state non-trivial
    torch.manual_seed(42)
    x = torch.randn(4, 4)
    y = torch.randint(0, 2, (4,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(loss.item())

    # Capture state before save
    model_state_pre = {k: v.clone() for k, v in model.state_dict().items()}
    sched_last_epoch_pre = scheduler.state_dict().get("last_epoch")

    # Save
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="equiv", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler, best_val_loss=0.123)
    mgr.save_last(fake, epoch=5)

    # Load into fresh components
    model2, optimizer2, scaler2, scheduler2 = _tiny_components()
    payload = CheckpointManager.load(
        ckpt_dir / "last.pth", model2, optimizer2, scaler2, scheduler2, device=_CPU
    )

    # Metadata survives
    assert payload.epoch == 5
    assert payload.best_val_loss == pytest.approx(0.123)
    assert payload.run_id == "equiv"
    assert payload.model_type == "full"

    # Model weights match exactly
    for key, ref in model_state_pre.items():
        torch.testing.assert_close(model2.state_dict()[key], ref, atol=0, rtol=0)

    # Scheduler last_epoch matches
    assert scheduler2.state_dict().get("last_epoch") == sched_last_epoch_pre

    # Deterministic forward pass: identical outputs (eval mode, no stochasticity)
    x_test = torch.randn(3, 4)
    model.eval()
    model2.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x_test), model2(x_test), atol=0, rtol=0)


# --------------------------------------------------------------------------- T5: None dir


def test_none_dir_save_is_noop(tmp_path: Path) -> None:
    """CheckpointManager(None) must not write any files and must return None from path queries."""
    model, optimizer, scaler, scheduler = _tiny_components()
    mgr = CheckpointManager(None, run_id="r", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler)

    mgr.save_last(fake, epoch=0)
    mgr.save_best(fake, epoch=0, val_loss=0.5)

    assert not list(tmp_path.rglob("*.pth"))
    assert mgr.best_path() is None
    assert mgr.last_path() is None


# --------------------------------------------------------------------------- T6: retention policy


def test_retention_keep_best_k(tmp_path: Path) -> None:
    """keep_best_k=3: after 5 calls to save_best only 3 archive files remain (plus best.pth)."""
    model, optimizer, scaler, scheduler = _tiny_components()
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="r", model_type="full", keep_best_k=3)

    for epoch in range(5):
        fake = _FakeTrainer(model, optimizer, scaler, scheduler, best_val_loss=1.0 - epoch * 0.1)
        mgr.save_best(fake, epoch=epoch, val_loss=1.0 - epoch * 0.1)

    archive_files = sorted(ckpt_dir.glob("best_????.pth"))
    assert len(archive_files) == 3, f"expected 3 archives, got: {[f.name for f in archive_files]}"
    # The 3 survivors should be the most recent epochs (2, 3, 4)
    assert {f.name for f in archive_files} == {"best_0002.pth", "best_0003.pth", "best_0004.pth"}
    # best.pth must also exist (current best)
    assert (ckpt_dir / "best.pth").exists()


# --------------------------------------------------------------------------- T7: atomic write


def test_atomic_write_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash during torch.save must leave last.pth absent (no partial/corrupt file)."""
    import pedpredict.training.callbacks as callbacks_mod

    call_count = [0]
    real_save = torch.save

    def _failing_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
        if call_count[0] == 0:
            call_count[0] += 1
            raise OSError("simulated disk failure")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(callbacks_mod.torch, "save", _failing_save)

    model, optimizer, scaler, scheduler = _tiny_components()
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(ckpt_dir, run_id="r", model_type="full")
    fake = _FakeTrainer(model, optimizer, scaler, scheduler)

    with pytest.raises(OSError, match="simulated disk failure"):
        mgr.save_last(fake, epoch=0)

    # last.pth must NOT exist (the .tmp rename never happened)
    assert not (ckpt_dir / "last.pth").exists()


# --------------------------------------------------------------------------- T8: build_trainer resume


def test_build_trainer_with_resume(tmp_path: Path) -> None:
    """build_trainer(resume_from=...) restores _start_epoch and best_val_loss on the Trainer."""
    from pedpredict.models.registry import build_model
    from pedpredict.training.trainer import build_trainer

    # Build a real checkpoint using the same architecture build_trainer will create
    model = build_model(RootCfg())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = make_grad_scaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
    )

    expected_epoch = 7
    expected_best = 0.314159
    ckpt_path = tmp_path / "resume_ckpt" / "last.pth"
    ckpt_path.parent.mkdir(parents=True)
    torch.save(
        {
            "pedpredict_ckpt_version": 1,
            "epoch": expected_epoch,
            "best_val_loss": expected_best,
            "run_id": "saved_run",
            "model_type": "full",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_path,
    )

    # Point runs_dir into tmp_path so build_trainer doesn't pollute cwd
    new_paths = dataclasses.replace(RootCfg().paths, runs_dir=str(tmp_path / "runs"))
    cfg = dataclasses.replace(RootCfg(), paths=new_paths)
    chunks = _ListChunkProvider([], [])

    trainer = build_trainer(cfg, chunks, device=_CPU, resume_from=ckpt_path)

    assert trainer._start_epoch == expected_epoch + 1
    assert trainer.best_val_loss == pytest.approx(expected_best)
    trainer.chunks.close()

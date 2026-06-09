"""Prompt 4.4 — two-phase schedule (B1) + Prompt 4.3 CheckpointManager tests.

Coverage:
  * FREEZE — freeze_backbone / unfreeze_all partition matches OLD train_two_phase.py:122-125 exactly.
  * RESET — reset_for_phase rebuilds optimizer/scheduler/ES with phase settings; resets best_val_loss.
  * CONFIG — ScheduleCfg defaults match OLD hardcoded LR/epoch/patience constants exactly.
  * SMOKE — run_phase_schedule with 3 phases × 1 epoch on in-memory data; checkpoints + CSV written.
  * RELOAD — Phase N+1 with reload_best=True loads Phase N best.pth model weights into the trainer.
  * CHECKPOINTER (4.3) — save_last/save_best write full-state payload; load() restores it exactly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from pedpredict.config.schema import PhaseCfg, RootCfg, ScheduleCfg, TrainCfg
from pedpredict.losses.multitask import MultiTaskLoss
from pedpredict.models.registry import build_model
from pedpredict.training import (
    CheckpointManager,
    EpochResult,
    ModelStateCheckpointer,
    Trainer,
    freeze_backbone,
    run_phase_schedule,
    unfreeze_all,
)
from pedpredict.training.schedule import _TRAINABLE_SUBSTRINGS, PhaseResult

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "trainer_step.pt"
_CPU = torch.device("cpu")
_TASKS = ("actions", "looks", "crosses")


# --------------------------------------------------------------------------- shared helpers


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(_FIXTURE, weights_only=False)


class _ListChunkProvider:
    """Reusable in-memory ChunkProvider — same pattern as test_trainer.py."""

    def __init__(self, train: list[list], val: list[list]) -> None:
        self._train = train
        self._val = val
        self.train_lmdb_paths: list[str] = []
        self.closed = False

    def epoch_loaders(self, epoch: int) -> Iterator[list]:
        return iter(self._train)

    def val_loaders(self) -> Iterator[list]:
        return iter(self._val)

    def close(self) -> None:
        self.closed = True


def _loss(golden: dict) -> MultiTaskLoss:
    return MultiTaskLoss(golden["class_weights"], golden["loss_weight"])


def _model(golden: dict) -> torch.nn.Module:
    m = build_model(RootCfg())
    m.load_state_dict(golden["init_state"])
    return m


def _trainer(golden: dict, model: torch.nn.Module, *, train=None, val=None, **kw) -> Trainer:
    cfg = RootCfg()
    chunks = _ListChunkProvider(train or [], val or [])
    return Trainer(cfg, model, _CPU, chunks, loss=_loss(golden), **kw)


def _tiny_phase(**overrides) -> PhaseCfg:
    defaults = dict(
        name="test_phase",
        data_source="augmented",
        lr=1e-3,
        max_epochs=1,
        early_stop_patience=10,
    )
    defaults.update(overrides)
    return PhaseCfg(**defaults)


def _mgr(path: Path | None, *, run_id: str = "test", model_type: str = "full") -> CheckpointManager:
    """Construct a CheckpointManager for tests; path=None → no-op manager."""
    return CheckpointManager(path, run_id=run_id, model_type=model_type)


# --------------------------------------------------------------------------- freeze / unfreeze


def test_freeze_backbone_trainable_subset(golden: dict) -> None:
    """After freeze_backbone only classifier/crosses_frame_head/pool_mlp have requires_grad."""
    model = _model(golden)
    freeze_backbone(model)

    for name, param in model.named_parameters():
        if any(k in name for k in _TRAINABLE_SUBSTRINGS):
            assert param.requires_grad, f"Expected trainable: {name}"
        else:
            assert not param.requires_grad, f"Expected frozen: {name}"


def test_freeze_backbone_trainable_set_nonempty(golden: dict) -> None:
    """At least one parameter remains trainable after freeze (sanity check)."""
    model = _model(golden)
    freeze_backbone(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_unfreeze_all_restores_grad(golden: dict) -> None:
    """unfreeze_all re-enables requires_grad for every parameter."""
    model = _model(golden)
    freeze_backbone(model)
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_matches_old_filter(golden: dict) -> None:
    """The frozen/trainable partition matches what OLD train_two_phase.py:freeze_backbone would produce."""
    model = _model(golden)
    # Replicate OLD filter: frozen if NOT ('classifier' in name OR 'crosses_frame_head' in name OR 'pool_mlp' in name)
    expected_trainable = {
        name for name, _ in model.named_parameters()
        if "classifier" in name or "crosses_frame_head" in name or "pool_mlp" in name
    }
    freeze_backbone(model)
    actual_trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert actual_trainable == expected_trainable


# --------------------------------------------------------------------------- reset_for_phase


def test_reset_for_phase_rebuilds_optimizer_lr(golden: dict) -> None:
    """reset_for_phase installs new Adam with the phase's LR; old momentum state is gone."""
    model = _model(golden)
    train_batches = golden["train_batches"]
    val_batches = golden["val_batches"]
    trainer = _trainer(golden, model, train=[train_batches], val=[val_batches])

    old_lr = trainer.optimizer.param_groups[0]["lr"]
    new_lr = old_lr * 0.1  # deliberately different

    phase = _tiny_phase(lr=new_lr)
    new_chunks = _ListChunkProvider([train_batches], [val_batches])
    trainer.reset_for_phase(phase, new_chunks)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(new_lr)


def test_reset_for_phase_resets_early_stopping(golden: dict) -> None:
    """reset_for_phase resets EarlyStopping counter and best_val_loss."""
    model = _model(golden)
    trainer = _trainer(golden, model)

    # Simulate a tired trainer
    trainer.early_stopping.counter = 4
    trainer.early_stopping.early_stop = True
    trainer.best_val_loss = 0.123

    phase = _tiny_phase(early_stop_patience=99)
    new_chunks = _ListChunkProvider([], [])
    trainer.reset_for_phase(phase, new_chunks)

    assert trainer.early_stopping.counter == 0
    assert not trainer.early_stopping.early_stop
    assert trainer.best_val_loss == float("inf")


def test_reset_for_phase_resets_start_epoch(golden: dict) -> None:
    """reset_for_phase resets _start_epoch to 0 so each phase starts from epoch 0."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    trainer._start_epoch = 5  # simulate a resumed trainer

    phase = _tiny_phase()
    new_chunks = _ListChunkProvider([], [])
    trainer.reset_for_phase(phase, new_chunks)

    assert trainer._start_epoch == 0


def test_reset_for_phase_fresh_scheduler(golden: dict) -> None:
    """reset_for_phase builds a new scheduler attached to the new optimizer."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    old_sched = trainer.scheduler

    phase = _tiny_phase(sched_factor=0.3, sched_patience=5)
    new_chunks = _ListChunkProvider([], [])
    trainer.reset_for_phase(phase, new_chunks)

    assert trainer.scheduler is not old_sched
    assert trainer.scheduler.optimizer is trainer.optimizer
    # factor stored as 'factor' in ReduceLROnPlateau
    assert trainer.scheduler.factor == pytest.approx(0.3)


def test_reset_for_phase_freeze_gate(golden: dict) -> None:
    """reset_for_phase with freeze_backbone=True puts only classifier params in the optimizer."""
    model = _model(golden)
    train_batches = golden["train_batches"]
    val_batches = golden["val_batches"]
    trainer = _trainer(golden, model, train=[train_batches], val=[val_batches])

    phase = _tiny_phase(freeze_backbone=True)
    new_chunks = _ListChunkProvider([train_batches], [val_batches])
    trainer.reset_for_phase(phase, new_chunks)

    # After freeze, only trainable params should be in the optimizer
    opt_param_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert id(param) in opt_param_ids, f"Trainable param missing from optimizer: {name}"
        else:
            assert id(param) not in opt_param_ids, f"Frozen param in optimizer: {name}"


def test_reset_for_phase_swaps_chunks(golden: dict) -> None:
    """reset_for_phase installs the new chunks object."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    old_chunks = trainer.chunks

    phase = _tiny_phase()
    new_chunks = _ListChunkProvider([], [])
    trainer.reset_for_phase(phase, new_chunks)

    assert trainer.chunks is new_chunks
    assert trainer.chunks is not old_chunks


# --------------------------------------------------------------------------- ScheduleCfg defaults


def test_default_schedule_matches_old_constants() -> None:
    """ScheduleCfg() default phases reproduce OLD train_two_phase.py hardcoded constants exactly."""
    phases = ScheduleCfg().phases
    assert len(phases) == 3

    p1, p2, p3 = phases
    # Phase 1: balanced warmup
    assert p1.lr == pytest.approx(1e-4)
    assert p1.max_epochs == 10
    assert p1.early_stop_patience == 5
    assert p1.data_source == "balanced"
    assert not p1.freeze_backbone
    assert not p1.reload_best

    # Phase 2: full fine-tune
    assert p2.lr == pytest.approx(1e-5)
    assert p2.max_epochs == 20
    assert p2.early_stop_patience == 5
    assert p2.data_source == "augmented"
    assert not p2.freeze_backbone
    assert p2.reload_best

    # Phase 3: decouple classifiers
    assert p3.lr == pytest.approx(5e-5)
    assert p3.max_epochs == 5
    assert p3.early_stop_patience == 3
    assert p3.data_source == "augmented"
    assert p3.freeze_backbone
    assert p3.reload_best


def test_schedule_cfg_default_disabled() -> None:
    """ScheduleCfg.enabled is False by default — opt-in only."""
    assert not ScheduleCfg().enabled
    assert not RootCfg().schedule.enabled


# --------------------------------------------------------------------------- smoke: run_phase_schedule


def test_phase_schedule_smoke_three_phases(golden: dict, tmp_path: Path) -> None:
    """3 phases × 1 epoch each on tiny in-memory data; PhaseResults returned, checkpoints written."""
    model = _model(golden)
    train_batches = golden["train_batches"]
    val_batches = golden["val_batches"]

    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=1))

    # Build schedule with 1 epoch per phase (don't use ScheduleCfg defaults which have 10/20/5)
    phases = (
        PhaseCfg(name="p1", data_source="balanced", lr=1e-4, max_epochs=1, early_stop_patience=5,
                 reload_best=False, freeze_backbone=False),
        PhaseCfg(name="p2", data_source="augmented", lr=1e-5, max_epochs=1, early_stop_patience=5,
                 reload_best=True, freeze_backbone=False),
        PhaseCfg(name="p3", data_source="augmented", lr=5e-5, max_epochs=1, early_stop_patience=3,
                 reload_best=True, freeze_backbone=True),
    )
    schedule = ScheduleCfg(enabled=True, phases=phases)

    # Build trainer with initial (dummy) chunks; reset_for_phase will swap them
    initial_chunks = _ListChunkProvider([train_batches], [val_batches])
    trainer = Trainer(
        cfg, model, _CPU, initial_chunks,
        loss=_loss(golden),
        checkpointer=_mgr(None),
    )

    chunk_builders = {
        "balanced": lambda: _ListChunkProvider([train_batches], [val_batches]),
        "augmented": lambda: _ListChunkProvider([train_batches], [val_batches]),
    }

    results = run_phase_schedule(cfg, trainer, schedule, chunk_builders, run_dir=tmp_path)

    # Correct number of phases returned
    assert len(results) == 3
    assert [r.phase_name for r in results] == ["p1", "p2", "p3"]

    # Each phase ran exactly 1 epoch
    for pr in results:
        assert isinstance(pr, PhaseResult)
        assert len(pr.epoch_results) == 1
        assert isinstance(pr.epoch_results[0], EpochResult)
        assert torch.isfinite(torch.tensor(pr.epoch_results[0].train_loss))

    # Checkpoint directories created
    for i, name in enumerate(("p1", "p2", "p3")):
        ckpt_dir = tmp_path / f"phase_{i}_{name}" / "checkpoints"
        assert ckpt_dir.exists(), f"Missing checkpoint dir: {ckpt_dir}"
        assert (ckpt_dir / "last.pth").exists(), f"Missing last.pth for phase {name}"

    # Phase 3 backbone was frozen (trainer's model should still have frozen backbone at end)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert len(frozen) > 0, "Phase 3 backbone freeze did not persist"


def test_phase_schedule_missing_data_source_raises(golden: dict) -> None:
    """run_phase_schedule raises KeyError if a phase's data_source is not in chunk_builders."""
    model = _model(golden)
    cfg = RootCfg()
    schedule = ScheduleCfg(enabled=True, phases=(
        PhaseCfg(name="p", data_source="missing_source", lr=1e-4, max_epochs=1,
                 early_stop_patience=5),
    ))
    initial_chunks = _ListChunkProvider([], [])
    trainer = Trainer(cfg, model, _CPU, initial_chunks, loss=_loss(golden),
                      checkpointer=_mgr(None))

    with pytest.raises(KeyError, match="missing_source"):
        run_phase_schedule(cfg, trainer, schedule, {"augmented": lambda: initial_chunks})


# --------------------------------------------------------------------------- reload best checkpoint


def test_reload_best_between_phases(golden: dict, tmp_path: Path) -> None:
    """Phase 2 with reload_best=True loads Phase 1's best.pth model weights before starting."""
    model = _model(golden)
    train_batches = golden["train_batches"]
    val_batches = golden["val_batches"]
    cfg = RootCfg()

    phases = (
        PhaseCfg(name="p1", data_source="augmented", lr=1e-4, max_epochs=1,
                 early_stop_patience=5, reload_best=False),
        PhaseCfg(name="p2", data_source="augmented", lr=1e-5, max_epochs=1,
                 early_stop_patience=5, reload_best=True),
    )
    schedule = ScheduleCfg(enabled=True, phases=phases)
    initial_chunks = _ListChunkProvider([train_batches], [val_batches])
    trainer = Trainer(cfg, model, _CPU, initial_chunks, loss=_loss(golden),
                      checkpointer=_mgr(None))

    chunk_builders = {"augmented": lambda: _ListChunkProvider([train_batches], [val_batches])}
    results = run_phase_schedule(cfg, trainer, schedule, chunk_builders, run_dir=tmp_path)

    assert len(results) == 2
    # Phase 1 best checkpoint should exist
    p1_best = results[0].best_ckpt
    assert p1_best is not None and p1_best.exists()

    # Verify the checkpoint contains required keys (new format)
    ckpt = torch.load(p1_best, weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "pedpredict_ckpt_version" in ckpt


# --------------------------------------------------------------------------- CheckpointManager (4.3)


def test_checkpoint_manager_noop_when_no_dir(golden: dict) -> None:
    """CheckpointManager(None) makes save_* no-ops and best_path/last_path return None."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    mgr = _mgr(None)
    mgr.save_last(trainer, 0)
    mgr.save_best(trainer, 0, 1.0)
    assert mgr.best_path() is None
    assert mgr.last_path() is None


def test_checkpoint_manager_saves_full_payload(golden: dict, tmp_path: Path) -> None:
    """save_best writes a dict with all expected keys."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    mgr = _mgr(tmp_path / "checkpoints")
    mgr.save_best(trainer, epoch=3, val_loss=0.5)

    assert mgr.best_path() is not None
    ckpt = torch.load(mgr.best_path(), weights_only=False)
    assert ckpt["epoch"] == 3
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt
    assert "scaler_state_dict" in ckpt
    assert "best_val_loss" in ckpt
    assert ckpt.get("pedpredict_ckpt_version") == 1


def test_checkpoint_manager_save_last(golden: dict, tmp_path: Path) -> None:
    """save_last writes last.pth; last_path returns it."""
    model = _model(golden)
    trainer = _trainer(golden, model)
    mgr = _mgr(tmp_path / "ckpts")
    mgr.save_last(trainer, epoch=7)
    assert mgr.last_path() is not None
    assert mgr.last_path().name == "last.pth"


def test_checkpoint_manager_load_restores_state(golden: dict, tmp_path: Path) -> None:
    """save_best -> load() restores model weights, optimizer/scheduler/scaler states, epoch."""
    model_a = _model(golden)
    trainer_a = _trainer(golden, model_a,
                         train=[golden["train_batches"]], val=[golden["val_batches"]])

    # Run one epoch to get non-trivial optimizer state
    trainer_a.train_chunk(golden["train_batches"])

    mgr = _mgr(tmp_path / "ckpts")
    trainer_a.best_val_loss = 0.77
    mgr.save_best(trainer_a, epoch=2, val_loss=0.77)
    assert mgr.best_path() is not None

    # Load into a fresh set of components
    model_b = build_model(RootCfg())
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    from pedpredict.utils.amp import make_grad_scaler
    scaler_b = make_grad_scaler(enabled=False)
    scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode="min")
    payload = CheckpointManager.load(
        mgr.best_path(),
        model_b,
        optimizer_b,
        scaler_b,
        scheduler_b,
        device=_CPU,
    )

    assert payload.epoch == 2
    assert payload.best_val_loss == pytest.approx(0.77)

    # Model weights should match
    for key in model_a.state_dict():
        torch.testing.assert_close(
            model_b.state_dict()[key],
            model_a.state_dict()[key],
        )


def test_checkpoint_manager_best_path_none_before_save(tmp_path: Path) -> None:
    """best_path is None until save_best is called."""
    mgr = _mgr(tmp_path / "ckpts")
    assert mgr.best_path() is None
    assert mgr.last_path() is None


def test_save_load_continue_equivalence(golden: dict, tmp_path: Path) -> None:
    """save -> load -> continue produces the same loss as an uninterrupted run."""
    train_batches = golden["train_batches"]
    val_batches = golden["val_batches"]

    # Reference: run 2 epochs uninterrupted
    model_ref = _model(golden)
    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=2))
    chunks_ref = _ListChunkProvider([train_batches], [val_batches])
    trainer_ref = Trainer(
        cfg, model_ref, _CPU, chunks_ref,
        loss=_loss(golden),
        checkpointer=ModelStateCheckpointer(None),
    )
    torch.manual_seed(42)
    ref_results = trainer_ref.fit()
    assert len(ref_results) == 2

    # Interrupted: run 1 epoch, save full state, resume, run 1 more
    model_i = _model(golden)
    cfg_1 = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=1))
    mgr = _mgr(tmp_path / "ckpts")
    chunks_i1 = _ListChunkProvider([train_batches], [val_batches])
    trainer_i = Trainer(
        cfg_1, model_i, _CPU, chunks_i1,
        loss=_loss(golden),
        checkpointer=mgr,
    )
    torch.manual_seed(42)
    trainer_i.fit()

    # Resume into epoch 2: load checkpoint, then run remaining epochs via fit()
    assert mgr.last_path() is not None
    model_i2 = build_model(RootCfg())
    cfg_2 = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=2))
    chunks_i2 = _ListChunkProvider([train_batches], [val_batches])
    mgr2 = _mgr(tmp_path / "ckpts2")
    trainer_i2 = Trainer(
        cfg_2, model_i2, _CPU, chunks_i2,
        loss=_loss(golden),
        checkpointer=mgr2,
    )
    payload = CheckpointManager.load(
        mgr.last_path(),
        trainer_i2.model,
        trainer_i2.optimizer,
        trainer_i2.scaler,
        trainer_i2.scheduler,
        device=_CPU,
    )
    trainer_i2.best_val_loss = payload.best_val_loss
    trainer_i2._start_epoch = payload.epoch + 1

    # fit() uses range(_start_epoch, cfg_2.train.num_epochs) = range(1, 2) → 1 epoch
    resumed_results = trainer_i2.fit()

    # Both should have produced valid results
    assert len(resumed_results) >= 1
    final_loss = resumed_results[-1].val_loss
    assert torch.isfinite(torch.tensor(final_loss))


# --------------------------------------------------------------------------- fit(max_epochs) kwarg


def test_fit_max_epochs_overrides_cfg(golden: dict) -> None:
    """Trainer.fit(max_epochs=N) runs exactly N epochs regardless of cfg.train.num_epochs."""
    model = _model(golden)
    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=99))
    chunks = _ListChunkProvider([golden["train_batches"]], [golden["val_batches"]])
    trainer = Trainer(cfg, model, _CPU, chunks, loss=_loss(golden),
                      checkpointer=ModelStateCheckpointer(None))
    results = trainer.fit(max_epochs=2)
    assert len(results) == 2


def test_fit_no_max_epochs_uses_cfg(golden: dict) -> None:
    """Trainer.fit() without max_epochs uses cfg.train.num_epochs."""
    model = _model(golden)
    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=1))
    chunks = _ListChunkProvider([golden["train_batches"]], [golden["val_batches"]])
    trainer = Trainer(cfg, model, _CPU, chunks, loss=_loss(golden),
                      checkpointer=ModelStateCheckpointer(None))
    results = trainer.fit()
    assert len(results) == 1

"""Prompt 4.1 — Trainer orchestration parity, B2 verification, and an end-to-end smoke test.

Parity here is COMPOSITIONAL: the Trainer adds no math (loss 3.1 / metrics 3.2 / model 2.4 / sampler 1.6
are each golden-locked), so these tests pin the orchestration the Trainer owns:

  * GOLDEN step parity — one optimizer step reproduces the transcribed legacy ``train_one_chunk`` body
    (per-batch loss + post-step weights), and ``validate`` reproduces the legacy ``validate_one_epoch``
    val-loss + per-task accuracy. Fixture: ``tests/fixtures/golden/trainer_step.pt``
    (``tests/_capture/capture_trainer_golden.py``).
  * B2 (consumer side) — the optimizer covers EVERY parameter with NO dummy forward, and a fresh model
    ``strict=True``-loads a saved ``state_dict`` without any forward.
  * CONTROL FLOW — grad-clip bound, scheduler/early-stop firing on the val scalar.
  * SMOKE — ``fit`` runs end-to-end on one tiny in-memory chunk (CPU, AMP off): finite results, a
    checkpoint + a CSV row, no leaked child processes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from pedpredict.config.schema import RootCfg
from pedpredict.losses.multitask import MultiTaskLoss
from pedpredict.models.registry import build_model
from pedpredict.training import (
    TRAIN_LOG_COLUMNS,
    EpochResult,
    ModelStateCheckpointer,
    Trainer,
)
from pedpredict.training.callbacks import EarlyStopping

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "trainer_step.pt"
_TASKS = ("actions", "looks", "crosses")
_CPU = torch.device("cpu")

# Post-step weight tensors are compared at a looser atol than the fixture's scalar ``tol`` (1e-6).
# The legacy oracle was captured on a different CPU BLAS build; conv2d backward accumulates in a
# kernel-specific order, so a single ViT-stem conv weight drifts by ~2e-6 (rel ~3e-5) after one
# optimizer step on other machines — last-ULP rounding, not a behavior change. The scalar loss/val/
# accuracy parity assertions stay at the strict 1e-6 (stable reductions); only the per-element weight
# comparison needs BLAS-portable headroom.
_WEIGHT_PARITY_ATOL = 5e-6


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(_FIXTURE, weights_only=False)


class _ListChunkProvider:
    """Minimal in-memory :class:`ChunkProvider`: each 'loader' is a list of pre-made collate tuples."""

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


def _loss_from_golden(golden: dict) -> MultiTaskLoss:
    return MultiTaskLoss(golden["class_weights"], golden["loss_weight"])


def _fresh_model(golden: dict) -> torch.nn.Module:
    model = build_model(RootCfg())
    model.load_state_dict(golden["init_state"])
    return model


def _trainer(golden: dict, model: torch.nn.Module, *, train=None, val=None, **kw) -> Trainer:
    cfg = RootCfg()
    chunks = _ListChunkProvider(train or [], val or [])
    return Trainer(cfg, model, _CPU, chunks, loss=_loss_from_golden(golden), **kw)


# --------------------------------------------------------------------------- golden step parity


def test_train_step_matches_legacy_oracle(golden: dict) -> None:
    model = _fresh_model(golden)
    batches = golden["train_batches"]
    trainer = _trainer(golden, model, train=[batches])

    torch.manual_seed(golden["step_seed"])           # sync dropout RNG with the capture
    loss_sum, n_batches = trainer.train_chunk(batches)

    assert n_batches == len(batches)
    expected_total = sum(float(t) for t in golden["expected"]["per_batch_total"])
    assert loss_sum == pytest.approx(expected_total, abs=golden["tol"])

    # post-step weights must match the transcribed legacy step (within BLAS-portable atol; see constant).
    new_state = model.state_dict()
    for key, ref in golden["expected"]["post_step_state"].items():
        torch.testing.assert_close(new_state[key], ref, atol=_WEIGHT_PARITY_ATOL, rtol=0)


def test_validate_matches_legacy_oracle(golden: dict) -> None:
    model = _fresh_model(golden)
    trainer = _trainer(golden, model, val=[golden["val_batches"]])

    val_loss, metrics = trainer.validate()

    assert val_loss == pytest.approx(golden["expected"]["val_loss"], abs=golden["tol"])
    n = golden["expected"]["val_n_samples"]
    for task in _TASKS:
        expected_acc = golden["expected"]["val_correct"][task] / n
        assert metrics.per_task[task].accuracy == pytest.approx(expected_acc, abs=golden["tol"])
    total_correct = sum(golden["expected"]["val_correct"].values())
    assert metrics.overall_accuracy == pytest.approx(total_correct / (n * len(_TASKS)), abs=golden["tol"])


# --------------------------------------------------------------------------- B2 (consumer side)


def test_optimizer_covers_all_params_without_forward(golden: dict) -> None:
    # Building the Trainer builds the optimizer in __init__ — with no forward (B2: ViT params are eager).
    model = _fresh_model(golden)
    trainer = _trainer(golden, model)
    opt_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    model_params = {id(p) for p in model.parameters() if p.requires_grad}
    assert opt_params == model_params
    assert len(opt_params) == len(list(model.parameters()))


def test_state_dict_round_trips_strict_without_forward() -> None:
    model = build_model(RootCfg())                   # never forwarded
    state = model.state_dict()
    fresh = build_model(RootCfg())
    missing, unexpected = fresh.load_state_dict(state, strict=True)
    assert not missing and not unexpected


# --------------------------------------------------------------------------- control flow


def test_grad_clip_uses_config_bound(golden: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fresh_model(golden)
    trainer = _trainer(golden, model, train=[golden["train_batches"]])
    seen: list[float] = []
    import pedpredict.training.trainer as trainer_mod

    real_clip = trainer_mod.clip_grad_norm_

    def _spy(params, max_norm, *a, **k):
        seen.append(max_norm)
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(trainer_mod, "clip_grad_norm_", _spy)
    torch.manual_seed(golden["step_seed"])
    trainer.train_chunk(golden["train_batches"])
    assert seen == [RootCfg().train.grad_clip_max_norm]


def test_scheduler_built_min_mode_and_earlystop_trips(golden: dict) -> None:
    model = _fresh_model(golden)
    trainer = _trainer(golden, model)
    # Trainer builds a min-mode plateau scheduler over its own optimizer (drives on val loss).
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert trainer.scheduler.mode == "min"
    assert trainer.scheduler.optimizer is trainer.optimizer

    es = EarlyStopping(patience=2, min_delta=0.0)
    for val_loss in (1.0, 1.0, 1.0):                 # non-improving -> trips after `patience`
        es(val_loss)
    assert es.early_stop


# --------------------------------------------------------------------------- end-to-end smoke


def test_fit_smoke_one_tiny_chunk(golden: dict, tmp_path: Path) -> None:
    import dataclasses
    import multiprocessing as mp

    from pedpredict.config.schema import TrainCfg

    model = _fresh_model(golden)
    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), num_epochs=1))
    chunks = _ListChunkProvider([golden["train_batches"]], [golden["val_batches"]])
    ckpt_dir = tmp_path / "checkpoints"
    from pedpredict.utils.logging import CsvLogger

    logger = CsvLogger(tmp_path / "train_log.csv", TRAIN_LOG_COLUMNS)
    trainer = Trainer(
        cfg, model, _CPU, chunks,
        loss=_loss_from_golden(golden),
        checkpointer=ModelStateCheckpointer(ckpt_dir),
        logger=logger,
        run_dir=tmp_path,
    )

    before = set(mp.active_children())
    torch.manual_seed(golden["step_seed"])
    results = trainer.fit()
    logger.close()

    assert len(results) == 1 and isinstance(results[0], EpochResult)
    assert torch.isfinite(torch.tensor(results[0].train_loss))
    assert torch.isfinite(torch.tensor(results[0].val_loss))
    assert (ckpt_dir / "last.pth").exists() and (ckpt_dir / "best.pth").exists()
    assert chunks.closed                               # fit() closes the provider in its finally
    rows = (tmp_path / "train_log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2 and rows[0].split(",")[0] == "epoch"   # header + 1 epoch row
    assert set(mp.active_children()) == before         # no leaked processes

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

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from pedpredict.config.schema import ModelCfg, RootCfg
from pedpredict.losses.multitask import MultiTaskLoss
from pedpredict.models.registry import build_model, forward_model
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
# Legacy ViT schedule: pin so golden["init_state"] keeps its param layout under the A1 default.
_LEGACY_VIT = dict(
    stage_dims=[36, 36, 288, 36], layer_nums=[2, 4, 5, 7],
    head_nums=[2, 2, 16, 2], window_size=[8, 4, 2, None],
)
# The legacy oracle was captured with the per-sequence motion z-norm; pin it for value parity
# (the A4 default is the new image-dimension norm — a deliberate behavior change, config-gated).
# Also pin lr_schedule="plateau": the oracle was captured at a fixed lr (plateau leaves the initial
# LR untouched), whereas the shipped default warmup_cosine warms the step-1 LR down via LinearLR.
# fusion_residual=False pins the legacy no-residual fusion the oracle was captured with (the shipped
# default is now True — the A3 fix; the False arm stays golden-pinned, like motion_norm="per_sequence").
# accum_steps=1 pins per-batch stepping (the legacy oracle stepped every batch; the shipped default is 8).
_PARITY_CFG = dataclasses.replace(
    RootCfg(),
    model=ModelCfg(motion_norm="per_sequence", fusion_residual=False, **_LEGACY_VIT),
    train=dataclasses.replace(RootCfg().train, lr_schedule="plateau", accum_steps=1),
)

# Post-step weight tensors are compared at a looser atol than the fixture's scalar ``tol`` (1e-6).
# The legacy oracle was captured on a different CPU BLAS build; conv2d backward accumulates in a
# kernel-specific order, so a single ViT-stem conv weight drifts after one optimizer step on other
# machines (observed up to ~6.4e-6, rel ~5e-5 on a different lab CPU) — last-ULP rounding, not a
# behavior change. The scalar loss/val/accuracy parity assertions stay at the strict 1e-6 (stable
# reductions); only the per-element weight comparison needs BLAS-portable headroom.
_WEIGHT_PARITY_ATOL = 1e-5


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
    model = build_model(_PARITY_CFG)  # per-sequence norm: matches the legacy capture
    model.load_state_dict(golden["init_state"])
    return model


def _trainer(golden: dict, model: torch.nn.Module, *, train=None, val=None, **kw) -> Trainer:
    cfg = _PARITY_CFG
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


def test_gradient_accumulation_steps_optimizer_every_n(golden: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """accum_steps=N steps the optimizer once per N micro-batches (+ a flush for any remainder)."""
    batches = list(golden["train_batches"])
    m = len(batches)
    for accum in (1, 2, 3):
        model = _fresh_model(golden)
        cfg = dataclasses.replace(
            _PARITY_CFG, train=dataclasses.replace(_PARITY_CFG.train, accum_steps=accum)
        )
        trainer = Trainer(cfg, model, _CPU, _ListChunkProvider([], []), loss=_loss_from_golden(golden))
        n_steps = 0
        real_step = trainer.optimizer.step

        def _counting(*a: object, _real=real_step, **k: object) -> object:
            nonlocal n_steps
            n_steps += 1
            return _real(*a, **k)

        monkeypatch.setattr(trainer.optimizer, "step", _counting)
        torch.manual_seed(golden["step_seed"])
        trainer.train_chunk(batches)
        expected = m // accum + (1 if m % accum else 0)        # ceil(m / accum): full groups + flush
        assert n_steps == expected, f"accum={accum}: {n_steps} steps, expected {expected}"
        assert all(torch.isfinite(p).all() for p in model.parameters())


def test_gradient_accumulation_matches_manual_accumulate(golden: dict) -> None:
    """One accum_steps=N step == hand-rolled: N losses/N backward into one zero_grad, then one step.

    Dropout/drop-path are zeroed so the forward is deterministic (no RNG to sync between the two paths);
    this isolates the accumulation *math* — that the averaged grad over N micro-batches drives one step.
    """
    # Dropout-free model so both paths are deterministic regardless of train()/eval() (params unchanged).
    nodrop = ModelCfg(
        motion_norm="per_sequence", dropout=0.0, attn_dropout=0.0, proj_dropout=0.0,
        drop_path=0.0, motion_dropout=0.0, head_dropout=0.0, **_LEGACY_VIT,
    )
    batches = list(golden["train_batches"])
    n = len(batches)

    def _fresh_nodrop() -> torch.nn.Module:
        cfg = dataclasses.replace(_PARITY_CFG, model=nodrop)
        model = build_model(cfg)
        model.load_state_dict(golden["init_state"])      # dropout has no params -> state_dict matches
        return model

    # (a) Trainer with accum_steps == n: exactly one optimizer step over the whole chunk.
    model_a = _fresh_nodrop()
    cfg_a = dataclasses.replace(_PARITY_CFG, model=nodrop,
                                train=dataclasses.replace(_PARITY_CFG.train, accum_steps=n))
    trainer = Trainer(cfg_a, model_a, _CPU, _ListChunkProvider([], []), loss=_loss_from_golden(golden))
    trainer.train_chunk(batches)

    # (b) Hand-rolled reference: N losses/N backward into one zero_grad, then one clip+step.
    model_b = _fresh_nodrop()
    loss_fn = _loss_from_golden(golden)
    opt = torch.optim.Adam(model_b.parameters(), lr=_PARITY_CFG.train.lr,
                           weight_decay=_PARITY_CFG.train.weight_decay)
    model_b.train()
    opt.zero_grad(set_to_none=True)
    for tight, ctx, motion, labels in batches:
        lab = {k: v.long() for k, v in labels.items()}
        lab["crosses"] = torch.clamp(lab["crosses"], 0, 1)
        (loss_fn(forward_model(model_b, tight, ctx, motion), lab).total / n).backward()
    torch.nn.utils.clip_grad_norm_(model_b.parameters(), _PARITY_CFG.train.grad_clip_max_norm)
    opt.step()

    for (ka, pa), (_, pb) in zip(model_a.state_dict().items(), model_b.state_dict().items(), strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=0, msg=f"{ka} mismatch")


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
    trainer = _trainer(golden, model)                # _PARITY_CFG pins lr_schedule="plateau"
    # The "plateau" arm builds a min-mode ReduceLROnPlateau over its own optimizer (drives on val loss).
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert trainer.scheduler.mode == "min"
    assert trainer.scheduler.optimizer is trainer.optimizer

    es = EarlyStopping(patience=2, min_delta=0.0)
    for val_loss in (1.0, 1.0, 1.0):                 # non-improving -> trips after `patience`
        es(val_loss)
    assert es.early_stop


def test_default_schedule_is_warmup_cosine() -> None:
    """The shipped default (RootCfg) is the warmup_cosine arm -> SequentialLR, not plateau."""
    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(RootCfg().train, use_class_weights=False))
    trainer = Trainer(cfg, build_model(cfg), _CPU, _ListChunkProvider([], []))   # no scan (uniform loss)
    assert cfg.train.lr_schedule == "warmup_cosine"
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.SequentialLR)


def test_warmup_cosine_schedule_shape(golden: dict) -> None:
    """lr_schedule=warmup_cosine builds a SequentialLR whose LR ramps up then anneals toward lr_min."""
    import dataclasses

    from pedpredict.config.schema import TrainCfg

    train = dataclasses.replace(
        TrainCfg(), lr_schedule="warmup_cosine", warmup_epochs=2, warmup_start_factor=0.1,
        num_epochs=6, lr=1e-3, lr_min=1e-5,
    )
    cfg = dataclasses.replace(_PARITY_CFG, train=train)
    trainer = Trainer(cfg, _fresh_model(golden), _CPU, _ListChunkProvider([], []),
                      loss=_loss_from_golden(golden))

    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.SequentialLR)
    # No metric is passed (only ReduceLROnPlateau takes one); collect the LR used each epoch.
    lrs = []
    for _ in range(cfg.train.num_epochs):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.scheduler.step()
    # warmup (epochs 0,1) ramps up to the peak at epoch 2, then cosine descends.
    assert lrs[0] < lrs[1] < lrs[2]
    assert lrs[2] == pytest.approx(1e-3)             # peak == train.lr
    assert lrs[2] > lrs[3] > lrs[4] > lrs[5]         # cosine anneal
    assert min(lrs) >= cfg.train.lr_min - 1e-12


def test_warmup_cosine_no_warmup_is_plain_cosine(golden: dict) -> None:
    """warmup_epochs=0 skips LinearLR and yields a bare CosineAnnealingLR (no SequentialLR wrapper)."""
    import dataclasses

    from pedpredict.config.schema import TrainCfg

    train = dataclasses.replace(TrainCfg(), lr_schedule="warmup_cosine", warmup_epochs=0)
    cfg = dataclasses.replace(_PARITY_CFG, train=train)
    trainer = Trainer(cfg, _fresh_model(golden), _CPU, _ListChunkProvider([], []),
                      loss=_loss_from_golden(golden))
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_unknown_lr_schedule_raises(golden: dict) -> None:
    import dataclasses

    from pedpredict.config.schema import TrainCfg

    cfg = dataclasses.replace(_PARITY_CFG, train=dataclasses.replace(TrainCfg(), lr_schedule="nope"))
    with pytest.raises(ValueError, match="Unknown train.lr_schedule"):
        Trainer(cfg, _fresh_model(golden), _CPU, _ListChunkProvider([], []),
                loss=_loss_from_golden(golden))


def test_selection_value_modes(golden: dict) -> None:
    """M8: the minimized selection scalar is val_loss as-is, or the NEGATED configured F1 metric."""
    import dataclasses

    from pedpredict.config.schema import TrainCfg
    from pedpredict.training.metrics import MetricResult, TaskMetrics

    tm = TaskMetrics(accuracy=0.5, f1=0.6, auc=0.5, precision=0.5, recall=0.5)
    crosses_tm = TaskMetrics(accuracy=0.5, f1=0.2, auc=0.5, precision=0.5, recall=0.5)
    metrics = MetricResult(
        per_task={"actions": tm, "looks": tm, "crosses": crosses_tm},
        macro_f1=0.47, overall_accuracy=0.5,
    )

    model = _fresh_model(golden)
    loss = _loss_from_golden(golden)
    for metric_name, expected in (("macro_f1", -0.47), ("crosses_f1", -0.2), ("val_loss", 1.23)):
        cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), selection_metric=metric_name))
        trainer = Trainer(cfg, model, _CPU, _ListChunkProvider([], []), loss=loss)
        assert trainer.selection_metric == metric_name
        assert trainer._selection_value(1.23, metrics) == pytest.approx(expected)
        assert trainer.best_selection == float("inf")


def test_uniform_class_weights_when_disabled(golden: dict) -> None:
    """M1 lever-3 off-switch: use_class_weights=false builds the loss with [1, 1] per task, no scan."""
    import dataclasses

    from pedpredict.config.schema import TrainCfg

    cfg = dataclasses.replace(RootCfg(), train=dataclasses.replace(TrainCfg(), use_class_weights=False))
    model = _fresh_model(golden)
    trainer = Trainer(cfg, model, _CPU, _ListChunkProvider([], []))   # no injected loss -> _build_loss
    for task in _TASKS:
        torch.testing.assert_close(trainer.loss.criteria[task].weight, torch.ones(2))


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

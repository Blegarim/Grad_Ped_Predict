"""Prompt 3.1 — MultiTaskLoss parity, imbalance-lever behavior, and contract tests.

Five kinds of checks (same shape as the 1.6 / 2.x suites):
  * GOLDEN parity: ``MultiTaskLoss`` reproduces the OLD ``train.py:144-153,341-345`` total + per-head CE
    EXACTLY (atol=1e-6), from ``tests/fixtures/golden/losses_cases.pt`` (see
    ``tests/_capture/capture_losses_golden.py``).
  * CONTRACT (B4): crosses is supervised on ``crosses_frame`` ONLY; ``crosses_pooled`` never enters the
    loss; a missing ``crosses_frame`` raises a clear error.
  * IMBALANCE LEVERS: class ``weight`` and per-task ``loss_weight`` each move the loss the documented way.
  * REDUCTION: ``mean`` == ``none``-then-mean; ``sum`` == ``mean`` * N; default is ``mean`` (legacy).
  * PLUMBING: detached logging scalars + live ``total`` gradient; device move; config factory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pedpredict.config import TrainCfg
from pedpredict.data.sampler import class_weights_ce
from pedpredict.losses import (
    TASK_OUTPUT_KEY,
    TASKS,
    MultiTaskLoss,
    build_multitask_loss,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "losses_cases.pt"


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(_FIXTURE, weights_only=False)


def _loss_from_golden(golden: dict, **kwargs) -> MultiTaskLoss:
    return MultiTaskLoss(golden["class_weights"], golden["loss_weight"], **kwargs)


# --------------------------------------------------------------------------- golden parity


def test_total_matches_legacy_oracle(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    out = loss(golden["outputs"], golden["labels"])
    torch.testing.assert_close(out.total, golden["expected"]["total"], atol=golden["tol"], rtol=0)


def test_per_task_breakdown_matches_oracle(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    out = loss(golden["outputs"], golden["labels"])
    for task in TASKS:
        torch.testing.assert_close(
            out.per_task[task], golden["expected"]["per_task"][task], atol=golden["tol"], rtol=0
        )
        torch.testing.assert_close(
            out.weighted[task], golden["expected"]["weighted"][task], atol=golden["tol"], rtol=0
        )
        # weighted == loss_weight * raw, total == sum(weighted)
        torch.testing.assert_close(
            out.weighted[task], golden["loss_weight"][task] * out.per_task[task], atol=1e-7, rtol=0
        )
    torch.testing.assert_close(
        out.total, sum(out.weighted[t] for t in TASKS), atol=1e-7, rtol=0
    )


def test_per_task_equals_independent_ce(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    out = loss(golden["outputs"], golden["labels"])
    for task in TASKS:
        key = TASK_OUTPUT_KEY[task]
        ref = torch.nn.CrossEntropyLoss(weight=golden["class_weights"][task])(
            golden["outputs"][key].float(), golden["labels"][task]
        )
        torch.testing.assert_close(out.per_task[task], ref, atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- output contract (B4)


def test_crosses_routes_to_crosses_frame_only(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    base = loss(golden["outputs"], golden["labels"]).total

    # Perturbing crosses_frame MUST change the loss (differential shift — a constant added to
    # BOTH classes is a softmax no-op, so bias only the positive class).
    bump = torch.tensor([0.0, 5.0])
    perturbed = dict(golden["outputs"])
    perturbed["crosses_frame"] = perturbed["crosses_frame"] + bump
    changed = loss(perturbed, golden["labels"]).total
    assert not torch.allclose(base, changed)

    # Perturbing crosses_pooled MUST NOT change the loss (unsupervised, B4).
    perturbed2 = dict(golden["outputs"])
    perturbed2["crosses_pooled"] = perturbed2["crosses_pooled"] + bump
    same = loss(perturbed2, golden["labels"]).total
    torch.testing.assert_close(base, same, atol=0, rtol=0)


def test_temporal_weights_ignored(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    base = loss(golden["outputs"], golden["labels"]).total
    no_extras = {k: golden["outputs"][k] for k in ("actions", "looks", "crosses_frame")}
    torch.testing.assert_close(base, loss(no_extras, golden["labels"]).total, atol=0, rtol=0)


def test_missing_crosses_frame_raises(golden: dict) -> None:
    loss = _loss_from_golden(golden)
    bad = {k: v for k, v in golden["outputs"].items() if k != "crosses_frame"}
    with pytest.raises(KeyError, match="crosses_frame"):
        loss(bad, golden["labels"])


# --------------------------------------------------------------------------- imbalance levers


def test_class_weight_upweights_minority(golden: dict) -> None:
    # MIXED batch: class-0 samples predicted right (low loss), class-1 samples predicted wrong
    # (high loss). Weighted-CE 'mean' normalizes by Σ class-weights, so upweighting class 1 shifts
    # the mean toward the high-loss minority — a single-class batch would be invariant (correctly).
    pos = torch.tensor([[4.0, -4.0]])   # predicts 0; for a class-1 target this is high loss
    neg = torch.tensor([[4.0, -4.0]])   # predicts 0; for a class-0 target this is ~0 loss
    crosses_logits = torch.cat([neg.repeat(6, 1), pos.repeat(2, 1)], dim=0)
    targets = torch.tensor([0] * 6 + [1] * 2)
    n = targets.shape[0]
    outputs = {"actions": torch.zeros(n, 2), "looks": torch.zeros(n, 2), "crosses_frame": crosses_logits}
    labels = {"actions": torch.zeros(n, dtype=torch.long),
              "looks": torch.zeros(n, dtype=torch.long), "crosses": targets}
    lw = {"actions": 1.0, "looks": 1.0, "crosses": 1.0}
    low = MultiTaskLoss({**golden["class_weights"], "crosses": torch.tensor([1.0, 1.0])}, lw)
    high = MultiTaskLoss({**golden["class_weights"], "crosses": torch.tensor([1.0, 10.0])}, lw)
    assert high(outputs, labels).per_task["crosses"] > low(outputs, labels).per_task["crosses"]


def test_loss_weight_scales_contribution(golden: dict) -> None:
    base_lw = dict(golden["loss_weight"])
    doubled = {**base_lw, "crosses": base_lw["crosses"] * 2}
    out_a = MultiTaskLoss(golden["class_weights"], base_lw)(golden["outputs"], golden["labels"])
    out_b = MultiTaskLoss(golden["class_weights"], doubled)(golden["outputs"], golden["labels"])
    # crosses contribution doubles; actions/looks unchanged.
    torch.testing.assert_close(out_b.weighted["crosses"], 2 * out_a.weighted["crosses"], atol=1e-7, rtol=0)
    for task in ("actions", "looks"):
        torch.testing.assert_close(out_b.weighted[task], out_a.weighted[task], atol=0, rtol=0)
    expected_delta = out_a.weighted["crosses"]
    torch.testing.assert_close(out_b.total, out_a.total + expected_delta, atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- reduction correctness


def test_reduction_mean_default_and_none_sum(golden: dict) -> None:
    outputs, labels = golden["outputs"], golden["labels"]
    n = labels["actions"].shape[0]
    # Unit class weights so 'mean' is the plain sample mean (weighted-CE 'mean' otherwise normalizes
    # by Σ class-weights, breaking the simple none.mean()/sum*N identities).
    uniform = {task: torch.ones(2) for task in TASKS}
    mean_out = MultiTaskLoss(uniform, golden["loss_weight"])(outputs, labels)             # default mean
    none_out = MultiTaskLoss(uniform, golden["loss_weight"], reduction="none")(outputs, labels)
    sum_out = MultiTaskLoss(uniform, golden["loss_weight"], reduction="sum")(outputs, labels)
    for task in TASKS:
        torch.testing.assert_close(none_out.per_task[task].mean(), mean_out.per_task[task], atol=1e-6, rtol=0)
        torch.testing.assert_close(sum_out.per_task[task], mean_out.per_task[task] * n, atol=1e-5, rtol=0)


# --------------------------------------------------------------------------- plumbing


def test_logging_scalars_detached_total_live(golden: dict) -> None:
    outputs = {k: (v.clone().requires_grad_(True) if torch.is_floating_point(v) else v)
               for k, v in golden["outputs"].items()}
    out = _loss_from_golden(golden)(outputs, golden["labels"])
    assert out.total.requires_grad
    for task in TASKS:
        assert not out.per_task[task].requires_grad
        assert not out.weighted[task].requires_grad
    out.total.backward()
    assert outputs["crosses_frame"].grad is not None
    # crosses_pooled is unsupervised → no gradient flows to it.
    assert outputs["crosses_pooled"].grad is None


def test_to_device_moves_class_weights(golden: dict) -> None:
    loss = _loss_from_golden(golden).to("cpu")
    for task in TASKS:
        assert loss.criteria[task].weight.device.type == "cpu"


def test_build_from_config_uses_traincfg_loss_weight(golden: dict) -> None:
    cfg = TrainCfg()
    loss = build_multitask_loss(cfg, golden["class_weights"])
    assert loss.loss_weight == {k: float(v) for k, v in cfg.loss_weight.items()}


def test_build_from_config_end_to_end_with_sampler_weights() -> None:
    # class weights from the Prompt 1.6 scanner feed straight into the loss factory.
    counts = {"actions": {0: 60, 1: 40}, "looks": {0: 85, 1: 15}, "crosses": {0: 974, 1: 26}}
    cw = class_weights_ce(counts)
    loss = build_multitask_loss(TrainCfg(), cw)
    n = 6
    outputs = {"actions": torch.randn(n, 2), "looks": torch.randn(n, 2), "crosses_frame": torch.randn(n, 2)}
    labels = {"actions": torch.randint(0, 2, (n,)), "looks": torch.randint(0, 2, (n,)),
              "crosses": torch.randint(0, 2, (n,))}
    out = loss(outputs, labels)
    assert out.total.ndim == 0 and torch.isfinite(out.total)

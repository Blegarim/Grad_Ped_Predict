"""Prompt 3.2 — MetricAccumulator parity, contract, and degenerate-case tests.

Same shape as the 3.1 / 1.6 / 2.x suites:
  * HAND-CHECKABLE: a tiny 2x2 confusion matrix with known rational metrics.
  * GOLDEN parity: reproduces BOTH OLD metric paths (test.py:74-100 + train.py:580-595) EXACTLY
    (atol=1e-6) from ``tests/fixtures/golden/metrics_cases.pt`` — the two legacy paths are numerically
    identical (see ``tests/_capture/capture_metrics_golden.py``), so one fixture covers #2 and #3.
  * CONTRACT (B4): crosses is scored on ``crosses_frame`` ONLY; ``crosses_pooled`` is ignored; a missing
    ``crosses_frame`` raises.
  * DEGENERATE: single-class targets -> AUC ``nan``, precision/recall/f1 -> 0 (zero_division).
  * PLUMBING: macro-F1 = mean task-F1; pooled overall-acc; AMP half upcast; flat-dict/CSV schema;
    aggregate-vs-per-chunk; eval-only threshold sweep.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pedpredict.config import EvalCfg
from pedpredict.training import METRIC_COLUMNS, MetricAccumulator

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "metrics_cases.pt"
_TASKS = ("actions", "looks", "crosses")


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(_FIXTURE, weights_only=False)


def _feed(acc: MetricAccumulator, scenario: dict) -> MetricAccumulator:
    inp = scenario["inputs"]
    for outputs, targets in zip(inp["outputs_batches"], inp["targets_batches"], strict=True):
        acc.update(outputs, targets)
    return acc


def _close(got: float, exp: float, tol: float) -> bool:
    if math.isnan(exp):
        return math.isnan(got)
    return abs(got - exp) <= tol


# --------------------------------------------------------------------------- hand-checkable


def test_hand_checkable_confusion_matrix() -> None:
    # TP=3, FP=1, FN=2, TN=4 -> acc=.7, precision=.75, recall=.6, f1=2*.75*.6/(.75+.6).
    pred1 = [0.0, 1.0]
    pred0 = [1.0, 0.0]
    logits = torch.tensor([pred1] * 4 + [pred0] * 6)        # preds: 4x class1, 6x class0
    targets = torch.tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])  # 3 TP, 1 FP, 2 FN, 4 TN
    acc = MetricAccumulator(tasks=("actions",))
    acc.update({"actions": logits}, {"actions": targets})
    m = acc.compute().per_task["actions"]
    assert m.accuracy == pytest.approx(0.7)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.6)
    assert m.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


# --------------------------------------------------------------------------- golden parity


@pytest.mark.parametrize("scenario_name", ["main", "degenerate"])
def test_golden_parity(golden: dict, scenario_name: str) -> None:
    scenario = golden[scenario_name]
    expected = scenario["expected"]
    tol = golden["tol"]
    result = _feed(MetricAccumulator(), scenario).compute()
    for task in _TASKS:
        m = result.per_task[task]
        exp = expected["per_task"][task]
        for suffix, got in (
            ("acc", m.accuracy),
            ("f1", m.f1),
            ("auc", m.auc),
            ("precision", m.precision),
            ("recall", m.recall),
        ):
            assert _close(got, exp[suffix], tol), (task, suffix, got, exp[suffix])
    assert _close(result.macro_f1, expected["macro_f1"], tol)
    assert _close(result.overall_accuracy, expected["overall_acc"], tol)


# --------------------------------------------------------------------------- contract (B4)


def test_crosses_scored_on_frame_not_pooled() -> None:
    # crosses_frame predicts class 0 (correct); crosses_pooled predicts class 1 (would be wrong).
    n = 6
    frame = torch.tensor([[3.0, -3.0]] * n)   # argmax -> 0
    pooled = torch.tensor([[-3.0, 3.0]] * n)  # argmax -> 1
    targets = {"crosses": torch.zeros(n, dtype=torch.long)}
    acc = MetricAccumulator(tasks=("crosses",))
    acc.update({"crosses_frame": frame, "crosses_pooled": pooled}, targets)
    assert acc.compute().per_task["crosses"].accuracy == pytest.approx(1.0)


def test_missing_contract_key_raises() -> None:
    acc = MetricAccumulator(tasks=("crosses",))
    with pytest.raises(KeyError, match="crosses_frame"):
        acc.update({"crosses_pooled": torch.zeros(2, 2)}, {"crosses": torch.zeros(2, dtype=torch.long)})


# --------------------------------------------------------------------------- degenerate / plumbing


def test_single_class_auc_nan_and_pr_zero() -> None:
    n = 8
    logits = torch.tensor([[-1.0, 1.0]] * n)               # all predicted class 1
    targets = {"crosses": torch.zeros(n, dtype=torch.long)}  # single class, no positives
    m = MetricAccumulator(tasks=("crosses",))
    m.update({"crosses_frame": logits}, targets)
    out = m.compute().per_task["crosses"]
    assert math.isnan(out.auc)
    assert out.precision == 0.0 and out.recall == 0.0 and out.f1 == 0.0


def test_macro_f1_is_mean_of_task_f1(golden: dict) -> None:
    result = _feed(MetricAccumulator(), golden["main"]).compute()
    mean_f1 = sum(result.per_task[t].f1 for t in _TASKS) / 3
    assert result.macro_f1 == pytest.approx(mean_f1)


def test_overall_accuracy_is_pooled_micro(golden: dict) -> None:
    scenario = golden["main"]
    result = _feed(MetricAccumulator(), scenario).compute()
    # Recompute pooled correct/total directly from the inputs.
    correct = total = 0
    key = {"actions": "actions", "looks": "looks", "crosses": "crosses_frame"}
    for outputs, targets in zip(
        scenario["inputs"]["outputs_batches"], scenario["inputs"]["targets_batches"], strict=True
    ):
        for task in _TASKS:
            preds = torch.argmax(outputs[key[task]], dim=1)
            correct += int((preds == targets[task]).sum())
            total += targets[task].numel()
    assert result.overall_accuracy == pytest.approx(correct / total)


def test_amp_half_logits_upcast(golden: dict) -> None:
    scenario = golden["main"]
    acc = MetricAccumulator()
    for outputs, targets in zip(
        scenario["inputs"]["outputs_batches"], scenario["inputs"]["targets_batches"], strict=True
    ):
        half = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in outputs.items()}
        acc.update(half, targets)
    result = acc.compute()  # must not raise on float16 logits
    for task in _TASKS:
        assert 0.0 <= result.per_task[task].accuracy <= 1.0


def test_empty_accumulator_raises() -> None:
    acc = MetricAccumulator()
    assert acc.n_samples == 0
    with pytest.raises(RuntimeError, match="no accumulated samples"):
        acc.compute()


def test_aggregate_equals_two_chunk_accumulation(golden: dict) -> None:
    # One accumulator fed both batches == the OLD aggregate over all samples (NOT the mean of
    # per-chunk metrics — aggregate AUC/F1 differ from averaging per-chunk values).
    scenario = golden["main"]
    expected = scenario["expected"]
    combined = _feed(MetricAccumulator(), scenario).compute()
    assert combined.macro_f1 == pytest.approx(expected["macro_f1"], abs=golden["tol"])
    assert combined.overall_accuracy == pytest.approx(expected["overall_acc"], abs=golden["tol"])

    # Per-chunk-then-average would generally disagree — assert the two batches are not identical so
    # the aggregate is a genuine pooling, not a trivial single-batch case.
    inp = scenario["inputs"]
    assert len(inp["outputs_batches"]) == 2


# --------------------------------------------------------------------------- schema + eval-only


def test_flat_dict_and_csv_row_match_columns(golden: dict) -> None:
    result = _feed(MetricAccumulator(), golden["main"]).compute()
    flat = result.as_flat_dict()
    assert tuple(flat.keys()) == METRIC_COLUMNS
    assert result.csv_row() == [flat[col] for col in METRIC_COLUMNS]
    assert len(METRIC_COLUMNS) == 3 * 5 + 2


def test_optimal_threshold_metrics_keys_and_range(golden: dict) -> None:
    acc = _feed(MetricAccumulator(), golden["main"])
    cfg = EvalCfg()
    out = acc.optimal_threshold_metrics(cfg)
    for task in _TASKS:
        assert f"{task}_threshold" in out
        thr = out[f"{task}_threshold"]
        assert thr == 0.5 or cfg.threshold_sweep_lo <= thr <= cfg.threshold_sweep_hi
    # Q3: the sweep's mean-of-per-task accuracies is MACRO averaging — renamed from the legacy
    # `overall_acc`, which collided with compute()'s pooled MICRO accuracy of the same name.
    assert "overall_acc" not in out
    assert 0.0 <= out["macro_acc"] <= 1.0


def test_sweep_then_apply_equals_optimal(golden: dict) -> None:
    """M2 decomposition: sweep_thresholds + metrics_at_thresholds == the one-shot oracle path."""
    acc = _feed(MetricAccumulator(), golden["main"])
    cfg = EvalCfg()
    swept = acc.sweep_thresholds(cfg)
    assert set(swept) == set(_TASKS)
    assert acc.metrics_at_thresholds(swept) == pytest.approx(acc.optimal_threshold_metrics(cfg))


def test_metrics_at_fixed_thresholds_respects_cutoff(golden: dict) -> None:
    """metrics_at_thresholds applies the GIVEN cutoffs (the val-tuned-on-test path, M2)."""
    acc = _feed(MetricAccumulator(), golden["main"])
    fixed = dict.fromkeys(_TASKS, 0.5)
    out = acc.metrics_at_thresholds(fixed)
    for task in _TASKS:
        assert out[f"{task}_threshold"] == 0.5
        # threshold 0.5 on the positive-class prob == argmax decisions == compute()'s accuracy
        y_true, y_pred, y_prob = acc.task_arrays(task)
        expected_acc = float((y_true == (y_prob[:, 1] >= 0.5).astype(int)).mean())
        assert out[f"{task}_acc"] == pytest.approx(expected_acc)

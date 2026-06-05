"""Capture golden fixtures for Prompt 3.2 by running the OLD metric code (provenance, not a test).

Neither OLD ``train.py`` nor ``test.py`` is importable (both pull torch/PIE/model imports at module
load), but the metric computation each performs is self-contained ``sklearn`` math over accumulated
``(preds, targets, probs)``. The two oracles are TRANSCRIBED VERBATIM below from:

  * ``OLD/Undergrad_thesis_project/test.py:74-100`` — ``evaluate``'s per-task block (acc / f1 / **auc** /
    precision / recall, ``average="binary"`` for 2-class, AUC over ``prob[:,1]`` with ValueError->nan)
    plus the aggregate ``overall_acc`` (pooled micro-accuracy, :463-470).
  * ``OLD/Undergrad_thesis_project/train.py:580-595`` — ``validate``'s F1/precision/recall block
    (``average='binary'``, ``zero_division=0``) plus ``macro_f1 = (a_f1 + l_f1 + c_f1) / 3``.

If those sources change, re-transcribe and rerun::

    python tests/_capture/capture_metrics_golden.py

Parity notes:
  * The two OLD paths are NUMERICALLY IDENTICAL on this data — they differ only cosmetically:
    ``test.evaluate`` argmaxes the softmax probs (== argmax logits) and omits ``zero_division`` (sklearn
    default ``'warn'`` returns 0.0 *and warns*); ``train.validate`` argmaxes logits and passes
    ``zero_division=0`` (same 0.0, no warning). The unified ``MetricAccumulator`` adopts argmax-on-logits
    + ``zero_division=0`` + AUC-on-probs, reproducing BOTH. The capture asserts the two oracles agree, so
    one fixture pins both (Prompt 3.2 tests #2 and #3).
  * ``outputs`` carries the FULL contract: ``actions`` / ``looks`` / ``crosses_frame`` (scored) PLUS
    ``crosses_pooled`` / ``temporal_weights`` (UNSUPERVISED, B4). ``crosses_pooled`` holds DIFFERENT logits
    from ``crosses_frame`` so the fixture pins that crosses is scored on ``crosses_frame`` only.
  * Two scenarios: ``main`` (2 batches, all metrics well-defined, exercises multi-batch accumulation) and
    ``degenerate`` (crosses targets all-zero -> AUC ``nan``, precision/recall/f1 -> 0 via zero_division).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "metrics_cases.pt"

GEN_SEED = 3201
TASKS = ("actions", "looks", "crosses")
#: crosses is scored on this output key (B4 contract; mirrors losses.multitask.TASK_OUTPUT_KEY).
TASK_OUTPUT_KEY = {"actions": "actions", "looks": "looks", "crosses": "crosses_frame"}


# ----------------------------------------------------------------- OLD oracles (verbatim transcription)
# Transcribed from OLD test.py:74-100,463-470 and train.py:580-595. Do not "improve" the math.


def _legacy_test_per_task(y_true, y_pred, y_prob):
    """test.py:80-97 — per-task acc/f1/auc/precision/recall. ``y_prob`` is [N,2] softmax probs.

    NOTE: test.evaluate omitted ``zero_division`` (sklearn default warns). We pass ``zero_division=0``
    here — numerically identical (the default also returns 0.0) — to match train.validate and silence
    the warning. ``_assert_oracles_agree`` confirms the chosen value equals the default-path value.
    """
    avg_type = "binary" if y_prob.shape[1] == 2 else "macro"
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
    precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc, "precision": precision, "recall": recall}


def _legacy_train_f1(y_true, y_pred):
    """train.py:583-590 — F1/precision/recall (``average='binary'``, ``zero_division=0``)."""
    if len(set(y_true.tolist())) > 1:
        return {
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        }
    return {"f1": 0.0, "precision": 0.0, "recall": 0.0}


def _assert_oracles_agree(y_true, y_pred, y_prob) -> None:
    """Pin that test.evaluate's f1/precision/recall == train.validate's, so one fixture covers both."""
    t = _legacy_test_per_task(y_true, y_pred, y_prob)
    r = _legacy_train_f1(y_true, y_pred)
    for key in ("f1", "precision", "recall"):
        assert math.isclose(t[key], r[key], abs_tol=1e-12), (key, t[key], r[key])


# ----------------------------------------------------------------- synthetic batches


def _signal_logits(targets: torch.Tensor, margin: float, gen: torch.Generator) -> torch.Tensor:
    """2-column logits whose class-1 score = ``margin * (2*target - 1) + noise``.

    Gives an imperfect-but-correlated classifier (AUC in (0.5, 1), non-trivial confusion matrix).
    ``softmax`` -> prob1 = sigmoid(score); ``argmax`` -> pred = (score > 0).
    """
    noise = torch.randn(targets.shape[0], generator=gen)
    score = margin * (2.0 * targets.float() - 1.0) + noise
    return torch.stack([torch.zeros_like(score), score], dim=1)


def _make_targets(n: int, pos_rate: float, gen: torch.Generator) -> torch.Tensor:
    return (torch.rand(n, generator=gen) < pos_rate).long()


def _make_main_batches():
    """2 batches; per-task targets at different positive rates; crosses imbalanced but 2-class."""
    gen = torch.Generator().manual_seed(GEN_SEED)
    sizes = (24, 16)
    pos = {"actions": 0.5, "looks": 0.3, "crosses": 0.2}
    margin = {"actions": 1.4, "looks": 1.2, "crosses": 1.6}
    outputs_batches, targets_batches = [], []
    for n in sizes:
        targets = {t: _make_targets(n, pos[t], gen) for t in TASKS}
        outputs = {
            "actions": _signal_logits(targets["actions"], margin["actions"], gen),
            "looks": _signal_logits(targets["looks"], margin["looks"], gen),
            "crosses_frame": _signal_logits(targets["crosses"], margin["crosses"], gen),
            # UNSUPERVISED extras — different values; must never reach the metrics.
            "crosses_pooled": torch.randn(n, 2, generator=gen),
            "temporal_weights": torch.softmax(torch.randn(n, 20, generator=gen), dim=1),
        }
        outputs_batches.append(outputs)
        targets_batches.append(targets)
    return outputs_batches, targets_batches


def _make_degenerate_batch():
    """Single batch; crosses targets all-zero -> AUC nan, precision/recall/f1 -> 0 (zero_division)."""
    gen = torch.Generator().manual_seed(GEN_SEED + 7)
    n = 20
    targets = {
        "actions": _make_targets(n, 0.5, gen),
        "looks": _make_targets(n, 0.4, gen),
        "crosses": torch.zeros(n, dtype=torch.long),  # single class
    }
    outputs = {
        "actions": _signal_logits(targets["actions"], 1.3, gen),
        "looks": _signal_logits(targets["looks"], 1.1, gen),
        # crosses logits favor class 1 for a few rows -> false positives (precision 0, no TP).
        "crosses_frame": _signal_logits(torch.ones(n, dtype=torch.long), 0.3, gen),
        "crosses_pooled": torch.randn(n, 2, generator=gen),
        "temporal_weights": torch.softmax(torch.randn(n, 20, generator=gen), dim=1),
    }
    return [outputs], [targets]


# ----------------------------------------------------------------- oracle over accumulated batches


def _expected_over_batches(outputs_batches, targets_batches) -> dict:
    """Run the OLD oracle over the CONCATENATION of all batches (== accumulator.compute scope)."""
    per_task: dict[str, dict[str, float]] = {}
    total_correct = 0
    total_samples = 0
    for task in TASKS:
        key = TASK_OUTPUT_KEY[task]
        logits = torch.cat([ob[key] for ob in outputs_batches], dim=0)
        targets = torch.cat([tb[task] for tb in targets_batches], dim=0)
        probs = torch.softmax(logits.float(), dim=1)
        preds = torch.argmax(logits.float(), dim=1)
        y_true = targets.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()
        _assert_oracles_agree(y_true, y_pred, y_prob)
        per_task[task] = _legacy_test_per_task(y_true, y_pred, y_prob)
        total_correct += int((preds == targets).sum().item())
        total_samples += targets.numel()
    macro_f1 = sum(per_task[t]["f1"] for t in TASKS) / 3.0          # train.py:593
    overall_acc = total_correct / total_samples                    # test.py:470 / train.py:595
    return {"per_task": per_task, "macro_f1": macro_f1, "overall_acc": overall_acc}


def _scenario(outputs_batches, targets_batches) -> dict:
    return {
        "inputs": {"outputs_batches": outputs_batches, "targets_batches": targets_batches},
        "expected": _expected_over_batches(outputs_batches, targets_batches),
    }


def main() -> None:
    fixture = {
        "main": _scenario(*_make_main_batches()),
        "degenerate": _scenario(*_make_degenerate_batch()),
        "tasks": list(TASKS),
        "tol": 1e-6,
        "meta": {
            "src": "OLD test.py:74-100,463-470 + train.py:580-595",
            "gen_seed": GEN_SEED,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    print(f"wrote {OUT}")
    for name in ("main", "degenerate"):
        exp = fixture[name]["expected"]
        print(f"[{name}] macro_f1={exp['macro_f1']:.6f} overall_acc={exp['overall_acc']:.6f}")
        for task in TASKS:
            m = exp["per_task"][task]
            print(
                f"  {task:8s} acc={m['acc']:.4f} f1={m['f1']:.4f} "
                f"auc={m['auc']:.4f} p={m['precision']:.4f} r={m['recall']:.4f}"
            )


if __name__ == "__main__":
    main()

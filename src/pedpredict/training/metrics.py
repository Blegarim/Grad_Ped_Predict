"""Single metric implementation shared by training-validation and test/eval (Prompt 3.2).

Consolidates the metric code duplicated across the OLD repo (band-aid **B1**):

* ``OLD train.py:186-234`` ``validate_one_epoch`` — per-task accuracy via ``argmax(logits)``, accumulated
  preds/targets; and ``:580-595`` the F1/precision/recall (``average='binary'``, ``zero_division=0``) +
  ``macro_f1 = (a + l + c) / 3`` block.
* ``OLD test.py:74-100`` ``evaluate`` — accuracy, F1, **AUC** (over softmax ``prob[:,1]``,
  ``ValueError -> nan``), precision, recall (``binary`` for 2-class else ``macro``); and ``:463-470`` the
  pooled ``overall_acc``.

The two paths computed the same five metrics with only cosmetic differences. The unified
:class:`MetricAccumulator` makes each divergence a deliberate canonical choice (see ``capture`` script and
MIGRATION.md "Metrics decisions (3.2)"):

* preds via ``argmax(logits)`` (== ``argmax(softmax)``; cheaper);
* ``zero_division=0`` everywhere (adopts ``train``'s explicit form; silences ``test``'s warning;
  value-identical) — the explicit ``len(set)>1`` guard is then redundant and dropped;
* **AUC is computed on the validation path too** (the accumulator holds the probs) — an enrichment over
  OLD ``validate``, which logged none;
* ``crosses`` is scored on ``crosses_frame`` ONCE via the shared :data:`TASK_OUTPUT_KEY` contract (B4) —
  ``crosses_pooled`` / ``temporal_weights`` are never scored.

Deliberately OUT of scope (documented, owned elsewhere): **loss** aggregation (Trainer, 4.1, via
:class:`~pedpredict.losses.multitask.MultiTaskLoss`), **temporal-weight** collection (viz, 6.2), and the
eval-only **threshold sweep** (exposed as :meth:`MetricAccumulator.optimal_threshold_metrics`, so train-val
and test still share the *same* core :meth:`MetricAccumulator.compute`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor

# Singular output contract — reuse the loss module's declaration so crosses->crosses_frame routing
# (B4) cannot drift between loss (3.1), metrics (3.2) and eval (5.1).
from pedpredict.config.schema import EvalCfg
from pedpredict.losses.multitask import TASK_OUTPUT_KEY, TASKS
from pedpredict.utils.amp import to_float_logits

__all__ = ["TaskMetrics", "MetricResult", "MetricAccumulator", "METRIC_COLUMNS"]

_METRIC_SUFFIXES: tuple[str, ...] = ("acc", "f1", "auc", "precision", "recall")

#: Canonical flat column order (task-major) — the single schema both train-val (4.5) and test (5.1)
#: emit. Writers prepend their own context columns (epoch / timestamp+chunk) and own the loss columns.
METRIC_COLUMNS: tuple[str, ...] = tuple(
    f"{task}_{suffix}" for task in TASKS for suffix in _METRIC_SUFFIXES
) + ("macro_f1", "overall_acc")


@dataclass(frozen=True)
class TaskMetrics:
    """The five per-task metrics. ``auc`` is ``float('nan')`` when undefined (single-class targets)."""

    accuracy: float
    f1: float
    auc: float
    precision: float
    recall: float


@dataclass(frozen=True)
class MetricResult:
    """Full metric bundle: per-task metrics + macro-F1 + pooled micro-accuracy."""

    per_task: dict[str, TaskMetrics]
    macro_f1: float
    overall_accuracy: float

    def as_flat_dict(self) -> dict[str, float]:
        """Flatten to :data:`METRIC_COLUMNS` keys (``actions_acc`` ... ``macro_f1``, ``overall_acc``)."""
        flat: dict[str, float] = {}
        for task, metrics in self.per_task.items():
            flat[f"{task}_acc"] = metrics.accuracy
            flat[f"{task}_f1"] = metrics.f1
            flat[f"{task}_auc"] = metrics.auc
            flat[f"{task}_precision"] = metrics.precision
            flat[f"{task}_recall"] = metrics.recall
        flat["macro_f1"] = self.macro_f1
        flat["overall_acc"] = self.overall_accuracy
        return flat

    def csv_row(self) -> list[float]:
        """Metric values in :data:`METRIC_COLUMNS` order (unrounded; logging layer rounds)."""
        flat = self.as_flat_dict()
        return [flat[col] for col in METRIC_COLUMNS]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC with degenerate handling: single-class targets -> ``nan`` (no raise, no warning).

    Binary uses the positive-class column; the (unreachable today) >2-class branch keeps OLD's
    ``multi_class='ovr'`` form. Both the legacy ``ValueError`` (older sklearn) and the newer
    warn-and-return-``nan`` path resolve to ``nan``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            if y_prob.shape[1] == 2:
                return float(roc_auc_score(y_true, y_prob[:, 1]))
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        return float("nan")


class MetricAccumulator:
    """Ingest per-batch ``(outputs, targets)`` per task; yield Acc/F1/AUC/Precision/Recall at the end.

    The SAME instance/implementation is used by training-validation (4.1) and test/eval (5.1) — there is
    no second metric code path. ``update`` routes ``crosses -> crosses_frame`` once and stores CPU
    softmax probs + targets; ``compute`` runs ``sklearn`` once over the concatenation.
    """

    def __init__(self, tasks: tuple[str, ...] = TASKS) -> None:
        self._tasks = tuple(tasks)
        self._probs: dict[str, list[Tensor]] = {t: [] for t in self._tasks}
        self._targets: dict[str, list[Tensor]] = {t: [] for t in self._tasks}

    def reset(self) -> None:
        """Drop all accumulated batches (reuse one accumulator across epochs / chunks)."""
        self._probs = {t: [] for t in self._tasks}
        self._targets = {t: [] for t in self._tasks}

    def update(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> None:
        """Accumulate one batch. ``outputs`` must hold each task's contract key (:data:`TASK_OUTPUT_KEY`)."""
        floated = to_float_logits(outputs)  # B8: single upcast site (no-op outside autocast)
        for task in self._tasks:
            key = TASK_OUTPUT_KEY[task]
            if key not in floated:
                raise KeyError(
                    f"MetricAccumulator: task '{task}' requires output key '{key}', "
                    f"missing from outputs (have: {sorted(outputs)})."
                )
            probs = torch.softmax(floated[key], dim=1)
            self._probs[task].append(probs.detach().cpu())
            self._targets[task].append(targets[task].detach().cpu().long())

    @property
    def n_samples(self) -> int:
        """Number of accumulated samples (equal across tasks)."""
        if not self._tasks:
            return 0
        return sum(t.shape[0] for t in self._targets[self._tasks[0]])

    def task_arrays(self, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate accumulated batches for ``task`` -> ``(y_true, y_pred, y_prob)`` numpy arrays.

        Public so eval (5.1) can build its per-sample predictions NPZ from the SAME canonical store the
        metrics are computed over — no second accumulation, no divergence (B1).
        """
        probs = torch.cat(self._probs[task], dim=0)
        targets = torch.cat(self._targets[task], dim=0)
        y_pred = torch.argmax(probs, dim=1)  # == argmax(logits); softmax is monotonic
        return targets.numpy(), y_pred.numpy(), probs.numpy()

    def compute(self) -> MetricResult:
        """Run the metric set over all accumulated batches. Raises if nothing was accumulated."""
        if self.n_samples == 0:
            raise RuntimeError("MetricAccumulator.compute() called with no accumulated samples.")
        per_task: dict[str, TaskMetrics] = {}
        total_correct = 0
        total_samples = 0
        for task in self._tasks:
            y_true, y_pred, y_prob = self.task_arrays(task)
            avg = "binary" if y_prob.shape[1] == 2 else "macro"
            per_task[task] = TaskMetrics(
                accuracy=float(accuracy_score(y_true, y_pred)),
                f1=float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
                auc=_safe_auc(y_true, y_prob),
                precision=float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            )
            total_correct += int((y_true == y_pred).sum())
            total_samples += y_true.shape[0]
        macro_f1 = sum(per_task[t].f1 for t in self._tasks) / len(self._tasks)
        return MetricResult(
            per_task=per_task,
            macro_f1=macro_f1,
            overall_accuracy=total_correct / total_samples,
        )

    # ----------------------------------------------------------------- eval-only (5.1) enrichment

    def optimal_threshold_metrics(self, cfg: EvalCfg) -> dict[str, float]:
        """Per-task F1-optimal decision threshold + the metrics at it (OLD test.py:110-146,472-497).

        Eval-only (NOT part of the shared core :meth:`compute`). Sweeps ``[lo, hi]`` by ``step`` from
        ``EvalCfg`` over the positive-class probability. Single-class tasks keep threshold ``0.5``.
        """
        thresholds = _sweep(cfg.threshold_sweep_lo, cfg.threshold_sweep_hi, cfg.threshold_sweep_step)
        out: dict[str, float] = {}
        accs: list[float] = []
        for task in self._tasks:
            y_true, _, y_prob = self.task_arrays(task)
            if y_prob.shape[1] != 2 or len(set(y_true.tolist())) < 2:
                thr = 0.5
            else:
                thr = _best_f1_threshold(y_true, y_prob[:, 1], thresholds)
            preds = (y_prob[:, 1] >= thr).astype(int)
            acc = float(accuracy_score(y_true, preds))
            out[f"{task}_threshold"] = thr
            out[f"{task}_acc"] = acc
            out[f"{task}_f1"] = float(f1_score(y_true, preds, average="binary", zero_division=0))
            out[f"{task}_precision"] = float(precision_score(y_true, preds, average="binary", zero_division=0))
            out[f"{task}_recall"] = float(recall_score(y_true, preds, average="binary", zero_division=0))
            accs.append(acc)
        out["overall_acc"] = sum(accs) / len(accs) if accs else 0.0  # test.py:495-497 (mean of per-task)
        return out


def _sweep(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive threshold grid ``[lo, hi]`` by ``step`` (rounded; OLD used 0.10..0.90 @ 0.05)."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def _best_f1_threshold(y_true: np.ndarray, pos_prob: np.ndarray, thresholds: list[float]) -> float:
    """Threshold in ``thresholds`` maximizing binary F1 (ties -> lowest threshold; default 0.5)."""
    best_thr, best_f1 = 0.5, 0.0
    for thr in thresholds:
        preds = (pos_prob >= thr).astype(int)
        f1 = float(f1_score(y_true, preds, average="binary", zero_division=0))
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr

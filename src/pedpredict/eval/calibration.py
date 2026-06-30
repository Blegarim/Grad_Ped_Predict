"""Calibration scaffolding (M10 / RQ6): reliability curves, ECE, and temperature scaling.

Answers "are the model's crossing probabilities *honest*?" — independent of any decision threshold. The
M1 imbalance stack trains on a `crosses` prior far from the 2.8% deployment rate, so the raw probabilities
are expected to be inflated; this module makes that visible (reliability curve + ECE) and provides the
standard cheap correction (temperature scaling, a single scalar fit on **val**).

Pure post-hoc analysis over arrays — it never touches the model or training. The inputs are exactly what
:meth:`MetricAccumulator.task_arrays` and the predictions NPZ already expose: per-sample binary labels
``y_true`` and positive-class probabilities ``pos_prob`` (``= y_prob[:, 1] = {task}_prob_1``). Binary tasks
only (the 3 pedpredict tasks); temperature is fit by minimizing NLL with L-BFGS.

Policy (RQ6, mirrors the M2 threshold rule): fit the temperature on **val**, store it, apply it to test —
never fit on test. Reliability diagrams are rendered by ``viz.plots.plot_reliability``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pedpredict.losses.multitask import TASKS

__all__ = [
    "ReliabilityCurve",
    "TaskCalibration",
    "reliability_curve",
    "expected_calibration_error",
    "fit_temperature",
    "apply_temperature",
    "calibrate_task",
    "calibrate_predictions_npz",
]

_EPS = 1e-7


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin reliability data for a diagram. Arrays are length ``n_bins`` (empty bins kept as NaN)."""

    bin_edges: np.ndarray   # [n_bins + 1] equal-width bin boundaries in [0, 1]
    confidence: np.ndarray  # [n_bins] mean predicted positive prob per bin (NaN if empty)
    accuracy: np.ndarray    # [n_bins] observed positive frequency per bin (NaN if empty)
    count: np.ndarray       # [n_bins] sample count per bin


@dataclass(frozen=True)
class TaskCalibration:
    """One task's calibration summary: temperature + ECE before/after, and the pre-scaling curve."""

    temperature: float
    ece_before: float
    ece_after: float
    curve: ReliabilityCurve


def _pos_logit(pos_prob: np.ndarray) -> np.ndarray:
    """Recover the binary logit ``z`` from ``p1`` (``p1 = sigmoid(z)``); clipped to avoid ``±inf``."""
    p = np.clip(np.asarray(pos_prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def reliability_curve(y_true: np.ndarray, pos_prob: np.ndarray, *, n_bins: int = 15) -> ReliabilityCurve:
    """Equal-width reliability binning of ``pos_prob`` vs observed positive frequency (``y_true``)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pos_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; clip the right edge (prob == 1.0) into the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = idx == b
        count[b] = int(mask.sum())
        if count[b]:
            conf[b] = float(p[mask].mean())
            acc[b] = float(y[mask].mean())
    return ReliabilityCurve(bin_edges=edges, confidence=conf, accuracy=acc, count=count)


def expected_calibration_error(y_true: np.ndarray, pos_prob: np.ndarray, *, n_bins: int = 15) -> float:
    """ECE: count-weighted mean ``|accuracy - confidence|`` over non-empty bins (Guo et al., 2017)."""
    curve = reliability_curve(y_true, pos_prob, n_bins=n_bins)
    total = curve.count.sum()
    if total == 0:
        return float("nan")
    nonempty = curve.count > 0
    gap = np.abs(curve.accuracy[nonempty] - curve.confidence[nonempty])
    return float((curve.count[nonempty] / total * gap).sum())


def fit_temperature(y_true: np.ndarray, pos_prob: np.ndarray, *, max_iter: int = 200) -> float:
    """Temperature ``T > 0`` minimizing NLL of ``sigmoid(z / T)`` vs ``y_true`` (val-only; M2 rule).

    Single-class targets -> ``1.0`` (temperature is undefined). Optimized over ``log T`` so ``T`` stays
    positive; L-BFGS on the reconstructed binary logits ``z = logit(pos_prob)``.
    """
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return 1.0
    z = torch.tensor(_pos_logit(pos_prob), dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.float64)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)  # T = exp(0) = 1 at init
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure() -> torch.Tensor:
        opt.zero_grad()
        loss = bce(z / torch.exp(log_t), target)
        loss.backward()
        return loss

    opt.step(_closure)
    return float(torch.exp(log_t).detach())


def apply_temperature(pos_prob: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale positive probabilities by ``T``: ``p' = sigmoid(logit(p) / T)``."""
    z = _pos_logit(pos_prob) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_task(
    y_true: np.ndarray, pos_prob: np.ndarray, *, n_bins: int = 15, temperature: float | None = None
) -> TaskCalibration:
    """Fit (or apply a given) temperature and report ECE before/after + the pre-scaling reliability curve.

    Pass ``temperature`` (fit on val) to APPLY it to another split; leave ``None`` to fit on these data.
    """
    t = fit_temperature(y_true, pos_prob) if temperature is None else float(temperature)
    return TaskCalibration(
        temperature=t,
        ece_before=expected_calibration_error(y_true, pos_prob, n_bins=n_bins),
        ece_after=expected_calibration_error(y_true, apply_temperature(pos_prob, t), n_bins=n_bins),
        curve=reliability_curve(y_true, pos_prob, n_bins=n_bins),
    )


def calibrate_predictions_npz(
    path: str | Path,
    *,
    tasks: tuple[str, ...] = TASKS,
    n_bins: int = 15,
    temperatures: dict[str, float] | None = None,
) -> dict[str, TaskCalibration]:
    """Per-task calibration from a predictions NPZ (``{task}_true`` / ``{task}_prob_1``).

    ``temperatures`` (e.g. fit on val and stored) are APPLIED when given; otherwise fit per task on this
    file's data. Returns ``{task: TaskCalibration}``.
    """
    data = np.load(path)
    out: dict[str, TaskCalibration] = {}
    for task in tasks:
        y_true = data[f"{task}_true"]
        pos_prob = data[f"{task}_prob_1"]
        given = None if temperatures is None else temperatures.get(task)
        out[task] = calibrate_task(y_true, pos_prob, n_bins=n_bins, temperature=given)
    return out

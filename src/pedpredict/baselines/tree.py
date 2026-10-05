"""Gradient-boosted trees on window summaries, fitted as a binary classifier or as a discrete-time hazard.

Why trees at all: on the identical 58-dim pose + motion input, a boosted-tree probe on per-window summaries
detects far more crossers than every deep configuration at matched false-alarm budgets, with a ~1-point seed
spread against 3-9 for the deep arms, and two tree fits score the same test pedestrians alike (paired noise
~2 points vs 6-12) — see ``outputs/runs/RESULTS_MATRIX.md``. The deep learner memorizes ~240 crossing events
within 1-3 epochs; a tree on summaries cannot, so it is the instrument stable enough to compare *objectives*.

Two objectives, one learner (:class:`~sklearn.ensemble.HistGradientBoostingClassifier`, same settings):

* **binary** — one row per window, the target a horizon label (the stored ``crosses`` at ``H = 32``).
* **hazard** — the person-period form of discrete-time survival analysis: one row per *(window, observed
  bin)* with the bin index as an extra feature, the target "the first crossing starts in this bin". The rows
  are exactly the bins :func:`~pedpredict.data.onset_target.hazard_targets` marks observed, so an unobserved
  bin contributes nothing — the tree twin of ``OnsetHazardLoss`` — and the readout is the same survival
  product, ``P(onset <= H) = 1 - prod_{k < H/w} (1 - h_k)``.

Features are ``[last, mean, last - first, std]`` over the window's frames per channel. Iteration count is
not tuned by early stopping on a random split of the training rows (windows of one pedestrian overlap 17 of
20 frames, so such a split leaks); the study picks it on the validation split through :func:`truncate`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from pedpredict.config.schema import MOTION_STORE_DIM
from pedpredict.data.onset_target import OnsetSpec, hazard_targets
from pedpredict.eval.onset_timing import hazard_horizon_prob

__all__ = [
    "SUMMARY_STATS",
    "TreeParams",
    "summary_features",
    "cue_groups",
    "horizon_label",
    "person_period",
    "hazard_design",
    "balanced_weight",
    "make_classifier",
    "fit_binary",
    "fit_hazard",
    "staged_binary_scores",
    "staged_hazard_readout",
    "truncate",
    "hazard_logits",
]

SUMMARY_STATS: tuple[str, ...] = ("last", "mean", "delta", "std")
_ROW_BLOCK = 200_000   # rows filled per step when expanding windows to person-period rows (bounds peak RAM)


@dataclass(frozen=True)
class TreeParams:
    """Learner settings shared by every arm. ``max_features < 1`` is the only source of seed randomness."""

    learning_rate: float = 0.1     # sklearn's default; the 2026-09-25 probe ran at defaults (<= 100 iterations)
    max_iter: int = 300            # an upper bound: each fit is cut at its validation-best iteration
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 50
    l2_regularization: float = 1.0
    max_features: float = 0.5


def summary_features(X: np.ndarray, channels: Sequence[int] | None = None) -> np.ndarray:
    """``X [N, T, C]`` -> ``[N, 4 * C']`` float32: per channel last value, mean, last - first, std."""
    if channels is not None:
        X = X[:, :, list(channels)]
    X = X.astype(np.float64, copy=False)
    blocks = [X[:, -1], X.mean(axis=1), X[:, -1] - X[:, 0], X.std(axis=1)]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def cue_groups(n_channels: int) -> dict[str, tuple[int, ...]]:
    """Channel subsets of the read-path vector: box (cx..dh), ego speed, pose, and their combinations."""
    box = tuple(range(MOTION_STORE_DIM - 1))
    ego = (MOTION_STORE_DIM - 1,)
    pose = tuple(range(MOTION_STORE_DIM, n_channels))
    return {"all": box + ego + pose, "no_ego": box + pose, "box_ego": box + ego, "box": box, "pose": pose,
            "ego": ego}


def horizon_label(onset: np.ndarray, observed: np.ndarray, horizon: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Binary "crossing starts within ``horizon`` frames" label and which windows train on it.

    ``mode="drop"`` keeps only windows whose answer is knowable (a crossing was seen, or ``horizon`` frames
    of future were) — the generator's M4 rule moved to ``horizon``; ``mode="zero"`` keeps every window and
    calls an unseen future "no crossing". Already-crossed windows stay negatives in both, as in the stored
    label, so the binary arms differ from the hazard arm in everything the hazard formulation changes.
    """
    onset, observed = np.asarray(onset), np.asarray(observed)
    label = ((onset >= 0) & (onset < horizon)).astype(np.int64)
    if mode == "drop":
        keep = (onset >= 0) | (observed >= horizon)
    elif mode == "zero":
        keep = np.ones(len(onset), dtype=bool)
    else:
        raise ValueError(f"mode must be 'drop' or 'zero', got {mode!r}")
    return label, keep


def person_period(onset: np.ndarray, observed: np.ndarray, ever: np.ndarray,
                  spec: OnsetSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(window, bin, target)`` for every observed bin, from the same rule the deep hazard loss uses."""
    labels = {"onset_offset": torch.as_tensor(np.asarray(onset), dtype=torch.long),
              "future_observed": torch.as_tensor(np.asarray(observed), dtype=torch.long),
              "track_crosses": torch.as_tensor(np.asarray(ever), dtype=torch.long)}
    targets = hazard_targets(labels, spec)
    window, k = np.nonzero(targets.mask.numpy() > 0)
    return window, k, targets.target.numpy()[window, k].astype(np.int64)


def hazard_design(features: np.ndarray, window: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rows ``[features[window], k]`` as float64 (the learner's own dtype, so it makes no second copy)."""
    out = np.empty((len(window), features.shape[1] + 1), dtype=np.float64)
    for start in range(0, len(window), _ROW_BLOCK):
        stop = start + _ROW_BLOCK
        out[start:stop, :-1] = features[window[start:stop]]
    out[:, -1] = k
    return out


def _query_design(features: np.ndarray, n_bins: int) -> np.ndarray:
    """Every window at bins ``0..n_bins-1``, window-major (row ``i * n_bins + k``)."""
    window = np.repeat(np.arange(len(features)), n_bins)
    return hazard_design(features, window, np.tile(np.arange(n_bins), len(features)))


def balanced_weight(y: np.ndarray) -> np.ndarray:
    """sklearn's ``class_weight="balanced"`` as per-row weights: ``n / (2 * n_class)``."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    return (len(y) / (2.0 * np.maximum(counts, 1.0)))[y]


def make_classifier(params: TreeParams, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=params.learning_rate, max_iter=params.max_iter, max_leaf_nodes=params.max_leaf_nodes,
        min_samples_leaf=params.min_samples_leaf, l2_regularization=params.l2_regularization,
        max_features=params.max_features, early_stopping=False, random_state=seed,
    )


def _weights(y: np.ndarray, weighting: str) -> np.ndarray | None:
    if weighting == "none":
        return None
    if weighting == "balanced":
        return balanced_weight(y)
    raise ValueError(f"weighting must be 'none' or 'balanced', got {weighting!r}")


def fit_binary(features: np.ndarray, y: np.ndarray, *, weighting: str, params: TreeParams,
               seed: int) -> HistGradientBoostingClassifier:
    model = make_classifier(params, seed)
    return model.fit(features.astype(np.float64), y, sample_weight=_weights(y, weighting))


def fit_hazard(features: np.ndarray, onset: np.ndarray, observed: np.ndarray, ever: np.ndarray,
               spec: OnsetSpec, *, weighting: str, params: TreeParams, seed: int) -> HistGradientBoostingClassifier:
    window, k, target = person_period(onset, observed, ever, spec)
    design = hazard_design(features, window, k)
    model = make_classifier(params, seed)
    return model.fit(design, target, sample_weight=_weights(target, weighting))


def staged_binary_scores(model: HistGradientBoostingClassifier, features: np.ndarray):
    """Raw score per window after each boosting iteration (rank-equivalent to the probability)."""
    yield from model.staged_decision_function(features.astype(np.float64))


def staged_hazard_readout(model: HistGradientBoostingClassifier, features: np.ndarray, horizon_bins: int):
    """``P(onset within horizon_bins bins)`` per window after each boosting iteration."""
    for raw in model.staged_decision_function(_query_design(features, horizon_bins)):
        yield hazard_horizon_prob(raw.reshape(len(features), horizon_bins), horizon_bins)


def truncate(model: HistGradientBoostingClassifier, n_iter: int) -> HistGradientBoostingClassifier:
    """Keep only the first ``n_iter`` boosting iterations (in place), as if training had stopped there.

    Uses the fitted predictor list directly — pinned against ``staged_decision_function`` in the tests, so
    a scikit-learn change that breaks it fails loudly instead of silently predicting with every tree.
    """
    if not 1 <= n_iter <= len(model._predictors):
        raise ValueError(f"n_iter={n_iter} outside [1, {len(model._predictors)}]")
    model._predictors = model._predictors[:n_iter]   # n_iter_ is derived from this list
    assert model.n_iter_ == n_iter
    return model


def hazard_logits(model: HistGradientBoostingClassifier, features: np.ndarray, n_bins: int,
                  block: int = 5_000) -> np.ndarray:
    """``[N, n_bins]`` hazard logits, computed in window blocks to bound memory."""
    out = np.empty((len(features), n_bins), dtype=np.float32)
    for start in range(0, len(features), block):
        chunk = features[start:start + block]
        out[start:start + len(chunk)] = model.decision_function(_query_design(chunk, n_bins)).reshape(-1, n_bins)
    return out

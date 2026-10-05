"""Tree baselines: the hazard rows ARE the deep loss's observed bins; truncation and readout are exact."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from pedpredict.baselines.tree import (
    TreeParams,
    balanced_weight,
    cue_groups,
    fit_binary,
    fit_hazard,
    hazard_logits,
    horizon_label,
    person_period,
    staged_binary_scores,
    staged_hazard_readout,
    summary_features,
    truncate,
)
from pedpredict.data.onset_target import OnsetSpec, hazard_targets, readout_targets
from pedpredict.eval.onset_timing import hazard_horizon_prob

_SPEC = OnsetSpec(lookahead=96, bin_width=4, horizon=32)
_FAST = TreeParams(learning_rate=0.2, max_iter=12, max_leaf_nodes=7, min_samples_leaf=5, max_features=0.8)


def _fields(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    onset = np.where(rng.random(n) < 0.4, rng.integers(0, 200, n), -1)
    observed = rng.integers(0, 150, n)
    observed = np.where(onset >= 0, np.maximum(observed, onset + 1), observed)
    ever = np.where(onset >= 0, 1, (rng.random(n) < 0.2).astype(int))
    return onset, observed, ever


def test_person_period_rows_are_exactly_the_observed_bins() -> None:
    onset, observed, ever = _fields()
    window, k, target = person_period(onset, observed, ever, _SPEC)
    t = hazard_targets({"onset_offset": torch.as_tensor(onset), "future_observed": torch.as_tensor(observed),
                        "track_crosses": torch.as_tensor(ever)}, _SPEC)
    mask = t.mask.numpy()
    assert len(window) == int(mask.sum())
    np.testing.assert_array_equal(mask[window, k], 1.0)
    np.testing.assert_array_equal(target, t.target.numpy()[window, k])
    already = (onset < 0) & (ever > 0)
    assert not np.isin(window, np.flatnonzero(already)).any()       # case 4 leaves the risk set
    for i in np.flatnonzero((onset >= 0) & (onset < 96))[:20]:       # case 1: rows stop at the event bin
        rows = k[window == i]
        assert rows.max() == onset[i] // 4 and target[window == i].sum() == 1


def test_horizon_label_matches_the_readout_rule() -> None:
    onset, observed, ever = _fields(seed=1)
    for horizon in (32, 64):
        label, keep = horizon_label(onset, observed, horizon, "drop")
        spec = OnsetSpec(lookahead=horizon, bin_width=1, horizon=horizon)
        ref_label, ref_valid = readout_targets(
            {"onset_offset": torch.as_tensor(onset), "future_observed": torch.as_tensor(observed),
             "track_crosses": torch.as_tensor(ever)}, spec)
        np.testing.assert_array_equal(label, ref_label.numpy().astype(int))
        already = (onset < 0) & (ever > 0)
        # readout_targets also drops already-crossed windows; the binary label keeps them as negatives
        np.testing.assert_array_equal(keep & ~already, ref_valid.numpy())
        zero_label, zero_keep = horizon_label(onset, observed, horizon, "zero")
        np.testing.assert_array_equal(zero_label, label)
        assert zero_keep.all()


def test_summary_features() -> None:
    X = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    f = summary_features(X)
    assert f.shape == (2, 8)
    np.testing.assert_allclose(f[0], [4, 5, 2, 3, 4, 4, np.std([0, 2, 4]), np.std([1, 3, 5])], rtol=1e-6)
    np.testing.assert_allclose(summary_features(X, channels=[1])[0], [5, 3, 4, np.std([1, 3, 5])], rtol=1e-6)


def test_cue_groups_partition_the_vector() -> None:
    g = cue_groups(58)
    assert g["ego"] == (8,) and len(g["box"]) == 8 and len(g["pose"]) == 49
    assert sorted(g["no_ego"] + g["ego"]) == list(range(58)) == sorted(g["all"])


def test_balanced_weight_equalizes_class_mass() -> None:
    y = np.array([0, 0, 0, 1])
    w = balanced_weight(y)
    assert w[y == 0].sum() == pytest.approx(w[y == 1].sum()) == pytest.approx(2.0)


def _risk_data(n: int = 600, seed: int = 0):
    """Crossing sooner the larger feature 0 is; everything fully observed."""
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, 4)).astype(np.float32)
    soon = rng.random(n) < 1 / (1 + np.exp(-2.5 * feats[:, 0]))
    onset = np.where(soon, rng.integers(0, 40, n), np.where(rng.random(n) < 0.5, rng.integers(60, 200, n), -1))
    observed = np.full(n, 300)
    ever = (onset >= 0).astype(int)
    return feats, onset, observed, ever


def test_truncate_equals_the_staged_prediction() -> None:
    feats, onset, _, _ = _risk_data()
    y = ((onset >= 0) & (onset < 32)).astype(int)
    model = fit_binary(feats, y, weighting="balanced", params=_FAST, seed=0)
    staged = list(staged_binary_scores(model, feats))
    np.testing.assert_allclose(truncate(model, 5).decision_function(feats.astype(np.float64)), staged[4])
    with pytest.raises(ValueError):
        truncate(model, 99)


def test_hazard_readout_is_the_survival_product_and_learns_the_risk() -> None:
    feats, onset, observed, ever = _risk_data()
    model = fit_hazard(feats, onset, observed, ever, _SPEC, weighting="none", params=_FAST, seed=0)
    last = list(staged_hazard_readout(model, feats, _SPEC.horizon_bins))[-1]
    logits = hazard_logits(model, feats, _SPEC.num_bins, block=97)
    np.testing.assert_allclose(last, hazard_horizon_prob(logits, _SPEC.horizon_bins), rtol=1e-5, atol=1e-6)
    y = ((onset >= 0) & (onset < 32)).astype(int)
    assert roc_auc_score(y, last) > 0.75

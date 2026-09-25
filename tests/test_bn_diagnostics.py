"""Backbone BN diagnostic helpers (docs/RECIPE_V2_PLAN.md Task 1).

The script that uses these runs only on the research PC (it needs a checkpoint and the LMDBs), so its
correctness rests on these helpers: reading, swapping and re-estimating BN statistics must touch the
running statistics and nothing else, and the drift numbers must mean what their names say.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from pedpredict.eval import diagnostics as dg


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
    ).eval()


def _shift_stats(net: nn.Module, n: int = 3) -> None:
    """Move the running stats the way a train-mode forward does."""
    dg.reestimate_bn(net, (torch.randn(2, 3, 8, 8) * 3 + 1 for _ in range(n)), device_type="cpu")


def test_state_covers_every_bn_layer_with_running_stats() -> None:
    net = _net()
    net.add_module("no_stats", nn.BatchNorm2d(4, track_running_stats=False))
    assert set(dg.bn_state(net)) == {"1", "4"}


def test_load_restores_exactly_and_state_is_a_copy() -> None:
    net = _net()
    saved = dg.bn_state(net)
    _shift_stats(net)
    assert not torch.equal(dg.bn_state(net)["1"]["running_mean"], saved["1"]["running_mean"])
    dg.load_bn_state(net, saved)
    for name, stats in dg.bn_state(net).items():
        for key, value in stats.items():
            assert torch.equal(value, saved[name][key])


def test_load_rejects_a_different_layer_set() -> None:
    net = _net()
    state = dg.bn_state(net)
    state.pop("4")
    with pytest.raises(KeyError, match="layer mismatch"):
        dg.load_bn_state(net, state)


def test_reestimate_moves_stats_not_weights_and_ends_in_eval() -> None:
    net = _net()
    weights = {k: v.clone() for k, v in net.state_dict().items() if "running" not in k and "num_batches" not in k}
    before = dg.bn_state(net)
    assert dg.reestimate_bn(net, (torch.randn(2, 3, 8, 8) for _ in range(4)), device_type="cpu") == 4
    assert not net.training
    after = net.state_dict()
    for key, value in weights.items():
        assert torch.equal(after[key], value)
    assert not torch.equal(dg.bn_state(net)["1"]["running_var"], before["1"]["running_var"])


def test_drift_is_in_reference_sd_units() -> None:
    ref = {"bn": {"running_mean": torch.zeros(2), "running_var": torch.ones(2)}}
    cur = {"bn": {"running_mean": torch.tensor([0.5, -1.0]), "running_var": torch.tensor([4.0, 1.0])}}
    (row,) = dg.bn_drift(ref, cur, eps=0.0)
    assert row.channels == 2
    assert row.mean_shift_sd == pytest.approx(0.75)
    assert row.max_mean_shift_sd == pytest.approx(1.0)
    assert row.var_ratio_max == pytest.approx(4.0)
    assert row.var_ratio_min == pytest.approx(1.0)


def test_drift_is_zero_for_identical_state_and_summary_reads_it() -> None:
    state = dg.bn_state(_net())
    rows = dg.bn_drift(state, state)
    assert all(r.mean_shift_sd == 0.0 and r.var_ratio_median == pytest.approx(1.0) for r in rows)
    summary = dg.summarize_drift(rows)
    assert summary["layers"] == 2
    assert summary["share_layers_shift_over_0.5sd"] == 0.0
    assert summary["median_layer_var_ratio"] == pytest.approx(1.0)


def test_max_param_difference() -> None:
    a, b = _net(), _net()
    assert dg.max_param_difference(a, b) == 0.0
    with torch.no_grad():
        b[0].weight[0, 0, 0, 0] += 0.25
    assert dg.max_param_difference(a, b) == pytest.approx(0.25)


def test_score_summary_reports_offset_and_share() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.1])
    s = dg.score_summary(y, p)
    assert s["mean_prob"] == pytest.approx(0.45)
    assert s["mean_prob_positives"] == pytest.approx(0.9)
    assert s["mean_prob_negatives"] == pytest.approx(0.3)
    assert s["share_predicted_positive"] == pytest.approx(0.5)

"""Onset-timing report (docs/RECIPE_V2_PLAN.md Task 2).

Pins the parts a wrong number would hide in: the numpy readout equals the model's own readout, windows
count as knowable at a horizon by exactly the readout-target rule, onset groups partition the windows the
way the hazard loss's four cases do, a perfectly timed hazard scores AUC 1.0 at every horizon while a
horizon-32-only score does not, and the tuned threshold comes from val, not test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pedpredict.eval import onset_timing as ot
from pedpredict.models.heads import hazard_to_horizon_logits

_META = {"split": "test", "onset_bin_width": 4, "onset_lookahead": 96, "onset_horizon": 32}


def _arrays(onsets, observed, ever, **extra) -> dict[str, np.ndarray]:
    onsets = np.asarray(onsets)
    return {
        "onset_offset": onsets,
        "future_observed": np.asarray(observed),
        "track_crosses": np.asarray(ever),
        "crosses": ((onsets >= 0) & (onsets < 32)).astype(int),
        **extra,
    }


def _perfect_hazard(onsets: np.ndarray, bins: int = 24, width: int = 4) -> np.ndarray:
    """Hazard ~1 on the event bin, ~0 everywhere else (and everywhere for windows with no event)."""
    logits = np.full((len(onsets), bins), -20.0)
    for i, onset in enumerate(onsets):
        if 0 <= onset < bins * width:
            logits[i, onset // width] = 20.0
    return logits


def test_numpy_readout_matches_the_model_readout() -> None:
    logits = torch.randn(16, 24) * 3
    expected = torch.softmax(hazard_to_horizon_logits(logits, 8), dim=1)[:, 1].numpy()
    assert np.allclose(ot.hazard_horizon_prob(logits.numpy(), 8), expected, atol=1e-5)


def test_onset_groups_partition_every_case() -> None:
    arrays = _arrays(
        onsets=[3, 20, 40, 70, 150, -1, -1, -1],
        observed=[200, 200, 200, 200, 200, 120, 50, 200],
        ever=[1, 1, 1, 1, 1, 0, 0, 1],
    )
    assert ot.onset_groups(arrays).tolist() == [
        "0-15", "16-31", "32-63", "64-95", "96+", "none_in_96", "censored", "already_crossed",
    ]
    assert set(ot.group_order()) >= set(ot.onset_groups(arrays).tolist())


def test_knowable_follows_the_readout_target_rule() -> None:
    arrays = _arrays(onsets=[50, -1, -1, 10], observed=[200, 40, 40, 200], ever=[1, 0, 1, 1])
    label32, know32 = ot.horizon_targets(arrays, 32)
    label64, know64 = ot.horizon_targets(arrays, 64)
    assert label32.tolist() == [0, 0, 0, 1] and know32.tolist() == [True, True, False, True]
    # the censored window saw 40 frames: enough to rule out a crossing within 32, not within 64
    assert label64.tolist() == [1, 0, 0, 1] and know64.tolist() == [True, False, False, True]


def _timing_fixture(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    onsets = np.concatenate([rng.integers(0, 96, 60), np.full(40, -1)])
    arrays = _arrays(onsets, observed=np.full(100, 200), ever=(onsets >= 0).astype(int))
    arrays["hazard_logits"] = _perfect_hazard(onsets)
    arrays["p_readout"] = ot.hazard_horizon_prob(arrays["hazard_logits"], 8)
    arrays["p_frame"] = np.where(arrays["crosses"] == 1, 0.9, 0.1)       # knows "within 32" and nothing else
    return arrays


def test_perfect_timing_scores_one_at_every_horizon() -> None:
    report = ot.timing_report(_timing_fixture(), _META)
    for horizon, cell in report["horizon_auc"]["hazard_cumulative"].items():
        assert cell["auc"] == pytest.approx(1.0), horizon
    binary = report["horizon_auc"]["crosses_frame"]
    assert binary["32"]["auc"] == pytest.approx(1.0)
    assert binary["96"]["auc"] < 0.9          # a horizon-32-only score cannot rank 32–95-frame crossers


def test_group_means_fall_off_with_time_to_onset_for_a_timed_score() -> None:
    table = ot.timing_report(_timing_fixture(), _META)["score_by_onset_group"]["hazard_within_96"]
    assert table["0-15"]["mean"] > 0.99 and table["64-95"]["mean"] > 0.99
    assert table["none_in_96"]["mean"] < 0.01


def test_negative_split_separates_later_crossers_from_never() -> None:
    onsets = np.array([5, 10, 40, 60, -1, -1])
    arrays = _arrays(onsets, observed=np.full(6, 200), ever=np.array([1, 1, 1, 1, 0, 0]))
    arrays["p_frame"] = np.array([0.8, 0.7, 0.75, 0.9, 0.1, 0.2])   # later crossers look like positives
    split = ot.timing_report(arrays, _META)["negative_split_auc_32"]["crosses_frame"]
    assert split["no_crossing_seen"] == pytest.approx(1.0)
    assert split["later_crossers"] == pytest.approx(0.25)
    assert np.isnan(split["already_crossed"])


def test_tuned_threshold_comes_from_val() -> None:
    val = _arrays([5, -1, -1, -1], [200] * 4, [1, 0, 0, 0], p_frame=np.array([0.62, 0.4, 0.3, 0.2]))
    test = _arrays([5, 6, -1, -1], [200] * 4, [1, 1, 0, 0], p_frame=np.array([0.61, 0.5, 0.2, 0.1]))
    tuned = ot.timing_report(test, _META, val=val)["tuned_f1_stored_label"]["p_frame"]
    assert tuned["threshold"] == pytest.approx(0.45)   # lowest grid point clearing val's negatives
    assert tuned["recall"] == pytest.approx(1.0)       # 0.61 and 0.5 both clear it on test


def test_dump_roundtrip_keeps_arrays_strings_and_meta(tmp_path) -> None:
    arrays = _timing_fixture()
    arrays["track_id"] = np.array([f"ped_{i}" for i in range(100)])
    path = ot.save_dump(tmp_path / "d" / "dump.npz", arrays, _META)
    loaded, meta = ot.load_dump(path)
    assert meta == _META
    assert loaded["track_id"][7] == "ped_7"
    assert np.array_equal(loaded["hazard_logits"], arrays["hazard_logits"])


def test_dump_predictions_collects_heads_labels_and_track_ids() -> None:
    def item(i: int) -> dict:
        return {
            "images_tight": torch.zeros(2, 3, 4, 4), "images_context": torch.zeros(2, 3, 4, 4),
            "motions": torch.zeros(2, 5), "actions": torch.tensor(0), "looks": torch.tensor(0),
            "crosses": torch.tensor(i % 2), "track_id": f"ped_{i}",
            "onset_offset": torch.tensor(10 if i % 2 else -1), "future_observed": torch.tensor(90),
            "track_crosses": torch.tensor(i % 2),
        }

    batches = [ot.collate_with_track_ids([item(i), item(i + 1)], max_seq_len=2, motion_dim=5) for i in (0, 2)]

    def fake_forward(model, tight, context, motions):
        n = tight.shape[0]
        return {
            "crosses_frame": torch.tensor([[0.0, 1.0]] * n), "crosses_readout": torch.tensor([[1.0, 0.0]] * n),
            "crosses_hazard": torch.zeros(n, 24),
        }

    out = ot.dump_predictions(torch.nn.Identity(), [batches], torch.device("cpu"), forward=fake_forward)
    assert out["track_id"].tolist() == ["ped_0", "ped_1", "ped_2", "ped_3"]
    assert out["onset_offset"].tolist() == [-1, 10, -1, 10]
    assert out["hazard_logits"].shape == (4, 24)
    assert out["p_frame"][0] == pytest.approx(torch.softmax(torch.tensor([0.0, 1.0]), 0)[1].item())

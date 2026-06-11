"""M1 instrument — the effective sampler-draw distribution report.

The expected positive fraction of ``WeightedRandomSampler`` draws is exact in expectation
(``Σ(w·y)/Σw`` per chunk), so the tests pin it against hand-computed inverse-frequency weights and
against the sampler-off identity (effective == base under a plain shuffle).
"""

from __future__ import annotations

import dataclasses
import json
import pickle

import lmdb
import pytest

from pedpredict.config.schema import TrainCfg
from pedpredict.training.distribution import (
    DISTRIBUTION_FILENAME,
    effective_distribution,
    write_distribution_report,
)


def _write_label_lmdb(path, labels: list[tuple[int, int, int]]) -> None:
    """Tiny meta-only chunk: one ``<i>_meta`` pickle per (actions, looks, crosses) row."""
    env = lmdb.open(str(path), map_size=8 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for i, (act, lk, cr) in enumerate(labels):
                txn.put(f"{i}_meta".encode(), pickle.dumps({"actions": act, "looks": lk, "crosses": cr}))
    finally:
        env.close()


def test_effective_equals_base_without_sampler(tmp_path) -> None:
    p = tmp_path / "c0.lmdb"
    _write_label_lmdb(p, [(1, 0, 1), (0, 0, 0), (0, 1, 0), (1, 0, 0)])
    cfg = dataclasses.replace(TrainCfg(), use_weighted_sampler=False)
    out = effective_distribution([p], cfg)
    assert out["effective_rate"] == out["base_rate"]      # plain shuffle changes nothing
    assert out["base_rate"]["crosses"] == pytest.approx(0.25)
    assert out["n_samples"] == 4


def test_effective_rate_matches_hand_computation(tmp_path) -> None:
    # crosses labels [1,0,0]: inverse weights w(1)=3/(2·1)=1.5, w(0)=3/(2·2)=0.75 (sampler math, 1.6).
    # With powers {crosses:1, actions:0, looks:0}: expected positive draw rate = 1.5/(1.5+0.75+0.75) = 0.5.
    p = tmp_path / "c0.lmdb"
    _write_label_lmdb(p, [(0, 0, 1), (0, 0, 0), (0, 0, 0)])
    cfg = dataclasses.replace(
        TrainCfg(), sampler_powers={"crosses": 1.0, "actions": 0.0, "looks": 0.0}
    )
    out = effective_distribution([p], cfg)
    assert out["base_rate"]["crosses"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["effective_rate"]["crosses"] == pytest.approx(0.5, abs=1e-6)


def test_write_distribution_report(tmp_path) -> None:
    p = tmp_path / "c0.lmdb"
    _write_label_lmdb(p, [(1, 0, 1), (0, 1, 0)])
    run_dir = tmp_path / "run"
    path = write_distribution_report(run_dir, [p], TrainCfg())
    assert path == run_dir / DISTRIBUTION_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["base_rate"]) == {"actions", "looks", "crosses"}
    assert set(data["class_weights"]) == {"actions", "looks", "crosses"}  # lever-3 view recorded too
    assert data["use_weighted_sampler"] is True and data["use_class_weights"] is True


def test_empty_chunks_raise(tmp_path) -> None:
    p = tmp_path / "c0.lmdb"
    _write_label_lmdb(p, [])
    with pytest.raises(ValueError, match="no samples"):
        effective_distribution([p], TrainCfg())

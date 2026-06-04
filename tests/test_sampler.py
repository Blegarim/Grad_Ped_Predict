"""Prompt 1.6 — online sampler + unified-scan parity, monotonicity, and config tests.

Four kinds of checks (same shape as the 1.3 / 1.5 suites):
  * GOLDEN parity: ``class_weights_ce`` (global) reproduces OLD ``compute_class_weights_from_lmdb`` and
    per-chunk ``sample_weights`` reproduces OLD ``build_sampler_weights`` EXACTLY (atol=1e-6), from
    ``tests/fixtures/golden/sampler_cases.json`` (see ``tests/_capture/capture_sampler_golden.py``).
  * DEDUP: one ``scan_chunk_labels`` pass feeds both levers; ``LabelScanCache`` scans each chunk once and
    ``aggregate_counts`` equals a single global rescan.
  * MONOTONICITY: rarer class -> higher weight (loss and sampler), and a larger crosses power widens the
    rare/common spread.
  * CONFIG: ``sampler_min_weight`` loads + overrides + validates.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import lmdb
import pytest
import torch

from pedpredict.config import ConfigError, TrainCfg, load_config, validate_config
from pedpredict.data.sampler import (
    LabelScanCache,
    build_weighted_sampler,
    class_weights_ce,
    sample_weights,
    scan_chunk_labels,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "sampler_cases.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _write_label_lmdb(records: list[dict], path: str) -> None:
    """Mirror the capture script's label-only ``<i>_meta`` writer (cursor order == write order keys)."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for i, rec in enumerate(records):
                txn.put(f"{i}_meta".encode(), pickle.dumps(rec))
    finally:
        env.close()


@pytest.fixture
def chunk_paths(golden, tmp_path) -> list[str]:
    """Rebuild the fixture's chunks as tiny LMDBs under tmp_path; return their paths in order."""
    paths = []
    for i, chunk in enumerate(golden["chunks"]):
        p = str(tmp_path / f"chunk{i}.lmdb")
        _write_label_lmdb(chunk["records"], p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- golden parity


def test_scan_seq_id_order_matches_fixture(golden, chunk_paths):
    """scan_chunk_labels derives seq_ids in LMDB cursor order — same order the capture used."""
    for case, path in zip(golden["chunks"], chunk_paths, strict=True):
        scan = scan_chunk_labels(path)
        assert scan.seq_ids == case["seq_ids"]


def test_golden_sample_weights_per_chunk(golden, chunk_paths):
    """Per-chunk sample_weights == OLD build_sampler_weights, exactly (atol=1e-6)."""
    powers, min_w, tol = golden["powers"], golden["min_weight"], golden["tol"]
    for case, path in zip(golden["chunks"], chunk_paths, strict=True):
        scan = scan_chunk_labels(path)
        got = sample_weights(scan, powers, min_w)
        torch.testing.assert_close(
            torch.tensor(got, dtype=torch.double),
            torch.tensor(case["sample_weights"], dtype=torch.double),
            rtol=0.0, atol=tol,
        )


def test_golden_class_weights_global(golden, chunk_paths):
    """class_weights_ce(aggregate over all chunks) == OLD compute_class_weights_from_lmdb (atol=1e-6)."""
    cache = LabelScanCache()
    counts = cache.aggregate_counts(chunk_paths)
    weights = class_weights_ce(counts)
    for task in ("actions", "looks", "crosses"):
        torch.testing.assert_close(
            weights[task], torch.tensor(golden["class_weights"][task], dtype=torch.float32),
            rtol=0.0, atol=golden["tol"],
        )


def test_scan_counts_match_fixture(golden, chunk_paths):
    """Per-task observed counts from the single scan match the captured legacy Counters."""
    for case, path in zip(golden["chunks"], chunk_paths, strict=True):
        scan = scan_chunk_labels(path)
        for task in ("actions", "looks", "crosses"):
            got = {str(k): v for k, v in scan.counts[task].items()}
            assert got == case["counts"][task]


# --------------------------------------------------------------------------- dedup / cache


def test_cache_scans_each_chunk_once(golden, chunk_paths, monkeypatch):
    """LabelScanCache.get scans on miss only; repeated access + aggregate reuse the cached scan."""
    import pedpredict.data.sampler as sampler_mod

    calls: list[str] = []
    real = sampler_mod.scan_chunk_labels

    def _counting(path, seq_ids=None):
        calls.append(str(path))
        return real(path, seq_ids)

    monkeypatch.setattr(sampler_mod, "scan_chunk_labels", _counting)
    cache = LabelScanCache()
    cache.aggregate_counts(chunk_paths)        # one scan per chunk
    for p in chunk_paths:
        cache.get(p)                            # cache hits, no rescan
    assert sorted(calls) == sorted(chunk_paths)


def test_aggregate_counts_equals_single_rescan(chunk_paths):
    """aggregate_counts == summing independent per-chunk scans (no hidden divergence)."""
    cache = LabelScanCache()
    agg = cache.aggregate_counts(chunk_paths)
    manual: dict[str, dict[int, int]] = {"actions": {}, "looks": {}, "crosses": {}}
    for path in chunk_paths:
        scan = scan_chunk_labels(path)
        for task in ("actions", "looks", "crosses"):
            for label, count in scan.counts[task].items():
                manual[task][label] = manual[task].get(label, 0) + count
    assert agg == manual


def test_explicit_seq_ids_align_label_rows(chunk_paths):
    """Passing seq_ids reorders label_rows accordingly (per-sample weight alignment contract)."""
    default = scan_chunk_labels(chunk_paths[0])
    reordered = scan_chunk_labels(chunk_paths[0], seq_ids=list(reversed(default.seq_ids)))
    assert reordered.label_rows == list(reversed(default.label_rows))
    assert reordered.counts == default.counts   # counts are order-invariant


# --------------------------------------------------------------------------- monotonicity (synthetic)


def _build_lmdb(records: list[dict], path: str) -> None:
    _write_label_lmdb(records, path)


def test_class_weights_monotonic_rarer_higher(tmp_path):
    """Loss class weight is higher for the rarer class; rarer task -> larger class-1 weight."""
    records = (
        [{"actions": 1, "looks": 1, "crosses": 1}] * 5      # crosses rarest
        + [{"actions": 1, "looks": 1, "crosses": 0}] * 20
        + [{"actions": 0, "looks": 0, "crosses": 0}] * 75
    )
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    cache = LabelScanCache()
    w = class_weights_ce(cache.aggregate_counts([path]))
    # crosses positive (5/100) is rarer than its negative -> weight_1 > weight_0
    assert w["crosses"][1] > w["crosses"][0]
    # crosses (5% pos) rarer than actions (25% pos) -> larger positive-class weight
    assert w["crosses"][1] > w["actions"][1]


def test_sample_weights_monotonic_rare_sample_higher(tmp_path):
    """A sample carrying the rare crosses=1 gets a strictly higher sampler weight than a common one."""
    records = (
        [{"actions": 0, "looks": 0, "crosses": 1}]          # the single rare sample
        + [{"actions": 0, "looks": 0, "crosses": 0}] * 99
    )
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    scan = scan_chunk_labels(path)
    powers = {"crosses": 1.5, "actions": 0.3, "looks": 0.7}
    weights = sample_weights(scan, powers, 1e-6)
    rare = weights[scan.label_rows.index((0, 0, 1))]
    common = weights[scan.label_rows.index((0, 0, 0))]
    assert rare > common


def test_higher_cross_power_widens_spread(tmp_path):
    """Raising the crosses power increases the rare/common sampler-weight ratio."""
    records = (
        [{"actions": 0, "looks": 0, "crosses": 1}] * 3
        + [{"actions": 0, "looks": 0, "crosses": 0}] * 97
    )
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    scan = scan_chunk_labels(path)

    def _ratio(cross_pow: float) -> float:
        w = sample_weights(scan, {"crosses": cross_pow, "actions": 0.0, "looks": 0.0}, 1e-6)
        return w[scan.label_rows.index((0, 0, 1))] / w[scan.label_rows.index((0, 0, 0))]

    assert _ratio(2.0) > _ratio(1.0) > 1.0


# --------------------------------------------------------------------------- edge cases


def test_single_class_chunk_absent_class(tmp_path):
    """A chunk with only crosses=0 yields n_classes=1 and min_weight floor for the absent class."""
    records = [{"actions": 0, "looks": 0, "crosses": 0}] * 10
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    scan = scan_chunk_labels(path)
    assert scan.counts["crosses"] == {0: 10}      # single observed class
    w = sample_weights(scan, {"crosses": 1.5, "actions": 0.0, "looks": 0.0}, 1e-6)
    # invw[0] = total/(1*total) = 1.0 for every sample -> weight = 1.0**1.5 = 1.0
    assert all(abs(x - 1.0) < 1e-12 for x in w)


def test_zero_power_skips_task(tmp_path):
    """A power of 0 drops its task entirely (legacy `if pow > 0` guard)."""
    records = (
        [{"actions": 1, "looks": 1, "crosses": 1}] * 5
        + [{"actions": 0, "looks": 0, "crosses": 0}] * 20
    )
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    scan = scan_chunk_labels(path)
    only_cross = sample_weights(scan, {"crosses": 1.5, "actions": 0.0, "looks": 0.0}, 1e-6)
    with_all = sample_weights(scan, {"crosses": 1.5, "actions": 0.3, "looks": 0.7}, 1e-6)
    # actions/looks factors are constant within (0,0,0) vs (1,1,1); zeroing them changes values
    assert only_cross != with_all


def test_build_weighted_sampler_shape(tmp_path):
    """build_weighted_sampler returns a replacement sampler sized to the chunk."""
    records = [{"actions": i % 2, "looks": 0, "crosses": i % 5 == 0} for i in range(30)]
    path = str(tmp_path / "c.lmdb")
    _build_lmdb(records, path)
    scan = scan_chunk_labels(path)
    sampler = build_weighted_sampler(scan, TrainCfg())
    assert sampler.replacement is True
    assert sampler.num_samples == scan.n == 30
    drawn = list(iter(sampler))
    assert len(drawn) == 30
    assert all(0 <= idx < 30 for idx in drawn)


# --------------------------------------------------------------------------- config


def test_sampler_min_weight_loads_and_overrides():
    cfg = load_config("configs")
    assert cfg.train.sampler_min_weight == pytest.approx(1e-6)
    overridden = load_config("configs", overrides=["train.sampler_min_weight=1e-4"])
    assert overridden.train.sampler_min_weight == pytest.approx(1e-4)


def test_invalid_sampler_min_weight_rejected():
    root = load_config("configs")
    bad = dataclasses.replace(root, train=dataclasses.replace(root.train, sampler_min_weight=0.0))
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_invalid_sampler_powers_rejected():
    root = load_config("configs")
    bad = dataclasses.replace(
        root, train=dataclasses.replace(root.train, sampler_powers={"crosses": -1.0, "actions": 0.3, "looks": 0.7})
    )
    with pytest.raises(ConfigError):
        validate_config(bad)

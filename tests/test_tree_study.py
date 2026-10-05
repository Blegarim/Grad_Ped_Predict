"""Tree study end to end on synthetic exports: arms fit, dumps are standard, the report's numbers are the
detection curve's, and selection never reads the test split."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from pedpredict.baselines.tree import TreeParams
from pedpredict.eval import tree_study as ts
from pedpredict.eval.detection_curve import detection_curve
from pedpredict.eval.onset_timing import load_dump, save_dump
from pedpredict.eval.tree_study_report import best_threshold, build_results, format_report, intent_timing

_FAST = TreeParams(learning_rate=0.2, max_iter=15, max_leaf_nodes=7, min_samples_leaf=5, max_features=0.8)
_T, _C = 20, 58


def _streaming(rng: np.random.Generator, n_cross: int, n_never: int, prefix: str) -> dict[str, np.ndarray]:
    """Tracks sliding at stride 3; channel 2 (dx) ramps up as the crossing nears."""
    onset, observed, ever, tid = [], [], [], []
    for i in range(n_cross + n_never):
        crosser = i < n_cross
        n = int(rng.integers(15, 40))
        start = int(rng.integers(0, 150)) if crosser else 0
        offs = [start + 3 * (n - 1 - j) for j in range(n)] if crosser else [-1] * n
        if crosser:
            offs[-2:] = [-1, -1]
        onset += offs
        observed += [3 * (n - j) + 40 for j in range(n)]
        ever += [int(crosser)] * n
        tid += [f"{prefix}{i}"] * n
    onset = np.asarray(onset)
    X = rng.normal(size=(len(onset), _T, _C)).astype(np.float32)
    near = np.clip(1.0 - np.where(onset >= 0, onset, 400) / 120.0, 0, 1)
    X[:, :, 2] += (2.5 * near)[:, None]
    return {"X": X, "crosses": ((onset >= 0) & (onset < 32)).astype(np.int64), "onset_offset": onset,
            "future_observed": np.asarray(observed), "track_crosses": np.asarray(ever),
            "track_id": np.asarray(tid), "actions": np.zeros(len(onset), np.int64),
            "looks": np.zeros(len(onset), np.int64), "chunk": np.zeros(len(onset), np.int32)}


def _anchored(rng: np.random.Generator, n: int, prefix: str) -> dict[str, np.ndarray]:
    y = (rng.random(n) < 0.3).astype(np.int64)
    X = rng.normal(size=(n, _T, _C)).astype(np.float32)
    X[:, :, 5] += 1.5 * y[:, None]
    return {"X": X, "crosses": y, "track_id": np.asarray([f"{prefix}{i}" for i in range(n)]),
            "tte": np.full(n, 45), "actions": np.zeros(n, np.int64), "looks": np.zeros(n, np.int64),
            "chunk": np.zeros(n, np.int32)}


@pytest.fixture(scope="module")
def exports(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("exports")
    rng = np.random.default_rng(0)
    splits = {"train": _streaming(rng, 40, 60, "tr"), "val": _streaming(rng, 25, 40, "va"),
              "test": _streaming(rng, 30, 50, "te"), "anc_train": _anchored(rng, 400, "at"),
              "anc_val": _anchored(rng, 150, "av"), "anc_test": _anchored(rng, 150, "ae")}
    cens = _streaming(rng, 5, 15, "ce")
    cens["future_observed"] = np.minimum(cens["future_observed"], 25)
    cens["onset_offset"] = np.where(cens["onset_offset"] >= 25, -1, cens["onset_offset"])
    cens["crosses"][:] = 0
    splits["censored"] = cens
    for name, arrays in splits.items():
        save_dump(root / ts.EXPORT_FILES[name], arrays, {"split": name})
    return root


_SUBSET = ("B", "Hz", "HzC", "B_no_ego", "A", "B_matched", "Bd64", "Hz64")


@pytest.fixture(scope="module")
def fitted(exports, tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("tree_study")
    data = ts.load_exports(exports)
    arms = [dataclasses.replace(a, seeds=a.seeds[:2]) for a in ts.ARMS if a.name in _SUBSET]
    chosen = ts.run_arms(arms, data, out, _FAST, log=lambda *_: None)
    return out, chosen


def test_every_fit_writes_standard_dumps(fitted) -> None:
    out, chosen = fitted
    assert chosen["B_no_ego"] == chosen["B"] and chosen["B_matched"] == chosen["B"]
    for name in ("B", "Hz", "A"):
        test, meta = load_dump(out / name / "s42" / "onset_test.npz")
        key = "p_frame" if name != "Hz" else "p_readout"
        assert key in test and len(test[key]) == len(test["crosses"]) and meta["arm"] == name
        assert (out / name / "s42" / "anchored_test.npz").exists()
        fit = json.loads((out / name / "s42" / "fit.json").read_text())
        assert 1 <= fit["n_iter"] <= _FAST.max_iter
        assert fit["val_auc"] == pytest.approx(max(fit["val_auc_curve"]), abs=1e-5)
    hz = load_dump(out / "Hz" / "s42" / "onset_test.npz")[0]
    assert hz["hazard_logits"].shape == (len(hz["crosses"]), 24)
    assert json.loads((out / "B" / "arm.json").read_text())["weighting"] in ("none", "balanced")


def test_selection_never_reads_the_test_split(exports, tmp_path) -> None:
    """Corrupting the test labels changes no selection decision and no score."""
    data = ts.load_exports(exports)
    bad = dict(data)
    bad["test"] = data["test"].take(np.arange(len(data["test"])))
    bad["test"].labels["crosses"] = 1 - bad["test"].labels["crosses"]
    spec = dataclasses.replace(next(a for a in ts.ARMS if a.name == "Hz"), seeds=(42,))
    a = ts.fit_arm_seed(spec, data, 42, "none", _FAST)
    b = ts.fit_arm_seed(spec, bad, 42, "none", _FAST)
    assert a.n_iter == b.n_iter and a.val_auc == b.val_auc
    np.testing.assert_allclose(ts.score_split(a, spec, data["test"])["p_readout"],
                               ts.score_split(b, spec, bad["test"])["p_readout"])


def test_matched_arm_trains_on_as_many_windows_as_the_anchored_set(fitted, exports) -> None:
    out, _ = fitted
    fit = json.loads((out / "B_matched" / "s42" / "fit.json").read_text())
    assert fit["n_rows"] == len(ts.load_exports(exports, ["anc_train"])["anc_train"])


def test_report_numbers_are_the_detection_curve(fitted) -> None:
    out, _ = fitted
    results = build_results(out, n_boot=40)
    q1 = results["questions"]["Q1_hazard_vs_binary"]["per_window"]
    b_dumps = [load_dump(out / "B" / f"s{s}" / "onset_test.npz")[0] for s in (42, 43)]
    expected = np.mean([[p.detection_rate for p in detection_curve(d, "p_frame")] for d in b_dumps], axis=0)
    np.testing.assert_allclose(q1["rates"]["B"], expected)
    assert q1["verdict"] in ("Hz better", "B better", "equivalent within 5 pp", "inconclusive")
    assert results["questions"]["Q3_drop_ego"] is not None
    assert results["questions"]["Q4_f0.125_hazard_vs_binary"] is None    # arm not run in this test
    assert results["gap_decomposition"] is not None
    text = format_report(results)
    assert "Q1_hazard_vs_binary" in text and "_not run yet_" in text
    json.dumps(results, default=float)


def test_label_gate_flags_count_drift(exports) -> None:
    gate = ts.label_gate(ts.load_exports(exports, ["train"]))
    assert gate["train"]["ok"] is False and gate["train"]["crosses_vs_onset_rule_mismatches"] == 0


def test_best_threshold_matches_the_reference_rule() -> None:
    from pedpredict.training.metrics import _best_f1_threshold

    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.05).astype(int)
    p = np.clip(0.3 * y + rng.random(3000) * 0.5, 0, 1)
    grid = np.round(np.arange(1, 100) / 100.0, 2)
    assert best_threshold(y, p, grid) == _best_f1_threshold(y, p, list(grid))
    assert best_threshold(np.zeros(5), np.zeros(5)) == 0.5


def test_intent_timing_on_a_constructed_case() -> None:
    arrays = {"onset_offset": np.array([40, 10, -1, -1, -1, -1]), "track_crosses": np.array([1, 1, 1, 0, 0, 0]),
              "track_id": np.array(["a", "a", "a", "b", "b", "c"])}
    out = intent_timing(arrays, np.array([0.6, 0.9, 0.0, 0.1, 0.2, 0.3]))
    assert out["who"] == 1.0 and out["when"] == 1.0

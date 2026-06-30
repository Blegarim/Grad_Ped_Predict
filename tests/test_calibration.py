"""Calibration scaffolding tests (M10/RQ6): reliability binning, ECE, temperature fit/apply."""

from __future__ import annotations

import numpy as np
import pytest

from pedpredict.eval.calibration import (
    apply_temperature,
    calibrate_predictions_npz,
    calibrate_task,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def _overconfident(n: int = 20000, scale: float = 2.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from the CALIBRATED prob sigmoid(z); probs INFLATED to sigmoid(scale*z).

    The recovering temperature is then ``scale`` (dividing the inflated logit by it restores calibration).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=n)
    true_p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < true_p).astype(int)
    over_p = 1.0 / (1.0 + np.exp(-scale * z))
    return y, over_p


def test_reliability_curve_bins_and_counts() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.12, 0.18, 0.82, 0.88])              # two in decile bin 1, two in decile bin 8
    curve = reliability_curve(y, p, n_bins=10)
    assert curve.count.sum() == 4
    assert curve.count[1] == 2 and curve.count[8] == 2
    assert curve.accuracy[1] == pytest.approx(0.0) and curve.accuracy[8] == pytest.approx(1.0)
    assert curve.confidence[1] == pytest.approx(0.15) and curve.confidence[8] == pytest.approx(0.85)


def test_ece_zero_for_perfectly_calibrated() -> None:
    # Probabilities exactly equal to per-bin observed frequency -> ECE 0.
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # bin freq is 0.5, confidence 0.5
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_positive_for_overconfident() -> None:
    y, over_p = _overconfident()
    assert expected_calibration_error(y, over_p) > 0.05  # clearly miscalibrated


def test_fit_temperature_recovers_inflation_scale() -> None:
    y, over_p = _overconfident(scale=2.0)
    t = fit_temperature(y, over_p)
    assert t == pytest.approx(2.0, abs=0.25)  # recovers the inflation it must divide out


def test_apply_temperature_reduces_ece() -> None:
    y, over_p = _overconfident(scale=2.5)
    t = fit_temperature(y, over_p)
    before = expected_calibration_error(y, over_p)
    after = expected_calibration_error(y, apply_temperature(over_p, t))
    assert after < before * 0.5  # temperature scaling materially improves calibration


def test_fit_temperature_single_class_is_identity() -> None:
    y = np.zeros(50, dtype=int)
    p = np.full(50, 0.3)
    assert fit_temperature(y, p) == 1.0


def test_calibrate_task_applies_given_temperature() -> None:
    y, over_p = _overconfident()
    fit = calibrate_task(y, over_p)                      # fits on these data
    applied = calibrate_task(y, over_p, temperature=fit.temperature)
    assert applied.temperature == pytest.approx(fit.temperature)
    assert applied.ece_after == pytest.approx(fit.ece_after, abs=1e-9)


def test_calibrate_predictions_npz_roundtrip(tmp_path) -> None:
    y, over_p = _overconfident()
    path = tmp_path / "predictions.npz"
    payload = {}
    for task in ("actions", "looks", "crosses"):
        payload[f"{task}_true"] = y
        payload[f"{task}_prob_1"] = over_p
    np.savez(path, **payload)
    report = calibrate_predictions_npz(path)
    assert set(report) == {"actions", "looks", "crosses"}
    for cal in report.values():
        assert cal.temperature > 1.0                     # overconfident -> T > 1
        assert cal.ece_after < cal.ece_before

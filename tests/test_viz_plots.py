"""Prompt 6.1 — quantitative viz plots.

Viz parity is about correct *data extraction* + non-crashing *render* against the NEW artifact schemas
(not pixel-diffing). Tests build artifacts with the REAL writers/column tuples (``TRAIN_LOG_COLUMNS``,
``EVAL_LOG_COLUMNS``, ``save_predictions_npz``, ``save_temporal_weights_npz``) so the loaders are locked
to the contracts they consume — if 4.5/5.1 change a column, these break loudly. The final test is the
requested end-to-end regeneration smoke over a sample run dir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from pedpredict.eval.evaluate import (
    EVAL_LOG_COLUMNS,
    save_predictions_npz,
    save_temporal_weights_npz,
)
from pedpredict.training.trainer import TRAIN_LOG_COLUMNS
from pedpredict.utils.logging import CsvLogger, RunDir, create_run_dir
from pedpredict.viz.plots import (
    ABLATION_F1_PNG,
    figure_ablation_bars,
    figure_f1_threshold,
    figure_loss_curves,
    figure_per_head_f1_curves,
    figure_pr_curves,
    figure_temporal_attention,
    generate_ablation_figures,
    generate_run_figures,
    load_ablation_metrics,
    load_eval_row,
    load_predictions_npz,
    load_train_log,
)

_TASKS = ("actions", "looks", "crosses")
_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- artifact builders


def _write_train_log(path: Path, n_epochs: int = 4) -> None:
    """Write a ``train_log.csv`` with every ``TRAIN_LOG_COLUMNS`` column populated."""
    with CsvLogger(path, TRAIN_LOG_COLUMNS) as logger:
        for e in range(n_epochs):
            row = {col: 0.5 for col in TRAIN_LOG_COLUMNS}
            row["epoch"] = e + 1
            row["train_loss"] = 1.0 - 0.1 * e
            row["val_loss"] = 1.2 - 0.05 * e  # min at last epoch
            row["lr"] = 1e-4
            row["epoch_time_s"] = 10.0
            logger.log(row)


def _make_predictions(n: int = 200) -> dict[str, np.ndarray]:
    """Per-sample predictions dict with both classes present per task (so AUC/PR are defined)."""
    preds: dict[str, np.ndarray] = {}
    for task in _TASKS:
        y_true = (_RNG.random(n) < 0.4).astype(int)
        # separable-ish probs so best-F1 is finite and AP > chance
        prob1 = np.clip(0.2 + 0.6 * y_true + 0.2 * _RNG.standard_normal(n), 0.01, 0.99)
        preds[f"{task}_true"] = y_true
        preds[f"{task}_prob_1"] = prob1
        preds[f"{task}_prob_0"] = 1.0 - prob1
        preds[f"{task}_pred"] = (prob1 >= 0.5).astype(int)
    return preds


def _write_eval_log(path: Path, model_type: str, f1: float, auc: float) -> None:
    """Write one ``eval_log.csv`` row (partial dict; DictWriter blanks the rest of EVAL_LOG_COLUMNS)."""
    row: dict[str, object] = {"model_type": model_type, "split": "test", "n_samples": 100}
    for task in _TASKS:
        row[f"{task}_f1"] = f1
        row[f"{task}_auc"] = auc
    with CsvLogger(path, EVAL_LOG_COLUMNS) as logger:
        logger.log(row)


# --------------------------------------------------------------------------- loaders


def test_load_train_log_schema(tmp_path: Path) -> None:
    path = tmp_path / "train_log.csv"
    _write_train_log(path, n_epochs=3)
    log = load_train_log(path)
    assert log["epoch"].tolist() == [1, 2, 3]
    for col in ("train_loss", "val_loss", "macro_f1", *(f"{t}_f1" for t in _TASKS)):
        assert col in log and log[col].shape == (3,)


def test_load_predictions_npz_roundtrip(tmp_path: Path) -> None:
    preds = _make_predictions(50)
    path = save_predictions_npz(tmp_path / "predictions.npz", preds)
    loaded = load_predictions_npz(path)
    for key in preds:
        np.testing.assert_array_almost_equal(loaded[key], preds[key])


def test_load_eval_row_picks_last(tmp_path: Path) -> None:
    path = tmp_path / "eval_log.csv"
    with CsvLogger(path, EVAL_LOG_COLUMNS) as logger:
        logger.log({"model_type": "full", "crosses_f1": 0.10})
        logger.log({"model_type": "full", "crosses_f1": 0.42})  # most recent
    row = load_eval_row(path)
    assert row["model_type"] == "full"
    assert row["crosses_f1"] == pytest.approx(0.42)


def test_load_eval_row_empty_raises(tmp_path: Path) -> None:
    path = tmp_path / "eval_log.csv"
    CsvLogger(path, EVAL_LOG_COLUMNS).close()  # header only, no data rows
    with pytest.raises(ValueError, match="no data rows"):
        load_eval_row(path)


def test_load_ablation_metrics_no_summary_parsing(tmp_path: Path) -> None:
    """The OLD ``_parse_summary_table`` text-scraping is unnecessary: columns are read directly."""
    logs = {}
    for name, f1, auc in (("full", 0.6, 0.8), ("motion_only", 0.4, 0.7)):
        p = tmp_path / f"{name}.csv"
        _write_eval_log(p, name, f1, auc)
        logs[name] = p
    abl = load_ablation_metrics(logs)
    assert abl["full"]["crosses_f1"] == pytest.approx(0.6)
    assert abl["motion_only"]["actions_auc"] == pytest.approx(0.7)


# --------------------------------------------------------------------------- pure figures (smoke)


def test_figure_loss_curves_best_epoch() -> None:
    log = {"epoch": np.array([1, 2, 3]), "train_loss": np.array([1.0, 0.8, 0.6]),
           "val_loss": np.array([1.0, 0.7, 0.9])}  # min at epoch 2
    fig = figure_loss_curves(log)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    vlines = [ln for ln in ax.get_lines() if "Best val epoch" in (ln.get_label() or "")]
    assert vlines and vlines[0].get_xdata()[0] == 2


def test_figure_per_head_f1_lines() -> None:
    n = 4
    log = {"epoch": np.arange(1, n + 1), "macro_f1": np.linspace(0.3, 0.6, n)}
    for t in _TASKS:
        log[f"{t}_f1"] = np.linspace(0.2, 0.7, n)
    fig = figure_per_head_f1_curves(log)
    assert len(fig.axes[0].get_lines()) == len(_TASKS) + 1  # 3 tasks + macro


def test_figure_pr_and_threshold_smoke() -> None:
    preds = _make_predictions(300)
    for fig in (figure_pr_curves(preds), figure_f1_threshold(preds)):
        assert isinstance(fig, Figure)
        assert len(fig.axes) == len(_TASKS)


def test_figure_ablation_bars_counts() -> None:
    abl = {
        "full": {f"{t}_f1": 0.6 for t in _TASKS},
        "motion_only": {f"{t}_f1": 0.4 for t in _TASKS},
    }
    fig = figure_ablation_bars(abl, "f1", "F1")
    ax = fig.axes[0]
    assert len(ax.patches) == len(_TASKS) * 2  # 3 tasks x 2 models
    # full-model macro reference line present
    assert any("Full macro" in (ln.get_label() or "") for ln in ax.get_lines())


def test_figure_temporal_attention_hides_missing() -> None:
    weights = _RNG.random((40, 20))
    labels = {"actions_true": (_RNG.random(40) < 0.5).astype(int)}  # only one task present
    fig = figure_temporal_attention(weights, labels)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 1


# --------------------------------------------------------------------------- run-dir drivers


def _make_run_dir(tmp_path: Path, *, with_preds: bool, with_tw: bool) -> RunDir:
    run = RunDir(run_id="20260607_120000_full", path=create_run_dir(tmp_path, "20260607_120000_full"))
    _write_train_log(run.train_log_path)
    if with_preds:
        save_predictions_npz(run.plots_dir / "predictions.npz", _make_predictions(120))
    if with_tw:
        save_temporal_weights_npz(
            run.plots_dir / "temporal_weights.npz", _RNG.random((120, 20)), _make_predictions(120)
        )
    return run


def test_generate_run_figures_skips_missing(tmp_path: Path) -> None:
    run = _make_run_dir(tmp_path, with_preds=False, with_tw=False)
    written = generate_run_figures(run)
    names = {p.name for p in written}
    assert names == {"loss_curves.png", "per_head_f1_curves.png"}  # only train-log figures


def test_generate_run_figures_only_subset(tmp_path: Path) -> None:
    run = _make_run_dir(tmp_path, with_preds=True, with_tw=False)
    written = generate_run_figures(run, which=["loss"])
    assert {p.name for p in written} == {"loss_curves.png"}


def test_generate_ablation_figures(tmp_path: Path) -> None:
    logs = {}
    for name in ("full", "motion_only"):
        p = tmp_path / f"{name}.csv"
        _write_eval_log(p, name, f1=0.5, auc=0.7)
        logs[name] = p
    written = generate_ablation_figures(logs, tmp_path / "ablation")
    assert {p.name for p in written} == {ABLATION_F1_PNG, "ablation_auc.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_regenerate_from_sample_run(tmp_path: Path) -> None:
    """End-to-end: a full sample run dir regenerates all five per-run PNGs (the requested smoke test)."""
    run = _make_run_dir(tmp_path, with_preds=True, with_tw=True)
    written = generate_run_figures(run)
    assert {p.name for p in written} == {
        "loss_curves.png",
        "per_head_f1_curves.png",
        "pr_curves.png",
        "f1_threshold.png",
        "temporal_attention.png",
    }
    assert all(p.exists() and p.stat().st_size > 0 for p in written)

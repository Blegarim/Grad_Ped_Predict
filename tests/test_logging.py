"""Tests for the run-directory + CSV logging conventions (Prompt 4.5).

Infrastructure module (MIGRATION row 4.5: fixture = n/a) — these assert the run-dir scaffold, the
canonical CSV schemas, the config snapshot, and the cross-run ``index.csv`` machinery. CPU-only and
CI-safe; tests import the *installed* ``pedpredict`` package (src-layout editable install).
"""

from __future__ import annotations

import csv
import dataclasses
import math
from datetime import datetime

from pedpredict.config.loader import load_resolved_config
from pedpredict.config.schema import RootCfg
from pedpredict.training.metrics import METRIC_COLUMNS, MetricResult, TaskMetrics
from pedpredict.training.trainer import TRAIN_LOG_COLUMNS, EpochResult, Trainer
from pedpredict.utils.logging import (
    CONFIG_SNAPSHOT_FILENAME,
    INDEX_COLUMNS,
    INDEX_FILENAME,
    RunDir,
    append_index_row,
    build_index_row,
    git_sha,
    init_run,
    make_run_id,
    read_index,
    rebuild_index,
    round_row,
    snapshot_config,
)

_NOW = datetime(2026, 6, 6, 1, 2, 3)


def _cfg(runs_root, *, model_type: str = "full") -> RootCfg:
    """A RootCfg whose runs_dir points at a tmp path (and the chosen model_type)."""
    root = RootCfg()
    paths = dataclasses.replace(root.paths, runs_dir=str(runs_root))
    evalc = dataclasses.replace(root.eval, model_type=model_type)
    return dataclasses.replace(root, paths=paths, eval=evalc)


def _metric_result(crosses_f1: float = 0.4321) -> MetricResult:
    """A small MetricResult with distinct per-task F1s so column routing is checkable."""
    tm = lambda f1: TaskMetrics(accuracy=0.9, f1=f1, auc=0.8, precision=0.7, recall=0.6)  # noqa: E731
    return MetricResult(
        per_task={"actions": tm(0.11), "looks": tm(0.22), "crosses": tm(crosses_f1)},
        macro_f1=0.25,
        overall_accuracy=0.9,
    )


# --------------------------------------------------------------------------- schema sync (no drift)


def test_train_log_columns_compose_metric_columns() -> None:
    # 4.5 owns the context+enrichment columns; metric names stay owned by 3.2 (no duplication/drift).
    assert TRAIN_LOG_COLUMNS == ("epoch", "train_loss", "val_loss", "lr", "epoch_time_s", *METRIC_COLUMNS)


def test_index_columns_are_stable_and_lead_with_crosses() -> None:
    assert INDEX_COLUMNS[0] == "run_id"
    # primary metric (experiment-tracking SKILL) is the first headline metric column.
    headline = INDEX_COLUMNS[INDEX_COLUMNS.index("best_val_loss") + 1]
    assert headline == "crosses_f1"
    for col in ("crosses_f1", "crosses_auc", "macro_f1", "looks_f1", "actions_f1"):
        assert col in INDEX_COLUMNS


# --------------------------------------------------------------------------- run id + run dir


def test_make_run_id_timestamp_first() -> None:
    rid = make_run_id("motion_only", "two phase/exp", now=_NOW)
    assert rid == "20260606_010203_motion_only_two_phase_exp"
    assert " " not in rid and "/" not in rid


def test_init_run_scaffolds_and_snapshots(tmp_path) -> None:
    cfg = _cfg(tmp_path / "runs", model_type="vanilla_concat")
    run = init_run(cfg, tag="exp1", now=_NOW)
    assert run.run_id == "20260606_010203_vanilla_concat_exp1"
    assert run.checkpoints_dir.is_dir()
    assert run.plots_dir.is_dir()
    assert run.config_path.name == CONFIG_SNAPSHOT_FILENAME
    assert run.config_path.exists()
    # snapshot round-trips back to an equivalent config tree.
    restored = load_resolved_config(run.config_path)
    assert restored.eval.model_type == "vanilla_concat"


def test_init_run_folds_kind_into_id_for_non_train(tmp_path) -> None:
    cfg = _cfg(tmp_path / "runs")
    run = init_run(cfg, kind="eval", now=_NOW)
    assert run.run_id.endswith("_full_eval")


def test_snapshot_config_writes_resolved_yaml(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    out = snapshot_config(tmp_path, cfg)
    assert out.name == CONFIG_SNAPSHOT_FILENAME
    assert load_resolved_config(out).train.lr == cfg.train.lr


# --------------------------------------------------------------------------- round_row / git_sha


def test_round_row_rounds_floats_only() -> None:
    row = {"a": 0.123456, "b": 7, "c": "text", "d": True}
    out = round_row(row, ndigits=2)
    assert out == {"a": 0.12, "b": 7, "c": "text", "d": True}


def test_round_row_preserves_nan() -> None:
    assert math.isnan(round_row({"x": float("nan")})["x"])


def test_git_sha_type() -> None:
    sha = git_sha()
    assert sha is None or (isinstance(sha, str) and sha)


# --------------------------------------------------------------------------- index row + append


def test_build_index_row_routes_headline_metrics(tmp_path) -> None:
    run = RunDir(run_id="20260606_010203_full_exp", path=tmp_path / "run")
    metrics = _metric_result(crosses_f1=0.4321)
    row = build_index_row(
        run,
        model_type="full",
        tag="exp",
        kind="train",
        epochs_run=12,
        best_epoch=7,
        best_val_loss=0.123456,
        headline=metrics.as_flat_dict(),
        best_ckpt=None,
        git_rev="abc1234",
    )
    assert set(row) == set(INDEX_COLUMNS)
    assert row["timestamp"] == "20260606_010203"
    assert row["crosses_f1"] == 0.4321
    assert row["looks_f1"] == 0.22
    assert row["actions_f1"] == 0.11
    assert row["best_val_loss"] == 0.1235  # rounded to 4 dp
    assert row["git_sha"] == "abc1234"
    assert row["best_ckpt"] == ""


def test_append_index_row_header_once_and_monotonic(tmp_path) -> None:
    base = dict.fromkeys(INDEX_COLUMNS, "")
    for i in range(3):
        row = {**base, "run_id": f"run{i}", "crosses_f1": 0.1 * i}
        append_index_row(tmp_path, row)
    index_path = tmp_path / INDEX_FILENAME
    with open(index_path, newline="", encoding="utf-8") as f:
        lines = list(csv.reader(f))
    assert lines[0] == list(INDEX_COLUMNS)        # header written exactly once
    assert [r[0] for r in lines[1:]] == ["run0", "run1", "run2"]
    rows = read_index(tmp_path)
    assert len(rows) == 3 and rows[1]["run_id"] == "run1"


# --------------------------------------------------------------------------- rebuild_index recovery


def test_rebuild_index_reconstructs_from_run_dirs(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    # Two runs, each with a config snapshot + a 2-epoch train_log.csv (best = lower val_loss).
    for model_type, best_val in (("full", 0.20), ("motion_only", 0.50)):
        cfg = _cfg(runs_root, model_type=model_type)
        run = init_run(cfg, tag="t", now=_NOW)
        logger = run.train_logger(TRAIN_LOG_COLUMNS)
        for epoch, (val, cf1) in enumerate([(0.9, 0.10), (best_val, 0.30)], start=1):
            base = dict.fromkeys(TRAIN_LOG_COLUMNS, 0.0)
            logger.log({**base, "epoch": epoch, "val_loss": val, "crosses_f1": cf1})
        logger.close()

    rebuilt = rebuild_index(runs_root)
    assert rebuilt == runs_root / INDEX_FILENAME
    rows = {r["model_type"]: r for r in read_index(runs_root)}
    assert set(rows) == {"full", "motion_only"}
    assert rows["full"]["tag"] == "t"
    assert int(rows["full"]["epochs_run"]) == 2
    assert int(rows["full"]["best_epoch"]) == 2          # epoch with the min val_loss
    assert float(rows["full"]["best_val_loss"]) == 0.20
    assert float(rows["full"]["crosses_f1"]) == 0.30     # metrics at the best epoch


# --------------------------------------------------------------------------- trainer integration


def test_trainer_append_index_row_end_to_end(tmp_path) -> None:
    cfg = _cfg(tmp_path / "runs")
    run = init_run(cfg, tag="it", now=_NOW)
    # Construct a Trainer shell without running fit(): drive _append_index_row directly.
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.run_dir = run.path
    trainer.run_tag = "it"
    trainer.best_val_loss = 0.33
    trainer._best_epoch = 1
    results = [
        EpochResult(0, 1.0, 0.50, _metric_result(0.10)),
        EpochResult(1, 0.8, 0.33, _metric_result(0.40)),
    ]
    trainer._append_index_row(results, kind="train")
    rows = read_index(tmp_path / "runs")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run.run_id
    assert int(row["epochs_run"]) == 2
    assert int(row["best_epoch"]) == 2                   # _best_epoch (0-based 1) + 1
    assert float(row["crosses_f1"]) == 0.40              # headline from the best epoch

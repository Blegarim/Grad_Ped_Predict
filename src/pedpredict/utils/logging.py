"""Run-directory + CSV logging conventions (Prompts 0.3 + 4.5).

Two layers live here:

* **Primitives (0.3):** a filesystem-safe :func:`make_run_id`, the per-run directory scaffold
  (:func:`create_run_dir` -> ``outputs/runs/{run_id}/{checkpoints,plots}``), and a generic append-only
  :class:`CsvLogger`. These hardcode no column schema — callers pass their ``fieldnames`` in.

* **Experiment-tracking conventions (4.5):** the :class:`RunDir` handle, :func:`init_run` (run id +
  scaffold + resolved-config snapshot), the cross-run :data:`INDEX_COLUMNS` table and its
  :func:`build_index_row` / :func:`append_index_row` / :func:`rebuild_index` helpers, plus the
  :func:`round_row` / :func:`git_sha` utilities. This replaces the OLD ad-hoc
  ``training_log/training_log_%m%d_%H%M.csv`` sprawl + ``model_suffix`` naming (B1 / B11).

The *metric* column names (``{task}_{acc,f1,auc,...}``) are owned by ``training/metrics.py``
(:data:`~pedpredict.training.metrics.METRIC_COLUMNS`); the train-log schema that composes them
(:data:`~pedpredict.training.trainer.TRAIN_LOG_COLUMNS`) lives in the training layer. This module stays
free of any ``training`` import so it remains a low-level helper (importing ``training.metrics`` here
would cycle through ``training/__init__`` -> ``trainer`` -> ``utils.logging``). :data:`INDEX_COLUMNS` is
plain strings and :func:`build_index_row` takes a flat metric dict, so the cross-run index needs no metric
import.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pedpredict.config.schema import RootCfg

__all__ = [
    "CONFIG_SNAPSHOT_FILENAME",
    "EVAL_LOG_FILENAME",
    "INDEX_COLUMNS",
    "INDEX_FILENAME",
    "TRAIN_LOG_FILENAME",
    "CsvLogger",
    "RunDir",
    "append_index_row",
    "build_index_row",
    "create_run_dir",
    "get_csv_logger",
    "git_sha",
    "init_run",
    "make_run_id",
    "read_index",
    "rebuild_index",
    "round_row",
    "snapshot_config",
]

_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")

#: Canonical per-run artifact filenames (one place so writers/readers/viz/tests agree).
CONFIG_SNAPSHOT_FILENAME = "resolved_config.yaml"
TRAIN_LOG_FILENAME = "train_log.csv"
EVAL_LOG_FILENAME = "eval_log.csv"
INDEX_FILENAME = "index.csv"

#: Cross-run comparison table (one row per finished run), written at ``outputs/runs/index.csv``.
#: Headline metrics lead with ``crosses_*`` — the primary task per the experiment-tracking SKILL
#: ("What Better Means": primary ``crosses_f1``, secondary ``crosses_auc``; guard ``looks``/``actions``).
INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "model_type",
    "tag",
    "kind",
    "epochs_run",
    "best_epoch",
    "best_val_loss",
    "crosses_f1",
    "crosses_auc",
    "macro_f1",
    "looks_f1",
    "actions_f1",
    "run_dir",
    "best_ckpt",
    "git_sha",
)

_AUTO = object()  # sentinel: build_index_row auto-detects the git sha


def _sanitize(token: str) -> str:
    """Collapse filesystem-unsafe runs (spaces, slashes, ...) to single underscores."""
    return _UNSAFE.sub("_", token).strip("_")


def make_run_id(model_type: str, tag: str = "", *, now: datetime | None = None) -> str:
    """Build ``{YYYYMMDD_HHMMSS}_{model_type}[_{tag}]`` — filesystem-safe, no spaces/slashes.

    Timestamp-first (vs the OLD SKILL's ``{variant}_{date}_{note}``) so a plain ``ls`` sorts runs
    chronologically; matches the schematic's ``run_id = timestamp + model_type + tag``.
    """
    stamp = (now or datetime.now()).strftime(_TIMESTAMP_FMT)
    parts = [stamp, _sanitize(model_type)]
    tag = _sanitize(tag)
    if tag:
        parts.append(tag)
    return "_".join(parts)


def create_run_dir(runs_root: Path, run_id: str) -> Path:
    """Create ``runs_root/run_id`` plus its ``checkpoints/`` and ``plots/`` subdirs; return it."""
    run_dir = Path(runs_root) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


class CsvLogger:
    """Append-only CSV writer with a fixed header.

    The header is written exactly once: on first creation of the file (or when it is empty).
    Re-opening an existing populated file appends rows without duplicating the header, so a
    resumed run extends its log. Each :meth:`log` flushes, so a crashed run leaves a readable CSV.
    """

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()
            self._handle.flush()

    def log(self, row: Mapping[str, object]) -> None:
        """Append one row (keys must be a subset of ``fieldnames``); flush immediately."""
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def get_csv_logger(path: Path, fieldnames: Sequence[str]) -> CsvLogger:
    """Convenience constructor mirroring the schematic's ``get_csv_logger(...)`` name."""
    return CsvLogger(path, fieldnames)


# --------------------------------------------------------------------------- run directory (4.5)


@dataclass(frozen=True)
class RunDir:
    """A resolved per-run directory + the canonical artifact paths inside it.

    Construct via :func:`init_run` (which also snapshots the config), or directly from an existing
    ``outputs/runs/{run_id}`` path for read/rebuild tooling.
    """

    run_id: str
    path: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def plots_dir(self) -> Path:
        return self.path / "plots"

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_SNAPSHOT_FILENAME

    @property
    def train_log_path(self) -> Path:
        return self.path / TRAIN_LOG_FILENAME

    @property
    def eval_log_path(self) -> Path:
        return self.path / EVAL_LOG_FILENAME

    @property
    def best_ckpt_path(self) -> Path:
        return self.checkpoints_dir / "best.pth"

    def csv_logger(self, name: str, fieldnames: Sequence[str]) -> CsvLogger:
        """A :class:`CsvLogger` writing ``<run_dir>/<name>`` with the given columns."""
        return CsvLogger(self.path / name, fieldnames)

    def train_logger(self, fieldnames: Sequence[str]) -> CsvLogger:
        """The per-epoch train+val log (caller passes ``trainer.TRAIN_LOG_COLUMNS``)."""
        return CsvLogger(self.train_log_path, fieldnames)

    def eval_logger(self, fieldnames: Sequence[str]) -> CsvLogger:
        """The test/eval log (5.1 passes its ``EVAL_LOG_COLUMNS``)."""
        return CsvLogger(self.eval_log_path, fieldnames)


def snapshot_config(run_dir: Path, cfg: RootCfg) -> Path:
    """Write the resolved config to ``<run_dir>/resolved_config.yaml`` for reproducibility.

    Thin wrapper over ``config.loader.dump_config`` (imported lazily so this low-level module never
    couples ``utils`` to ``config`` at import time). OLD training never snapshotted config.
    """
    from pedpredict.config.loader import dump_config

    return dump_config(cfg, run_dir)


def init_run(cfg: RootCfg, *, tag: str = "", kind: str = "train", now: datetime | None = None) -> RunDir:
    """Make a run id, scaffold ``cfg.paths.runs_dir/{run_id}``, snapshot the config; return a RunDir.

    ``kind`` distinguishes non-training runs (e.g. ``"eval"``): for ``kind != "train"`` it is folded
    into the run id so standalone runs are visually distinct (``..._full_eval``). The model type comes
    from ``cfg.eval.model_type`` (the shared selector).
    """
    full_tag = tag if kind == "train" else (f"{tag}_{kind}" if tag else kind)
    run_id = make_run_id(cfg.eval.model_type, full_tag, now=now)
    path = create_run_dir(Path(cfg.paths.runs_dir), run_id)
    snapshot_config(path, cfg)
    return RunDir(run_id=run_id, path=path)


# --------------------------------------------------------------------------- cross-run index (4.5)


def round_row(row: Mapping[str, object], ndigits: int = 4) -> dict[str, object]:
    """Round every ``float`` cell to ``ndigits`` (standardized at 4 dp); pass non-floats through.

    ``bool`` is excluded (it is an ``int`` subclass but not a metric); ``nan`` rounds to ``nan``.
    """
    return {
        k: (round(v, ndigits) if isinstance(v, float) else v)
        for k, v in row.items()
    }


def git_sha(*, short: bool = True) -> str | None:
    """Current commit sha for reproducibility, or ``None`` outside a git work tree / on error."""
    cmd = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _run_timestamp(run_id: str) -> str:
    """The ``YYYYMMDD_HHMMSS`` prefix of a run id (first two underscore tokens)."""
    return "_".join(run_id.split("_")[:2])


def build_index_row(
    run_dir: RunDir,
    *,
    model_type: str,
    tag: str = "",
    kind: str = "train",
    epochs_run: int,
    best_epoch: int,
    best_val_loss: float,
    headline: Mapping[str, float],
    best_ckpt: str | Path | None = None,
    git_rev: str | None | object = _AUTO,
    now: datetime | None = None,  # noqa: ARG001 (reserved; timestamp derives from run_id)
) -> dict[str, object]:
    """Assemble one :data:`INDEX_COLUMNS` row from a run's headline numbers.

    ``headline`` is a flat metric dict (``MetricResult.as_flat_dict()`` or a parsed train-log row):
    ``crosses_f1 / crosses_auc / macro_f1 / looks_f1 / actions_f1`` are picked out. ``git_rev`` defaults
    to auto-detecting the current sha; pass an explicit string (or ``None``) to override.
    """
    rev = git_sha() if git_rev is _AUTO else git_rev
    row: dict[str, object] = {
        "run_id": run_dir.run_id,
        "timestamp": _run_timestamp(run_dir.run_id),
        "model_type": model_type,
        "tag": tag,
        "kind": kind,
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "crosses_f1": _as_float(headline.get("crosses_f1")),
        "crosses_auc": _as_float(headline.get("crosses_auc")),
        "macro_f1": _as_float(headline.get("macro_f1")),
        "looks_f1": _as_float(headline.get("looks_f1")),
        "actions_f1": _as_float(headline.get("actions_f1")),
        "run_dir": str(run_dir.path),
        "best_ckpt": str(best_ckpt) if best_ckpt is not None else "",
        "git_sha": rev or "",
    }
    return round_row(row)


def _as_float(value: object) -> float:
    """Coerce a metric cell (number or CSV string) to ``float``; ``None``/blank -> ``nan``."""
    if value is None or value == "":
        return float("nan")
    return float(value)  # type: ignore[arg-type]


def append_index_row(
    runs_root: str | Path,
    row: Mapping[str, object],
    *,
    columns: Sequence[str] = INDEX_COLUMNS,
) -> Path:
    """Append one row to ``runs_root/index.csv`` (header written once); return the index path."""
    path = Path(runs_root) / INDEX_FILENAME
    with CsvLogger(path, columns) as logger:
        logger.log(row)
    return path


def read_index(runs_root: str | Path) -> list[dict[str, str]]:
    """Read ``runs_root/index.csv`` as a list of row dicts (empty list if it does not exist)."""
    path = Path(runs_root) / INDEX_FILENAME
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rebuild_index(runs_root: str | Path) -> Path:
    """Regenerate ``runs_root/index.csv`` from every run dir's ``train_log.csv`` + config snapshot.

    Recovery tool (CI-friendly, B11): scans each ``outputs/runs/{run_id}`` that has a populated
    ``train_log.csv``, derives the best epoch (min ``val_loss``), reads ``model_type`` from the config
    snapshot, and rewrites the index from scratch. Runs without a train log (e.g. eval-only) are skipped.
    """
    runs_root = Path(runs_root)
    rows: list[dict[str, object]] = []
    for run_path in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        recs = _read_train_log(run_path / TRAIN_LOG_FILENAME)
        if not recs:
            continue
        run = RunDir(run_id=run_path.name, path=run_path)
        best = min(recs, key=lambda r: _as_float(r.get("val_loss")))
        model_type = _model_type_from_snapshot(run.config_path)
        tag = _tag_from_run_id(run.run_id, model_type)
        kind = "schedule" if any(run_path.glob("phase_*")) else "train"
        rows.append(
            build_index_row(
                run,
                model_type=model_type,
                tag=tag,
                kind=kind,
                epochs_run=len(recs),
                best_epoch=int(_as_float(best.get("epoch"))),
                best_val_loss=_as_float(best.get("val_loss")),
                headline=best,
                best_ckpt=run.best_ckpt_path if run.best_ckpt_path.exists() else None,
            )
        )
    path = runs_root / INDEX_FILENAME
    path.unlink(missing_ok=True)
    with CsvLogger(path, INDEX_COLUMNS) as logger:
        for row in rows:
            logger.log(row)
    return path


def _read_train_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _model_type_from_snapshot(config_path: Path) -> str:
    """Read ``eval.model_type`` from a ``resolved_config.yaml`` snapshot (``"unknown"`` if absent)."""
    if not config_path.exists():
        return "unknown"
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        snapshot = yaml.safe_load(handle) or {}
    return str(snapshot.get("eval", {}).get("model_type", "unknown"))


def _tag_from_run_id(run_id: str, model_type: str) -> str:
    """Recover the ``tag`` suffix from a run id given its (underscore-containing) model type."""
    prefix = f"{_run_timestamp(run_id)}_{model_type}"
    if run_id.startswith(prefix) and len(run_id) > len(prefix) + 1:
        return run_id[len(prefix) + 1 :]
    return ""

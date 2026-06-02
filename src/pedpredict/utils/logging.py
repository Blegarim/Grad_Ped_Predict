"""Run-directory + CSV logging primitives (Prompt 0.3).

This module owns only the *primitives*: a filesystem-safe run id, the per-run directory
scaffold (``outputs/runs/{run_id}/{checkpoints,plots}``), and a generic append-only
:class:`CsvLogger`. The concrete ``train_log.csv`` / ``eval_log.csv`` / ``index.csv`` column
schemas are owned by ``training/metrics.py`` (Prompt 3.2) and the logging conventions in
Prompt 4.5 — they pass their column lists into :class:`CsvLogger`; nothing here hardcodes them.

Replaces the OLD ad-hoc ``datetime_str = '%m%d_%H%M'`` + ``model_suffix`` naming and the inline
``csv.writer`` blocks in ``train.py`` (B1/B11).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

__all__ = ["make_run_id", "create_run_dir", "CsvLogger", "get_csv_logger"]

_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")


def _sanitize(token: str) -> str:
    """Collapse filesystem-unsafe runs (spaces, slashes, ...) to single underscores."""
    return _UNSAFE.sub("_", token).strip("_")


def make_run_id(model_type: str, tag: str = "", *, now: datetime | None = None) -> str:
    """Build ``{YYYYMMDD_HHMMSS}_{model_type}[_{tag}]`` — filesystem-safe, no spaces/slashes."""
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

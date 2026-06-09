"""Path resolution.

Single home for turning the *relative* strings in :class:`~pedpredict.config.schema.PathsCfg`
into absolute :class:`pathlib.Path` objects rooted at the project root. No module elsewhere
should call ``os.path.join`` on a config path literal — they go through :func:`resolve_paths`.

Pure: ``resolve_paths`` performs no I/O (no ``mkdir``); directory creation for a run lives in
``utils.logging.create_run_dir``. Absolute config entries pass through unchanged so an operator
can point ``lmdb_*`` at an out-of-tree dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config.schema import PathsCfg

__all__ = ["find_project_root", "ResolvedPaths", "resolve_paths"]

_ROOT_MARKER = "pyproject.toml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this file) to the directory holding ``pyproject.toml``.

    Falls back to ``start`` resolved if no marker is found (e.g. an installed wheel with no
    source tree), so callers always get an absolute, existing directory.
    """
    base = (start or Path(__file__)).resolve()
    for candidate in (base, *base.parents):
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    return base if base.is_dir() else base.parent


def _resolve_one(root: Path, rel: str) -> Path:
    """Join ``rel`` under ``root`` unless it is already absolute."""
    path = Path(rel)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True, slots=True)
class ResolvedPaths:
    """Absolute counterparts of every :class:`PathsCfg` field, rooted at ``root``."""

    root: Path
    pie_root: Path
    sequences_dir: Path
    lmdb_train: tuple[Path, ...]
    lmdb_val: Path
    lmdb_test: Path
    log_dir: Path
    ckpt_dir: Path
    run_ckpt_dir: Path
    runs_dir: Path


def resolve_paths(cfg: PathsCfg, root: Path | None = None) -> ResolvedPaths:
    """Resolve a :class:`PathsCfg` against ``root`` (default :func:`find_project_root`)."""
    base = (root or find_project_root()).resolve()
    return ResolvedPaths(
        root=base,
        pie_root=_resolve_one(base, cfg.pie_root),
        sequences_dir=_resolve_one(base, cfg.sequences_dir),
        lmdb_train=tuple(_resolve_one(base, p) for p in cfg.lmdb_train),
        lmdb_val=_resolve_one(base, cfg.lmdb_val),
        lmdb_test=_resolve_one(base, cfg.lmdb_test),
        log_dir=_resolve_one(base, cfg.log_dir),
        ckpt_dir=_resolve_one(base, cfg.ckpt_dir),
        run_ckpt_dir=_resolve_one(base, cfg.run_ckpt_dir),
        runs_dir=_resolve_one(base, cfg.runs_dir),
    )

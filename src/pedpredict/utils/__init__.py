"""Cross-cutting helpers: seed, device, amp, memory, logging.

Re-exports the flat helper API so callers write ``from pedpredict.utils import set_seed``
rather than reaching into submodules. Path resolution lives one level up in
``pedpredict.paths`` (it depends on the config schema, not on these runtime helpers).
"""

from __future__ import annotations

from .amp import autocast_ctx, make_grad_scaler, resolve_amp, to_float_logits
from .device import enable_perf_flags, get_device
from .logging import (
    INDEX_COLUMNS,
    CsvLogger,
    RunDir,
    append_index_row,
    build_index_row,
    create_run_dir,
    get_csv_logger,
    git_sha,
    init_run,
    make_run_id,
    read_index,
    rebuild_index,
    round_row,
    snapshot_config,
)
from .memory import free_cuda, wait_for_memory
from .seed import set_seed

__all__ = [
    "set_seed",
    "get_device",
    "enable_perf_flags",
    "resolve_amp",
    "autocast_ctx",
    "make_grad_scaler",
    "to_float_logits",
    "wait_for_memory",
    "free_cuda",
    "make_run_id",
    "create_run_dir",
    "CsvLogger",
    "get_csv_logger",
    "RunDir",
    "init_run",
    "snapshot_config",
    "round_row",
    "git_sha",
    "INDEX_COLUMNS",
    "build_index_row",
    "append_index_row",
    "read_index",
    "rebuild_index",
]

"""Cross-cutting helpers: seed, device, amp, memory, logging (Prompt 0.3).

Re-exports the flat helper API so callers write ``from pedpredict.utils import set_seed``
rather than reaching into submodules. Path resolution lives one level up in
``pedpredict.paths`` (it depends on the config schema, not on these runtime helpers).
"""

from __future__ import annotations

from .amp import autocast_ctx, make_grad_scaler, resolve_amp, to_float_logits
from .device import enable_perf_flags, get_device
from .logging import CsvLogger, create_run_dir, get_csv_logger, make_run_id
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
]

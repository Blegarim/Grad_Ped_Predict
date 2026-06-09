"""Memory-pressure helpers.

Extracts two idioms repeated through the OLD god-script:
  - RAM-pressure polling before spawning a prefetch process (``scripts/train_utils.py:74-77``)
  - the ``gc.collect(); torch.cuda.empty_cache()`` cleanup pair
    (train.py:174,475-476,503-504,568-570 and elsewhere)

Only the memory primitive lands here; the crash-safe ``ChunkPrefetcher`` that consumes it is
deferred to Prompt 4.2 (the rest of B9).
"""

from __future__ import annotations

import gc
import time

import psutil
import torch

__all__ = ["wait_for_memory", "free_cuda"]


def wait_for_memory(
    threshold: float = 96.0,
    interval: float = 1.0,
    *,
    timeout: float | None = None,
) -> None:
    """Block while system RAM usage exceeds ``threshold`` percent (OLD train_utils.py:74-77).

    ``timeout=None`` preserves the legacy infinite wait. A finite ``timeout`` (seconds) is an
    opt-in safety net: it returns once the deadline passes even if memory is still high.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while psutil.virtual_memory().percent > threshold:
        if deadline is not None and time.monotonic() >= deadline:
            return
        time.sleep(interval)


def free_cuda(device: torch.device | None = None) -> None:
    """Run ``gc.collect()`` then empty the CUDA cache when CUDA is active/available.

    ``device=None`` empties the cache whenever CUDA is available (matches the OLD
    unconditional cleanup sites); pass a CPU device to skip the CUDA call.
    """
    gc.collect()
    use_cuda = torch.cuda.is_available() if device is None else device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()

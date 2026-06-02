"""Device selection and CUDA performance flags (Prompt 0.3).

Port of the device/perf block from OLD ``train.py:244-255``. Behavior-preserving: the exact
same four backend toggles, only relocated out of the god-script (part of B1). No-op on CPU.
"""

from __future__ import annotations

import torch

__all__ = ["get_device", "enable_perf_flags"]


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return ``cuda`` if available and requested, else ``cpu`` (OLD ``train.py:244``)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enable_perf_flags(device: torch.device) -> None:
    """Enable the CUDA perf flags from OLD ``train.py:250-255``. No-op on CPU.

    Sets cudnn benchmark autotuning + TF32 matmul/cudnn paths + high float32 matmul
    precision. ``cudnn.benchmark`` is skipped when ``cudnn.deterministic`` is set (by
    ``seed.set_seed(deterministic=True)``) — the autotuner is incompatible with determinism.
    """
    if device.type != "cuda":
        return
    if not torch.backends.cudnn.deterministic:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

"""Automatic mixed precision: one context + one dtype helper.

Consolidates the AMP machinery scattered through the OLD repo:
  - ``torch.amp.autocast('cuda', enabled=use_amp)``  (train.py:141,204; test.py:45)
  - ``torch.amp.GradScaler('cuda', enabled=use_amp)`` (train.py:347)
  - per-call ``logits.float()`` upcasts                (train.py:152,215; test.py:49,60)

``use_amp = device.type == 'cuda'`` (OLD ``train.py:247``) becomes :func:`resolve_amp`, ANDed
with the config *request* (``TrainCfg.use_amp``; Config decision Q2 — schema stores intent,
runtime resolves it). :func:`to_float_logits` replaces the scattered casts with one dtype-safe
pass over the output dict, so consumers never sprinkle ``.float()`` again.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager

import torch

__all__ = ["resolve_amp", "autocast_ctx", "make_grad_scaler", "to_float_logits"]


def resolve_amp(requested: bool, device: torch.device) -> bool:
    """Resolve the AMP *request* against runtime reality: AMP only on CUDA (OLD train.py:247)."""
    return requested and device.type == "cuda"


def autocast_ctx(enabled: bool, device_type: str = "cuda") -> AbstractContextManager:
    """Wrapper over ``torch.amp.autocast`` — single import site for the AMP context."""
    return torch.amp.autocast(device_type, enabled=enabled)


def make_grad_scaler(enabled: bool, device_type: str = "cuda") -> torch.amp.GradScaler:
    """Construct a ``GradScaler`` (a disabled scaler is a transparent pass-through)."""
    return torch.amp.GradScaler(device_type, enabled=enabled)


def to_float_logits(outputs: Mapping[str, object]) -> dict[str, object]:
    """Return a new dict with every *floating* tensor upcast to float32 (B8).

    Non-tensor values and integer/bool tensors pass through untouched. A no-op outside
    autocast (values are already fp32), so it is behavior-neutral vs the OLD per-key casts.
    Does not mutate ``outputs``.
    """
    result: dict[str, object] = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.is_floating_point():
            result[key] = value.float()
        else:
            result[key] = value
    return result

"""Deterministic seeding (Prompt 0.3).

Net-new: the OLD repo had no global seed (only a local ``torch.manual_seed`` in
``scripts/augment_sequences.py:80``). This adds one entry point that seeds ``random``,
``numpy``, and ``torch`` (CPU + all CUDA devices).

``deterministic=True`` is mutually exclusive with ``device.enable_perf_flags`` — it sets
``cudnn.deterministic=True`` / ``cudnn.benchmark=False`` and asks torch to use deterministic
algorithms. Call order is **seed first, then perf flags** so the latter sees ``deterministic``.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int = 0, deterministic: bool = False) -> None:
    """Seed all RNGs. If ``deterministic``, also force deterministic CUDA kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops (e.g. some pooling) lack deterministic kernels; warn, don't crash.
        torch.use_deterministic_algorithms(True, warn_only=True)

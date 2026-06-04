"""Pure spatial-geometry helpers for the hierarchical ViT (no torch — importable by the config layer).

Single source of truth for the stem/downsample schedule used both to BUILD the ViT (``models/vit.py``)
and to VALIDATE that every stage window tiles the feature map (``config/loader.py``). Kept torch-free so
config loading never pulls in torch.
"""

from __future__ import annotations

# Stem + per-stage downsample geometry (kernel, stride, padding).
STEM = (7, 4, 3)         # conv7-s4 stem
DOWNSAMPLE = (3, 2, 1)   # each inter-stage conv3-s2 (stage 0 uses Identity)


def conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    """Spatial output size of a conv (PyTorch floor formula, dilation=1)."""
    return (size + 2 * padding - kernel) // stride + 1


def feature_map_size(img_size: int, stage_idx: int) -> int:
    """Side length of the feature map entering ``stage_idx``'s blocks for a square ``img_size`` input.

    Applies the stem, then ``stage_idx`` inter-stage downsamples (stage 0's downsample is Identity).
    """
    size = conv_out(img_size, *STEM)
    for _ in range(stage_idx):
        size = conv_out(size, *DOWNSAMPLE)
    return size


def is_global(window: int | str | None) -> bool:
    """A window spec is 'global' when it is ``None`` or the string ``"global"``."""
    return window is None or window == "global"

"""Per-channel statistics of the pose arm's read-path input — the numbers ``pose.input_stats`` points at.

The pose read path (:class:`~pedpredict.data.pose.PoseMotionTransform`) emits a ``[T, 9 + feature_dim]``
vector that otherwise reaches the network unscaled: measured 2026-09-25 on the streaming train split,
per-channel std spans ~770x, with the four velocity channels (dx, dy, dw, dh) smallest. This module
computes the mean/std that ``pose.input_mean/input_std`` standardize with.

Metadata-only: reads the ``_meta`` keys (pose + motions) and never touches an image blob. Statistics are
taken over every frame of every window in the given dirs, from the RAW transform — any standardization
already configured on ``root`` is stripped first, so a stats file can never be computed on its own output.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch

from pedpredict.config.schema import RootCfg
from pedpredict.data.pose import PoseMotionTransform

__all__ = ["STD_FLOOR", "compute_input_stats", "raw_transform"]

#: Lower bound on a channel's std, so a (near-)constant channel cannot turn into a division by ~0.
STD_FLOOR = 1e-6


def raw_transform(root: RootCfg) -> PoseMotionTransform:
    """The read-path transform with any configured standardization removed."""
    raw_pose = dataclasses.replace(root.pose, input_stats="", input_mean=(), input_std=())
    return PoseMotionTransform(dataclasses.replace(root, pose=raw_pose))


def _chunk_frames(chunk: Path, transform: PoseMotionTransform):
    """Yield the ``[T, C]`` read-path vector of every window in one chunk (metas only)."""
    env = lmdb.open(str(chunk), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            keys = [k for k in txn.cursor().iternext(keys=True, values=False) if k.endswith(b"_meta")]
            for key in keys:
                meta = pickle.loads(txn.get(key))
                if meta.get("pose") is None:
                    raise ValueError(f"{chunk}: window {key.decode()!r} has no pose meta — not a pose build")
                yield transform(
                    torch.as_tensor(np.asarray(meta["pose"]), dtype=torch.float32),
                    torch.as_tensor(np.asarray(meta["motions"]), dtype=torch.float32),
                )
    finally:
        env.close()


def compute_input_stats(dirs: list[Path], root: RootCfg) -> dict[str, object]:
    """Mean/std per read-path channel over every frame in ``dirs`` (float64 accumulation).

    Non-finite entries are counted and excluded channel-wise rather than silently poisoning the mean.
    """
    transform = raw_transform(root)
    total = sq = count = None
    windows = nonfinite = 0
    for directory in dirs:
        chunks = sorted(p for p in Path(directory).iterdir() if p.name.startswith("chunk_"))
        if not chunks:
            raise FileNotFoundError(f"no chunk_*.lmdb in {directory}")
        for chunk in chunks:
            for frames in _chunk_frames(chunk, transform):
                x = frames.double()
                ok = torch.isfinite(x)
                nonfinite += int((~ok).sum())
                x = torch.where(ok, x, torch.zeros_like(x))
                s, q, c = x.sum(0), (x * x).sum(0), ok.sum(0).double()
                total, sq, count = (s, q, c) if total is None else (total + s, sq + q, count + c)
                windows += 1
    if total is None:
        raise ValueError(f"no windows found in {dirs}")
    n = count.clamp_min(1.0)
    mean = total / n
    std = ((sq / n) - mean * mean).clamp_min(0.0).sqrt().clamp_min(STD_FLOOR)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "width": int(mean.numel()),
        "n_windows": windows,
        "n_frames": int(count.max().item()),
        "nonfinite": nonfinite,
        "sources": [Path(d).name for d in dirs],
    }

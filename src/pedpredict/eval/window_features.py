"""Export the pixel-free read path's per-window inputs to plain arrays, so models can train without LMDBs.

The tree-model study runs on a machine that has no LMDBs. :func:`export_windows` turns one or more LMDB
dirs into arrays holding, per window and in the stable order every eval dump uses (chunks sorted per dir,
records in index order — the order :func:`pedpredict.eval.onset_timing.dump_loaders` reads):

* ``X [N, T, C]`` float32 — the ``pose_kinematics`` input exactly as the read path builds it
  (:class:`~pedpredict.data.pose.PoseMotionTransform`: the image-normalized 9-dim motion block, then the
  pose features), **before** the optional ``pose.input_stats`` standardization. Trees are invariant to it;
  a model that needs it re-applies the mean/std its run recorded in ``resolved_config.yaml``.
* ``actions`` / ``looks`` int64 as stored, ``crosses`` clipped to ``{0, 1}`` exactly as the dumps do,
  ``track_id`` str, ``chunk`` (index into the meta's chunk list),
* ``tte`` on anchored (benchmark) sets and the three S1 onset fields on streaming sets — each only when
  every window carries it (anchored windows carry no onset fields by construction).

:func:`check_against_dump` is the alignment gate: an export of a split must reproduce the label arrays and
``track_id`` of an existing prediction dump row for row, or the two cannot be compared window by window.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from pedpredict.config.schema import MOTION_STORE_DIM, PoseCfg, RootCfg
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.pie_sequences import ONSET_FIELDS
from pedpredict.data.pose import kept_joints, pose_motion_transform

__all__ = ["MOTION_CHANNELS", "channel_names", "collate_export", "export_windows", "check_against_dump"]

Arrays = dict[str, np.ndarray]

#: The 9 stored motion channels (data/transforms.py), image-normalized by the read path.
MOTION_CHANNELS: tuple[str, ...] = ("cx", "cy", "dx", "dy", "w", "h", "dw", "dh", "ego")
_JOINT_NAMES: dict[int, str] = {
    0: "nose", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow", 8: "r_elbow", 9: "l_wrist", 10: "r_wrist",
    11: "l_hip", 12: "r_hip", 13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle", 17: "l_bigtoe",
    18: "l_smalltoe", 19: "l_heel", 20: "r_bigtoe", 21: "r_smalltoe", 22: "r_heel",
}
_OPTIONAL_KEYS: tuple[str, ...] = ("tte", *ONSET_FIELDS)


def channel_names(pose: PoseCfg) -> tuple[str, ...]:
    """Name of every channel of the read-path vector, in order (``build_pose_features``' block order)."""
    joints = [_JOINT_NAMES[j] for j in kept_joints(pose.include_arms)]
    names = list(MOTION_CHANNELS)
    names += [f"{j}_{axis}" for j in joints for axis in ("x", "y")]
    if pose.conf_channel:
        names += [f"{j}_conf" for j in joints]
    names += ["head_sin", "head_cos", "body_sin", "body_cos"]
    assert len(names) == MOTION_STORE_DIM + pose.feature_dim()
    return tuple(names)


def collate_export(batch: list[dict]) -> Arrays:
    """Stack one batch of dataset samples into export arrays (no images; optional keys all-or-nothing)."""
    out: Arrays = {
        "X": torch.stack([b["motions"] for b in batch]).to(torch.float32).numpy(),
        "track_id": np.asarray([b["track_id"] for b in batch], dtype=str),
        "actions": np.asarray([int(b["actions"]) for b in batch], dtype=np.int64),
        "looks": np.asarray([int(b["looks"]) for b in batch], dtype=np.int64),
        "crosses": np.clip(np.asarray([int(b["crosses"]) for b in batch], dtype=np.int64), 0, 1),
    }
    for key in _OPTIONAL_KEYS:
        present = sum(key in b for b in batch)
        if present == len(batch):
            out[key] = np.asarray([int(b[key]) for b in batch], dtype=np.int64)
        elif present:
            raise ValueError(f"collate_export: '{key}' on {present}/{len(batch)} samples — mixed-vintage chunk")
    return out


def export_windows(cfg: RootCfg, chunk_paths: Sequence[str], *, batch_size: int = 512,
                   num_workers: int = 0) -> Arrays:
    """Read every window of ``chunk_paths`` through the pixel-free pose read path into arrays."""
    if cfg.pose.input_mean or cfg.pose.input_std:
        raise ValueError("export_windows: the export is RAW by contract — leave pose.input_stats unset")
    if cfg.data.visual_input != "none":
        raise ValueError(f"export_windows: needs data.visual_input=none, got {cfg.data.visual_input!r}")
    transform = pose_motion_transform(cfg)
    if transform is None:
        raise ValueError("export_windows: needs pose.enabled=true (the pose_kinematics read path)")
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, path in enumerate(chunk_paths):
        dataset = LMDBChunkDataset.from_config(path, cfg.data, pose_transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_export)
        for batch in loader:
            for key, value in batch.items():
                parts[key].append(value)
            parts["chunk"].append(np.full(len(batch["track_id"]), i, dtype=np.int32))
        dataset.close()
    n_batches = len(parts["X"])
    ragged = sorted(k for k, v in parts.items() if len(v) != n_batches)
    if ragged:
        raise ValueError(f"export_windows: {ragged} present in some chunks but not others — mixed-vintage dirs")
    return {key: np.concatenate(chunks) for key, chunks in parts.items()}


def check_against_dump(export: Arrays, dump: Arrays) -> list[str]:
    """Row-for-row label + ``track_id`` agreement with a prediction dump; returns the mismatches."""
    problems = []
    if len(export["crosses"]) != len(dump["crosses"]):
        return [f"length: export {len(export['crosses'])} vs dump {len(dump['crosses'])}"]
    for key in ("crosses", "track_id", *ONSET_FIELDS):
        if key not in dump:
            continue
        if key not in export:
            problems.append(f"{key}: in the dump but not the export")
            continue
        bad = int((np.asarray(export[key]) != np.asarray(dump[key])).sum())
        if bad:
            problems.append(f"{key}: {bad} rows differ")
    return problems

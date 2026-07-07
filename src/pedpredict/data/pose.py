"""Pose-keypoint feature math (docs/POSE_ENCODER.md) — single source of truth, like
``transforms.compute_motion`` for the motion contract.

Storage contract: the extraction cache and the LMDB meta hold **raw** keypoints ``[T, 23, 3]``
(x, y, confidence — absolute frame px), the first 23 COCO-WholeBody joints (body-17 + feet-6).
Features are built at read time by :func:`build_pose_features` so the normalization/angle math stays
iterable without re-running extraction:

======  ====  =========================================================================
block   dims  definition
======  ====  =========================================================================
coords  2n    kept-joint (x, y), bbox-normalized: ``(xy - (cx, cy)) / h`` from the
              stored motion channels (idx 0, 1, 5) — position/scale live ONLY in the
              motion vector
conf    n     raw per-joint confidence ``[0, 1]`` (net gates; never pre-zeroed)
head    2     head-facing (sin, cos): ear-midpoint -> nose, scaled by the product of
              the contributing joints' confidences (a dead angle reads ~(0, 0))
body    2     body-facing (sin, cos): perpendicular of the summed shoulder + hip
              R-L vectors, confidence-scaled the same way
======  ====  =========================================================================

``n`` = 15 core kept joints (nose, shoulders, hips, knees, ankles, feet) or 19 with arms
(``pose.include_arms``); eyes/ears are stored but collapse into the head angle only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from pedpredict.config.schema import MOTION_STORE_DIM, RootCfg
from pedpredict.paths import find_project_root

__all__ = [
    "POSE_STORE_JOINTS",
    "CORE_JOINTS",
    "ARM_JOINTS",
    "kept_joints",
    "build_pose_features",
    "smooth_pose",
    "flip_pose",
    "PoseMotionTransform",
    "pose_motion_transform",
    "PoseCache",
]

#: Stored joint layout = COCO-WholeBody indices 0..22 (body-17 + feet-6).
POSE_STORE_JOINTS: int = 23
_NOSE, _L_EAR, _R_EAR = 0, 3, 4
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 5, 6, 11, 12
#: Kept coordinate joints: nose, shoulders, hips, knees, ankles, feet ×6 (15 core).
CORE_JOINTS: tuple[int, ...] = (0, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
#: Optional arm joints (elbows, wrists) — first to ablate.
ARM_JOINTS: tuple[int, ...] = (7, 8, 9, 10)
#: Horizontal-flip left/right reindexing (eyes, ears, shoulders, elbows, wrists, hips, knees,
#: ankles, big toes, small toes, heels swap; nose fixed).
_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22),
)
def _flip_index() -> Tensor:
    index = list(range(POSE_STORE_JOINTS))
    for a, b in _FLIP_PAIRS:
        index[a], index[b] = b, a
    return torch.tensor(index)


_FLIP_INDEX: Tensor = _flip_index()

_EPS = 1e-6


def kept_joints(include_arms: bool) -> tuple[int, ...]:
    """Kept coordinate-joint indices into the 23-joint storage layout (15 core, 19 with arms)."""
    return CORE_JOINTS + ARM_JOINTS if include_arms else CORE_JOINTS


def _facing(vec: Tensor, conf: Tensor) -> Tensor:
    """``vec [T, 2]`` + contributing-joint confidences ``[T, k]`` -> confidence-scaled unit
    ``(sin, cos) [T, 2]``; zeros where the vector is degenerate."""
    norm = vec.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > _EPS, vec / norm.clamp_min(_EPS), torch.zeros_like(vec))
    gate = conf.prod(dim=-1, keepdim=True)
    return torch.stack([unit[:, 1], unit[:, 0]], dim=-1) * gate  # (sin, cos) = (vy, vx)/|v|


def build_pose_features(
    pose: Tensor, motions: Tensor, *, include_arms: bool = False, conf_channel: bool = True
) -> Tensor:
    """``pose [T, 23, 3]`` (absolute px) + ``motions [T, >=9]`` (raw px) -> features ``[T, dim]``.

    ``dim`` == ``PoseCfg.feature_dim()`` for matching flags. Translation/scale-invariant by
    construction: coords are centered on the bbox center and divided by bbox height, angles are
    normalized direction vectors.
    """
    kept = list(kept_joints(include_arms))
    center = motions[:, 0:2]                                  # (cx, cy)
    h = motions[:, 5].clamp_min(1.0)                          # bbox height
    xy_n = (pose[:, kept, :2] - center[:, None, :]) / h[:, None, None]

    ears = pose[:, [_L_EAR, _R_EAR], :2]
    head = _facing(pose[:, _NOSE, :2] - ears.mean(dim=1), pose[:, [_NOSE, _L_EAR, _R_EAR], 2])
    shoulder = pose[:, _R_SHOULDER, :2] - pose[:, _L_SHOULDER, :2]
    hip = pose[:, _R_HIP, :2] - pose[:, _L_HIP, :2]
    line = shoulder + hip
    body = _facing(
        torch.stack([line[:, 1], -line[:, 0]], dim=-1),       # perpendicular of the R-L line
        pose[:, [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP], 2],
    )

    blocks = [xy_n.flatten(1)]
    if conf_channel:
        blocks.append(pose[:, kept, 2])
    blocks += [head, body]
    return torch.cat(blocks, dim=-1)


def smooth_pose(pose: Tensor, *, window: int, min_conf: float) -> Tensor:
    """Interpolate low-confidence joint coordinates over time, then moving-average them.

    Frames where a joint's confidence is below ``min_conf`` get coordinates linearly interpolated
    from that joint's nearest confident frames (edges hold the nearest value); confidences are kept
    raw so the net can still gate. ``window`` > 1 then applies a centered, replicate-padded moving
    average to the coordinates only.
    """
    out = pose.clone()
    t = pose.shape[0]
    idx = np.arange(t, dtype=np.float64)
    for j in range(pose.shape[1]):
        valid = pose[:, j, 2] >= min_conf
        if valid.any() and not valid.all():
            valid_np = valid.numpy()
            for c in (0, 1):
                out[:, j, c] = torch.from_numpy(
                    np.interp(idx, idx[valid_np], pose[valid, j, c].numpy())
                ).to(pose.dtype)
    if window > 1 and t > 1:
        coords = out[..., :2].permute(1, 2, 0).reshape(-1, 1, t)   # [23*2, 1, T]
        coords = F.pad(coords, (window // 2, window - 1 - window // 2), mode="replicate")
        coords = F.avg_pool1d(coords, kernel_size=window, stride=1)
        out[..., :2] = coords.reshape(pose.shape[1], 2, t).permute(2, 0, 1)
    return out


def flip_pose(pose: Tensor, source_width: float) -> Tensor:
    """Horizontal flip: reflect x about the frame width and swap left/right joints."""
    out = pose.clone()
    out[..., 0] = source_width - out[..., 0]
    return out[:, _FLIP_INDEX]


class PoseMotionTransform:
    """Picklable read-time builder: ``(pose [T, 23, 3], motions [T, 9]) -> motions [T, 9 + dim]``.

    Image-normalizes the stored motion block with the same fixed per-channel scale
    ``KinematicsEncoder(motion_norm="image")`` applies in-forward, then concats the pose features —
    which is why pose models run with ``motion_norm="none"``.
    """

    def __init__(self, root: RootCfg) -> None:
        w, h = float(root.data.source_width), float(root.data.source_height)
        self.scale = torch.tensor(
            [w, h, w, h, w, h, w, h, float(root.model.ego_speed_scale)]
        ).view(1, MOTION_STORE_DIM)
        self.include_arms = root.pose.include_arms
        self.conf_channel = root.pose.conf_channel

    def __call__(self, pose: Tensor, motions: Tensor) -> Tensor:
        feats = build_pose_features(
            pose, motions, include_arms=self.include_arms, conf_channel=self.conf_channel
        )
        return torch.cat([motions / self.scale, feats], dim=-1)


def pose_motion_transform(root: RootCfg) -> PoseMotionTransform | None:
    """The read-path gate: a :class:`PoseMotionTransform` when ``pose.enabled``, else ``None``."""
    return PoseMotionTransform(root) if root.pose.enabled else None


class PoseCache:
    """Read side of the extraction cache: ``{cache_dir}/{set}/{video}.npz`` with keys
    ``f"{frame}_{pid}" -> [23, 3]`` float32 (written by ``scripts/extract_pose.py``).

    Per-video files are materialized into plain dicts on first access (small: keypoints only) and
    dropped from the pickle state, so instances survive DataLoader ``spawn`` workers.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._videos: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    @classmethod
    def from_config(cls, root: RootCfg) -> PoseCache | None:
        """A cache rooted at ``pose.cache_dir`` (project-root-relative) when enabled, else ``None``."""
        if not root.pose.enabled:
            return None
        path = Path(root.pose.cache_dir)
        return cls(path if path.is_absolute() else find_project_root() / path)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_videos"] = {}
        return state

    def _video(self, set_id: str, video_id: str) -> dict[str, np.ndarray]:
        key = (set_id, video_id)
        if key not in self._videos:
            path = self.cache_dir / set_id / f"{video_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"[PoseCache] missing {path} — run scripts/extract_pose.py")
            with np.load(path) as npz:
                self._videos[key] = dict(npz)
        return self._videos[key]

    def sequence(self, images: list[str], track_id: str) -> Tensor:
        """Assemble ``[T, 23, 3]`` for one window from its frame paths + pedestrian id."""
        frames = []
        for img_path in images:
            p = Path(img_path)
            set_id, video_id, frame = p.parts[-3], p.parts[-2], int(p.stem)
            video = self._video(set_id, video_id)
            kpts = video.get(f"{frame}_{track_id}")
            if kpts is None:
                raise KeyError(
                    f"[PoseCache] no keypoints for frame {frame} pid {track_id!r} in "
                    f"{set_id}/{video_id} — stale cache, re-run scripts/extract_pose.py"
                )
            frames.append(torch.from_numpy(kpts.astype(np.float32)))
        return torch.stack(frames)

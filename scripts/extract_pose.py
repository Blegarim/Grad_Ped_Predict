"""Extract 23-joint whole-body pose for every (frame, pedestrian) a sequence pickle touches.

Runs the bbox-conditioned extractor (PIE GT boxes as the person prior, full frame — never the stored
crop) once per unique frame, smooths/interpolates per track (``pose.smooth_window`` /
``pose.min_conf``), and writes ``{pose.cache_dir}/{set}/{video}.npz`` keyed ``f"{frame}_{pid}"`` ->
``[23, 3]`` float32 (x, y, conf, absolute px). Dedupe by (video, frame, pid) means overlapping
windows never re-extract; existing video files are merge-updated, so splits can be run separately.
Decoupled from windowing — the LMDB build (``pose.enabled``) reads this cache afterwards.

The real extractor needs the PIE frames + ``pip install rtmlib onnxruntime-gpu`` (lab PC).
``--dry-run`` fabricates deterministic random keypoints inside each GT box so the whole pose
pipeline is exercisable end-to-end without frames or the extractor installed.

    python scripts/extract_pose.py --split train
    python scripts/extract_pose.py --split all --dry-run
    python scripts/extract_pose.py --split train_benchmark   # anchored windows share the same cache
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pedpredict.config import build_argparser, load_config
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.data.pose import POSE_STORE_JOINTS, smooth_pose
from pedpredict.paths import find_project_root, resolve_paths

_SPLITS = ("train", "val", "test")
_EXTRA_SPLITS = ("train_benchmark", "val_benchmark", "test_benchmark")

# (set, video) -> {f"{frame}_{pid}": (bbox [x1, y1, x2, y2], frame image path)}
_VideoTasks = dict[tuple[str, str], dict[str, tuple[list[float], str]]]


def collect_tasks(pkl_paths: list[Path]) -> _VideoTasks:
    """Dedupe every record frame into per-video (frame, pid) -> (bbox, path) lookup tasks."""
    tasks: _VideoTasks = defaultdict(dict)
    for pkl in pkl_paths:
        for rec in load_sequences(pkl):
            for img_path, bbox in zip(rec["images"], rec["bboxes"], strict=True):
                p = Path(img_path)
                key = f"{int(p.stem)}_{rec['track_id']}"
                tasks[(p.parts[-3], p.parts[-2])].setdefault(key, (list(bbox), str(img_path)))
    return tasks


def _fabricate(bbox: list[float], key: str) -> np.ndarray:
    """--dry-run keypoints: deterministic uniform points inside the GT box, conf in [0.5, 1]."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = bbox
    out = np.empty((POSE_STORE_JOINTS, 3), dtype=np.float32)
    out[:, 0] = rng.uniform(x1, x2, POSE_STORE_JOINTS)
    out[:, 1] = rng.uniform(y1, y2, POSE_STORE_JOINTS)
    out[:, 2] = rng.uniform(0.5, 1.0, POSE_STORE_JOINTS)
    return out


class DwposeExtractor:
    """rtmlib whole-body (DWPose/RTMW family) run bbox-conditioned; keeps the first 23 joints."""

    def __init__(self, device: str = "cuda") -> None:
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise SystemExit(
                "rtmlib is required for real extraction: pip install rtmlib onnxruntime-gpu "
                "(or use --dry-run)"
            ) from exc
        self._pose = Wholebody(to_openpose=False, backend="onnxruntime", device=device).pose_model

    def __call__(self, img_rgb: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        """``img_rgb [H, W, 3]`` + N xyxy boxes -> ``[N, 23, 3]`` (x, y, conf)."""
        keypoints, scores = self._pose(img_rgb, bboxes=boxes)
        kpts = np.asarray(keypoints)[:, :POSE_STORE_JOINTS]
        conf = np.asarray(scores)[:, :POSE_STORE_JOINTS, None]
        return np.concatenate([kpts, conf], axis=-1).astype(np.float32)


def extract_video(
    set_id: str,
    video_id: str,
    frames: dict[str, tuple[list[float], str]],
    *,
    extractor: DwposeExtractor | None,
) -> dict[str, np.ndarray]:
    """Run one video's unique frames through the extractor (or fabrication)."""
    if extractor is None:
        return {key: _fabricate(bbox, f"{set_id}/{video_id}/{key}") for key, (bbox, _) in frames.items()}
    from pedpredict.data.transforms import load_rgb  # PIL import stays off the --dry-run path

    by_frame: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for key, (bbox, img_path) in frames.items():
        by_frame[img_path].append((key, bbox))
    out: dict[str, np.ndarray] = {}
    for img_path, items in sorted(by_frame.items()):
        img = np.asarray(load_rgb(img_path))
        results = extractor(img, [bbox for _, bbox in items])
        for (key, _), kpts in zip(items, results, strict=True):
            out[key] = kpts
    return out


def smooth_tracks(raw: dict[str, np.ndarray], *, window: int, min_conf: float) -> dict[str, np.ndarray]:
    """Group keys by pid, smooth each track's time series in frame order (data/pose.smooth_pose)."""
    tracks: dict[str, list[int]] = defaultdict(list)
    for key in raw:
        frame, pid = key.split("_", 1)
        tracks[pid].append(int(frame))
    out: dict[str, np.ndarray] = {}
    for pid, frame_ids in tracks.items():
        frame_ids.sort()
        stacked = torch.from_numpy(np.stack([raw[f"{f}_{pid}"] for f in frame_ids]))
        smoothed = smooth_pose(stacked, window=window, min_conf=min_conf).numpy()
        for f, kpts in zip(frame_ids, smoothed, strict=True):
            out[f"{f}_{pid}"] = kpts
    return out


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", choices=(*_SPLITS, *_EXTRA_SPLITS, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Fabricate keypoints (no frames/extractor).")
    parser.add_argument("--device", default="cuda", help="onnxruntime device for the extractor.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides, validate=False)  # pose may be off yet
    if cfg.pose.extractor != "dwpose" and not args.dry_run:
        raise SystemExit(f"pose.extractor={cfg.pose.extractor!r} is not implemented (dwpose only)")
    paths = resolve_paths(cfg.paths)
    cache_root = Path(cfg.pose.cache_dir)
    cache_root = cache_root if cache_root.is_absolute() else find_project_root() / cache_root

    splits = _SPLITS if args.split == "all" else (args.split,)
    pkls = [paths.sequences_dir / f"sequences_{s}.pkl" for s in splits]
    missing = [p for p in pkls if not p.exists()]
    if missing:
        raise SystemExit(f"missing sequence pickle(s): {missing} — run make_sequences.py first")

    tasks = collect_tasks(pkls)
    extractor = None if args.dry_run else DwposeExtractor(args.device)
    total = 0
    for (set_id, video_id), frames in sorted(tasks.items()):
        raw = extract_video(set_id, video_id, frames, extractor=extractor)
        smoothed = smooth_tracks(raw, window=cfg.pose.smooth_window, min_conf=cfg.pose.min_conf)
        out_path = cache_root / set_id / f"{video_id}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():  # merge-update so separate split runs accumulate
            with np.load(out_path) as existing:
                smoothed = dict(existing) | smoothed
        np.savez(out_path, **smoothed)
        total += len(raw)
        print(f"[{set_id}/{video_id}] {len(raw)} (frame, ped) entries -> {out_path}")
    print(f"done: {total} entries across {len(tasks)} video(s) in {cache_root}")


if __name__ == "__main__":
    main()

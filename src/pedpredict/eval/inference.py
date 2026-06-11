"""Video / frame-sequence inference — ports OLD ``main.py``.

Reuses the Phase-1 preprocessing math (``crop_tight`` / ``crop_context`` / ``compute_motion`` /
``build_read_transforms``, 1.2) instead of re-running ``PIESequenceDataset`` on in-memory crops
(B5/B7); the typed registry ``build_model`` / ``forward_model`` (2.4, B10) + ``load_eval_weights``
(5.1, B2) for build/load/forward. Detection/tracking is YOLO via the optional ``[infer]`` extra
(lazy import). Five separable stages: detect -> assemble -> preprocess -> forward -> overlay.

Behavior changes vs OLD (no E2E golden oracle — non-deterministic YOLO + a trained ckpt + a video;
parity is proven at the reused-math seams instead, see ``tests/test_inference.py`` and MIGRATION 5.3):

* **``context_scale`` 2.0 -> 3.0** — OLD ``main.py`` built the dataset with ``context_scale=2.0`` while
  the model trained on 3.0 crops; the rebuild uses ``cfg.data.context_scale`` (matches training).
* **BGR -> RGB at frame read** — OLD fed cv2 BGR crops to an RGB-trained model; ``FrameSource`` converts
  to RGB once, and detection re-derives BGR only for YOLO.
* **context crop from the full frame** — OLD's in-memory dataset cropped "context" from the ped crop (a
  degenerate crop-of-a-crop); the rebuild crops both tight + context from the full frame at detect time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from pedpredict.config.schema import DataCfg, InferenceCfg, RootCfg
from pedpredict.data.transforms import build_read_transforms, compute_motion, crop_context, crop_tight
from pedpredict.eval.evaluate import load_eval_weights
from pedpredict.models.registry import build_model, forward_model
from pedpredict.utils.amp import autocast_ctx, resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device

__all__ = [
    "Detection", "Track", "TrackWindow", "FramePrediction", "InferenceResult",
    "FrameSource", "VideoFrameSource", "DirFrameSource", "open_frame_source",
    "crop_from_frame", "detect_tracks", "assemble_windows",
    "preprocess_window", "predict_windows", "aggregate_by_frame",
    "render_overlays", "run_video_inference",
]

_BoxInt = tuple[int, int, int, int]
_DEFAULT_FPS = 30.0                       # DirFrameSource fps fallback (InferenceCfg.default_fps drives it)
_IMG_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

# Overlay colors (BGR, drawn on a BGR frame — parity with OLD main.py 235-316).
_LABEL_COLORS = {"action": (0, 255, 255), "look": (255, 0, 255), "cross": (255, 255, 0)}
_TEXT_COLOR = (0, 0, 0)
_BBOX_COLOR = (0, 255, 0)
_CHIP_OFF = (50, 50, 50)
# Inline label text (drops the PIE toolkit runtime dep — B11).
_ACTION_TEXT = {0: "standing", 1: "walking"}
_LOOK_TEXT = {0: "not-look", 1: "look"}
_CROSS_TEXT = {0: "not-cross", 1: "cross"}


# ---- data types (typed replacements for OLD's dict-of-dicts) --------------------------------
@dataclass(slots=True, frozen=True, eq=False)
class Detection:
    """One tracked detection in one frame. ``eq=False`` because ``tight``/``context`` are ndarrays."""

    frame_idx: int
    bbox: _BoxInt                        # (x1, y1, x2, y2), int, clamped to frame bounds
    tight: np.ndarray                    # RGB HxWx3 tight crop from the FULL frame (crop_tight)
    context: np.ndarray                  # RGB HxWx3 context crop from the FULL frame (crop_context)


Track = list[Detection]                  # one tracked pedestrian, frame-ordered


@dataclass(slots=True, frozen=True)
class TrackWindow:
    """A fixed-length (``seq_len``) slice of one track — the unit of a single model forward."""

    track_id: int
    detections: tuple[Detection, ...]

    @property
    def frame_idxs(self) -> tuple[int, ...]:
        return tuple(d.frame_idx for d in self.detections)

    @property
    def bboxes(self) -> tuple[_BoxInt, ...]:
        return tuple(d.bbox for d in self.detections)


@dataclass(slots=True, frozen=True)
class FramePrediction:
    """One pedestrian's per-frame prediction (argmax {0,1}); ``crosses`` from ``crosses_frame`` (B4)."""

    track_id: int
    bbox: _BoxInt
    actions: int
    looks: int
    crosses: int


@dataclass(frozen=True)
class InferenceResult:
    """Result of one video pass."""

    frame_results: dict[int, list[FramePrediction]]
    n_frames: int
    n_windows: int
    out_video: Path | None


# ---- stage 0: frame source (video file OR dir of frames; folds extract_frames.py) -----------
class FrameSource(Sequence[np.ndarray]):
    """Abstract RGB-frame provider — ``__len__`` / ``__getitem__`` / ``__iter__`` + fps/size metadata."""

    fps: float
    width: int
    height: int

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]


class VideoFrameSource(FrameSource):
    """``cv2.VideoCapture``-backed source; yields RGB frames (cv2 reads BGR -> converted once)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or _DEFAULT_FPS
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> np.ndarray:
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"Frame {index} unreadable in {self.path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(self.path)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


class DirFrameSource(FrameSource):
    """Sorted directory of frame images (``qualitative_visualize/``); yields RGB frames, fixed fps."""

    def __init__(self, path: str | Path, fps: float = _DEFAULT_FPS) -> None:
        root = Path(path)
        files: list[Path] = []
        for pat in _IMG_GLOBS:
            files.extend(root.glob(pat))
        self.files = sorted(files)
        if not self.files:
            raise ValueError(f"No frame images ({', '.join(_IMG_GLOBS)}) found in {root}")
        self.fps = fps
        first = self[0]
        self.height, self.width = first.shape[0], first.shape[1]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> np.ndarray:
        frame = cv2.imread(str(self.files[index]), cv2.IMREAD_COLOR)
        if frame is None:
            raise IndexError(f"Unreadable frame image: {self.files[index]}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def open_frame_source(path: str | Path, *, fps: float | None = None) -> FrameSource:
    """Dispatch on ``path``: a directory -> :class:`DirFrameSource`, a file -> :class:`VideoFrameSource`."""
    p = Path(path)
    if p.is_dir():
        return DirFrameSource(p, fps=fps if fps is not None else _DEFAULT_FPS)
    return VideoFrameSource(p)


# ---- preprocessing helper (full-frame -> tight + context RGB crops) -------------------------
def crop_from_frame(frame_rgb: np.ndarray, bbox: _BoxInt, context_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Crop tight + context RGB arrays from a full RGB frame via the canonical 1.2 geometry.

    Both crops come from the FULL frame (not a crop-of-a-crop) using ``crop_tight`` / ``crop_context``,
    so the in-memory inference path is pixel-identical to the on-disk ``process_record`` path.
    """
    pil = Image.fromarray(frame_rgb)
    tight, box = crop_tight(pil, bbox)
    context = crop_context(pil, box, context_scale)
    return np.asarray(tight), np.asarray(context)


# ---- stage 1: detection + tracking (lazy ultralytics; [infer] extra) ------------------------
def _load_yolo():
    """Lazy-import ``ultralytics.YOLO`` with a clear ``[infer]``-extra hint on ImportError."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatched import
        raise ImportError(
            "Video detection needs the optional YOLO dependency. Install it with "
            "`pip install pedpredict[infer]` (ultralytics + lap)."
        ) from exc
    return YOLO


def detect_tracks(source: FrameSource, icfg: InferenceCfg, dcfg: DataCfg) -> dict[int, Track]:
    """YOLO detect+track over ``source``; returns ``{track_id: Track}`` (RGB crops, clamped bbox).

    Ports OLD ``pedestrian_detection.extract_tracks_from_video``. The full RGB frame is converted to BGR
    only for YOLO (which expects cv2 channel order); tight + context crops are taken from the RGB frame
    using ``dcfg.context_scale`` (the §2.3 fix). Lazy-imports ultralytics so the module imports without
    the ``[infer]`` extra and errors clearly only when actually invoked.
    """
    yolo_cls = _load_yolo()
    model = yolo_cls(icfg.detector_weights)
    tracks: dict[int, Track] = {}
    for frame_idx, frame_rgb in enumerate(source):
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = model.track(bgr, persist=True, classes=[icfg.detector_class_idx], conf=icfg.detector_conf)[0]
        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame_rgb.shape[1] - 1, x2)
            y2 = min(frame_rgb.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:                      # guard degenerate boxes (OLD line 45)
                continue
            bbox = (x1, y1, x2, y2)
            tight, context = crop_from_frame(frame_rgb, bbox, dcfg.context_scale)
            tracks.setdefault(track_id, []).append(
                Detection(frame_idx=frame_idx, bbox=bbox, tight=tight, context=context)
            )
    return tracks


# Q5: the legacy smooth_track (centered moving-average of bbox centers) was deleted — it wrote
# smoothed cx/cy that nothing downstream read (motion is computed from the raw bbox), i.e. verified
# dead computation preserved from OLD. Re-introduce as a bbox-level smoother if ever actually wanted.


# ---- stage 2: track -> fixed-length windows (pure) ------------------------------------------
def assemble_windows(tracks: dict[int, Track], cfg: DataCfg, icfg: InferenceCfg) -> list[TrackWindow]:
    """Sliding window (len=``cfg.seq_len``, step=``icfg.window_stride``) over each track >= ``seq_len``.

    Each window carries its detections (hence frame_idxs + bboxes), so aggregation needs no bbox
    re-matching (removes the OLD O(n²) ``bboxes == bboxes`` lookup).
    """
    seq_len, stride = cfg.seq_len, icfg.window_stride
    windows: list[TrackWindow] = []
    for track_id, track in tracks.items():
        if len(track) < seq_len:
            continue
        for i in range(0, len(track) - seq_len + 1, stride):
            windows.append(TrackWindow(track_id=track_id, detections=tuple(track[i : i + seq_len])))
    return windows


# ---- stage 3: preprocess one window via Phase-1 math (no PIESequenceDataset) ----------------
def preprocess_window(
    win: TrackWindow, cfg: DataCfg, transform_tight, transform_context
) -> tuple[Tensor, Tensor, Tensor]:
    """One window -> ``(images_tight[T,3,h,w], images_context[T,3,H,W], motions[T,8])``.

    Wraps the stored RGB crops as PIL, applies the read transforms (resize + ImageNet normalize), and
    computes motion via the canonical ``compute_motion`` (B7 — same 8-dim fn, no ``[..., :8]`` slice).
    """
    tights = [transform_tight(Image.fromarray(d.tight)) for d in win.detections]
    contexts = [transform_context(Image.fromarray(d.context)) for d in win.detections]
    motions = compute_motion([d.bbox for d in win.detections])
    return torch.stack(tights, dim=0), torch.stack(contexts, dim=0), motions


# ---- stage 4: batched forward -------------------------------------------------------------
def predict_windows(
    model: nn.Module, windows: list[TrackWindow], cfg: RootCfg, device: torch.device, *, use_amp: bool
) -> list[dict[str, int]]:
    """Batch windows -> ``forward_model`` -> argmax per task. ``crosses`` from ``crosses_frame`` only (B4)."""
    if not windows:
        return []
    transform_tight, transform_context = build_read_transforms(cfg.data)
    processed = [preprocess_window(w, cfg.data, transform_tight, transform_context) for w in windows]
    preds: list[dict[str, int]] = []
    bs = cfg.infer.batch_size
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(processed), bs):
            chunk = processed[start : start + bs]
            tight = torch.stack([p[0] for p in chunk]).to(device)
            context = torch.stack([p[1] for p in chunk]).to(device)
            motions = torch.stack([p[2] for p in chunk]).to(device)
            with autocast_ctx(use_amp, device.type):
                outputs = forward_model(model, tight, context, motions)
            actions = outputs["actions"].argmax(dim=1).cpu().tolist()
            looks = outputs["looks"].argmax(dim=1).cpu().tolist()
            crosses = outputs["crosses_frame"].argmax(dim=1).cpu().tolist()
            for a, lk, c in zip(actions, looks, crosses, strict=True):
                preds.append({"actions": int(a), "looks": int(lk), "crosses": int(c)})
    return preds


# ---- stage 5: aggregate + overlay ---------------------------------------------------------
def aggregate_by_frame(windows: list[TrackWindow], preds: list[dict[str, int]]) -> dict[int, list[FramePrediction]]:
    """Scatter each window's prediction to its frames (carried indices; no bbox-equality matching).

    A frame covered by several windows of one track collects one ``FramePrediction`` per window (OLD's
    list-append behavior); the renderer draws them in order, so the last window wins visually.
    """
    frame_results: dict[int, list[FramePrediction]] = {}
    for win, pred in zip(windows, preds, strict=True):
        for frame_idx, bbox in zip(win.frame_idxs, win.bboxes, strict=True):
            frame_results.setdefault(frame_idx, []).append(
                FramePrediction(
                    track_id=win.track_id, bbox=bbox,
                    actions=pred["actions"], looks=pred["looks"], crosses=pred["crosses"],
                )
            )
    return frame_results


def _draw_prediction(frame_bgr: np.ndarray, res: FramePrediction, icfg: InferenceCfg) -> None:
    """Draw one bbox + ID + action/look/cross labels + optional color chips (OLD main.py 257-316)."""
    x1, y1, x2, y2 = res.bbox
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _BBOX_COLOR, 2)

    id_text = f"ID {res.track_id}"
    (tw, _), _ = cv2.getTextSize(id_text, font, scale, thick)
    y1_label = max(y1 - 22, 0)
    cv2.rectangle(frame_bgr, (x1, y1_label), (x1 + tw, y1), (200, 200, 200), -1)
    cv2.putText(frame_bgr, id_text, (x1, y1 - 7), font, scale, _TEXT_COLOR, thick)

    labels = [
        ("action", _ACTION_TEXT[res.actions]),
        ("look", _LOOK_TEXT[res.looks]),
        ("cross", _CROSS_TEXT[res.crosses]),
    ]
    x_off, y_off = x1, y1_label - 22
    for name, text in labels:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame_bgr, (x_off, y_off), (x_off + tw + 6, y_off + th + 8), _LABEL_COLORS[name], -1)
        cv2.putText(frame_bgr, text, (x_off + 3, y_off + th + 3), font, scale, _TEXT_COLOR, thick)
        x_off += tw + 10

    if icfg.draw_color_chips:
        chips = [
            _LABEL_COLORS["action"] if res.actions else _CHIP_OFF,
            _LABEL_COLORS["look"] if res.looks else _CHIP_OFF,
            _LABEL_COLORS["cross"] if res.crosses else _CHIP_OFF,
        ]
        for i, color in enumerate(chips):
            cv2.rectangle(frame_bgr, (x1 + i * 15, y2 + 5), (x1 + (i + 1) * 15, y2 + 20), color, -1)


def render_overlays(
    source: FrameSource, frame_results: dict[int, list[FramePrediction]], icfg: InferenceCfg, out_path: Path
) -> Path:
    """Overlay predictions on each frame and write an mp4 at ``source.fps`` (ports OLD main.py 230-325)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, source.fps, (source.width, source.height))
    try:
        for frame_idx, frame_rgb in enumerate(source):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)   # draw + write in BGR
            for res in frame_results.get(frame_idx, []):
                _draw_prediction(frame_bgr, res, icfg)
            writer.write(frame_bgr)
    finally:
        writer.release()
    return out_path


# ---- top-level orchestrator ----------------------------------------------------------------
def run_video_inference(
    cfg: RootCfg, *, video: str | Path, checkpoint: str | Path,
    out_video: str | Path | None = None, device: torch.device | None = None, strict: bool = True,
) -> InferenceResult:
    """open -> detect -> window -> build+load -> predict -> aggregate -> render."""
    device = device if device is not None else get_device()
    enable_perf_flags(device)

    source = open_frame_source(video, fps=cfg.infer.default_fps)
    tracks = detect_tracks(source, cfg.infer, cfg.data)
    windows = assemble_windows(tracks, cfg.data, cfg.infer)

    model = build_model(cfg, cfg.eval.model_type).to(device)
    load_eval_weights(model, checkpoint, device=device, strict=strict)
    use_amp = resolve_amp(cfg.train.use_amp, device)
    preds = predict_windows(model, windows, cfg, device, use_amp=use_amp)

    frame_results = aggregate_by_frame(windows, preds)
    out_path = (
        render_overlays(source, frame_results, cfg.infer, Path(out_video)) if out_video is not None else None
    )
    return InferenceResult(
        frame_results=frame_results, n_frames=len(source), n_windows=len(windows), out_video=out_path
    )

"""Qualitative visualizations: GT overlays, pred-vs-GT comparison, temporal-attention overlay.

Ports OLD/Undergrad_thesis_project/visualize_gt.py (``visualize_from_pickle``,
``draw_future_labels``, ``draw_sequence_frame``) and visualize_comparison.py
(``process_pie_dataset``, ``draw_labels``, ``draw_comparison``) into one clean module (B11).
Adds temporal_weights attention overlay that has no OLD equivalent (requires 5.1 eval artifacts).

Band-aids resolved:
    B11 – two root-level one-off scripts promoted here; both are now dead at the OLD root.
    B6/B7 – SEQ_LEN/FUTURE_OFFSET/TOL literals → DataCfg.seq_len / .future_offset / .tol.
    B4 – crosses read from crosses_frame only; the OLD fallback to the 'crosses' key is dropped.

Intentional behavior changes vs OLD (documented in MIGRATION.md):
    context_scale 2.0 in OLD visualize_comparison.py vs DataCfg 3.0 – irrelevant here because
        render_comparison consumes pre-computed predictions (does not re-extract crops).
    process_custom_video (YOLO live-inference path) dropped – belongs to eval/inference.py (5.3).
    generate_and_visualize (on-the-fly PIE generation) dropped – callers pass pre-built sequences.
    visualize_blank_with_labels demo helper dropped.

Sequence format consumed by render_gt_sequences / render_comparison / render_attention_overlay:
    SequenceRecord (TypedDict from data/pie_sequences.py):
        images  – list[str], full PIE frame paths, length = seq_len
        bboxes  – list[list[float]], [[x1,y1,x2,y2]] per frame
        actions – int, binary future label (1 = will walk)
        looks   – int
        crosses – int
    Missing image files fall back silently to black placeholder frames.
"""

from __future__ import annotations

import enum
import warnings
from pathlib import Path

import cv2
import numpy as np

from pedpredict.config.schema import DataCfg, RootCfg
from pedpredict.utils.logging import RunDir

__all__ = [
    "ComparisonMode",
    "LABEL_COLORS",
    "BOX_COLOR",
    "MISMATCH_COLOR",
    "draw_bbox_with_labels",
    "draw_temporal_bar",
    "render_gt_sequences",
    "render_comparison",
    "render_attention_overlay",
    "generate_qualitative_figures",
]

#: Task label BGR colors — identical to OLD LABEL_COLORS in both scripts.
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "action": (0, 255, 255),   # yellow
    "look":   (255, 0, 255),   # magenta
    "cross":  (255, 255, 0),   # cyan
}
BOX_COLOR: tuple[int, int, int]      = (0, 255, 0)   # green
MISMATCH_COLOR: tuple[int, int, int] = (0, 0, 255)   # red

#: Task names in output-contract order (mirrors losses.TASKS / eval/plots TASKS).
_TASKS: tuple[str, ...] = ("actions", "looks", "crosses")
#: Maps plural task name → LABEL_COLORS key (singular).
_COLOR_KEY: dict[str, str] = {"actions": "action", "looks": "look", "crosses": "cross"}
#: Maps task name → display text for pills.
_TASK_TEXT: dict[str, str] = {"actions": "WALK", "looks": "LOOK", "crosses": "CROSS"}

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_THICKNESS  = 2

_DEFAULT_W, _DEFAULT_H = 1920, 1080   # fallback when no frames can be probed


class ComparisonMode(enum.Enum):
    GT   = "gt"
    PRED = "pred"
    BOTH = "both"
    DIFF = "diff"


# ---------------------------------------------------------------------------
# Public rendering primitives
# ---------------------------------------------------------------------------


def draw_bbox_with_labels(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    labels: dict[str, int],
    *,
    prefix: str = "",
    mismatch_mask: dict[str, bool] | None = None,
) -> np.ndarray:
    """Draw a green bbox + a row of colored task-pill labels above it (in-place; returns frame).

    Merges OLD ``draw_labels()`` + ``draw_sequence_frame()`` (visualize_gt.py:42, 241).
    For the combined GT|PRED format used in BOTH/DIFF modes call ``render_comparison`` directly
    which invokes the private ``_draw_comparison_pills`` helper.

    Args:
        frame:          BGR frame array, modified in-place.
        bbox:           (x1, y1, x2, y2) ints.
        labels:         {"actions": 0|1, "looks": 0|1, "crosses": 0|1}.
        prefix:         prepended to each label text (e.g. "GT:" or "PD:").
        mismatch_mask:  per-task True overrides pill background to MISMATCH_COLOR.
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    y_off = y1 - 22
    x_off = x1

    for task in _TASKS:
        val      = labels.get(task, 0)
        text_key = _TASK_TEXT[task]
        color    = LABEL_COLORS[_COLOR_KEY[task]]

        if mismatch_mask is not None and mismatch_mask.get(task, False):
            bg = MISMATCH_COLOR
        else:
            bg = color if val == 1 else (50, 50, 50)

        status    = f"{prefix}{text_key}" if val == 1 else f"{prefix}---"
        txt_color = (0, 0, 0) if val == 1 else (100, 100, 100)

        (tw, th), _ = cv2.getTextSize(status, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(frame, (x_off, y_off), (x_off + tw + 6, y_off + th + 8), bg, -1)
        cv2.putText(frame, status, (x_off + 3, y_off + th + 3),
                    _FONT, _FONT_SCALE, txt_color, _THICKNESS)
        x_off += tw + 10

    return frame


def draw_temporal_bar(
    frame: np.ndarray,
    weights: np.ndarray,
    t: int,
    *,
    bar_h: int = 12,
) -> np.ndarray:
    """Render a horizontal attention-weight bar in a strip at the bottom of frame (in-place).

    Each of the T cells represents one timestep in the sequence.  Brightness (dark→bright red)
    encodes normalized attention magnitude; the current timestep ``t`` is outlined in white.
    New primitive with no OLD equivalent.

    Args:
        frame:   BGR frame array, modified in-place.
        weights: 1-D array of length T; need not sum to 1.
        t:       index of the current frame being rendered (0 ≤ t < T).
        bar_h:   height of the bottom strip in pixels.
    """
    h, w = frame.shape[:2]
    T = len(weights)
    if T == 0 or bar_h <= 0:
        return frame

    w_max  = float(weights.max())
    norm   = (weights / w_max) if w_max > 0.0 else np.zeros_like(weights, dtype=float)
    cell_w = max(1, w // T)
    bar_y  = h - bar_h

    for j in range(T):
        x0 = j * cell_w
        x1 = min(x0 + cell_w, w - 1)
        a  = float(norm[j])
        blue  = int(20  * (1.0 - a))
        green = int(20  * (1.0 - a))
        red   = int(50  + 205 * a)
        cv2.rectangle(frame, (x0, bar_y), (x1, h - 1), (blue, green, red), -1)
        if j == t:
            cv2.rectangle(frame, (x0, bar_y), (x1, h - 1), (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------


def _probe_frame_size(seq: dict) -> tuple[int, int]:
    """Return (height, width) from the first readable image in seq, else (1080, 1920)."""
    for img_path in seq.get("images", []):
        img = cv2.imread(str(img_path))
        if img is not None:
            return int(img.shape[0]), int(img.shape[1])
    return _DEFAULT_H, _DEFAULT_W


def _load_frame(path: str, w: int, h: int) -> np.ndarray:
    """Load a BGR frame; return a black placeholder if the path is missing or unreadable."""
    frame = cv2.imread(str(path))
    if frame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return frame


def _draw_future_label_panel(
    frame: np.ndarray,
    action_lbl: int,
    look_lbl: int,
    cross_lbl: int,
    frame_width: int,
) -> None:
    """Draw the FUTURE PREDICTION corner panel (top-right). Ports OLD ``draw_future_labels``."""
    x_start = max(0, frame_width - 200)
    y_offset = 40
    cv2.rectangle(frame, (x_start - 10, 10), (frame_width - 10, y_offset + 80), (0, 0, 0), -1)
    cv2.putText(frame, "FUTURE PREDICTION", (x_start, 30), _FONT, 0.6, (255, 255, 255), 2)

    for task, text, val in (
        ("actions", "WILL WALK",  action_lbl),
        ("looks",   "WILL LOOK",  look_lbl),
        ("crosses", "WILL CROSS", cross_lbl),
    ):
        bg = LABEL_COLORS[_COLOR_KEY[task]] if val == 1 else (50, 50, 50)
        fg = (0, 0, 0) if val == 1 else (100, 100, 100)
        status = text if val == 1 else "---"
        cv2.rectangle(frame, (x_start, y_offset), (x_start + 180, y_offset + 22), bg, -1)
        cv2.putText(frame, status, (x_start + 5, y_offset + 16), _FONT, 0.7, fg, 2)
        y_offset += 28


def _draw_seq_info(
    frame: np.ndarray,
    seq_idx: int,
    total_seqs: int,
    seq_len: int,
    future_offset: int,
    frame_height: int,
) -> None:
    """Draw sequence/context info at bottom-left. Ports OLD last-frame info overlay."""
    info_y = max(10, frame_height - 100)
    cv2.rectangle(frame, (10, info_y - 10), (450, frame_height - 10), (0, 0, 0), -1)
    cv2.putText(frame, f"Sequence {seq_idx}/{total_seqs}", (20, info_y + 20),
                _FONT, 0.6, (255, 255, 255), 2)
    cv2.putText(frame,
                f"Input: {seq_len} frames | Future: {future_offset} frames",
                (20, info_y + 50), _FONT, 0.5, (200, 200, 200), 1)


def _draw_comparison_pills(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    gt_labels: dict[str, int],
    pred_labels: dict[str, int],
) -> None:
    """Render combined GT|PRED pills above bbox. Ports OLD ``draw_comparison`` (line 273)."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    y_off = y1 - 22
    x_off = x1

    for task in _TASKS:
        text_key = _TASK_TEXT[task]
        gt_val   = gt_labels.get(task, 0)
        pred_val = pred_labels.get(task, 0)
        match    = (gt_val == pred_val)

        gt_str   = text_key if gt_val   == 1 else "---"
        pred_str = text_key if pred_val == 1 else "---"
        text     = f"{gt_str}|{pred_str}"

        bg = LABEL_COLORS[_COLOR_KEY[task]] if match else MISMATCH_COLOR
        (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _THICKNESS)
        cv2.rectangle(frame, (x_off, y_off), (x_off + tw + 6, y_off + th + 8), bg, -1)
        cv2.putText(frame, text, (x_off + 3, y_off + th + 3),
                    _FONT, _FONT_SCALE, (0, 0, 0), _THICKNESS)
        x_off += tw + 10


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------


def render_gt_sequences(
    sequences: list[dict],
    cfg: DataCfg,
    out_path: Path,
    *,
    max_sequences: int = 50,
    fps: int = 5,
) -> Path:
    """Write a GT-only sequence visualization MP4 (ports OLD ``visualize_from_pickle``).

    Renders bbox on every frame; draws the FUTURE PREDICTION panel on the last frame only
    (matching OLD: ``if frame_idx == seq_len - 1``).  Missing PIE frames → black placeholders.

    Args:
        sequences:      list of SequenceRecord dicts (images/bboxes/actions/looks/crosses).
        cfg:            DataCfg for seq_len / future_offset constants.
        out_path:       destination MP4 path.
        max_sequences:  cap on sequences rendered.
        fps:            output video frame rate.

    Returns:
        out_path (written).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seqs = sequences[:max_sequences]
    if not seqs:
        warnings.warn(
            f"render_gt_sequences: empty sequence list; nothing written to {out_path}", stacklevel=2
        )
        return out_path

    h, w      = _probe_frame_size(seqs[0])
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    try:
        for seq_idx, seq in enumerate(seqs):
            images  = seq["images"]
            bboxes  = seq["bboxes"]
            seq_len = len(images)

            for t in range(seq_len):
                frame = _load_frame(images[t], w, h)
                bbox  = tuple(map(int, bboxes[t]))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BOX_COLOR, 2)

                if t == seq_len - 1:
                    _draw_future_label_panel(
                        frame, int(seq["actions"]), int(seq["looks"]), int(seq["crosses"]), w
                    )
                    _draw_seq_info(frame, seq_idx + 1, len(seqs), seq_len, cfg.future_offset, h)

                writer.write(frame)
    finally:
        writer.release()

    return out_path


def render_comparison(
    sequences: list[dict],
    predictions: dict[str, np.ndarray],
    cfg: DataCfg,
    out_path: Path,
    *,
    mode: ComparisonMode = ComparisonMode.BOTH,
    fps: int = 10,
) -> Path:
    """Write a GT-vs-prediction comparison MP4 (ports OLD ``process_pie_dataset``).

    One prediction applies to every frame in its sequence (the model output is sequence-level).
    ``predictions`` must be aligned with ``sequences`` by index:
    ``predictions["{task}_pred"][i]`` is the binary prediction for ``sequences[i]``.

    Modes:
        GT   – bbox + GT future labels only.
        PRED – bbox + model predictions only.
        BOTH – bbox + combined "GT|PRED" pills per task; red when they disagree.
        DIFF – identical to BOTH (OLD process_pie_dataset sent both modes to draw_comparison).

    Missing image files → black placeholder frames.

    Args:
        sequences:   list of SequenceRecord dicts (images/bboxes/actions/looks/crosses).
        predictions: dict with "{task}_pred" arrays of shape [N] (from predictions.npz).
        cfg:         DataCfg (for future_offset; not strictly needed for rendering but kept
                     consistent with the other render functions).
        out_path:    destination MP4 path.
        mode:        ComparisonMode variant.
        fps:         output video frame rate.

    Returns:
        out_path (written).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pred = len(next(iter(predictions.values()))) if predictions else 0
    seqs   = sequences[:n_pred] if n_pred else sequences
    if not seqs:
        warnings.warn(
            f"render_comparison: no sequences to render; nothing written to {out_path}", stacklevel=2
        )
        return out_path

    h, w   = _probe_frame_size(seqs[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    try:
        for i, seq in enumerate(seqs):
            images  = seq["images"]
            bboxes  = seq["bboxes"]
            seq_len = len(images)

            gt_labels = {task: int(seq[task]) for task in _TASKS}
            pred_labels = {
                task: int(predictions.get(f"{task}_pred", np.zeros(i + 1, int))[i])
                for task in _TASKS
            }

            for t in range(seq_len):
                frame = _load_frame(images[t], w, h)
                bbox  = tuple(map(int, bboxes[t]))

                if mode is ComparisonMode.GT:
                    draw_bbox_with_labels(frame, bbox, gt_labels, prefix="GT:")
                elif mode is ComparisonMode.PRED:
                    draw_bbox_with_labels(frame, bbox, pred_labels, prefix="PD:")
                else:
                    _draw_comparison_pills(frame, bbox, gt_labels, pred_labels)

                cv2.putText(frame, f"Mode: {mode.value.upper()}", (10, 30),
                            _FONT, 0.7, (255, 255, 255), 2)
                writer.write(frame)
    finally:
        writer.release()

    return out_path


def render_attention_overlay(
    sequences: list[dict],
    temporal_weights: np.ndarray,
    cfg: DataCfg,
    out_path: Path,
    *,
    fps: int = 5,
) -> Path:
    """Write per-frame attention-weight overlay MP4 (new; no OLD equivalent).

    Each frame shows the bbox, GT future labels, and a horizontal attention bar at the
    bottom (bright-red = high attention).  On the last frame of each sequence, the
    peak-attention timestep is annotated.

    ``sequences[i]`` is paired with ``temporal_weights[i]`` by index; the caller ensures
    alignment (typically: sequences from the test split pkl, weights from temporal_weights.npz
    produced by the eval run over the same split in the same chunk order).

    Args:
        sequences:        list of SequenceRecord dicts.
        temporal_weights: float array of shape [N, T].
        cfg:              DataCfg (for future_offset).
        out_path:         destination MP4 path.
        fps:              output video frame rate.

    Returns:
        out_path (written).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(sequences), len(temporal_weights))
    seqs = sequences[:n]
    if not seqs:
        warnings.warn(
            f"render_attention_overlay: no sequences to render; nothing written to {out_path}",
            stacklevel=2,
        )
        return out_path

    h, w   = _probe_frame_size(seqs[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    try:
        for i, seq in enumerate(seqs):
            images  = seq["images"]
            bboxes  = seq["bboxes"]
            seq_len = len(images)
            tw      = temporal_weights[i].astype(float)   # [T]

            peak_t = int(np.argmax(tw))

            for t in range(seq_len):
                frame = _load_frame(images[t], w, h)
                bbox  = tuple(map(int, bboxes[t]))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), BOX_COLOR, 2)

                draw_temporal_bar(frame, tw, t)

                if t == seq_len - 1:
                    _draw_future_label_panel(
                        frame, int(seq["actions"]), int(seq["looks"]), int(seq["crosses"]), w
                    )
                    _draw_seq_info(frame, i + 1, n, seq_len, cfg.future_offset, h)
                    cv2.putText(frame, f"peak attention @ t={peak_t}", (10, 30),
                                _FONT, 0.6, (255, 255, 255), 2)

                writer.write(frame)
    finally:
        writer.release()

    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_qualitative_figures(
    run: RunDir,
    cfg: RootCfg,
    *,
    which: list[str] | None = None,
    sequences: list[dict] | None = None,
    max_sequences: int = 20,
    fps: int = 5,
) -> list[Path]:
    """Load eval artifacts from ``run.plots_dir`` and generate qualitative MP4 videos.

    Reads ``predictions.npz`` and ``temporal_weights.npz`` written by :func:`evaluate.run_evaluation`
    (5.1).  ``sequences`` must be provided by the caller for GT / comparison modes (load from the
    test-split pkl produced by ``scripts/make_sequences.py``); they should be aligned with the eval
    run's sample order (same split, same chunk order) so that ``sequences[i]`` matches
    ``predictions[*][i]`` and ``temporal_weights[i]``.

    Args:
        run:           RunDir handle (plots_dir is read / written).
        cfg:           RootCfg for DataCfg constants.
        which:         subset of ``["gt", "comparison", "attention"]``; None = all available.
        sequences:     pre-built SequenceRecord list; required for "gt" and "comparison" modes.
        max_sequences: max sequences per video.
        fps:           output video frame rate.

    Returns:
        list of paths actually written (skipped modes produce no entry).
    """
    plots_dir = run.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    want = set(which) if which is not None else {"gt", "comparison", "attention"}
    generated: list[Path] = []

    predictions_path = plots_dir / "predictions.npz"
    tw_path          = plots_dir / "temporal_weights.npz"

    predictions: dict[str, np.ndarray] | None = None
    if predictions_path.exists():
        raw = np.load(predictions_path)
        predictions = {k: raw[k] for k in raw.files}
    else:
        if "comparison" in want:
            warnings.warn(
                f"generate_qualitative_figures: {predictions_path} not found; "
                "skipping comparison mode. Run evaluate.py with --save-predictions first.",
                stacklevel=2,
            )

    temporal_weights: np.ndarray | None = None
    if tw_path.exists():
        raw = np.load(tw_path)
        if "temporal_weights" in raw.files:
            temporal_weights = raw["temporal_weights"].astype(float)
    else:
        if "attention" in want:
            warnings.warn(
                f"generate_qualitative_figures: {tw_path} not found; "
                "skipping attention mode. Run evaluate.py with --save-temporal-weights first.",
                stacklevel=2,
            )

    data_cfg = cfg.data
    seqs = sequences[:max_sequences] if sequences is not None else None

    if "gt" in want and seqs is not None:
        out = render_gt_sequences(
            seqs, data_cfg, plots_dir / "qualitative_gt.mp4",
            max_sequences=max_sequences, fps=fps,
        )
        if out.exists():
            generated.append(out)

    if "comparison" in want and seqs is not None and predictions is not None:
        out = render_comparison(
            seqs, predictions, data_cfg,
            plots_dir / "qualitative_comparison_both.mp4",
            fps=fps,
        )
        if out.exists():
            generated.append(out)

    if "attention" in want and temporal_weights is not None:
        seqs_for_attn = seqs if seqs is not None else []
        if not seqs_for_attn:
            warnings.warn(
                "generate_qualitative_figures: sequences not provided; "
                "skipping attention overlay.",
                stacklevel=2,
            )
        else:
            out = render_attention_overlay(
                seqs_for_attn, temporal_weights, data_cfg,
                plots_dir / "qualitative_attention.mp4",
                fps=fps,
            )
            if out.exists():
                generated.append(out)

    return generated

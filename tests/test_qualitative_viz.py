"""Prompt 6.2 — qualitative visualization tests.

All tests use synthetic sequences with non-existent image paths so they exercise the black-frame
fallback path without requiring PIE images or LMDB data on disk.  No pixel-level comparison of MP4
output is done (codec behaviour is platform-specific); correctness is verified through:
    - file existence / non-zero size
    - frame count via cv2.VideoCapture (structural)
    - pixel-level assertions on the rendering primitives directly (draw_bbox_with_labels,
      draw_temporal_bar) using small in-memory numpy arrays
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pedpredict.config.schema import DataCfg, RootCfg
from pedpredict.utils.logging import RunDir, create_run_dir
from pedpredict.viz.qualitative import (
    BOX_COLOR,
    LABEL_COLORS,
    MISMATCH_COLOR,
    ComparisonMode,
    draw_bbox_with_labels,
    draw_temporal_bar,
    generate_qualitative_figures,
    render_attention_overlay,
    render_comparison,
    render_gt_sequences,
)

_RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _blank_frame(h: int = 200, w: int = 300) -> np.ndarray:
    """Return a black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_sequences(n: int = 3, seq_len: int = 4) -> list[dict]:
    """Synthetic SequenceRecord list with non-existent image paths → black frames."""
    seqs = []
    for i in range(n):
        seqs.append({
            "images":  [f"/nonexistent/frame_{i}_{t}.jpg" for t in range(seq_len)],
            "bboxes":  [[10, 20, 60, 90]] * seq_len,
            "actions": i % 2,
            "looks":   1,
            "crosses": 0,
        })
    return seqs


def _make_run_dir(tmp_path: Path) -> RunDir:
    path = create_run_dir(tmp_path, "20260608_000000_full")
    return RunDir(run_id="20260608_000000_full", path=path)


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# ---------------------------------------------------------------------------
# draw_bbox_with_labels
# ---------------------------------------------------------------------------


def test_draw_bbox_with_labels_smoke():
    frame = _blank_frame()
    original = frame.copy()
    labels = {"actions": 1, "looks": 0, "crosses": 1}
    result = draw_bbox_with_labels(frame, (10, 50, 80, 150), labels)
    assert result is frame, "should return the same array"
    assert not np.array_equal(frame, original), "frame should be modified"
    assert frame.shape == (200, 300, 3)


def test_draw_bbox_with_labels_returns_frame_shape():
    frame = _blank_frame(100, 120)
    labels = {"actions": 0, "looks": 0, "crosses": 0}
    out = draw_bbox_with_labels(frame, (5, 30, 40, 80), labels)
    assert out.shape == (100, 120, 3)


def test_draw_bbox_with_labels_prefix():
    """Calling with prefix should not crash and should still modify the frame."""
    frame = _blank_frame()
    orig  = frame.copy()
    draw_bbox_with_labels(frame, (5, 50, 100, 150), {"actions": 1, "looks": 1, "crosses": 0},
                          prefix="GT:")
    assert not np.array_equal(frame, orig)


def test_draw_bbox_with_labels_mismatch_color():
    """When mismatch_mask is True for a task, MISMATCH_COLOR pixels should appear."""
    frame = _blank_frame(200, 400)
    labels = {"actions": 1, "looks": 0, "crosses": 1}
    mask   = {"actions": True, "looks": False, "crosses": True}
    draw_bbox_with_labels(frame, (10, 100, 100, 180), labels, mismatch_mask=mask)
    # At least one pixel in the frame must equal MISMATCH_COLOR (BGR red).
    found = np.any(
        (frame[:, :, 0] == MISMATCH_COLOR[0]) &
        (frame[:, :, 1] == MISMATCH_COLOR[1]) &
        (frame[:, :, 2] == MISMATCH_COLOR[2])
    )
    assert found, "MISMATCH_COLOR should appear when mismatch_mask is True"


def test_draw_bbox_with_labels_no_mismatch_uses_task_color():
    """A matching active task should use its task color, not MISMATCH_COLOR."""
    frame = _blank_frame(200, 400)
    draw_bbox_with_labels(frame, (10, 100, 100, 180), {"actions": 1, "looks": 0, "crosses": 0})
    # LABEL_COLORS["action"] = (0, 255, 255) — at least one pixel should have this color.
    yellow = LABEL_COLORS["action"]
    found = np.any(
        (frame[:, :, 0] == yellow[0]) &
        (frame[:, :, 1] == yellow[1]) &
        (frame[:, :, 2] == yellow[2])
    )
    assert found, "Active task pill should use LABEL_COLORS color"


# ---------------------------------------------------------------------------
# draw_temporal_bar
# ---------------------------------------------------------------------------


def test_draw_temporal_bar_smoke():
    frame = _blank_frame()
    weights = np.ones(20) / 20
    result  = draw_temporal_bar(frame, weights, t=5)
    assert result is frame
    assert frame.shape == (200, 300, 3)


def test_draw_temporal_bar_bottom_strip_modified():
    """Bottom ``bar_h`` rows must differ from the original black frame."""
    frame  = _blank_frame(100, 200)
    bar_h  = 12
    weights = np.linspace(0, 1, 10)
    draw_temporal_bar(frame, weights, t=3, bar_h=bar_h)
    bottom = frame[100 - bar_h:, :, :]
    assert np.any(bottom != 0), "bottom strip should be non-black after draw_temporal_bar"


def test_draw_temporal_bar_top_unchanged():
    """Pixels above the bar strip should remain black."""
    frame  = _blank_frame(100, 200)
    bar_h  = 12
    weights = _RNG.random(10)
    draw_temporal_bar(frame, weights, t=0, bar_h=bar_h)
    top = frame[:100 - bar_h, :, :]
    assert np.all(top == 0), "pixels above the attention bar should be untouched"


def test_draw_temporal_bar_current_frame_tick():
    """The current frame column must contain at least one white pixel (tick outline)."""
    frame   = _blank_frame(60, 200)
    T       = 10
    weights = np.zeros(T)
    t       = 4
    draw_temporal_bar(frame, weights, t=t, bar_h=12)
    cell_w = max(1, 200 // T)
    x0, x1 = t * cell_w, min(t * cell_w + cell_w, 200)
    col_strip = frame[60 - 12:, x0:x1, :]
    white = np.array([255, 255, 255])
    found = np.any(np.all(col_strip == white, axis=2))
    assert found, "white tick should appear at the current frame column"


def test_draw_temporal_bar_empty_weights_noop():
    """Zero-length weights should not crash and leave frame unchanged."""
    frame = _blank_frame()
    orig  = frame.copy()
    draw_temporal_bar(frame, np.array([]), t=0)
    assert np.array_equal(frame, orig)


# ---------------------------------------------------------------------------
# render_gt_sequences
# ---------------------------------------------------------------------------


def test_render_gt_sequences_writes_mp4(tmp_path):
    seqs = _make_sequences(n=2, seq_len=4)
    out  = render_gt_sequences(seqs, DataCfg(), tmp_path / "gt.mp4", max_sequences=2, fps=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_gt_sequences_frame_count(tmp_path):
    seq_len  = 5
    n_seqs   = 3
    seqs     = _make_sequences(n=n_seqs, seq_len=seq_len)
    out      = render_gt_sequences(seqs, DataCfg(), tmp_path / "gt.mp4", max_sequences=n_seqs, fps=5)
    assert out.exists()
    count = _video_frame_count(out)
    assert count == n_seqs * seq_len, f"expected {n_seqs * seq_len} frames, got {count}"


def test_render_gt_sequences_respects_max_sequences(tmp_path):
    seqs = _make_sequences(n=5, seq_len=3)
    out  = render_gt_sequences(seqs, DataCfg(), tmp_path / "gt.mp4", max_sequences=2, fps=5)
    assert out.exists()
    count = _video_frame_count(out)
    assert count == 2 * 3


def test_render_gt_sequences_missing_images_black_frames(tmp_path):
    """Non-existent images must not crash; output is still a valid non-empty file."""
    seqs = _make_sequences(n=1, seq_len=3)   # all images are /nonexistent/...
    out  = render_gt_sequences(seqs, DataCfg(), tmp_path / "gt.mp4", fps=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_gt_sequences_empty_list(tmp_path):
    """Empty sequence list should warn and return the path without crashing."""
    out = render_gt_sequences([], DataCfg(), tmp_path / "gt.mp4", fps=5)
    assert not out.exists() or out.stat().st_size == 0


# ---------------------------------------------------------------------------
# render_comparison
# ---------------------------------------------------------------------------


def _make_predictions(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        f"{task}_pred": rng.integers(0, 2, size=n).astype(int)
        for task in ("actions", "looks", "crosses")
    }


@pytest.mark.parametrize("mode", list(ComparisonMode))
def test_render_comparison_all_modes_write_mp4(tmp_path, mode):
    seqs  = _make_sequences(n=3, seq_len=4)
    preds = _make_predictions(3)
    out   = render_comparison(seqs, preds, DataCfg(),
                              tmp_path / f"cmp_{mode.value}.mp4", mode=mode, fps=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_comparison_frame_count(tmp_path):
    n_seqs, seq_len = 2, 5
    seqs  = _make_sequences(n=n_seqs, seq_len=seq_len)
    preds = _make_predictions(n_seqs)
    out   = render_comparison(seqs, preds, DataCfg(), tmp_path / "cmp.mp4", fps=5)
    count = _video_frame_count(out)
    assert count == n_seqs * seq_len


def test_render_comparison_mismatch_produces_red_pixels(tmp_path):
    """Diff mode with forced mismatches should produce MISMATCH_COLOR pixels."""
    seqs = _make_sequences(n=1, seq_len=4)
    seqs[0]["actions"] = 1   # GT = 1
    preds = {"actions_pred": np.array([0]),   # pred = 0 → mismatch
             "looks_pred":   np.array([1]),
             "crosses_pred": np.array([0])}
    out = render_comparison(seqs, preds, DataCfg(), tmp_path / "diff.mp4",
                            mode=ComparisonMode.DIFF, fps=5)
    assert out.exists()
    # Read a frame back and check for red pixels (MISMATCH_COLOR = (0,0,255) BGR).
    cap = cv2.VideoCapture(str(out))
    ret, frame = cap.read()
    cap.release()
    assert ret
    found = np.any(
        (frame[:, :, 0] == 0) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 255)
    )
    assert found, "MISMATCH_COLOR should appear in DIFF mode when labels disagree"


def test_render_comparison_empty_sequences(tmp_path):
    """Empty sequence list should warn but not crash."""
    out = render_comparison([], _make_predictions(0), DataCfg(), tmp_path / "cmp.mp4", fps=5)
    assert not out.exists() or out.stat().st_size == 0


# ---------------------------------------------------------------------------
# render_attention_overlay
# ---------------------------------------------------------------------------


def test_render_attention_overlay_writes_mp4(tmp_path):
    T       = 4
    seqs    = _make_sequences(n=3, seq_len=T)
    weights = _RNG.random((3, T)).astype(float)
    out     = render_attention_overlay(seqs, weights, DataCfg(), tmp_path / "attn.mp4", fps=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_attention_overlay_frame_count(tmp_path):
    n_seqs, seq_len = 2, 5
    seqs    = _make_sequences(n=n_seqs, seq_len=seq_len)
    weights = _RNG.random((n_seqs, seq_len)).astype(float)
    out     = render_attention_overlay(seqs, weights, DataCfg(), tmp_path / "attn.mp4", fps=5)
    count   = _video_frame_count(out)
    assert count == n_seqs * seq_len


def test_render_attention_overlay_extra_weights_truncated(tmp_path):
    """If temporal_weights has more rows than sequences, only sequences rows are rendered."""
    seqs    = _make_sequences(n=2, seq_len=4)
    weights = _RNG.random((10, 4)).astype(float)
    out     = render_attention_overlay(seqs, weights, DataCfg(), tmp_path / "attn.mp4", fps=5)
    count   = _video_frame_count(out)
    assert count == 2 * 4


def test_render_attention_overlay_empty(tmp_path):
    """Empty sequence list should warn and not crash."""
    weights = _RNG.random((5, 4)).astype(float)
    out     = render_attention_overlay([], weights, DataCfg(), tmp_path / "attn.mp4", fps=5)
    assert not out.exists() or out.stat().st_size == 0


# ---------------------------------------------------------------------------
# generate_qualitative_figures (orchestrator)
# ---------------------------------------------------------------------------


def test_generate_qualitative_figures_no_artifacts(tmp_path):
    """Empty run dir (no npz files) should return empty list without crashing."""
    run = _make_run_dir(tmp_path)
    cfg = RootCfg()
    result = generate_qualitative_figures(run, cfg, which=["comparison", "attention"])
    assert result == []


def test_generate_qualitative_figures_gt_only(tmp_path):
    """GT mode writes qualitative_gt.mp4 when sequences are provided."""
    run  = _make_run_dir(tmp_path)
    cfg  = RootCfg()
    seqs = _make_sequences(n=2, seq_len=3)
    result = generate_qualitative_figures(run, cfg, which=["gt"], sequences=seqs,
                                          max_sequences=2, fps=5)
    assert len(result) == 1
    assert result[0].name == "qualitative_gt.mp4"
    assert result[0].exists()


def test_generate_qualitative_figures_with_predictions_npz(tmp_path):
    """Comparison mode writes qualitative_comparison_both.mp4 when predictions.npz present."""
    run  = _make_run_dir(tmp_path)
    cfg  = RootCfg()
    seqs = _make_sequences(n=2, seq_len=3)

    # Write a minimal predictions.npz
    preds_path = run.plots_dir / "predictions.npz"
    n = 2
    np.savez_compressed(
        preds_path,
        actions_pred=np.zeros(n, int),
        looks_pred=np.ones(n, int),
        crosses_pred=np.zeros(n, int),
    )

    result = generate_qualitative_figures(run, cfg, which=["comparison"], sequences=seqs,
                                          max_sequences=2, fps=5)
    assert len(result) == 1
    assert result[0].name == "qualitative_comparison_both.mp4"


def test_generate_qualitative_figures_with_temporal_weights_npz(tmp_path):
    """Attention mode writes qualitative_attention.mp4 when temporal_weights.npz present."""
    run  = _make_run_dir(tmp_path)
    cfg  = RootCfg()
    seqs = _make_sequences(n=2, seq_len=4)

    tw_path = run.plots_dir / "temporal_weights.npz"
    np.savez_compressed(tw_path, temporal_weights=_RNG.random((5, 4)).astype(np.float32))

    result = generate_qualitative_figures(run, cfg, which=["attention"], sequences=seqs,
                                          max_sequences=2, fps=5)
    assert len(result) == 1
    assert result[0].name == "qualitative_attention.mp4"


def test_generate_qualitative_figures_no_sequences_skips_gt_comparison(tmp_path):
    """GT and comparison modes are skipped when sequences=None."""
    run = _make_run_dir(tmp_path)
    cfg = RootCfg()

    preds_path = run.plots_dir / "predictions.npz"
    np.savez_compressed(preds_path, actions_pred=np.zeros(3, int),
                        looks_pred=np.zeros(3, int), crosses_pred=np.zeros(3, int))

    result = generate_qualitative_figures(run, cfg, which=["gt", "comparison"], sequences=None)
    assert result == []


def test_generate_qualitative_figures_all_modes(tmp_path):
    """All three modes run together when both npz files and sequences are present."""
    run  = _make_run_dir(tmp_path)
    cfg  = RootCfg()
    seqs = _make_sequences(n=2, seq_len=3)

    np.savez_compressed(
        run.plots_dir / "predictions.npz",
        actions_pred=np.zeros(2, int),
        looks_pred=np.zeros(2, int),
        crosses_pred=np.zeros(2, int),
    )
    np.savez_compressed(
        run.plots_dir / "temporal_weights.npz",
        temporal_weights=_RNG.random((2, 3)).astype(np.float32),
    )

    result = generate_qualitative_figures(run, cfg, sequences=seqs, max_sequences=2, fps=5)
    names  = {p.name for p in result}
    assert "qualitative_gt.mp4" in names
    assert "qualitative_comparison_both.mp4" in names
    assert "qualitative_attention.mp4" in names
    assert len(result) == 3

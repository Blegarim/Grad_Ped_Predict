"""Prompt 5.3 — video-inference seam tests.

Full video inference is not golden-able end-to-end (non-deterministic YOLO + a trained checkpoint + a
real video), so parity is proven at the reused-math seams: the preprocessing math is already golden
(``test_transforms.py``) and the forward is already golden (``ensemble.pt``). These tests cover the NEW
glue — windowing, in-memory preprocess == the on-disk path, aggregation — plus the lazy
detector isolation and the two flagged fixes (context_scale, BGR). Legacy parity uses in-test verbatim
oracles (the repo convention, cf. ``_legacy_compute_motion``).
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pedpredict.config import DataCfg, InferenceCfg, RootCfg
from pedpredict.data.transforms import build_read_transforms, compute_motion, process_record
from pedpredict.eval import inference as inf
from pedpredict.eval.inference import (
    Detection,
    FramePrediction,
    TrackWindow,
    aggregate_by_frame,
    assemble_windows,
    crop_from_frame,
    detect_tracks,
    open_frame_source,
    predict_windows,
    preprocess_window,
    render_overlays,
    run_video_inference,
)
from pedpredict.models.registry import ModelType, build_model

_CPU = torch.device("cpu")


# --------------------------------------------------------------------------- helpers / oracles


def _dummy_crop(h: int = 8, w: int = 6) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_track(bboxes: list[tuple[int, int, int, int]]) -> list[Detection]:
    """Build a frame-ordered track (as detect_tracks produces it)."""
    return [
        Detection(frame_idx=i, bbox=b, tight=_dummy_crop(), context=_dummy_crop())
        for i, b in enumerate(bboxes)
    ]


def _legacy_windows(bboxes, seq_len):
    """Verbatim OLD windowing loop (main.py 102-103): step 1, range(len - T + 1)."""
    return [
        (tuple(range(i, i + seq_len)), tuple(bboxes[i : i + seq_len]))
        for i in range(len(bboxes) - seq_len + 1)
    ]


def _fast_cfg() -> RootCfg:
    """Small geometry so a full-model CPU forward stays cheap (mirrors test_benchmark)."""
    return dataclasses.replace(
        RootCfg(),
        data=dataclasses.replace(
            DataCfg(), img_height=64, img_width=64, read_context_height=64, read_context_width=64,
            seq_len=4, max_seq_len=4,
        ),
    )


_BBOXES = [(50 + 4 * i, 40 + 3 * i, 90 + 4 * i, 120 + 3 * i) for i in range(6)]


# (Q5: the legacy smooth_track parity test was deleted with the function — verified dead computation.)


# --------------------------------------------------------------------------- 2. windowing parity


def test_assemble_windows_matches_legacy() -> None:
    track = _make_track(_BBOXES)
    cfg = dataclasses.replace(DataCfg(), seq_len=4)
    windows = assemble_windows({7: track}, cfg, InferenceCfg(window_stride=1))
    oracle = _legacy_windows(_BBOXES, seq_len=4)
    assert len(windows) == len(oracle)
    for win, (frame_idxs, bboxes) in zip(windows, oracle, strict=True):
        assert win.track_id == 7
        assert win.frame_idxs == frame_idxs
        assert win.bboxes == bboxes


def test_assemble_windows_drops_short_tracks() -> None:
    cfg = dataclasses.replace(DataCfg(), seq_len=4)
    short = _make_track(_BBOXES[:3])
    assert assemble_windows({0: short}, cfg, InferenceCfg(window_stride=1)) == []


# --------------------------------------------------------------------------- 3/4/10. preprocess parity


def _synthetic_window(rng, n=5, hh=200, ww=300):
    """A window of n detections cropped from random full RGB frames; returns (window, frames, bboxes)."""
    cfg = RootCfg().data
    frames, dets, bboxes = [], [], []
    for i in range(n):
        frame = rng.integers(0, 255, size=(hh, ww, 3), dtype=np.uint8)
        bbox = (40 + 3 * i, 30 + 2 * i, 110 + 3 * i, 150 + 2 * i)
        tight, context = crop_from_frame(frame, bbox, cfg.context_scale)
        frames.append(frame)
        bboxes.append(bbox)
        dets.append(Detection(i, bbox, tight, context))
    return TrackWindow(0, tuple(dets)), frames, bboxes


def test_preprocess_equals_process_record(tmp_path) -> None:
    rng = np.random.default_rng(0)
    win, frames, bboxes = _synthetic_window(rng)
    cfg = RootCfg().data
    tt, tc = build_read_transforms(cfg)
    t_out, c_out, m_out = preprocess_window(win, cfg, tt, tc)

    paths = []
    for i, frame in enumerate(frames):
        p = tmp_path / f"f_{i}.png"             # PNG = lossless -> decoded pixels identical
        Image.fromarray(frame).save(p)
        paths.append(str(p))
    rec = {
        "images": paths,
        "bboxes": [list(b) for b in bboxes],
        "track_id": "ped_infer",
        "ego_speed": [0.0] * len(paths),  # the raw-video path has no OBD — ego is zeros there too
        "actions": 0, "looks": 0, "crosses": 0,
    }
    exp = process_record(rec, cfg, tt, tc)

    # inference slices to motion_dim exactly like the LMDB read path
    torch.testing.assert_close(m_out, exp.motions[:, : cfg.motion_dim], rtol=0, atol=0)
    torch.testing.assert_close(t_out, exp.images_tight, rtol=0, atol=1e-6)
    torch.testing.assert_close(c_out, exp.images_context, rtol=0, atol=1e-6)


def test_motion_is_canonical_compute_motion() -> None:
    win, _, bboxes = _synthetic_window(np.random.default_rng(1))
    cfg = RootCfg().data
    tt, tc = build_read_transforms(cfg)
    _, _, m_out = preprocess_window(win, cfg, tt, tc)
    assert m_out.shape[-1] == cfg.motion_dim
    torch.testing.assert_close(m_out, compute_motion(bboxes)[:, : cfg.motion_dim], rtol=0, atol=0)


def test_context_scale_uses_data_cfg() -> None:
    """Guards the §2.3 fix: DataCfg default is 3.0 (training distribution), not OLD's hardcoded 2.0."""
    assert RootCfg().data.context_scale == 3.0
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bbox = (50, 50, 90, 130)
    _, ctx3 = crop_from_frame(frame, bbox, 3.0)
    _, ctx2 = crop_from_frame(frame, bbox, 2.0)
    assert ctx3.shape[0] > ctx2.shape[0] and ctx3.shape[1] > ctx2.shape[1]   # 3.0 -> larger context


# --------------------------------------------------------------------------- 5. crosses from crosses_frame


class _StubModel(nn.Module):
    """Full-type stub: crosses_frame argmax -> 1, crosses_pooled argmax -> 0 (opposite)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_type = ModelType.FULL

    def forward(self, images_tight, images_context, motions, return_feats=False):
        b = images_tight.shape[0]
        hi = torch.tensor([0.0, 1.0]).repeat(b, 1)        # argmax -> 1
        lo = torch.tensor([1.0, 0.0]).repeat(b, 1)        # argmax -> 0
        return {
            "actions": hi, "looks": lo, "crosses_frame": hi, "crosses_pooled": lo,
            "temporal_weights": torch.zeros(b, motions.shape[1]),
        }


def test_predict_windows_reads_crosses_frame() -> None:
    cfg = _fast_cfg()
    win, _, _ = _synthetic_window(np.random.default_rng(2), n=cfg.data.seq_len)
    preds = predict_windows(_StubModel(), [win], cfg, _CPU, use_amp=False)
    assert len(preds) == 1
    # crosses == 1 comes from crosses_frame; crosses_pooled (argmax 0) is ignored (B4).
    assert preds[0] == {"actions": 1, "looks": 0, "crosses": 1}


# --------------------------------------------------------------------------- 6. aggregate scatter


def test_aggregate_by_frame_scatter() -> None:
    t1 = _make_track([(0, 0, 10, 10), (1, 1, 11, 11)])          # frames 0,1
    t2_dets = (
        Detection(1, (2, 2, 12, 12), _dummy_crop(), _dummy_crop()),
        Detection(2, (3, 3, 13, 13), _dummy_crop(), _dummy_crop()),
    )
    w1 = TrackWindow(5, tuple(t1))
    w2 = TrackWindow(5, t2_dets)
    preds = [{"actions": 1, "looks": 0, "crosses": 1}, {"actions": 0, "looks": 1, "crosses": 0}]
    out = aggregate_by_frame([w1, w2], preds)

    assert set(out) == {0, 1, 2}
    assert len(out[0]) == 1 and len(out[1]) == 2 and len(out[2]) == 1     # frame 1 covered by both
    assert out[0][0].bbox == (0, 0, 10, 10) and out[0][0].track_id == 5
    assert out[1][0].crosses == 1 and out[1][1].crosses == 0              # w1 then w2 (order preserved)
    assert out[2][0].actions == 0 and out[2][0].looks == 1


# --------------------------------------------------------------------------- 7. lazy detector import


def test_detect_tracks_lazy_import(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)      # force ImportError on `import ultralytics`
    with pytest.raises(ImportError, match=r"infer"):
        detect_tracks(None, InferenceCfg(), DataCfg())


# --------------------------------------------------------------------------- 8. end-to-end (stub detector)


def _write_frames(dir_path, n, h=64, w=64):
    rng = np.random.default_rng(3)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(dir_path / f"frame_{i:03d}.png")


def test_run_inference_with_stub_detector(tmp_path, monkeypatch) -> None:
    cfg = _fast_cfg()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frames(frames_dir, n=6)

    def _stub_detect(source, icfg, dcfg):
        track = []
        for i, frame in enumerate(source):
            bbox = (5 + i, 5 + i, 40 + i, 55 + i)
            tight, context = crop_from_frame(frame, bbox, dcfg.context_scale)
            track.append(Detection(i, bbox, tight, context))
        return {1: track}

    monkeypatch.setattr(inf, "detect_tracks", _stub_detect)

    model = build_model(cfg, "full")
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    out_mp4 = tmp_path / "out.mp4"
    res = run_video_inference(cfg, video=frames_dir, checkpoint=ckpt, out_video=out_mp4, device=_CPU)

    assert res.n_frames == 6
    assert res.n_windows == 3                       # 6 - 4 + 1
    assert res.frame_results                        # non-empty
    assert res.out_video == out_mp4 and out_mp4.exists()


# --------------------------------------------------------------------------- 9. render smoke


def test_render_overlays_writes_video(tmp_path) -> None:
    import cv2

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frames(frames_dir, n=4, h=48, w=64)
    source = open_frame_source(frames_dir)

    frame_results = {0: [FramePrediction(1, (5, 5, 30, 40), 1, 0, 1)],
                     2: [FramePrediction(1, (6, 6, 31, 41), 0, 1, 0)]}
    out_mp4 = tmp_path / "render.mp4"
    render_overlays(source, frame_results, InferenceCfg(), out_mp4)

    assert out_mp4.exists()
    cap = cv2.VideoCapture(str(out_mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n > 0


# --------------------------------------------------------------------------- frame source sanity


def test_open_frame_source_dir(tmp_path) -> None:
    _write_frames(tmp_path, n=3, h=48, w=64)
    source = open_frame_source(tmp_path)
    assert len(source) == 3
    assert source.fps == 30.0 and source.width == 64 and source.height == 48
    frame = next(iter(source))
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8

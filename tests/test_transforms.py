"""Prompt 1.2 — crop/motion geometry parity tests.

Three kinds of checks (same shape as the 1.1 suite):
  * PARITY vs a verbatim transcription of the OLD motion construction (``_legacy_compute_motion``,
    from OLD ``PIE_sequence_Dataset_1.py`` lines 95-108) — the behavior-preserving oracle.
  * GOLDEN parity: ``process_record`` reproduces OLD ``_process_sequence`` output captured in
    ``tests/fixtures/golden/lmdb_process_record.pt`` (see ``tests/_capture/capture_lmdb_golden.py``).
  * HAND-CHECKED motion channel semantics, incl. the frame-0 dw/dh quirk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pedpredict.config import DataCfg
from pedpredict.data.transforms import (
    build_write_transforms,
    compute_motion,
    imagenet_normalize,
    load_rgb,
    process_record,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "lmdb_process_record.pt"


# --------------------------------------------------------------------------- legacy motion oracle


def _legacy_compute_motion(boxes_int):
    """Verbatim transcription of OLD _process_sequence motion math (lines 95-108)."""
    centers = torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes_int], dtype=torch.float32)
    widths = torch.tensor([x2 - x1 for x1, _, x2, _ in boxes_int], dtype=torch.float32)
    heights = torch.tensor([y2 - y1 for _, y1, _, y2 in boxes_int], dtype=torch.float32)
    dt = centers[1:] - centers[:-1]
    dt = torch.cat([dt[0:1], dt], dim=0)
    dw = torch.cat([widths[0:1], widths[1:] - widths[:-1]], dim=0)
    dh = torch.cat([heights[0:1], heights[1:] - heights[:-1]], dim=0)
    return torch.cat(
        [centers, dt, widths.unsqueeze(1), heights.unsqueeze(1), dw.unsqueeze(1), dh.unsqueeze(1)], dim=1
    )


_TRACKS = [
    [(50, 40, 90, 120), (55, 45, 99, 127), (60, 50, 108, 134), (62, 52, 112, 140)],
    [(0, 0, 10, 20), (0, 0, 10, 20)],                       # static box, T=2
    [(5, 5, 15, 25), (10, 8, 30, 40), (3, 2, 9, 11)],       # shrinking/jumping
]


@pytest.mark.parametrize("boxes", _TRACKS)
def test_compute_motion_matches_legacy_oracle(boxes) -> None:
    torch.testing.assert_close(compute_motion(boxes), _legacy_compute_motion(boxes), rtol=0, atol=0)


# --------------------------------------------------------------------------- hand-checked semantics


def test_motion_channel_semantics() -> None:
    """Frame-0: dx/dy hold the first *delta*; dw/dh hold the *raw* w0/h0 (the legacy quirk)."""
    boxes = [(50, 40, 90, 120), (55, 45, 99, 127)]  # w0=40 h0=80 cx0=70 cy0=80; w1=44 h1=82 cx1=77 cy1=86
    m = compute_motion(boxes)
    assert m[0].tolist() == [70.0, 80.0, 7.0, 6.0, 40.0, 80.0, 40.0, 80.0]  # dw0=w0=40, dh0=h0=80
    assert m[1].tolist() == [77.0, 86.0, 7.0, 6.0, 44.0, 82.0, 4.0, 2.0]    # dw1=44-40, dh1=82-80
    assert m.shape == (2, 8) and m.dtype == torch.float32


def test_compute_motion_requires_two_frames() -> None:
    with pytest.raises(ValueError, match="T >= 2"):
        compute_motion([(0, 0, 10, 10)])


# --------------------------------------------------------------------------- golden parity


def test_process_record_matches_golden(tmp_path) -> None:
    fx = torch.load(_FIXTURE, weights_only=False)  # trusted local fixture (nested dict)
    paths = []
    for t, arr in enumerate(fx["inputs"]["frames"]):
        p = tmp_path / f"frame_{t}.png"
        Image.fromarray(arr.numpy()).save(p)       # PNG = lossless -> decoded pixels == capture's
        paths.append(str(p))
    rec = {
        "images": paths,
        "bboxes": fx["inputs"]["bboxes"],
        "actions": fx["inputs"]["actions"],
        "looks": fx["inputs"]["looks"],
        "crosses": fx["inputs"]["crosses"],
    }
    out = process_record(rec, DataCfg())  # DataCfg defaults (context_scale=3.0, 128px) match the capture
    exp = fx["outputs"]
    torch.testing.assert_close(out.motions, exp["motions"], rtol=0, atol=0)
    torch.testing.assert_close(out.images_tight, exp["images_tight"], rtol=0, atol=1e-6)
    torch.testing.assert_close(out.images_context, exp["images_context"], rtol=0, atol=1e-6)
    assert out.actions.item() == exp["actions"].item()
    assert out.looks.item() == exp["looks"].item()
    assert out.crosses.item() == exp["crosses"].item()
    for lbl in (out.actions, out.looks, out.crosses):
        assert lbl.dtype == torch.long and lbl.ndim == 0


# --------------------------------------------------------------------------- transforms


def test_write_transforms_shapes_and_range() -> None:
    cfg = DataCfg()  # context_scale 3.0 -> context 384px
    tight, context = build_write_transforms(cfg)
    img = Image.fromarray(np.full((50, 30, 3), 200, dtype=np.uint8))
    t, c = tight(img), context(img)
    assert t.shape == (3, 128, 128) and c.shape == (3, 384, 384)
    assert t.dtype == torch.float32 and 0.0 <= float(t.min()) and float(t.max()) <= 1.0  # NOT normalized


def test_imagenet_normalize_uses_config_stats() -> None:
    norm = imagenet_normalize(DataCfg())
    assert list(norm.mean) == [0.485, 0.456, 0.406]
    assert list(norm.std) == [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- load_rgb fallback


def test_load_rgb_zero_pad_fallback(tmp_path) -> None:
    """A record path '7.png' missing on disk resolves to the zero-padded '000007.png' (PIE naming)."""
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "000007.png")
    img = load_rgb(tmp_path / "7.png")  # exact path absent; stem is digit -> zfill(6)
    assert img.size == (4, 4) and img.mode == "RGB"


def test_load_rgb_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "nope.png")

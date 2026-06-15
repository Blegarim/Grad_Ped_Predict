"""Crop/motion geometry tests (v2 motion contract).

The crop/JPEG path remains golden-parity vs the legacy capture; the motion feature deliberately
departs from it (hole audit A4/M9): frame-0 deltas are true zeros (the legacy raw-size dw/dh quirk
is removed) and ego-speed is the 9th stored channel. The golden fixture is still the source of
truth for everything the v2 contract does NOT change — rows t>=1 of the 8 bbox channels, and all
pixel outputs — so the tests derive the v2 expectation FROM the legacy golden rather than
re-capturing it.
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


# --------------------------------------------------------------------------- v2 vs legacy motion math


def _legacy_compute_motion(boxes_int):
    """Verbatim transcription of OLD _process_sequence motion math — kept as the t>=1 oracle."""
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


def _v2_from_legacy(legacy: torch.Tensor, ego: list[float] | None) -> torch.Tensor:
    """Derive the v2 expectation from the legacy 8-channel tensor (A4: zero frame-0 deltas; M9: ego)."""
    out = legacy.clone()
    out[0, 2:4] = 0.0  # dx, dy
    out[0, 6:8] = 0.0  # dw, dh (legacy held RAW w0/h0 here)
    t = out.shape[0]
    ego_t = torch.zeros(t, 1) if ego is None else torch.tensor(ego, dtype=torch.float32).unsqueeze(1)
    return torch.cat([out, ego_t], dim=1)


_TRACKS = [
    [(50, 40, 90, 120), (55, 45, 99, 127), (60, 50, 108, 134), (62, 52, 112, 140)],
    [(0, 0, 10, 20), (0, 0, 10, 20)],                       # static box, T=2
    [(5, 5, 15, 25), (10, 8, 30, 40), (3, 2, 9, 11)],       # shrinking/jumping
]


@pytest.mark.parametrize("boxes", _TRACKS)
def test_compute_motion_v2_vs_legacy_oracle(boxes) -> None:
    """v2 == legacy everywhere EXCEPT the four frame-0 delta cells, plus the ego channel."""
    torch.testing.assert_close(
        compute_motion(boxes), _v2_from_legacy(_legacy_compute_motion(boxes), None), rtol=0, atol=0
    )


# --------------------------------------------------------------------------- hand-checked semantics


def test_motion_channel_semantics_v2() -> None:
    """Frame-0 deltas (dx, dy, dw, dh) are true zeros; ego-speed is channel 8."""
    boxes = [(50, 40, 90, 120), (55, 45, 99, 127)]  # w0=40 h0=80 cx0=70 cy0=80; w1=44 h1=82 cx1=77 cy1=86
    m = compute_motion(boxes, ego_speed=[12.5, 13.0])
    assert m[0].tolist() == [70.0, 80.0, 0.0, 0.0, 40.0, 80.0, 0.0, 0.0, 12.5]
    assert m[1].tolist() == [77.0, 86.0, 7.0, 6.0, 44.0, 82.0, 4.0, 2.0, 13.0]
    assert m.shape == (2, 9) and m.dtype == torch.float32


def test_compute_motion_ego_default_zeros() -> None:
    """ego_speed=None (the raw-video inference path) fills the ego channel with zeros."""
    m = compute_motion(_TRACKS[0])
    assert m.shape == (4, 9)
    assert m[:, 8].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_compute_motion_requires_two_frames() -> None:
    with pytest.raises(ValueError, match="T >= 2"):
        compute_motion([(0, 0, 10, 10)])


def test_compute_motion_ego_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="ego_speed"):
        compute_motion(_TRACKS[0], ego_speed=[1.0, 2.0])


# --------------------------------------------------------------------------- golden parity


def test_process_record_matches_golden(tmp_path) -> None:
    """Pixels stay golden-parity; motions are the v2 derivation of the golden (A4/M9)."""
    fx = torch.load(_FIXTURE, weights_only=False)  # trusted local fixture (nested dict)
    paths = []
    for t, arr in enumerate(fx["inputs"]["frames"]):
        p = tmp_path / f"frame_{t}.png"
        Image.fromarray(arr.numpy()).save(p)       # PNG = lossless -> decoded pixels == capture's
        paths.append(str(p))
    ego = [10.0 + t for t in range(len(paths))]
    rec = {
        "images": paths,
        "bboxes": fx["inputs"]["bboxes"],
        "track_id": "ped_golden",
        "ego_speed": ego,
        "actions": fx["inputs"]["actions"],
        "looks": fx["inputs"]["looks"],
        "crosses": fx["inputs"]["crosses"],
    }
    out = process_record(rec, DataCfg())  # DataCfg defaults (context_scale=3.0, 128px) match the capture
    exp = fx["outputs"]
    torch.testing.assert_close(out.motions, _v2_from_legacy(exp["motions"], ego), rtol=0, atol=0)
    torch.testing.assert_close(out.images_tight, exp["images_tight"], rtol=0, atol=1e-6)
    torch.testing.assert_close(out.images_context, exp["images_context"], rtol=0, atol=1e-6)
    assert out.actions.item() == exp["actions"].item()
    assert out.looks.item() == exp["looks"].item()
    assert out.crosses.item() == exp["crosses"].item()
    assert out.track_id == "ped_golden"
    assert out.tte is None  # standard (non-benchmark) record
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

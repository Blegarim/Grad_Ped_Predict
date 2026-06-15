"""Offline augmentation tests (v2 flip contract).

Three kinds of checks:
  * GOLDEN-DERIVED parity: color/noise/erase still reproduce the OLD ``SequenceAugmenter`` exactly
    (``tests/fixtures/golden/augment_cases.pt``); the FLIP expectation is derived from the golden
    with the A4 fix applied (absolute ``cx`` reflected about the source width — the legacy version
    left it unreflected, silently corrupting flipped copies' geometry).
  * INVARIANCE: ``flip(flip(x)) == x`` and the flip↔motion-channel coupling (``dx`` negated, ``cx``
    reflected, everything else untouched).
  * PLAN: ``plan_oversample`` is deterministic and respects the minority multipliers (negatives excluded).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pedpredict.config import AugmentCfg, ConfigError, DataCfg, RootCfg, validate_config
from pedpredict.data.augment import (
    _FLIP_NEGATE_IDX,
    _FLIP_REFLECT_IDX,
    AugItem,
    AugmentedCropSequenceDataset,
    SequenceAugmenter,
    TransformName,
    plan_oversample,
    summarize_plan,
)
from pedpredict.data.transforms import ProcessedSample, compute_motion, process_record

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "augment_cases.pt"
_SOURCE_WIDTH = DataCfg().source_width


def _sample_from(d: dict) -> ProcessedSample:
    """Wrap the golden input tensors as a ProcessedSample (labels are irrelevant to the transforms)."""
    return ProcessedSample(
        images_tight=d["images_tight"].clone(),
        images_context=d["images_context"].clone(),
        motions=d["motions"].clone(),
        actions=torch.tensor(1),
        looks=torch.tensor(0),
        crosses=torch.tensor(1),
    )


@pytest.fixture(scope="module")
def golden() -> dict:
    return torch.load(_FIXTURE, weights_only=False)


@pytest.fixture
def augmenter() -> SequenceAugmenter:
    return SequenceAugmenter(AugmentCfg(), _SOURCE_WIDTH)


def _assert_sample_close(got: ProcessedSample, exp: dict, *, atol: float) -> None:
    torch.testing.assert_close(got.images_tight, exp["images_tight"], rtol=0, atol=atol)
    torch.testing.assert_close(got.images_context, exp["images_context"], rtol=0, atol=atol)
    torch.testing.assert_close(got.motions, exp["motions"], rtol=0, atol=atol)


# --------------------------------------------------------------------------- golden parity per transform


def test_flip_matches_golden_with_a4_reflection(golden, augmenter) -> None:
    """Images + negated dx stay golden-exact; cx is additionally reflected (the A4 fix)."""
    out = augmenter.horizontal_flip(_sample_from(golden["input"]))
    exp = golden["outputs"]["flip"]
    torch.testing.assert_close(out.images_tight, exp["images_tight"], rtol=0, atol=0)
    torch.testing.assert_close(out.images_context, exp["images_context"], rtol=0, atol=0)
    exp_motions = exp["motions"].clone()  # legacy flip output: dx negated, cx NOT reflected
    exp_motions[:, _FLIP_REFLECT_IDX] = _SOURCE_WIDTH - exp_motions[:, _FLIP_REFLECT_IDX]
    torch.testing.assert_close(out.motions, exp_motions, rtol=0, atol=0)


def test_color_matches_golden(golden, augmenter) -> None:
    out = augmenter.apply(_sample_from(golden["input"]), TransformName.COLOR, golden["seeds"]["color"])
    _assert_sample_close(out, golden["outputs"]["color"], atol=1e-6)


def test_motion_noise_matches_golden(golden, augmenter) -> None:
    out = augmenter.apply(_sample_from(golden["input"]), TransformName.NOISE, golden["seeds"]["noise"])
    _assert_sample_close(out, golden["outputs"]["noise"], atol=1e-6)
    # only motions change
    torch.testing.assert_close(out.images_tight, golden["input"]["images_tight"], rtol=0, atol=0)


def test_random_erase_matches_golden(golden, augmenter) -> None:
    out = augmenter.apply(_sample_from(golden["input"]), TransformName.ERASE, golden["seeds"]["erase"])
    _assert_sample_close(out, golden["outputs"]["erase"], atol=0)  # averaging is exact given fixed frames
    # motions untouched by erase (faithful image/motion desync)
    torch.testing.assert_close(out.motions, golden["input"]["motions"], rtol=0, atol=0)


# --------------------------------------------------------------------------- flip invariance / coupling


def test_flip_involution(golden, augmenter) -> None:
    s = _sample_from(golden["input"])
    back = augmenter.horizontal_flip(augmenter.horizontal_flip(s))
    torch.testing.assert_close(back.images_tight, s.images_tight, rtol=0, atol=0)
    torch.testing.assert_close(back.images_context, s.images_context, rtol=0, atol=0)
    torch.testing.assert_close(back.motions, s.motions, rtol=0, atol=0)


def test_flip_touches_exactly_dx_and_cx(golden, augmenter) -> None:
    s = _sample_from(golden["input"])
    out = augmenter.horizontal_flip(s)
    for ch in range(s.motions.shape[1]):
        if ch == _FLIP_NEGATE_IDX:
            expected = -s.motions[:, ch]
        elif ch == _FLIP_REFLECT_IDX:
            expected = _SOURCE_WIDTH - s.motions[:, ch]
        else:
            expected = s.motions[:, ch]
        torch.testing.assert_close(out.motions[:, ch], expected, rtol=0, atol=0)


def test_flip_indices_match_motion_channel_def(augmenter) -> None:
    """Guard the cross-module coupling: negate ``dx`` (2), reflect ``cx`` (0) in compute_motion's layout."""
    assert _FLIP_NEGATE_IDX == 2
    assert _FLIP_REFLECT_IDX == 0
    boxes = [(50, 40, 90, 120), (55, 45, 99, 127)]  # cx = 70, 77; dx = 0, +7 (v2 frame-0 zero)
    motions = compute_motion(boxes, ego_speed=[30.0, 31.0])
    assert motions[:, _FLIP_NEGATE_IDX].tolist() == [0.0, 7.0]
    s = ProcessedSample(
        images_tight=torch.zeros(2, 3, 4, 4),
        images_context=torch.zeros(2, 3, 4, 4),
        motions=motions,
        actions=torch.tensor(0),
        looks=torch.tensor(0),
        crosses=torch.tensor(0),
    )
    flipped = augmenter.horizontal_flip(s).motions
    assert flipped[:, _FLIP_NEGATE_IDX].tolist() == [-0.0, -7.0]
    assert flipped[:, _FLIP_REFLECT_IDX].tolist() == [float(_SOURCE_WIDTH - 70), float(_SOURCE_WIDTH - 77)]
    assert flipped[:, 8].tolist() == [30.0, 31.0]  # ego speed is flip-invariant (M9)


# --------------------------------------------------------------------------- oversampling plan


def _records(n_cross: int, n_look: int, n_neg: int) -> list[dict]:
    recs = [{"crosses": 1, "looks": 0} for _ in range(n_cross)]
    recs += [{"crosses": 0, "looks": 1} for _ in range(n_look)]
    recs += [{"crosses": 0, "looks": 0} for _ in range(n_neg)]
    return recs


def test_plan_counts_respect_multipliers() -> None:
    cfg = AugmentCfg()  # crosses x6, looks x3
    recs = _records(n_cross=5, n_look=4, n_neg=100)
    items = plan_oversample(recs, cfg)
    assert len(items) == 5 * cfg.crosses_multiplier + 4 * cfg.looks_multiplier
    # negatives are never a source
    neg_idx = set(range(9, 109))
    assert not any(it.record_index in neg_idx for it in items)


def test_plan_is_deterministic() -> None:
    recs = _records(3, 3, 10)
    assert plan_oversample(recs, AugmentCfg()) == plan_oversample(recs, AugmentCfg())


def test_plan_summary() -> None:
    recs = _records(2, 2, 5)
    items = plan_oversample(recs, AugmentCfg())
    summ = summarize_plan(recs, items)
    assert summ["total"] == len(items)
    assert summ["identity"] + summ["augmented"] == len(items)
    assert summ["crosses_pos_sources"] == 2 and summ["looks_pos_sources"] == 2


# --------------------------------------------------------------------------- dataset integration


def _png_record(dirpath: Path, n_frames: int = 4) -> dict:
    paths = []
    for t in range(n_frames):
        yy, xx = np.mgrid[0:120, 0:120]
        arr = np.stack([(xx + t) % 256, (yy + t) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        p = dirpath / f"f{t}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    bboxes = [[10.0 + t, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(n_frames)]
    return {
        "images": paths,
        "bboxes": bboxes,
        "track_id": "ped_aug",
        "ego_speed": [float(t) for t in range(n_frames)],
        "actions": 1,
        "looks": 1,
        "crosses": 1,
    }


def test_dataset_identity_and_flip(tmp_path) -> None:
    cfg = DataCfg()
    rec = _png_record(tmp_path)
    items = [AugItem(0, None, 0), AugItem(0, TransformName.FLIP, 0)]
    ds = AugmentedCropSequenceDataset([rec], items, cfg, AugmentCfg())
    assert len(ds) == 2

    plain = process_record(rec, cfg)
    identity, flipped = ds[0], ds[1]
    torch.testing.assert_close(identity.images_tight, plain.images_tight, rtol=0, atol=0)
    torch.testing.assert_close(flipped.images_tight, torch.flip(plain.images_tight, dims=[3]), rtol=0, atol=0)
    dx, plain_dx = flipped.motions[:, _FLIP_NEGATE_IDX], plain.motions[:, _FLIP_NEGATE_IDX]
    torch.testing.assert_close(dx, -plain_dx, rtol=0, atol=0)


# --------------------------------------------------------------------------- config validation


def test_augment_config_validation() -> None:
    bad_cfgs = [
        AugmentCfg(p_flip=1.5),
        AugmentCfg(n_augs_min=3, n_augs_max=2),
        AugmentCfg(n_augs_max=5),
        AugmentCfg(crosses_multiplier=0),
        AugmentCfg(motion_noise_std=-0.1),
        AugmentCfg(erase_n_frames=-1),
    ]
    for bad in bad_cfgs:
        with pytest.raises(ConfigError):
            validate_config(dataclasses.replace(RootCfg(), augment=bad))
    validate_config(dataclasses.replace(RootCfg(), augment=AugmentCfg()))  # defaults are valid

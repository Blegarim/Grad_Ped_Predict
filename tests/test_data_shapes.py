"""Prompt 1.1 — PIE sequence-window math tests.

Three kinds of checks:
  * PARITY vs a verbatim transcription of the OLD windowing loop (``_legacy_window_track``,
    copied from OLD ``scripts/generate_sequences.py`` lines 52-84) over synthetic tracks — the
    behavior-preserving oracle. Independent reimplementation must agree exactly.
  * HAND-CHECKED window math (counts, filter #2, future-window clip, the ``-1`` clamp).
  * DRIFT GATE (slow): the legacy ``sequences_{split}.pkl`` label counts must equal the captured
    golden fixture, which equals the documented CLAUDE.md stat table.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pedpredict.config import DataCfg
from pedpredict.data.pie_sequences import (
    clamp_to_binary,
    load_sequences,
    pie_data_opts,
    save_sequences,
    window_track,
    windows_from_pie,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "pie_sequences_counts.json"

# Small, hand-verifiable window geometry (NOT the production 20/3/30/2).
_TINY = dataclasses.replace(DataCfg(), seq_len=4, stride=2, future_offset=3, tol=1)


# --------------------------------------------------------------------------- legacy oracle


def _legacy_window_track(images, bboxes, actions, looks, crosses, cfg):
    """Verbatim transcription of OLD generate_sequences windowing (lines 52-84) — parity oracle."""
    crosses = [1 if v == 1 else 0 for v in crosses]
    n = len(images)
    out = []
    if n < cfg.seq_len:
        return out
    for start in range(0, n - cfg.seq_len + 1, cfg.stride):
        end = start + cfg.seq_len
        if any(crosses[start:end]):
            continue
        future_start = end
        future_end = min(end + cfg.future_offset + cfg.tol, n)
        out.append(
            {
                "images": images[start:end],
                "bboxes": bboxes[start:end],
                "actions": 1 if any(actions[future_start:future_end]) else 0,
                "looks": 1 if any(looks[future_start:future_end]) else 0,
                "crosses": 1 if any(crosses[future_start:future_end]) else 0,
            }
        )
    return out


def _track(n, *, actions=None, looks=None, crosses=None):
    """Build one synthetic track of length ``n`` with string image names and 1-d bboxes."""
    images = [f"f{i}" for i in range(n)]
    bboxes = [[float(i)] * 4 for i in range(n)]
    zeros = [0] * n
    return images, bboxes, actions or zeros, looks or zeros, crosses or zeros


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize(
    "track",
    [
        _track(3),  # shorter than seq_len -> no windows
        _track(8),  # all-zero labels
        _track(12, actions=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        _track(10, crosses=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # crossing during observation
        _track(8, crosses=[-1, -1, -1, -1, -1, -1, -1, -1]),  # clamp: -1 -> 0
        _track(9, looks=[0, 0, 0, 0, 0, 0, 0, 0, 1]),  # positive only in far future (clip)
    ],
)
def test_window_track_matches_legacy_oracle(track) -> None:
    images, bboxes, actions, looks, crosses = track
    new = window_track(images, bboxes, actions, looks, crosses, _TINY)
    old = _legacy_window_track(images, bboxes, actions, looks, crosses, _TINY)
    assert new == old


def test_windows_from_pie_matches_per_track_sum() -> None:
    """The full-dict path equals concatenating per-track windows (and unwraps PIE's [[v]] nesting)."""
    tracks = [_track(8, actions=[0, 0, 0, 0, 1, 0, 0, 0]), _track(3), _track(10)]
    sequences = {
        "image": [t[0] for t in tracks],
        "bbox": [t[1] for t in tracks],
        "actions": [[[v] for v in t[2]] for t in tracks],
        "looks": [[[v] for v in t[3]] for t in tracks],
        "cross": [[[v] for v in t[4]] for t in tracks],
    }
    expected: list = []
    for t in tracks:
        expected.extend(window_track(*t, _TINY))
    assert windows_from_pie(sequences, _TINY) == expected


# --------------------------------------------------------------------------- hand-checked math


def test_window_count_and_stride() -> None:
    images, bboxes, a, lo, c = _track(8)
    recs = window_track(images, bboxes, a, lo, c, _TINY)
    # range(0, 8 - 4 + 1, 2) -> starts 0, 2, 4
    assert len(recs) == 3
    assert all(len(r["images"]) == _TINY.seq_len for r in recs)
    assert all(len(r["bboxes"]) == _TINY.seq_len for r in recs)
    assert recs[0]["images"] == ["f0", "f1", "f2", "f3"]


def test_future_window_labeling_and_clip() -> None:
    # action positive at idx 6 only.
    images, bboxes, _, lo, c = _track(8)
    a = [0, 0, 0, 0, 0, 0, 1, 0]
    recs = window_track(images, bboxes, a, lo, c, _TINY)
    # window starts 0,2,4 -> future [4,8), [6,8), [8,8)=empty
    assert [r["actions"] for r in recs] == [1, 1, 0]
    assert [r["looks"] for r in recs] == [0, 0, 0]
    assert [r["crosses"] for r in recs] == [0, 0, 0]


def test_filter2_drops_crossing_during_observation() -> None:
    images, bboxes, a, lo, _ = _track(8)
    c = [0, 0, 1, 0, 0, 0, 0, 0]  # cross at idx 2
    recs = window_track(images, bboxes, a, lo, c, _TINY)
    # starts 0 (obs [0:4] has idx2) and 2 (obs [2:6] has idx2) dropped; only start 4 survives
    assert len(recs) == 1
    assert recs[0]["images"] == ["f4", "f5", "f6", "f7"]
    assert recs[0]["crosses"] == 0  # future window [8,8) empty


def test_minus_one_is_clamped_not_truthy() -> None:
    """An unclamped ``-1`` is truthy and would falsely fire filter #2 + the cross label."""
    images, bboxes, a, lo, _ = _track(8)
    c = [-1] * 8
    recs = window_track(images, bboxes, a, lo, c, _TINY)
    assert len(recs) == 3  # no window dropped
    assert all(r["crosses"] == 0 for r in recs)


def test_short_track_yields_no_windows() -> None:
    assert window_track(*_track(_TINY.seq_len - 1), _TINY) == []


def test_clamp_to_binary_value_mapping() -> None:
    assert clamp_to_binary([-1, 0, 1, 1, -1, 0]) == [0, 0, 1, 1, 0, 0]


# --------------------------------------------------------------------------- PIE-call surface


def test_pie_data_opts_matches_legacy() -> None:
    assert pie_data_opts(DataCfg()) == {
        "fstride": 1,
        "data_split_type": "default",
        "seq_type": "all",
        "height_rng": [0.0, float("inf")],
        "squarify_ratio": 0.0,
        "min_track_size": 10,
    }


# --------------------------------------------------------------------------- I/O isolation


def test_save_load_roundtrip(tmp_path) -> None:
    records = window_track(*_track(8, actions=[0, 0, 0, 0, 0, 1, 0, 0]), _TINY)
    out = save_sequences(records, tmp_path / "nested" / "seq.pkl")
    assert out.exists()
    assert load_sequences(out) == records


# --------------------------------------------------------------------------- drift gate


def _golden() -> dict:
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_counts_fixture_matches_documented_table() -> None:
    """The captured fixture must equal the CLAUDE.md stat table (cheap; no pkls needed)."""
    splits = _golden()["splits"]
    assert splits["train"]["N"] == 95684
    assert splits["val"]["N"] == 22665
    assert splits["test"]["N"] == 76048
    # crosses pos-rate ~2.6 / 2.5 / 2.8 %
    assert round(splits["train"]["crosses"] / splits["train"]["N"] * 100, 1) == 2.6
    assert round(splits["val"]["crosses"] / splits["val"]["N"] * 100, 1) == 2.5
    assert round(splits["test"]["crosses"] / splits["test"]["N"] * 100, 1) == 2.8


@pytest.mark.slow
def test_legacy_pkl_counts_match_fixture() -> None:
    """Regenerating a split that changes these counts is a behavior break. Skips if pkls absent."""
    golden = _golden()
    repo = Path(golden["src_repo"])
    for split, exp in golden["splits"].items():
        pkl = repo / f"sequences_{split}.pkl"
        if not pkl.exists():
            pytest.skip(f"legacy pickle not available: {pkl}")
        records = load_sequences(pkl)
        assert len(records) == exp["N"]
        assert sum(r["actions"] for r in records) == exp["actions"]
        assert sum(r["looks"] for r in records) == exp["looks"]
        assert sum(r["crosses"] for r in records) == exp["crosses"]

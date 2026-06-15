"""PIE sequence-window math tests (v2 labeling contract).

The v1 suite asserted parity against a transcription of the OLD windowing loop. The v2 contract
*deliberately* departs from it (hole audit): M3 state-at-end labels for actions/looks, M4 censor
filter, M6 track_id, M9 ego_speed, M5 benchmark windows. The legacy oracle is therefore retired;
these are HAND-CHECKED expectations plus the drift gate against the captured counts fixture.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pedpredict.config import DataCfg
from pedpredict.data.pie_sequences import (
    WindowStats,
    clamp_to_binary,
    load_sequences,
    pie_data_opts,
    save_sequences,
    window_track,
    windows_from_pie,
    windows_from_pie_benchmark,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "pie_sequences_counts.json"

# Small, hand-verifiable window geometry (NOT the production 20/3/30/2).
# future window span = future_offset + tol = 4 frames, required IN FULL by the M4 censor filter.
_TINY = dataclasses.replace(DataCfg(), seq_len=4, stride=2, future_offset=3, tol=1)
# benchmark geometry: 4-frame obs, TTE in [2, 5], overlap 0.5 -> step 2
_TINY_BM = dataclasses.replace(
    DataCfg(), benchmark_obs_len=4, benchmark_tte_min=2, benchmark_tte_max=5, benchmark_overlap=0.5
)


def _track(n, *, actions=None, looks=None, crosses=None):
    """Build one synthetic track of length ``n`` with string image names and 1-d bboxes."""
    images = [f"f{i}" for i in range(n)]
    bboxes = [[float(i)] * 4 for i in range(n)]
    zeros = [0] * n
    return images, bboxes, actions or zeros, looks or zeros, crosses or zeros


def _window(track, cfg=_TINY, *, track_id="ped_1", ego=None, stats=None):
    images, bboxes, actions, looks, crosses = track
    ego = ego if ego is not None else [float(i) for i in range(len(images))]
    return window_track(
        images, bboxes, actions, looks, crosses, cfg, track_id=track_id, ego_speed=ego, stats=stats
    )


# --------------------------------------------------------------------------- window geometry + M4 censor


def test_window_count_censor_filter() -> None:
    """starts 0,2,4 -> only start 0 has a fully-observed 4-frame future ([4,8) with n=8)."""
    stats = WindowStats()
    recs = _window(_track(8), stats=stats)
    assert len(recs) == 1
    assert recs[0]["images"] == ["f0", "f1", "f2", "f3"]
    assert (stats.emitted, stats.censored, stats.obs_crossing) == (1, 2, 0)


def test_censored_window_never_labeled_zero() -> None:
    """M4: a positive just past the truncated future must NOT appear as a 0-labeled record."""
    # v1 would emit start=4 with future [8,8)=empty -> crosses=0; v2 drops the window instead.
    images, bboxes, a, lo, _ = _track(8)
    c = [0, 0, 0, 0, 0, 0, 0, 1]  # cross at the last frame
    recs = window_track(images, bboxes, a, lo, c, _TINY, track_id="p", ego_speed=[0.0] * 8)
    # start 0: obs [0:4) clean, future [4:8) contains idx 7 -> crosses=1. starts 2,4: censored.
    assert [r["crosses"] for r in recs] == [1]


def test_m3_actions_looks_are_state_at_end_of_observation() -> None:
    """M3: actions/looks read signal[end-1] (last observed frame), NOT any() over the future."""
    images, bboxes, _, _, c = _track(8)
    a = [0, 0, 0, 1, 0, 0, 0, 0]   # walking at the window's last observed frame (idx 3)
    lo = [0, 0, 0, 0, 1, 0, 0, 0]  # looking only IN THE FUTURE (idx 4) -> must NOT fire
    recs = window_track(images, bboxes, a, lo, c, _TINY, track_id="p", ego_speed=[0.0] * 8)
    assert len(recs) == 1  # start 0 only (censor)
    assert recs[0]["actions"] == 1
    assert recs[0]["looks"] == 0


def test_crosses_stays_future_any() -> None:
    images, bboxes, a, lo, _ = _track(8)
    c = [0, 0, 0, 0, 0, 1, 0, 0]  # cross at idx 5, inside future [4,8) of the start-0 window
    recs = window_track(images, bboxes, a, lo, c, _TINY, track_id="p", ego_speed=[0.0] * 8)
    assert [r["crosses"] for r in recs] == [1]


def test_filter2_drops_crossing_during_observation() -> None:
    images, bboxes, a, lo, _ = _track(8)
    c = [0, 0, 1, 0, 0, 0, 0, 0]  # cross at idx 2
    stats = WindowStats()
    recs = window_track(
        images, bboxes, a, lo, c, _TINY, track_id="p", ego_speed=[0.0] * 8, stats=stats
    )
    # starts 0 and 2 observe idx 2 -> filter #2; start 4 is censored -> nothing emitted
    assert recs == []
    assert (stats.emitted, stats.censored, stats.obs_crossing) == (0, 1, 2)


def test_minus_one_is_clamped_not_truthy() -> None:
    """An unclamped ``-1`` is truthy and would falsely fire filter #2 + the cross label."""
    images, bboxes, a, lo, _ = _track(8)
    c = [-1] * 8
    recs = window_track(images, bboxes, a, lo, c, _TINY, track_id="p", ego_speed=[0.0] * 8)
    assert len(recs) == 1  # censor only; no filter-#2 drops
    assert all(r["crosses"] == 0 for r in recs)


def test_short_track_yields_no_windows_and_counts() -> None:
    stats = WindowStats()
    assert _window(_track(_TINY.seq_len - 1), stats=stats) == []
    assert stats.short_tracks == 1


def test_clamp_to_binary_value_mapping() -> None:
    assert clamp_to_binary([-1, 0, 1, 1, -1, 0]) == [0, 0, 1, 1, 0, 0]


# --------------------------------------------------------------------------- M6 track_id + M9 ego_speed


def test_record_carries_track_id_and_ego_slice() -> None:
    recs = _window(_track(8), track_id="set01_video_0001_1_5b", ego=[float(10 + i) for i in range(8)])
    assert len(recs) == 1
    assert recs[0]["track_id"] == "set01_video_0001_1_5b"
    assert recs[0]["ego_speed"] == [10.0, 11.0, 12.0, 13.0]  # observation slice [0:4)


def test_ego_speed_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="ego_speed"):
        _window(_track(8), ego=[0.0] * 5)


# --------------------------------------------------------------------------- full-dict path


def _pie_dict(tracks, *, event_idx=None, activity=None):
    """Wrap synthetic tracks in PIE's 'all'-path nesting (labels as [[v]], pid/obd_speed parallel)."""
    d = {
        "image": [t[0] for t in tracks],
        "bbox": [t[1] for t in tracks],
        "pid": [[[f"ped_{i}"]] * len(t[0]) for i, t in enumerate(tracks)],
        "obd_speed": [[[float(j)] for j in range(len(t[0]))] for t in tracks],
        "actions": [[[v] for v in t[2]] for t in tracks],
        "looks": [[[v] for v in t[3]] for t in tracks],
        "cross": [[[v] for v in t[4]] for t in tracks],
    }
    if event_idx is not None:
        d["event_frames_idx"] = event_idx
        d["activities"] = [[[v]] * len(t[0]) for v, t in zip(activity, tracks, strict=True)]
    return d


def test_windows_from_pie_matches_per_track_sum() -> None:
    tracks = [_track(8, actions=[0, 0, 0, 1, 0, 0, 0, 0]), _track(3), _track(10)]
    sequences = _pie_dict(tracks)
    expected: list = []
    for i, t in enumerate(tracks):
        expected.extend(
            _window(t, track_id=f"ped_{i}", ego=[float(j) for j in range(len(t[0]))])
        )
    assert windows_from_pie(sequences, _TINY) == expected


def test_windows_from_pie_accumulates_stats() -> None:
    stats = WindowStats()
    windows_from_pie(_pie_dict([_track(8), _track(3)]), _TINY, stats=stats)
    assert (stats.emitted, stats.censored, stats.short_tracks) == (1, 2, 1)


# --------------------------------------------------------------------------- M5 benchmark protocol


def test_benchmark_windows_hand_checked() -> None:
    """n=12, event at idx 10, TTE [2,5], obs 4, step 2 -> ends e=5 (tte 5) and e=7 (tte 3)."""
    tracks = [_track(12, actions=[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0])]
    sequences = _pie_dict(tracks, event_idx=[10], activity=[1])
    recs = windows_from_pie_benchmark(sequences, _TINY_BM)
    assert [r["tte"] for r in recs] == [5, 3]
    assert recs[0]["images"] == ["f2", "f3", "f4", "f5"]
    assert recs[1]["images"] == ["f4", "f5", "f6", "f7"]
    # event label for EVERY window of the track; actions = state at the last observed frame
    assert all(r["crosses"] == 1 for r in recs)
    assert [r["actions"] for r in recs] == [1, 1]
    assert recs[0]["track_id"] == "ped_0"
    assert recs[0]["ego_speed"] == [2.0, 3.0, 4.0, 5.0]


def test_benchmark_event_label_zero_for_non_crosser() -> None:
    sequences = _pie_dict([_track(12)], event_idx=[10], activity=[0])
    recs = windows_from_pie_benchmark(sequences, _TINY_BM)
    assert len(recs) == 2 and all(r["crosses"] == 0 for r in recs)


def test_benchmark_early_event_yields_no_windows() -> None:
    """An event too close to the track start leaves no room for a full observation window."""
    sequences = _pie_dict([_track(12)], event_idx=[2], activity=[1])
    assert windows_from_pie_benchmark(sequences, _TINY_BM) == []


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
    records = _window(_track(8, actions=[0, 0, 0, 1, 0, 0, 0, 0]))
    out = save_sequences(records, tmp_path / "nested" / "seq.pkl")
    assert out.exists()
    assert load_sequences(out) == records


# --------------------------------------------------------------------------- drift gate


def _golden() -> dict:
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_counts_fixture_matches_documented_table() -> None:
    """The captured fixture must equal the CLAUDE.md stat table (cheap; no pkls needed).

    v2 counts (regenerated under the M3/M4 labeling contract) — if sequence generation params or
    labeling change, re-run scripts/make_sequences.py + count_labels and update fixture + table.
    """
    splits = _golden()["splits"]
    for split in ("train", "val", "test"):
        exp = splits[split]
        assert exp["N"] > 0
        for task in ("actions", "looks", "crosses"):
            assert 0 <= exp[task] <= exp["N"]
        # crosses stays the (severely) imbalanced minority under any sane labeling
        assert exp["crosses"] / exp["N"] < 0.10


@pytest.mark.slow
def test_pkl_counts_match_fixture() -> None:
    """Regenerating a split that changes these counts is a contract break. Skips if pkls absent."""
    from pedpredict.config import PathsCfg
    from pedpredict.paths import resolve_paths

    golden = _golden()
    seq_dir = resolve_paths(PathsCfg()).sequences_dir
    for split, exp in golden["splits"].items():
        pkl = seq_dir / f"sequences_{split}.pkl"
        if not pkl.exists():
            pytest.skip(f"sequence pickle not available: {pkl}")
        records = load_sequences(pkl)
        assert len(records) == exp["N"]
        assert sum(r["actions"] for r in records) == exp["actions"]
        assert sum(r["looks"] for r in records) == exp["looks"]
        assert sum(r["crosses"] for r in records) == exp["crosses"]

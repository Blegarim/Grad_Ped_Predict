"""Scene features: right-of-way timing, traffic, crosswalk geometry and oracles from a hand-built scene."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pedpredict.eval.scene_features import (
    SCENE_GROUPS,
    VideoScene,
    WindowRef,
    check_alignment,
    match_windows,
    scene_features,
)
from pedpredict.eval.tree_study import SplitData


def _scene() -> VideoScene:
    s = VideoScene()
    for f in range(0, 101):                       # regular light: red until 79, green from 80
        state = 1.0 if f < 80 else 3.0
        s.lights[f].append(("L1", "regular", state, 100.0))
        s.light_states["L1"][f] = state
    for f in range(90, 101):                      # a small pedestrian signal appears at 90, green
        s.lights[f].append(("P1", "pedestrian", 3.0, 20.0))
        s.light_states["P1"][f] = 3.0
    for f in (81, 100):
        s.vehicles[f] += [500.0, 1500.0]
        s.vehicle_h[f] += [200.0, 50.0]
    s.crosswalks[100].append((900.0, 700.0, 1100.0, 760.0))
    s.speed.update({100: 30.0, 115: 20.0})        # 130 missing on purpose
    return s


def _features(frame: int, bbox=(950.0, 600.0, 990.0, 720.0)) -> dict[str, float]:
    ref = WindowRef("set01", "video_0001", frame, bbox, "1_1_1", {})
    attrs = {"1_1_1": {"intention_prob": "0.9", "intersection": "four-way", "num_lanes": "2", "signalized": "CS",
                       "traffic_direction": "TW"}}
    out = scene_features([ref], {("set01", "video_0001"): _scene()}, attrs, before_cx=[600.0])
    return {k: float(v[0]) for k, v in out.items()}


def test_right_of_way_timing() -> None:
    f = _features(100)
    assert f["r_n_lights"] == 1 and f["r_state"] == 3.0 and f["r_any_green"] == 1 and f["r_any_red"] == 0
    assert f["r_since_change"] == 21 and f["r_changed_recent"] == 0      # green held 80..100 = 21 frames
    assert _features(90)["r_changed_recent"] == 1                        # 11 frames after the change
    assert f["p_visible"] == 1 and f["p_state"] == 3.0 and f["p_since_change"] == 11   # lower bound: no change seen
    early = _features(50)
    assert early["r_since_change"] == 51 and early["r_changed_recent"] == 0 and early["p_visible"] == 0
    assert math.isnan(early["p_since_change"])


def test_traffic_layout_and_oracles() -> None:
    f = _features(100)
    assert f["t_n_vehicles"] == 2 and f["t_n_close"] == 0
    assert f["t_near_dx"] == np.float32(abs(500.0 - 970.0) / 1920.0) and f["t_near_h"] == np.float32(200.0 / 1080.0)
    assert f["t_near_dx_change"] == np.float32(abs(500 - 970) / 1920 - abs(500 - 600) / 1920)   # vs frame 81
    assert f["s_cw_visible"] == 1 and f["s_cw_dist"] == 0.0 and f["s_cw_in_x"] == 1       # foot inside the box
    assert (f["s_intersection"], f["s_num_lanes"], f["s_signalized"], f["s_traffic_direction"]) == (4, 2, 3, 1)
    assert f["h_intention"] == np.float32(0.9) and f["e_dv15"] == -10.0 and math.isnan(f["e_dv30"])
    assert math.isnan(_features(50)["t_near_dx"])
    assert set(_features(100)) == {n for g in SCENE_GROUPS.values() for n in g}


def test_alignment_check() -> None:
    refs = [WindowRef("s", "v", 1, (0, 0, 1, 1), "a", {"crosses": 1, "onset_offset": 5}),
            WindowRef("s", "v", 2, (0, 0, 1, 1), "b", {"crosses": 0, "onset_offset": -1})]
    good = {"track_id": np.array(["a", "b"]), "crosses": np.array([1, 0]), "onset_offset": np.array([5, -1])}
    assert check_alignment(refs, good) == []
    bad = {**good, "onset_offset": np.array([5, 7])}
    assert check_alignment(refs, bad) == ["onset_offset: 1 rows differ"]


def test_extra_groups_address_appended_columns() -> None:
    split = SplitData("x", np.zeros((3, 4 * 58), np.float32), 58, {"crosses": np.zeros(3)})
    split = split.with_groups({"R": np.ones((3, 2)), "H": np.full(3, 7.0)})
    assert split.features.shape == (3, 235)
    cols = split.columns("all+R+H")
    assert len(cols) == 235 and list(cols[-3:]) == [232, 233, 234]
    assert (split.features[:, split.columns("all+H")[-1]] == 7.0).all()
    assert split.take(np.array([0, 2])).columns("all+R").tolist() == split.columns("all+R").tolist()


def test_match_windows_recovers_a_shuffled_order() -> None:
    refs = [WindowRef("s", "v", f, (100.0 + 3 * f, 50.0, 110.0 + 3 * f, 90.0), "a",
                      {"crosses": 0, "onset_offset": -1, "future_observed": 100 - f}) for f in range(6)]
    refs.append(WindowRef("s", "v", 9, (400.0, 50.0, 410.0, 90.0), "b", {"crosses": 1, "onset_offset": 3,
                                                                          "future_observed": 40}))
    order = np.array([6, 2, 0, 5, 1, 4, 3])
    X = np.zeros((7, 20, 58), np.float32)
    for i, j in enumerate(order):
        r = refs[j]
        X[i, -1, 0], X[i, -1, 1] = (r.bbox[0] + r.bbox[2]) / 2 / 1920, (r.bbox[1] + r.bbox[3]) / 2 / 1080
    export = {"X": X, "track_id": np.array([refs[j].track_id for j in order]),
              "future_observed": np.array([refs[j].labels["future_observed"] for j in order]),
              "onset_offset": np.array([refs[j].labels["onset_offset"] for j in order]),
              "crosses": np.array([refs[j].labels["crosses"] for j in order])}
    perm = match_windows(refs, export)
    assert perm.tolist() == order.tolist()
    assert check_alignment([refs[j] for j in perm], export) == []
    export["X"][0, -1, 0] += 5 / 1920
    with pytest.raises(ValueError, match="px away"):
        match_windows(refs, export)


def test_history_features_count_bouts_and_looks() -> None:
    from pedpredict.eval.scene_features import PedestrianHistory, history_features

    frames = np.arange(100, 131)                      # walk 100-109, stand 110-130; look at 112-114 and 125
    standing = frames >= 110
    looking = np.isin(frames, [112, 113, 114, 125])
    peds = {("s", "v"): {"p": PedestrianHistory(frames, standing, looking)}}
    ref = WindowRef("s", "v", 128, (0, 0, 1, 1), "p", {})
    f = {k: float(v[0]) for k, v in history_features([ref], peds).items()}
    assert f["w_age"] == 28 and f["w_standing"] == 1 and f["w_stand_bout"] == 19 and f["w_walk_bout"] == 0
    assert f["w_stand_total"] == 19 and f["w_n_stops"] == 1
    assert f["l_since_look"] == 3 and f["l_looks_90"] == 4 and f["l_n_look_bouts"] == 2
    early = {k: float(v[0]) for k, v in history_features([WindowRef("s", "v", 105, (0, 0, 1, 1), "p", {})],
                                                          peds).items()}
    assert early["w_walk_bout"] == 6 and early["w_stand_bout"] == 0 and math.isnan(early["l_since_look"])
    missing = history_features([WindowRef("s", "v", 128, (0, 0, 1, 1), "nobody", {})], peds)
    assert all(math.isnan(float(v[0])) for v in missing.values())

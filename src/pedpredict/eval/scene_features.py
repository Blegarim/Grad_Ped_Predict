"""Scene context for exported windows, from PIE's own per-frame annotations (the scene audit's inputs).

The pose + motion vector every model has seen so far describes the pedestrian and the ego car's past speed. PIE also
annotates, frame by frame, what decides right-of-way: traffic lights with their state (and pedestrian signals),
crosswalks, every other vehicle, and per pedestrian the intersection layout. This module turns those annotations
into per-window features at the window's last observed frame ``t``, in the five groups the scene audit tests
(``outputs/diagnostics/scene_audit/PREREG.md``):

* ``R`` right-of-way and ``T`` traffic: a camera sees them, frames ``<= t`` only;
* ``S`` layout (map attributes) and crosswalk proximity (camera);
* ``H`` human-judged intention and ``E`` the ego car's next second: oracles that bound the problem, never inputs.

Windows are matched to frames through the sequence files the LMDBs were built from; :func:`check_alignment` proves
the match row for row before any feature is trusted. Missing values are NaN (the tree learner handles them).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pedpredict.data.pie_sequences import ONSET_FIELDS

__all__ = ["SCENE_GROUPS", "HISTORY_GROUPS", "WindowRef", "window_refs", "match_windows", "check_alignment",
           "VideoScene", "parse_video", "load_attributes", "scene_features", "parse_pedestrians", "history_features"]

SCENE_GROUPS: dict[str, tuple[str, ...]] = {
    "R": ("r_n_lights", "r_state", "r_since_change", "r_changed_recent", "r_any_red", "r_any_green",
          "p_visible", "p_state", "p_since_change"),
    "T": ("t_n_vehicles", "t_near_dx", "t_near_h", "t_near_dx_change", "t_n_close"),
    "S": ("s_intersection", "s_num_lanes", "s_signalized", "s_traffic_direction", "s_cw_visible", "s_cw_dist",
          "s_cw_in_x"),
    "H": ("h_intention",),
    "E": ("e_dv15", "e_dv30"),
}
#: Behaviour history beyond the 20-frame window (waiting audit, ``outputs/diagnostics/waiting_audit/PREREG.md``).
HISTORY_GROUPS: dict[str, tuple[str, ...]] = {
    "W": ("w_age", "w_standing", "w_stand_bout", "w_walk_bout", "w_stand_total", "w_n_stops"),
    "L": ("l_since_look", "l_looks_90", "l_n_look_bouts"),
}
_LOOK_WINDOW = 90     # "looking frames in the last 3 s"
_STATE = {"red": 1.0, "yellow": 2.0, "green": 3.0}
_INTERSECTION = {"midblock": 0.0, "T": 1.0, "T-left": 2.0, "T-right": 3.0, "four-way": 4.0}
_SIGNALIZED = {"n/a": 0.0, "C": 1.0, "S": 2.0, "CS": 3.0}
_DIRECTION = {"OW": 0.0, "TW": 1.0}
_RECENT = 20          # "changed within the window" = the last 20 frames
_CLOSE = 0.15         # "close" vehicles: within this share of the image width


@dataclass(frozen=True)
class WindowRef:
    """Where a window ends: video, last observed frame, the pedestrian's box there, and the label fields."""

    set_id: str
    video_id: str
    frame: int
    bbox: tuple[float, float, float, float]
    track_id: str
    labels: dict[str, int]


def window_refs(records: Sequence[Mapping]) -> list[WindowRef]:
    """One :class:`WindowRef` per sequence record (the order the LMDB writer used)."""
    refs = []
    for r in records:
        path = Path(r["images"][-1])
        labels = {k: int(r[k]) for k in ("crosses", *ONSET_FIELDS) if k in r}
        refs.append(WindowRef(path.parts[-3], path.parts[-2], int(path.stem), tuple(map(float, r["bboxes"][-1])),
                              str(r["track_id"]), labels))
    return refs


def match_windows(refs: Sequence[WindowRef], export: Mapping[str, np.ndarray], *, width: float = 1920.0,
                  height: float = 1080.0, tol_px: float = 0.05) -> np.ndarray:
    """``perm`` with ``refs[perm[i]]`` = export row ``i``, matched by content, not position.

    The LMDB writer need not keep the sequence file's order (the train build does not), so each export row is
    matched on its pedestrian, ``future_observed``, ``onset_offset`` and the pedestrian's box centre in the last
    frame (the export's image-normalized ``cx, cy`` channels times the image size; the writer int-truncates boxes
    first, ``transforms.crop_tight``, so the record side does too). Raises unless every row
    finds exactly one record within ``tol_px`` and no record is used twice.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for j, r in enumerate(refs):
        groups[(r.track_id, r.labels.get("future_observed"), r.labels.get("onset_offset"))].append(j)
    ints = [tuple(int(v) for v in r.bbox) for r in refs]
    centre = np.asarray([((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in ints])
    cx = np.asarray(export["X"][:, -1, 0], dtype=np.float64) * width
    cy = np.asarray(export["X"][:, -1, 1], dtype=np.float64) * height
    perm = np.full(len(cx), -1, dtype=np.int64)
    for i in range(len(cx)):
        key = (str(export["track_id"][i]), int(export["future_observed"][i]), int(export["onset_offset"][i]))
        cands = groups.get(key, [])
        if not cands:
            raise ValueError(f"match_windows: export row {i} {key} has no sequence record")
        d = np.abs(centre[cands, 0] - cx[i]) + np.abs(centre[cands, 1] - cy[i])
        k = int(np.argmin(d))
        if d[k] > tol_px:
            raise ValueError(f"match_windows: export row {i} {key}: nearest record is {d[k]:.2f} px away")
        perm[i] = cands[k]
    if len(np.unique(perm)) != len(perm):
        raise ValueError("match_windows: a sequence record matched more than one export row")
    return perm


def check_alignment(refs: Sequence[WindowRef], export: Mapping[str, np.ndarray]) -> list[str]:
    """The export row i must be sequence record i: same pedestrian, same labels. Returns the mismatches."""
    if len(refs) != len(export["track_id"]):
        return [f"length: {len(refs)} records vs {len(export['track_id'])} export rows"]
    problems = []
    tid = np.asarray([r.track_id for r in refs])
    if (bad := int((tid != np.asarray(export["track_id"]).astype(str)).sum())):
        problems.append(f"track_id: {bad} rows differ")
    for key in ("crosses", *ONSET_FIELDS):
        if key in export:
            ref = np.asarray([r.labels[key] for r in refs])
            ref = np.clip(ref, 0, 1) if key == "crosses" else ref
            if (bad := int((ref != np.asarray(export[key])).sum())):
                problems.append(f"{key}: {bad} rows differ")
    return problems


@dataclass
class VideoScene:
    """Per-frame scene objects of one video (only boxes not marked ``outside``)."""

    lights: dict[int, list[tuple[str, str, float, float]]] = field(default_factory=lambda: defaultdict(list))
    light_states: dict[str, dict[int, float]] = field(default_factory=lambda: defaultdict(dict))
    vehicles: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    vehicle_h: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    crosswalks: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=lambda: defaultdict(list))
    speed: dict[int, float] = field(default_factory=dict)


def _box_attrs(box: ET.Element) -> dict[str, str]:
    return {a.get("name"): (a.text or "") for a in box.findall("attribute")}


def parse_video(annotation_xml: str | Path, obd_xml: str | Path | None = None) -> VideoScene:
    """Traffic lights (``(id, type, state, area)`` per frame), vehicles, crosswalks and OBD speed of one video."""
    scene = VideoScene()
    for track in ET.parse(annotation_xml).getroot().iter("track"):
        label = track.get("label")
        if label not in ("traffic_light", "vehicle", "crosswalk"):
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            f = int(box.get("frame"))
            x1, y1, x2, y2 = (float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            if label == "traffic_light":
                a = _box_attrs(box)
                state = _STATE.get(a.get("state", ""), 0.0)
                scene.lights[f].append((a.get("id", ""), a.get("type", ""), state, (x2 - x1) * (y2 - y1)))
                scene.light_states[a.get("id", "")][f] = state
            elif label == "vehicle":
                scene.vehicles[f].append((x1 + x2) / 2.0)
                scene.vehicle_h[f].append(y2 - y1)
            else:
                scene.crosswalks[f].append((x1, y1, x2, y2))
    if obd_xml is not None and Path(obd_xml).exists():
        for frame in ET.parse(obd_xml).getroot():
            scene.speed[int(frame.get("id"))] = float(frame.get("OBD_speed"))
    return scene


def load_attributes(attribute_dir: str | Path) -> dict[str, dict[str, str]]:
    """PIE per-pedestrian attributes (``intention_prob``, ``intersection``, ``signalized`` ...) keyed by id."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(Path(attribute_dir).glob("set*/*_attributes.xml")):
        for ped in ET.parse(path).getroot().iter("pedestrian"):
            out[ped.get("id")] = dict(ped.attrib)
    return out


def _since_change(states: Mapping[int, float], t: int) -> tuple[float, float]:
    """Frames the light has held its state at ``t`` (walking back along its own track), and whether it changed in
    the last ``_RECENT`` frames. Stops at the first missing frame, so a long-held state is a lower bound."""
    if t not in states:
        return math.nan, math.nan
    now, back = states[t], 0
    while (t - back - 1) in states and states[t - back - 1] == now:
        back += 1
    changed = (t - back - 1) in states   # the walk stopped on a different state, not on a gap
    return float(back + 1), float(changed and back + 1 <= _RECENT)


def _right_of_way(scene: VideoScene, t: int) -> list[float]:
    lights = scene.lights.get(t, [])
    regular = [x for x in lights if x[1] == "regular"]
    walk = [x for x in lights if x[1] == "pedestrian"]
    out = [float(len(regular))]
    if regular:
        lid, _, state, _ = max(regular, key=lambda x: x[3])
        since, recent = _since_change(scene.light_states[lid], t)
        out += [state, since, recent, float(any(x[2] == 1.0 for x in regular)),
                float(any(x[2] == 3.0 for x in regular))]
    else:
        out += [0.0, math.nan, math.nan, 0.0, 0.0]
    if walk:
        lid, _, state, _ = max(walk, key=lambda x: x[3])
        out += [1.0, state, _since_change(scene.light_states[lid], t)[0]]
    else:
        out += [0.0, 0.0, math.nan]
    return out


def _nearest(scene: VideoScene, t: int, cx: float, width: float) -> tuple[float, float, int]:
    """(nearest vehicle's horizontal distance / width, its box height, index) at frame ``t``; NaN when none."""
    xs = scene.vehicles.get(t, [])
    if not xs:
        return math.nan, math.nan, -1
    d = np.abs(np.asarray(xs) - cx) / width
    i = int(np.argmin(d))
    return float(d[i]), scene.vehicle_h[t][i], i


def _traffic(scene: VideoScene, t: int, cx: float, width: float, height: float, cx_before: float) -> list[float]:
    near, h, _ = _nearest(scene, t, cx, width)
    before, _, _ = _nearest(scene, t - (_RECENT - 1), cx_before, width)
    xs = np.asarray(scene.vehicles.get(t, []))
    return [float(len(xs)), near, h / height if not math.isnan(h) else math.nan, near - before,
            float((np.abs(xs - cx) / width < _CLOSE).sum()) if len(xs) else 0.0]


def _layout(attrs: Mapping[str, str], scene: VideoScene, t: int, bbox: Sequence[float], width: float) -> list[float]:
    def num(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    x1, _, x2, y2 = bbox
    px, py = (x1 + x2) / 2.0, y2
    boxes = scene.crosswalks.get(t, [])
    dist = math.nan
    if boxes:
        dist = min(math.hypot(max(bx1 - px, 0.0, px - bx2), max(by1 - py, 0.0, py - by2)) / width
                   for bx1, by1, bx2, by2 in boxes)
    return [_INTERSECTION.get(attrs.get("intersection", ""), 5.0), num(attrs.get("num_lanes", "")),
            _SIGNALIZED.get(attrs.get("signalized", ""), math.nan), _DIRECTION.get(attrs.get("traffic_direction", ""),
                                                                              math.nan),
            float(bool(boxes)), dist, float(any(bx1 <= px <= bx2 for bx1, _, bx2, _ in boxes))]


def _future_speed(scene: VideoScene, t: int) -> list[float]:
    now = scene.speed.get(t)
    if now is None:
        return [math.nan, math.nan]
    return [scene.speed[t + k] - now if (t + k) in scene.speed else math.nan for k in (15, 30)]


def scene_features(refs: Sequence[WindowRef], scenes: Mapping[tuple[str, str], VideoScene],
                   attributes: Mapping[str, Mapping[str, str]], *, width: float = 1920.0, height: float = 1080.0,
                   before_cx: Sequence[float] | None = None) -> dict[str, np.ndarray]:
    """``{feature name: [N] float32}`` for every name in :data:`SCENE_GROUPS`, in window order.

    ``before_cx`` is each window's pedestrian centre 19 frames before ``t`` (its first frame), for the change in
    the nearest-vehicle distance; defaults to the centre at ``t``.
    """
    rows = []
    for i, ref in enumerate(refs):
        scene = scenes[(ref.set_id, ref.video_id)]
        attrs = attributes.get(ref.track_id, {})
        cx = (ref.bbox[0] + ref.bbox[2]) / 2.0
        cx0 = cx if before_cx is None else before_cx[i]
        rows.append(_right_of_way(scene, ref.frame) + _traffic(scene, ref.frame, cx, width, height, cx0)
                    + _layout(attrs, scene, ref.frame, ref.bbox, width)
                    + [float(attrs["intention_prob"]) if "intention_prob" in attrs else math.nan]
                    + _future_speed(scene, ref.frame))
    matrix = np.asarray(rows, dtype=np.float32)
    names = [n for group in SCENE_GROUPS.values() for n in group]
    assert matrix.shape[1] == len(names)
    return {name: matrix[:, j] for j, name in enumerate(names)}


@dataclass(frozen=True)
class PedestrianHistory:
    """One pedestrian's annotated frames (sorted) with per-frame standing / looking flags."""

    frames: np.ndarray
    standing: np.ndarray
    looking: np.ndarray


def parse_pedestrians(annotation_xml: str | Path) -> dict[str, PedestrianHistory]:
    """Per-frame ``action`` (standing / walking) and ``look`` of every pedestrian in one video. ``cross`` is
    deliberately not read: it is the label."""
    rows: dict[str, list[tuple[int, bool, bool]]] = defaultdict(list)
    for track in ET.parse(annotation_xml).getroot().iter("track"):
        if track.get("label") != "pedestrian":
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            a = _box_attrs(box)
            rows[a.get("id", "")].append((int(box.get("frame")), a.get("action") == "standing",
                                          a.get("look") == "looking"))
    out = {}
    for pid, r in rows.items():
        r.sort()
        arr = np.asarray(r, dtype=np.int64)
        out[pid] = PedestrianHistory(arr[:, 0], arr[:, 1].astype(bool), arr[:, 2].astype(bool))
    return out


def _bout(flags: np.ndarray, frames: np.ndarray) -> int:
    """Length (frames) of the run of equal ``flags`` ending at the last element, contiguous in frame index."""
    n = 1
    while n < len(flags) and flags[-n - 1] == flags[-1] and frames[-n] - frames[-n - 1] == 1:
        n += 1
    return n


def _history_row(h: PedestrianHistory | None, t: int) -> list[float]:
    if h is None:
        return [math.nan] * sum(len(g) for g in HISTORY_GROUPS.values())
    k = int(np.searchsorted(h.frames, t, side="right"))
    if k == 0:
        return [math.nan] * sum(len(g) for g in HISTORY_GROUPS.values())
    f, st, lk = h.frames[:k], h.standing[:k], h.looking[:k]
    bout = float(_bout(st, f))
    stops = float(np.sum(st[1:] & ~st[:-1]))
    looked = np.flatnonzero(lk)
    since_look = float(t - f[looked[-1]]) if len(looked) else math.nan
    recent = float(np.sum(lk & (f > t - _LOOK_WINDOW)))
    look_bouts = float(np.sum(lk[1:] & ~lk[:-1]) + (1 if lk[0] else 0))
    return [float(t - f[0]), float(st[-1]), bout if st[-1] else 0.0, bout if not st[-1] else 0.0,
            float(st.sum()), stops, since_look, recent, look_bouts]


def history_features(refs: Sequence[WindowRef], peds: Mapping[tuple[str, str], Mapping[str, PedestrianHistory]]
                     ) -> dict[str, np.ndarray]:
    """``{feature name: [N] float32}`` for :data:`HISTORY_GROUPS`, from frames ``<= t`` of each window's pedestrian."""
    rows = [_history_row(peds[(r.set_id, r.video_id)].get(r.track_id), r.frame) for r in refs]
    matrix = np.asarray(rows, dtype=np.float32)
    names = [n for g in HISTORY_GROUPS.values() for n in g]
    return {name: matrix[:, j] for j, name in enumerate(names)}

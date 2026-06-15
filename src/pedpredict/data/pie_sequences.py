"""PIE -> fixed-length sequence windows (v2 labeling contract).

Stage one of the offline pipeline. The module keeps the Phase-A split of concerns:

* :func:`pie_data_opts` / :func:`iter_pie_tracks` -- the only PIE-facing adapter (the dataset
  toolkit). ``iter_pie_tracks`` unwraps PIE's nested per-frame label lists into :class:`PieTrack`.
* :func:`clamp_to_binary` / :func:`window_track` / :func:`windows_from_pie` -- pure, deterministic
  windowing logic (no PIE, no disk): the unit-test surface.
* :func:`window_track_benchmark` / :func:`windows_from_pie_benchmark` -- the M5 benchmark-protocol
  eval windows (TTE-sampled relative to the PIE ``crossing_point``, labeled by the crossing EVENT).
* :func:`save_sequences` / :func:`load_sequences` -- isolated pickle I/O.
* :func:`generate_sequences` -- the thin orchestrator; ``scripts/make_sequences.py`` is the CLI.

v2 labeling contract (hole audit, deliberate behavior changes vs the v1/legacy windowing):

* **M3** -- ``actions`` / ``looks`` are the pedestrian's state at the END of the observation
  (``signal[end - 1]``), matching the literature's per-frame attribute semantics. Only ``crosses``
  remains a *future* label: ``any(crosses[end : end + future_offset + tol])``.
* **M4** -- windows whose future window would extend past the end of the track are DROPPED, not
  silently labeled 0 (right-censoring fix). :class:`WindowStats` counts them so the exclusion is
  reportable.
* **M6** -- every record carries the PIE pedestrian id as ``track_id`` (eval-side track aggregation).
* **M9** -- every record carries per-frame OBD ego-vehicle speed (``ego_speed``, km/h); the writer
  stores it as the 9th motion channel (see ``transforms.compute_motion``).
* **M5** -- benchmark mode emits fixed-TTE windows (obs ``benchmark_obs_len`` ending
  ``benchmark_tte_min..benchmark_tte_max`` frames before ``crossing_point``), labeled by the
  event-level ``activities`` flag — the externally-comparable protocol (Kotseruba et al. WACV 2021).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, TypedDict

from pedpredict.config.schema import DataCfg

__all__ = [
    "SequenceRecord",
    "BenchmarkRecord",
    "PieTrack",
    "WindowStats",
    "clamp_to_binary",
    "pie_data_opts",
    "iter_pie_tracks",
    "window_track",
    "windows_from_pie",
    "window_track_benchmark",
    "windows_from_pie_benchmark",
    "generate_sequences",
    "save_sequences",
    "load_sequences",
]


class SequenceRecord(TypedDict):
    """One fixed-length observation window plus its binary labels (v2 contract).

    ``actions``/``looks`` = state at the last observed frame (M3); ``crosses`` = any crossing in the
    fully-observed future window (M4 guarantees it is never truncated). ``track_id`` is the PIE
    pedestrian id (M6); ``ego_speed`` is per-frame OBD speed in km/h over the observation (M9).
    """

    images: list[str]
    bboxes: list[list[float]]
    track_id: str
    ego_speed: list[float]
    actions: int
    looks: int
    crosses: int


class BenchmarkRecord(SequenceRecord, total=False):
    """M5 benchmark-protocol record: adds the window's time-to-event (frames to ``crossing_point``)."""

    tte: int


class PieTrack(NamedTuple):
    """One unwrapped PIE track: flat per-frame signals + identity + ego speed."""

    track_id: str
    images: list[str]
    bboxes: list[list[float]]
    ego_speed: list[float]
    actions: list[int]
    looks: list[int]
    crosses: list[int]


@dataclass(slots=True)
class WindowStats:
    """Windowing accounting (M4): how many candidate windows each filter consumed.

    ``censored`` is the thesis-reportable number — windows that passed every other filter but whose
    future window extends past the end of the track (unobserved future, previously mislabeled 0).
    """

    emitted: int = 0
    censored: int = 0       # M4: dropped — future window not fully observed
    obs_crossing: int = 0   # filter #2: dropped — crossing during the observation window
    short_tracks: int = 0   # tracks shorter than seq_len (zero candidate windows)

    def as_dict(self) -> dict[str, int]:
        return {
            "emitted": self.emitted,
            "censored": self.censored,
            "obs_crossing": self.obs_crossing,
            "short_tracks": self.short_tracks,
        }


class _PieLike(Protocol):
    """Minimal structural type for a PIE instance (avoids importing the toolkit in this module)."""

    def generate_data_trajectory_sequence(self, image_set: str, **opts: Any) -> dict[str, list]: ...


def clamp_to_binary(signal: Sequence[int]) -> list[int]:
    """Map a label signal to ``{0, 1}``: only the literal value ``1`` is positive (so ``-1`` -> 0).

    Applied to ``crosses`` only (raw PIE crosses are ``{-1, 0, 1}``), never to ``actions``/``looks``
    (already ``{0, 1}``). The ``-1`` case is why ``clamp`` matters: ``any([-1])`` is truthy, so an
    unclamped ``-1`` would corrupt both the observation filter and the future label.
    """
    return [1 if v == 1 else 0 for v in signal]


def pie_data_opts(cfg: DataCfg) -> dict[str, Any]:
    """Build the exact ``data_opts`` dict passed to PIE (unchanged from v1 — same source tracks)."""
    height_max = float("inf") if cfg.height_max is None else cfg.height_max
    return {
        "fstride": cfg.fstride,
        "data_split_type": cfg.data_split_type,
        "seq_type": cfg.seq_type,
        "height_rng": [cfg.height_min, height_max],
        "squarify_ratio": cfg.squarify_ratio,
        "min_track_size": cfg.min_track_size,
    }


def iter_pie_tracks(sequences: dict[str, list]) -> Iterator[PieTrack]:
    """Unwrap PIE's per-track parallel lists into flat per-frame signals, one track at a time.

    PIE's 'all' path nests behaviour labels as ``[[v], [v], ...]``, keys crosses as ``cross``
    (singular), pedestrian ids as ``pid`` and OBD speed as ``obd_speed`` (km/h, same nesting).
    """
    for i in range(len(sequences["image"])):
        yield PieTrack(
            track_id=str(sequences["pid"][i][0][0]),
            images=sequences["image"][i],
            bboxes=sequences["bbox"][i],
            ego_speed=[float(frame[0]) for frame in sequences["obd_speed"][i]],
            actions=[frame[0] for frame in sequences["actions"][i]],
            looks=[frame[0] for frame in sequences["looks"][i]],
            crosses=[frame[0] for frame in sequences["cross"][i]],
        )


def _label_window(
    actions: Sequence[int],
    looks: Sequence[int],
    crosses: Sequence[int],
    end: int,
    cfg: DataCfg,
) -> dict[str, int]:
    """v2 labeling rule (M3), the single place it lives.

    ``actions``/``looks`` = state at the last observed frame (``end - 1``); ``crosses`` = any
    crossing in the future window ``[end, end + future_offset + tol)``. Callers guarantee the
    future window is fully observed (M4 censor filter), so no clipping happens here.
    """
    return {
        "actions": int(actions[end - 1]),
        "looks": int(looks[end - 1]),
        "crosses": int(any(crosses[end : end + cfg.future_offset + cfg.tol])),
    }


def window_track(
    images: Sequence[str],
    bboxes: Sequence[Sequence[float]],
    actions: Sequence[int],
    looks: Sequence[int],
    crosses: Sequence[int],
    cfg: DataCfg,
    *,
    track_id: str,
    ego_speed: Sequence[float],
    stats: WindowStats | None = None,
) -> list[SequenceRecord]:
    """Slide a ``seq_len`` window (step ``stride``) over one track -> labeled v2 records.

    Filters, in order: clamp crosses; skip tracks shorter than ``seq_len``; drop any window with a
    crossing *during observation* (filter #2); drop windows whose future window is not fully
    observed (M4 right-censoring). ``stats`` (optional) accumulates per-filter counts.
    """
    crosses = clamp_to_binary(crosses)
    n = len(images)
    if len(ego_speed) != n:
        raise ValueError(f"track {track_id!r}: ego_speed has {len(ego_speed)} frames, expected {n}")
    records: list[SequenceRecord] = []
    if n < cfg.seq_len:
        if stats is not None:
            stats.short_tracks += 1
        return records
    for start in range(0, n - cfg.seq_len + 1, cfg.stride):
        end = start + cfg.seq_len
        if any(crosses[start:end]):  # filter #2: pedestrian crosses during observation
            if stats is not None:
                stats.obs_crossing += 1
            continue
        if end + cfg.future_offset + cfg.tol > n:  # M4: unobserved future — drop, don't label 0
            if stats is not None:
                stats.censored += 1
            continue
        records.append(
            {
                "images": list(images[start:end]),
                "bboxes": [list(b) for b in bboxes[start:end]],
                "track_id": track_id,
                "ego_speed": [float(v) for v in ego_speed[start:end]],
                **_label_window(actions, looks, crosses, end, cfg),
            }
        )
        if stats is not None:
            stats.emitted += 1
    return records


def windows_from_pie(
    sequences: dict[str, list], cfg: DataCfg, stats: WindowStats | None = None
) -> list[SequenceRecord]:
    """Pure: a full PIE 'all' output dict -> all windowed records (PIE-free given the dict)."""
    records: list[SequenceRecord] = []
    for track in iter_pie_tracks(sequences):
        records.extend(
            window_track(
                track.images,
                track.bboxes,
                track.actions,
                track.looks,
                track.crosses,
                cfg,
                track_id=track.track_id,
                ego_speed=track.ego_speed,
                stats=stats,
            )
        )
    return records


def window_track_benchmark(
    track: PieTrack,
    event_idx: int,
    activity: int,
    cfg: DataCfg,
) -> list[BenchmarkRecord]:
    """M5: fixed-TTE windows for one track, labeled by the crossing EVENT.

    Observation windows of ``benchmark_obs_len`` frames whose last frame ``e`` lies
    ``benchmark_tte_min..benchmark_tte_max`` frames before the PIE ``crossing_point`` (``event_idx``,
    an index into the track). Successive windows step ``round(obs_len * (1 - benchmark_overlap))``.
    ``crosses`` = the event-level ``activity`` flag (``attributes['crossing'] > 0``) for EVERY window
    of the track; ``actions``/``looks`` keep the M3 state-at-end semantics. ``tte`` records each
    window's exact time-to-event for TTE-stratified analysis.
    """
    n = len(track.images)
    obs = cfg.benchmark_obs_len
    step = max(1, round(obs * (1.0 - cfg.benchmark_overlap)))
    records: list[BenchmarkRecord] = []
    e_lo = event_idx - cfg.benchmark_tte_max
    e_hi = event_idx - cfg.benchmark_tte_min
    for e in range(e_lo, e_hi + 1, step):
        start = e - obs + 1
        if start < 0 or e >= n:
            continue
        end = e + 1
        records.append(
            {
                "images": list(track.images[start:end]),
                "bboxes": [list(b) for b in track.bboxes[start:end]],
                "track_id": track.track_id,
                "ego_speed": [float(v) for v in track.ego_speed[start:end]],
                "actions": int(track.actions[e]),
                "looks": int(track.looks[e]),
                "crosses": int(activity),
                "tte": event_idx - e,
            }
        )
    return records


def windows_from_pie_benchmark(sequences: dict[str, list], cfg: DataCfg) -> list[BenchmarkRecord]:
    """Pure: a full PIE 'all' output dict -> benchmark-protocol records (M5).

    Consumes the event annotations the 'all' path already emits: ``event_frames_idx`` (index of
    ``crossing_point`` within each track) and ``activities`` (per-track event label).
    """
    records: list[BenchmarkRecord] = []
    for i, track in enumerate(iter_pie_tracks(sequences)):
        event_idx = int(sequences["event_frames_idx"][i])
        activity = int(sequences["activities"][i][0][0])
        records.extend(window_track_benchmark(track, event_idx, activity, cfg))
    return records


def generate_sequences(
    imdb: _PieLike,
    split: str,
    cfg: DataCfg,
    *,
    mode: str = "standard",
    stats: WindowStats | None = None,
) -> list[SequenceRecord]:
    """Orchestrate: query PIE for ``split`` with ``cfg`` opts, then window. No disk I/O.

    ``mode='standard'`` -> the training/eval windows (v2 labels); ``mode='benchmark'`` -> the M5
    fixed-TTE eval windows (``stats`` is ignored — no sliding-window filters apply).
    """
    sequences = imdb.generate_data_trajectory_sequence(split, **pie_data_opts(cfg))
    if mode == "benchmark":
        return windows_from_pie_benchmark(sequences, cfg)
    if mode != "standard":
        raise ValueError(f"generate_sequences: unknown mode {mode!r} (expected 'standard' or 'benchmark')")
    return windows_from_pie(sequences, cfg, stats=stats)


def save_sequences(records: list[SequenceRecord], path: str | Path) -> Path:
    """Pickle ``records`` to ``path`` (parent dirs created). I/O isolated from the windowing logic."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(records, handle)
    return out


def load_sequences(path: str | Path) -> list[SequenceRecord]:
    """Load a pickled sequence list written by :func:`save_sequences`."""
    with open(path, "rb") as handle:
        return pickle.load(handle)

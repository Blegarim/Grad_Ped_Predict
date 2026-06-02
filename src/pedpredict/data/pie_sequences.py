"""PIE -> fixed-length sequence windows (Prompt 1.1).

Ports OLD ``scripts/generate_sequences.py``. The legacy script fused three concerns into one
function with hardcoded literals and a ``__main__`` that dumped three pickles. Here they split so
the windowing logic is testable without the PIE toolkit or disk (B5 — canonicalize stage one):

* :func:`pie_data_opts` / :func:`iter_pie_tracks` -- the only PIE-facing adapter (the dataset
  toolkit). ``iter_pie_tracks`` unwraps PIE's nested per-frame label lists.
* :func:`clamp_to_binary` / :func:`window_track` / :func:`windows_from_pie` -- pure, deterministic
  windowing logic (no PIE, no disk): the unit-test surface.
* :func:`save_sequences` / :func:`load_sequences` -- isolated pickle I/O.
* :func:`generate_sequences` -- the thin orchestrator; ``scripts/make_sequences.py`` is the CLI.

Behavior is preserved *exactly* vs the legacy windowing: labels are integers and the PIE 'all'
path is deterministic (sorted iteration, no RNG), so parity is exact equality, not float
tolerance. Every window/source parameter comes from :class:`DataCfg` (no literals). The legacy
``has_onset`` dead code is dropped; the future-window labeling rule lives in one helper
(:func:`_label_future_window`) so an onset-based scheme can replace it in Phase B without touching
the windowing loop.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict

from pedpredict.config.schema import DataCfg

__all__ = [
    "SequenceRecord",
    "clamp_to_binary",
    "pie_data_opts",
    "iter_pie_tracks",
    "window_track",
    "windows_from_pie",
    "generate_sequences",
    "save_sequences",
    "load_sequences",
]


class SequenceRecord(TypedDict):
    """One fixed-length observation window plus its future-window binary labels."""

    images: list[str]
    bboxes: list[list[float]]
    actions: int
    looks: int
    crosses: int


class _PieLike(Protocol):
    """Minimal structural type for a PIE instance (avoids importing the toolkit in this module)."""

    def generate_data_trajectory_sequence(self, image_set: str, **opts: Any) -> dict[str, list]: ...


def clamp_to_binary(signal: Sequence[int]) -> list[int]:
    """Map a label signal to ``{0, 1}``: only the literal value ``1`` is positive (so ``-1`` -> 0).

    Mirrors OLD ``generate_sequences.clamp_to_binary``; applied to ``crosses`` only (raw PIE
    crosses are ``{-1, 0, 1}``), never to ``actions``/``looks`` (already ``{0, 1}``). The ``-1``
    case is why ``clamp`` matters: ``any([-1])`` is truthy, so an unclamped ``-1`` would corrupt
    both the observation filter and the future label.
    """
    return [1 if v == 1 else 0 for v in signal]


def pie_data_opts(cfg: DataCfg) -> dict[str, Any]:
    """Build the exact ``data_opts`` dict the legacy script passed to PIE (the PIE-call parity surface)."""
    height_max = float("inf") if cfg.height_max is None else cfg.height_max
    return {
        "fstride": cfg.fstride,
        "data_split_type": cfg.data_split_type,
        "seq_type": cfg.seq_type,
        "height_rng": [cfg.height_min, height_max],
        "squarify_ratio": cfg.squarify_ratio,
        "min_track_size": cfg.min_track_size,
    }


def iter_pie_tracks(
    sequences: dict[str, list],
) -> Iterator[tuple[list[str], list[list[float]], list[int], list[int], list[int]]]:
    """Unwrap PIE's per-track parallel lists into flat per-frame signals, one track at a time.

    PIE's 'all' path nests behaviour labels as ``[[v], [v], ...]`` and keys crosses as ``cross``
    (singular). This flattens ``frame[0] for frame in ...`` exactly as the legacy code did.
    """
    for i in range(len(sequences["image"])):
        images = sequences["image"][i]
        bboxes = sequences["bbox"][i]
        actions = [frame[0] for frame in sequences["actions"][i]]
        looks = [frame[0] for frame in sequences["looks"][i]]
        crosses = [frame[0] for frame in sequences["cross"][i]]
        yield images, bboxes, actions, looks, crosses


def _label_future_window(
    actions: Sequence[int],
    looks: Sequence[int],
    crosses: Sequence[int],
    end: int,
    n: int,
    cfg: DataCfg,
) -> dict[str, int]:
    """Label the future window ``[end, min(end + future_offset + tol, n))`` via ``any(...)`` per task.

    The single place the labeling rule lives -- swap this for an onset-based scheme in Phase B
    without touching the windowing loop. An empty future window (``end == n``) yields all zeros.
    """
    future_end = min(end + cfg.future_offset + cfg.tol, n)
    return {
        "actions": int(any(actions[end:future_end])),
        "looks": int(any(looks[end:future_end])),
        "crosses": int(any(crosses[end:future_end])),
    }


def window_track(
    images: Sequence[str],
    bboxes: Sequence[Sequence[float]],
    actions: Sequence[int],
    looks: Sequence[int],
    crosses: Sequence[int],
    cfg: DataCfg,
) -> list[SequenceRecord]:
    """Slide a ``seq_len`` window (step ``stride``) over one track -> labeled records.

    Preserves the legacy logic exactly: clamp crosses; skip tracks shorter than ``seq_len``;
    drop any window with a crossing *during observation* (legacy filter #2); label from the
    future window.
    """
    crosses = clamp_to_binary(crosses)
    n = len(images)
    records: list[SequenceRecord] = []
    if n < cfg.seq_len:
        return records
    for start in range(0, n - cfg.seq_len + 1, cfg.stride):
        end = start + cfg.seq_len
        if any(crosses[start:end]):  # filter #2: pedestrian crosses during observation
            continue
        labels = _label_future_window(actions, looks, crosses, end, n, cfg)
        records.append(
            {
                "images": list(images[start:end]),
                "bboxes": [list(b) for b in bboxes[start:end]],
                **labels,
            }
        )
    return records


def windows_from_pie(sequences: dict[str, list], cfg: DataCfg) -> list[SequenceRecord]:
    """Pure: a full PIE 'all' output dict -> all windowed records (PIE-free given the dict)."""
    records: list[SequenceRecord] = []
    for images, bboxes, actions, looks, crosses in iter_pie_tracks(sequences):
        records.extend(window_track(images, bboxes, actions, looks, crosses, cfg))
    return records


def generate_sequences(imdb: _PieLike, split: str, cfg: DataCfg) -> list[SequenceRecord]:
    """Orchestrate: query PIE for ``split`` with ``cfg`` opts, then window. No disk I/O."""
    sequences = imdb.generate_data_trajectory_sequence(split, **pie_data_opts(cfg))
    return windows_from_pie(sequences, cfg)


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

"""Scene context for the deep net: encode the scene audit's per-window features, and look a window's row up.

The scene audit (``outputs/diagnostics/scene_audit/PREREG.md``) gave the boosted trees lights, traffic and layout as
extra columns, raw: categories as integer codes, missing values as NaN, counts unscaled. A network needs them
encoded, so :func:`encode_scene` turns them into a dense block, with the statistics fitted on the training windows
only:

* numeric features are standardized (``(x - mean) / std``, clipped to ``+-clip``), and a missing value becomes 0
  plus a 0/1 "was missing" column (one per feature that is ever missing in training);
* categorical features (light states, intersection type, signalization, traffic direction) become one-hot columns,
  and a missing or unseen category is all zeros.

``scripts/build_scene_sidecar.py`` writes that block with each window's key to one sidecar file for every split.
At read time :class:`SceneLookup` finds a window's row by its pedestrian (``track_id``, which carries the PIE set and
video) and two of its onset fields (``future_observed``, ``onset_offset``), a key that is unique per window except
where PIE splits one pedestrian into segments. Those collisions are resolved by the last frame's box centre, with a
horizontally flipped copy matched at its reflection. Offline-augmented copies keep the key of the window they were
made from, and every scene feature is unchanged by a horizontal flip (distances are absolute values), so a copy
reads its source's scene.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pedpredict.paths import find_project_root

__all__ = ["CATEGORICAL", "SCENE_KEY_FIELDS", "encode_scene", "SceneLookup", "open_scene_lookup"]

#: Integer-coded categories (codes from ``eval.scene_features``) and the codes each one-hot block covers.
CATEGORICAL: dict[str, tuple[int, ...]] = {
    "r_state": (0, 1, 2, 3),            # largest regular light: none / red / yellow / green
    "p_state": (0, 1, 2, 3),            # largest pedestrian light: none / red / yellow / green
    "s_intersection": (0, 1, 2, 3, 4, 5),   # midblock / T / T-left / T-right / four-way / unknown
    "s_signalized": (0, 1, 2, 3),       # n/a / C / S / CS
    "s_traffic_direction": (0, 1),      # one-way / two-way
}
#: The per-window fields (besides ``track_id``) that key a sidecar row; both come from the S1 onset annotation.
SCENE_KEY_FIELDS: tuple[str, ...] = ("future_observed", "onset_offset")
_MATCH_TOL_PX = 2.0   # stored copies differ from their source by a flip or N(0, 0.02 px) noise, never more


def encode_scene(raw: Mapping[str, np.ndarray], names: Sequence[str], fit_rows: np.ndarray, *,
                 clip: float = 5.0) -> tuple[np.ndarray, dict]:
    """``raw[name] [N]`` -> ``(block [N, K] float32, spec)``; statistics come from ``raw[...][fit_rows]`` only.

    ``spec`` records the encoded column names and the fitted means/stds, so the block can be explained and
    re-derived. Column order: for each name in ``names``, its one-hot block (categorical) or its standardized value
    followed by its missing flag (numeric, flag only if the feature is ever missing in the fit rows).
    """
    columns, col_names = [], []
    spec: dict = {"names": col_names, "clip": clip, "mean": {}, "std": {}, "categorical": {}}
    for name in names:
        x = np.asarray(raw[name], dtype=np.float64)
        if name in CATEGORICAL:
            codes = CATEGORICAL[name]
            spec["categorical"][name] = list(codes)
            for c in codes:
                columns.append((x == c).astype(np.float32))   # NaN or an unseen code -> all zeros
                col_names.append(f"{name}={c}")
            continue
        fit = x[fit_rows]
        mean = float(np.nanmean(fit)) if np.isfinite(fit).any() else 0.0
        std = float(np.nanstd(fit)) if np.isfinite(fit).any() else 0.0
        std = std if std > 1e-8 else 1.0                       # a constant feature stays 0, never inf
        spec["mean"][name], spec["std"][name] = mean, std
        missing = ~np.isfinite(x)
        z = np.clip((np.where(missing, mean, x) - mean) / std, -clip, clip)
        columns.append(z.astype(np.float32))
        col_names.append(name)
        if (~np.isfinite(fit)).any():
            columns.append(missing.astype(np.float32))
            col_names.append(f"{name}_missing")
    return np.stack(columns, axis=1).astype(np.float32), spec


class SceneLookup:
    """One sidecar file in memory: the encoded block plus a key index; :meth:`vector` returns a window's row."""

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as z:
            self.features = torch.from_numpy(np.ascontiguousarray(z["features"], dtype=np.float32))
            track_id = z["track_id"].astype(str)
            keys = [z[k].astype(np.int64) for k in SCENE_KEY_FIELDS]
            self._cx = z["cx"].astype(np.float64)
            self._cy = z["cy"].astype(np.float64)
            self.spec = json.loads(str(z["spec"]))
        self.width = int(self.features.shape[1])
        self.source_width = float(self.spec.get("source_width", 1920.0))
        self._index: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for i, key in enumerate(zip(track_id, *keys, strict=True)):
            self._index[(str(key[0]), int(key[1]), int(key[2]))].append(i)

    def row(self, track_id: str, future_observed: int, onset_offset: int, cx: float, cy: float) -> int:
        """The sidecar row of the window with this key whose last-frame box centre is at (or mirrors) ``cx, cy``."""
        key = (str(track_id), int(future_observed), int(onset_offset))
        cands = self._index.get(key)
        if not cands:
            raise KeyError(f"SceneLookup: no scene row for window {key} - the sidecar does not cover this LMDB dir")
        c = np.asarray(cands)
        dx = np.minimum(np.abs(self._cx[c] - cx), np.abs(self._cx[c] - (self.source_width - cx)))
        d = dx + np.abs(self._cy[c] - cy)
        k = int(np.argmin(d))
        if d[k] > _MATCH_TOL_PX:
            raise KeyError(f"SceneLookup: window {key} matches no row's box (nearest {d[k]:.2f} px away)")
        return int(c[k])

    def vector(self, track_id: str, future_observed: int, onset_offset: int, cx: float, cy: float) -> Tensor:
        """``[width]`` float32: the window's encoded scene."""
        return self.features[self.row(track_id, future_observed, onset_offset, cx, cy)]


_CACHE: dict[str, SceneLookup] = {}


def open_scene_lookup(path: str | Path) -> SceneLookup:
    """Per-process cached :class:`SceneLookup` (a relative path resolves against the project root).

    Datasets hold only the path, so the block never travels in a worker pickle; each process loads it once.
    """
    p = Path(path)
    if not p.is_absolute():
        p = find_project_root() / p
    key = str(p)
    if key not in _CACHE:
        if not p.is_file():
            raise FileNotFoundError(f"data.scene_features sidecar not found: {p} - run scripts/build_scene_sidecar.py")
        _CACHE[key] = SceneLookup(p)
    return _CACHE[key]

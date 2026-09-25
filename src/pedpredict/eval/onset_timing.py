"""Onset-timing evaluation — does a model order windows by *when* the person starts crossing?

Check (c) of docs/RECIPE_V2_PLAN.md, and the timing instrument for the onset arms after it. The binary
crossing metrics score every window against one question — "does a crossing start within 32 frames?" —
so a model that has learned "this person crosses in ~40 frames" is scored as a false alarm and whatever
timing it learned is invisible. This module scores the same predictions against the S1 onset fields:

* **per-horizon AUC** at several horizons ``H``, each over the windows whose answer is knowable at ``H``
  — exactly the readout-target rule (:func:`pedpredict.data.onset_target.readout_targets`), so a window
  whose future ran out before ``H`` is excluded rather than counted as a non-crosser;
* **score by true onset group** — a timing-aware score should fall off as the crossing gets further away;
* **negatives split by kind** — "crosses within 32" against later crossers, against windows with no
  crossing seen, and against people who already crossed, separately (the binary label lumps all three).

Two halves: a **dump** (model -> per-window arrays; research PC, needs data + checkpoint) and a **report**
(arrays -> numbers; runs anywhere). The threshold grid and F1 tie-breaking are the ones ``evaluate.py``
uses, so a tuned F1 here is comparable with ``eval_log.csv``.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from pedpredict.config.schema import RootCfg
from pedpredict.data.collate import collate_sequences
from pedpredict.data.feature_cache import open_chunk_features
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.onset_target import OnsetSpec, readout_targets
from pedpredict.data.pie_sequences import ONSET_FIELDS
from pedpredict.data.pose import pose_motion_transform
from pedpredict.models.registry import forward_model
from pedpredict.training.metrics import _best_f1_threshold, _sweep
from pedpredict.utils.amp import autocast_ctx
from pedpredict.utils.memory import free_cuda

__all__ = [
    "LABEL_KEYS",
    "collate_with_track_ids",
    "dump_loaders",
    "dump_predictions",
    "save_dump",
    "load_dump",
    "hazard_horizon_prob",
    "onset_groups",
    "group_order",
    "horizon_targets",
    "timing_report",
    "format_report",
]

#: Per-window label arrays every dump carries (``crosses`` + the three S1 onset fields).
LABEL_KEYS: tuple[str, ...] = ("crosses", *ONSET_FIELDS)
_EDGES: tuple[int, ...] = (0, 16, 32, 64, 96)

Arrays = dict[str, np.ndarray]


# --------------------------------------------------------------------------- dump (model -> arrays)


def collate_with_track_ids(batch: list[dict], *, max_seq_len: int, motion_dim: int) -> tuple:
    """The standard collate, plus the per-window ``track_id`` strings it normally drops."""
    return (*collate_sequences(batch, max_seq_len=max_seq_len, motion_dim=motion_dim), [b["track_id"] for b in batch])


def dump_loaders(cfg: RootCfg, chunk_paths: Iterable[str], device: torch.device) -> Iterator[DataLoader]:
    """Stable-order eval loaders (as ``evaluate._eval_chunk_loaders``) whose batches keep ``track_id``."""
    collate = functools.partial(
        collate_with_track_ids, max_seq_len=cfg.data.max_seq_len, motion_dim=cfg.data.motion_dim
    )
    pose_transform = pose_motion_transform(cfg)
    for path in chunk_paths:
        dataset = LMDBChunkDataset.from_config(
            path, cfg.data, pose_transform=pose_transform, features=open_chunk_features(cfg, path)
        )
        yield DataLoader(
            dataset, batch_size=cfg.eval.batch_size, shuffle=False, num_workers=cfg.eval.num_workers,
            pin_memory=device.type == "cuda", collate_fn=collate,
        )
        dataset.close()
        free_cuda(device)


def _positive_prob(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()


def dump_predictions(
    model: nn.Module,
    loaders: Iterable[Iterable[tuple]],
    device: torch.device,
    *,
    use_amp: bool = False,
    forward: Callable[..., dict[str, torch.Tensor]] = forward_model,
) -> Arrays:
    """Run ``model`` and keep, per window: ``p_frame`` (the binary head), ``p_readout`` + ``hazard_logits``
    (when the onset head exists), the labels and ``track_id``. Labels must include the S1 onset fields."""
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}

    def keep(key: str, value: np.ndarray) -> None:
        parts.setdefault(key, []).append(value)

    with torch.inference_mode():
        for loader in loaders:
            for tight, context, motions, labels, track_ids in loader:
                missing = [k for k in LABEL_KEYS if k not in labels]
                if missing:
                    raise KeyError(f"dump_predictions: batch labels lack {missing} — pre-S1 chunks? Backfill them.")
                with autocast_ctx(use_amp, device.type):
                    out = forward(model, tight.to(device), context.to(device), motions.to(device))
                keep("p_frame", _positive_prob(out["crosses_frame"]))
                if "crosses_readout" in out:
                    keep("p_readout", _positive_prob(out["crosses_readout"]))
                    keep("hazard_logits", out["crosses_hazard"].float().cpu().numpy())
                for key in LABEL_KEYS:
                    value = labels[key].cpu().numpy()
                    keep(key, np.clip(value, 0, 1) if key == "crosses" else value)
                keep("track_id", np.asarray(track_ids, dtype=str))
    return {key: np.concatenate(chunks) for key, chunks in parts.items()}


def save_dump(path: str | Path, arrays: Arrays, meta: dict[str, object]) -> Path:
    """Compressed ``.npz`` with the arrays plus a JSON ``meta`` record (checkpoint, split, geometry, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrays)
    return path


def load_dump(path: str | Path) -> tuple[Arrays, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "meta"}
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    return arrays, meta


# --------------------------------------------------------------------------- labels and groups


def hazard_horizon_prob(hazard_logits: np.ndarray, horizon_bins: int) -> np.ndarray:
    """``P(onset within the first horizon_bins bins) = 1 - prod(1 - h_k)``, computed in log space.

    The numpy twin of :func:`pedpredict.models.heads.hazard_to_horizon_logits`: ``log(1 - sigmoid(z))`` is
    ``-softplus(z)``, so the product is ``exp(-sum softplus(z_k))``.
    """
    logits = np.asarray(hazard_logits, dtype=np.float64)
    if not 1 <= horizon_bins <= logits.shape[1]:
        raise ValueError(f"horizon_bins={horizon_bins} outside [1, K={logits.shape[1]}]")
    return -np.expm1(-np.logaddexp(0.0, logits[:, :horizon_bins]).sum(axis=1))


def group_order(edges: tuple[int, ...] = _EDGES) -> list[str]:
    spans = [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:], strict=False)]
    return [*spans, f"{edges[-1]}+", f"none_in_{edges[-1]}", "censored", "already_crossed"]


def onset_groups(arrays: Arrays, edges: tuple[int, ...] = _EDGES) -> np.ndarray:
    """True onset group per window: onset frame spans, a crossing beyond the last edge, no crossing seen
    across the last edge, a future too short to tell (censored), or already crossed before the window."""
    onset, observed, ever = (np.asarray(arrays[k]) for k in ONSET_FIELDS)
    order = group_order(edges)
    groups = np.full(onset.shape, "censored", dtype=object)
    for i, (lo, hi) in enumerate(zip(edges, edges[1:], strict=False)):
        groups[(onset >= lo) & (onset < hi)] = order[i]
    groups[onset >= edges[-1]] = f"{edges[-1]}+"
    no_crossing = onset < 0
    groups[no_crossing & (observed >= edges[-1])] = f"none_in_{edges[-1]}"
    groups[no_crossing & (ever > 0)] = "already_crossed"
    return groups.astype(str)


def horizon_targets(arrays: Arrays, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """``(label, knowable)`` for "crossing starts within ``horizon`` frames" — the readout-target rule."""
    tensors = {k: torch.as_tensor(np.asarray(arrays[k]), dtype=torch.long) for k in ONSET_FIELDS}
    label, valid = readout_targets(tensors, OnsetSpec(lookahead=horizon, bin_width=1, horizon=horizon))
    return label.numpy().astype(int), valid.numpy().astype(bool)


# --------------------------------------------------------------------------- report (arrays -> numbers)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _scores(arrays: Arrays, meta: dict[str, object], horizon: int) -> dict[str, np.ndarray]:
    """The fixed per-window scores in a dump; the hazard score is the cumulative probability at ``horizon``."""
    scores = {"crosses_frame": arrays["p_frame"]}
    if "p_readout" in arrays:
        scores["readout"] = arrays["p_readout"]
    if "hazard_logits" in arrays:
        width = int(meta.get("onset_bin_width", 0) or 0)
        if width and horizon % width == 0 and horizon // width <= arrays["hazard_logits"].shape[1]:
            scores[f"hazard_within_{horizon}"] = hazard_horizon_prob(arrays["hazard_logits"], horizon // width)
    return scores


def _horizon_auc(arrays: Arrays, meta: dict[str, object], horizons: Iterable[int]) -> dict[str, dict]:
    table: dict[str, dict] = {}
    for horizon in horizons:
        label, knowable = horizon_targets(arrays, horizon)
        for name, score in _scores(arrays, meta, horizon).items():
            key = "hazard_cumulative" if name.startswith("hazard_within_") else name
            table.setdefault(key, {})[str(horizon)] = {
                "auc": _auc(label[knowable], score[knowable]),
                "n": int(knowable.sum()),
                "positives": int(label[knowable].sum()),
            }
    return table


def _group_table(score: np.ndarray, groups: np.ndarray, order: list[str]) -> dict[str, dict]:
    return {
        g: {"n": int((groups == g).sum()), "mean": float(score[groups == g].mean()),
            "median": float(np.median(score[groups == g]))}
        for g in order if (groups == g).any()
    }


def _negative_split(arrays: Arrays, score: np.ndarray, horizon: int) -> dict[str, float]:
    """AUC of within-``horizon`` crossers against each kind of negative separately."""
    onset, observed, ever = (np.asarray(arrays[k]) for k in ONSET_FIELDS)
    positive = (onset >= 0) & (onset < horizon)
    kinds = {
        "later_crossers": onset >= horizon,
        "no_crossing_seen": (onset < 0) & (ever <= 0) & (observed >= horizon),
        "already_crossed": (onset < 0) & (ever > 0),
    }
    out = {}
    for name, negative in kinds.items():
        both = positive | negative
        out[name] = _auc(positive[both].astype(int), score[both])
    return out


def _tuned_f1(val: Arrays, test: Arrays, key: str, grid: list[float]) -> dict[str, float]:
    """Threshold swept on ``val`` (evaluate.py's grid + tie-break), applied to ``test`` on the stored label."""
    threshold = _best_f1_threshold(val["crosses"].astype(int), val[key], grid)
    y, pred = test["crosses"].astype(int), (test[key] >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def timing_report(
    test: Arrays,
    meta: dict[str, object],
    *,
    val: Arrays | None = None,
    horizons: Iterable[int] = (16, 32, 64, 96),
    group_horizon: int = 32,
    sweep: tuple[float, float, float] = (0.10, 0.90, 0.05),
) -> dict[str, object]:
    """Every timing number for one dump (``test``); ``val`` adds F1 at val-tuned thresholds."""
    horizons = tuple(horizons)
    groups, order = onset_groups(test), group_order()
    scores = _scores(test, meta, group_horizon)
    long_range = _scores(test, meta, max(horizons))
    scores.update({k: v for k, v in long_range.items() if k.startswith("hazard_within_")})
    stored = test["crosses"].astype(int)
    report: dict[str, object] = {
        "meta": meta,
        "n": int(stored.size),
        "stored_label_auc": {name: _auc(stored, s) for name, s in scores.items()},
        "horizon_auc": _horizon_auc(test, meta, horizons),
        "score_by_onset_group": {name: _group_table(s, groups, order) for name, s in scores.items()},
        f"negative_split_auc_{group_horizon}": {
            name: _negative_split(test, s, group_horizon) for name, s in scores.items()
        },
    }
    if val is not None:
        grid = _sweep(*sweep)
        keys = [k for k in ("p_frame", "p_readout") if k in test and k in val]
        report["tuned_f1_stored_label"] = {k: _tuned_f1(val, test, k, grid) for k in keys}
    return report


def format_report(report: dict[str, object]) -> str:
    """Markdown tables for a :func:`timing_report` result."""
    lines = [f"# Onset timing — {report['meta'].get('split', '?')} ({report['n']} windows)", ""]
    lines += ["## AUC on the stored label (crossing within 32 frames)", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in report["stored_label_auc"].items()]
    if "tuned_f1_stored_label" in report:
        lines += ["", "## F1 at val-tuned thresholds", ""]
        lines += [f"- {k}: {json.dumps({m: round(x, 4) for m, x in v.items()})}"
                  for k, v in report["tuned_f1_stored_label"].items()]
    lines += ["", "## AUC by horizon (knowable windows only)", ""]
    for name, per_h in report["horizon_auc"].items():
        cells = ", ".join(f"H={h}: {v['auc']:.4f} (n={v['n']}, pos={v['positives']})" for h, v in per_h.items())
        lines.append(f"- {name}: {cells}")
    lines += ["", "## Mean score by true onset group", ""]
    for name, table in report["score_by_onset_group"].items():
        cells = ", ".join(f"{g}: {v['mean']:.4f} (n={v['n']})" for g, v in table.items())
        lines.append(f"- {name}: {cells}")
    split_key = next(k for k in report if k.startswith("negative_split_auc_"))
    lines += ["", f"## Within-horizon crossers vs each kind of negative ({split_key})", ""]
    for name, table in report[split_key].items():
        lines.append(f"- {name}: " + ", ".join(f"{k}: {v:.4f}" for k, v in table.items()))
    return "\n".join(lines) + "\n"

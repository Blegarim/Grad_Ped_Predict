"""Field-level diff between two resolved configs — the comparability check for multi-arm campaigns.

CLAUDE.md: an accidental config difference between two legs invalidates a comparison. This makes that
mechanical: flatten both configs to dotted keys, drop the fields that cannot change a trained model, and
report every remaining difference that is not one of the arm's intended flags.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from pedpredict.config.schema import RootCfg

__all__ = ["RESULT_NEUTRAL_KEYS", "config_diff", "flatten_config", "unexpected_diffs"]

#: Keys that never change what a run learns: eval-time knobs (eval.model_type is the exception — it picks
#: the architecture) and worker counts (sampler order is drawn in the main process; runtime augmentation is
#: seeded per run + epoch + index, so the batches do not depend on them).
RESULT_NEUTRAL_KEYS: tuple[str, ...] = ("eval.", "train.num_workers", "paths.runs_dir")
_ARCHITECTURE_KEYS: tuple[str, ...] = ("eval.model_type",)


def flatten_config(root: RootCfg) -> dict[str, object]:
    """``{"section.field": value}`` over every field; lists become tuples so values compare by content."""
    flat: dict[str, object] = {}
    for section, fields in dataclasses.asdict(root).items():
        for name, value in fields.items():
            flat[f"{section}.{name}"] = _freeze(value)
    return flat


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _neutral(key: str) -> bool:
    return key not in _ARCHITECTURE_KEYS and any(key.startswith(p) for p in RESULT_NEUTRAL_KEYS)


def config_diff(reference: RootCfg, candidate: RootCfg) -> dict[str, tuple[object, object]]:
    """Every result-relevant key whose value differs: ``{key: (reference, candidate)}``."""
    ref, cand = flatten_config(reference), flatten_config(candidate)
    return {k: (ref.get(k), cand.get(k)) for k in sorted(ref.keys() | cand.keys())
            if not _neutral(k) and ref.get(k) != cand.get(k)}


def unexpected_diffs(diff: dict[str, tuple[object, object]], allowed: Iterable[str]) -> list[str]:
    """Keys in ``diff`` not covered by an allowed key or ``section.`` / ``prefix_`` prefix."""
    prefixes = tuple(allowed)
    return [k for k in diff if not any(k == p or (p.endswith((".", "_")) and k.startswith(p)) for p in prefixes)]

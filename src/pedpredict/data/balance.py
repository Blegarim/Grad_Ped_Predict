"""Offline class-balancing of sequence pkls (Prompt 1.3) — the OPT-IN majority-downsample lever.

Consolidates OLD ``scripts/balance_sequences.py`` (equal 50/50 preset) and
``scripts/split_balance_sequences_all.py`` (configurable-ratio preset, default 30/70) into one
pure, deterministic module + a thin ``scripts/balance_dataset.py`` CLI (B5). It reads only the
three label fields of each record and returns a *subset* in the identical pkl format, so a balanced
pkl flows through the LMDB writer (1.2) → dataset (1.5) → training exactly like any other sequence
pkl — balancing is a transform on the artifact, never a branch in the model/training path.

Imbalance policy (B3): this lever is **OFF by default** (``BalanceCfg.enabled=False``). The default
runtime policy is the online ``WeightedRandomSampler`` (1.6) + inverse-freq loss class weights (3.1),
both already ON in ``TrainCfg``, layered on offline *augmentation* (1.4). Offline balance is the
opt-in *alternative* to augmentation for ablation; when enabled, relax the online levers so the
three do not silently triple-stack. See ``MIGRATION.md`` "Imbalance policy".

Parity (Phase A): the two legacy behaviors are reproduced exactly by the ``BALANCE_EQUAL`` /
``BALANCE_RATIO_30_70`` presets (golden fixture ``tests/fixtures/golden/balance_cases.json``).
Determinism is stdlib ``random.Random(seed)`` with the legacy group/pick/shuffle call order — the
parity contract; numpy would diverge.

⚠️ Behavior change (flagged, justified in MIGRATION.md): the default ``solve_cross0_counts`` ships the
**corrected** count constraint. OLD ``split_balance_sequences_all.solve_exact`` had a sign-flipped
``x00`` bound (``n0 - a - l`` instead of ``a + l - n0``) that could silently miscount in the 30/70
regime; it is reachable only via ``legacy_x00_sign_bug=True`` (parity-test surface), dropped in Phase B.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from pedpredict.config.schema import BalanceCfg
from pedpredict.data.pie_sequences import SequenceRecord, load_sequences, save_sequences

__all__ = [
    "X11Select",
    "clamp_cross",
    "group_by_labels",
    "solve_cross0_counts",
    "solve_cross0_counts_approx",
    "choose_cross1",
    "balance_indices",
    "balance_records",
    "summarize",
    "balance_sequence_file",
    "BALANCE_EQUAL",
    "BALANCE_RATIO_30_70",
]

GroupTable = dict[int, dict[tuple[int, int], list[int]]]   # {crosses: {(actions, looks): [idx, ...]}}
Counts4 = tuple[int, int, int, int]                        # (x00, x01, x10, x11)
_AL_ORDER: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


class X11Select(str, Enum):
    """Which end of the feasible ``x11`` interval to pick (the two legacy solvers differ here)."""

    LOWER = "lower"   # split_balance_sequences_all.solve_exact
    UPPER = "upper"   # balance_sequences._solve_cross0_counts


# --------------------------------------------------------------------------- pure label utilities


def clamp_cross(value: int) -> int:
    """Map raw ``crosses`` to ``{0, 1}`` (only literal ``1`` is positive); matches both OLD clampers."""
    return 1 if value == 1 else 0


def group_by_labels(records: Sequence[SequenceRecord]) -> GroupTable:
    """Group record indices by ``(clamped crosses, (actions, looks))``.

    Insertion order ``(0,0),(0,1),(1,0),(1,1)`` and record-enumeration order are parity-critical:
    ``random.Random`` consumes population order downstream.
    """
    groups: GroupTable = {0: {al: [] for al in _AL_ORDER}, 1: {al: [] for al in _AL_ORDER}}
    for idx, item in enumerate(records):
        a = int(item["actions"])
        looks = int(item["looks"])
        c = clamp_cross(int(item["crosses"]))
        groups[c][(a, looks)].append(idx)
    return groups


# --------------------------------------------------------------------------- count solvers


def solve_cross0_counts(
    n0: int,
    a_target: int,
    l_target: int,
    c00: int,
    c01: int,
    c10: int,
    c11: int,
    *,
    x11_select: X11Select,
    legacy_x00_sign_bug: bool = False,
) -> Counts4 | None:
    """Closed-form pick of ``(x00, x01, x10, x11)`` from the cross=0 group; ``None`` if infeasible.

    Constraints: ``x10+x11 == a_target``, ``x01+x11 == l_target``, ``Σx == n0``, ``0 ≤ xij ≤ cij``.
    The correct ``x00`` bound is ``a_target + l_target - n0``; ``legacy_x00_sign_bug`` flips it to the
    OLD ``solve_exact`` value (``n0 - a_target - l_target``) for golden parity only.
    """
    x00_term = (n0 - a_target - l_target) if legacy_x00_sign_bug else (a_target + l_target - n0)
    lower = max(0, a_target - c10, l_target - c01, x00_term)
    upper = min(c11, a_target, l_target, x00_term + c00)
    if lower > upper:
        return None
    x11 = lower if x11_select is X11Select.LOWER else upper
    x10 = a_target - x11
    x01 = l_target - x11
    x00 = n0 - x11 - x10 - x01
    return x00, x01, x10, x11


def solve_cross0_counts_approx(
    n0: int, a_target: int, l_target: int, c00: int, c01: int, c10: int, c11: int
) -> Counts4 | None:
    """Greedy fallback minimizing ``|actions-target| + |looks-target|`` (verbatim OLD ``solve_approx``)."""
    best: Counts4 | None = None
    best_error: int | None = None
    for x11 in range(min(c11, n0) + 1):
        remaining = n0 - x11
        x10_min = max(0, remaining - (c01 + c00))
        x10_max = min(c10, remaining)
        if x10_min > x10_max:
            continue
        x10 = min(max(a_target - x11, x10_min), x10_max)
        remaining2 = remaining - x10
        x01_min = max(0, remaining2 - c00)
        x01_max = min(c01, remaining2)
        if x01_min > x01_max:
            continue
        x01 = min(max(l_target - x11, x01_min), x01_max)
        x00 = remaining2 - x01
        error = abs(x10 + x11 - a_target) + abs(x01 + x11 - l_target)
        if best_error is None or error < best_error:
            best_error = error
            best = (x00, x01, x10, x11)
            if error == 0:
                break
    return best


# --------------------------------------------------------------------------- selection


def choose_cross1(groups: GroupTable, n1: int, rng: random.Random) -> list[int]:
    """Pick ``n1`` cross=1 indices in priority order ``(0,0),(0,1),(1,0),(1,1)`` (OLD ``choose_cross1``)."""
    selected: list[int] = []
    remaining = n1
    for combo in _AL_ORDER:
        if remaining <= 0:
            break
        pool = groups[1][combo]
        if len(pool) <= remaining:
            selected.extend(pool)
            remaining -= len(pool)
        else:
            selected.extend(rng.sample(pool, remaining))
            remaining = 0
    return selected


def _pick(pool: list[int], k: int, rng: random.Random, *, strict: bool) -> list[int]:
    """Sample ``k`` of ``pool``. ``strict`` (equal preset) raises on over-request; else truncates."""
    if strict:
        if k == 0:
            return []
        if k < 0 or k > len(pool):
            raise RuntimeError("Insufficient samples for requested balance.")
        return list(pool) if k == len(pool) else rng.sample(pool, k)
    if k <= 0:
        return []
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)


def balance_indices(records: Sequence[SequenceRecord], cfg: BalanceCfg) -> list[int]:
    """Select a balanced subset of record indices, deterministic given ``cfg.seed``.

    Reproduces ``balance_sequences.balance_dataset`` (``BALANCE_EQUAL``) and
    ``split_balance_sequences_all.balance_split`` (``BALANCE_RATIO_30_70``) exactly. ``on_infeasible``
    decides whether an unsolvable target raises or returns ``[]``.
    """
    rng = random.Random(cfg.seed)
    groups = group_by_labels(records)
    c1_total = sum(len(v) for v in groups[1].values())
    c0_total = sum(len(v) for v in groups[0].values())
    if c1_total == 0 or c0_total == 0:
        return []

    r = cfg.cross_pos_ratio
    if cfg.subsample_cross1:
        n1 = min(c1_total, int(c0_total * r / (1.0 - r)))
        n0 = int(round(n1 * (1.0 - r) / r))
        if n0 > c0_total:
            n0 = c0_total
            n1 = int(round(n0 * r / (1.0 - r)))
        selected1 = choose_cross1(groups, n1, rng)
    else:
        n1 = c1_total
        n0 = int(round(n1 * (1.0 - r) / r))
        selected1 = [idx for combo in groups[1].values() for idx in combo]

    a1 = sum(int(records[i]["actions"]) for i in selected1)
    l1 = sum(int(records[i]["looks"]) for i in selected1)
    total_n = n0 + n1
    a_target = max(0, min(n0, int(round(cfg.target_action_rate * total_n)) - a1))
    l_target = max(0, min(n0, int(round(cfg.target_look_rate * total_n)) - l1))

    g0 = groups[0]
    counts = (len(g0[(0, 0)]), len(g0[(0, 1)]), len(g0[(1, 0)]), len(g0[(1, 1)]))
    solved = solve_cross0_counts(
        n0, a_target, l_target, *counts,
        x11_select=X11Select(cfg.x11_select), legacy_x00_sign_bug=cfg.legacy_x00_sign_bug,
    )
    if solved is None and cfg.allow_approx:
        solved = solve_cross0_counts_approx(n0, a_target, l_target, *counts)
    if solved is None:
        if cfg.on_infeasible == "raise":
            raise RuntimeError("No feasible balanced subset found with current data.")
        return []

    x00, x01, x10, x11 = solved
    strict = cfg.on_infeasible == "raise"
    selected0 = (
        _pick(g0[(0, 0)], x00, rng, strict=strict)
        + _pick(g0[(0, 1)], x01, rng, strict=strict)
        + _pick(g0[(1, 0)], x10, rng, strict=strict)
        + _pick(g0[(1, 1)], x11, rng, strict=strict)
    )
    selected = selected1 + selected0
    rng.shuffle(selected)
    return selected


def balance_records(records: Sequence[SequenceRecord], cfg: BalanceCfg) -> list[SequenceRecord]:
    """Convenience: materialize the balanced subset as a new record list (order = ``balance_indices``)."""
    return [records[i] for i in balance_indices(records, cfg)]


# --------------------------------------------------------------------------- reporting + I/O


def summarize(records: Sequence[SequenceRecord], indices: Sequence[int]) -> dict[str, float]:
    """Total + per-task positive rate over ``indices`` (``crosses`` clamped — fixes OLD unclamped sum)."""
    total = len(indices)
    if total == 0:
        return {}
    a = sum(int(records[i]["actions"]) for i in indices)
    looks = sum(int(records[i]["looks"]) for i in indices)
    c = sum(clamp_cross(int(records[i]["crosses"])) for i in indices)
    return {
        "total": total,
        "actions_pos_rate": a / total,
        "looks_pos_rate": looks / total,
        "crosses_pos_rate": c / total,
    }


def balance_sequence_file(in_path: str | Path, out_path: str | Path, cfg: BalanceCfg) -> dict[str, float]:
    """Load a sequence pkl, balance it, save the subset, and return its summary. I/O isolated."""
    records = load_sequences(in_path)
    indices = balance_indices(records, cfg)
    save_sequences([records[i] for i in indices], out_path)
    return summarize(records, indices)


# --------------------------------------------------------------------------- legacy-equivalent presets

#: Reproduces OLD ``scripts/balance_sequences.py`` (keep all cross=1, 50/50, x11=upper, raise on infeasible).
BALANCE_EQUAL = BalanceCfg(
    enabled=True,
    cross_pos_ratio=0.5,
    x11_select="upper",
    subsample_cross1=False,
    allow_approx=False,
    on_infeasible="raise",
    legacy_x00_sign_bug=False,
)

#: Reproduces OLD ``scripts/split_balance_sequences_all.py`` balance step (30/70, x11=lower, approx,
#: empty on infeasible) — including the ``solve_exact`` sign bug, for golden parity only.
BALANCE_RATIO_30_70 = BalanceCfg(
    enabled=True,
    cross_pos_ratio=0.30,
    x11_select="lower",
    subsample_cross1=True,
    allow_approx=True,
    on_infeasible="empty",
    legacy_x00_sign_bug=True,
)

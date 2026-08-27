"""What the streaming negatives are actually made of, and how that moves with the horizon.

Two pure reports over the S1 onset fields (``onset_offset`` / ``future_observed`` / ``track_crosses``),
both computed from sequence records alone — no LMDB, no GPU, no images.

**Composition.** ``crosses == 0`` lumps together three different situations, and the method work is
sized by which one dominates:

* *genuine* — the pedestrian never crosses anywhere in the track;
* *hard-temporal* — they do cross, but later than the horizon asks about. Visually these are the
  windows nearest the positives, and they are the ones the event-anchored benchmark largely omits;
* *already-crossed* — the crossing is behind them.

**Horizon sweep.** The same records re-labelled at any horizon H, which the S1 fields make free: a
window is positive at H when ``0 <= onset_offset < H``, and it is *usable* at H only when the answer
is knowable — either a crossing was seen, or ``future_observed >= H``. Windows failing that are
H-censored and are excluded rather than labelled 0 (the M4 rule, applied at an arbitrary H). The sweep
also reports the *confusable band*: negatives whose crossing falls just past the boundary, which is
the population that matched-pair training would draw on.

One caveat the caller owns: records generated at the canonical horizon were already filtered to
``future_observed >= future_offset + tol``, so a sweep over them is exact for H at or above that and
conservative below it (windows censored at generation are not recoverable here).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "NEGATIVE_KINDS",
    "OnsetComposition",
    "HorizonPoint",
    "classify_window",
    "is_usable",
    "compose",
    "horizon_sweep",
    "format_composition_table",
    "format_sweep_table",
]

#: The three ways a streaming window can be a negative (see module docstring).
NEGATIVE_KINDS = ("genuine", "hard_temporal", "already_crossed")


def classify_window(record: dict[str, Any], horizon: int) -> str:
    """``'positive'`` or one of :data:`NEGATIVE_KINDS` for one window at anticipation horizon ``horizon``."""
    onset = int(record["onset_offset"])
    if 0 <= onset < horizon:
        return "positive"
    if not int(record["track_crosses"]):
        return "genuine"
    return "hard_temporal" if onset >= horizon else "already_crossed"


def is_usable(record: dict[str, Any], horizon: int) -> bool:
    """Is this window's horizon-``horizon`` label knowable? (M4's rule, at an arbitrary horizon.)

    Yes when a crossing was actually observed, or when enough future was observed to rule one out.
    """
    return int(record["onset_offset"]) >= 0 or int(record["future_observed"]) >= horizon


@dataclass(slots=True)
class OnsetComposition:
    """Per-split window counts split by negative kind at one horizon."""

    split: str
    horizon: int
    total: int
    positive: int
    genuine: int
    hard_temporal: int
    already_crossed: int

    @property
    def negatives(self) -> int:
        return self.genuine + self.hard_temporal + self.already_crossed

    @property
    def imbalance(self) -> float:
        """Negatives per positive (the ``~37:1`` figure)."""
        return self.negatives / self.positive if self.positive else float("nan")

    def rate(self, kind: str) -> float:
        """Fraction of all windows in ``kind`` (``'positive'`` or a :data:`NEGATIVE_KINDS` member)."""
        return getattr(self, kind) / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "negatives": self.negatives, "imbalance": self.imbalance}


@dataclass(slots=True)
class HorizonPoint:
    """One row of the horizon sweep."""

    horizon: int
    usable: int
    positive: int
    band: int          # negatives crossing within `band_frames` past the boundary
    band_frames: int
    censored_out: int  # usable at the canonical horizon, not knowable at this one

    @property
    def imbalance(self) -> float:
        return (self.usable - self.positive) / self.positive if self.positive else float("nan")

    @property
    def band_per_positive(self) -> float:
        return self.band / self.positive if self.positive else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "imbalance": self.imbalance, "band_per_positive": self.band_per_positive}


def compose(split: str, records: Sequence[dict[str, Any]], horizon: int) -> OnsetComposition:
    """Count ``records`` by window kind at ``horizon`` (records are assumed usable at that horizon)."""
    counts = dict.fromkeys(("positive", *NEGATIVE_KINDS), 0)
    for record in records:
        counts[classify_window(record, horizon)] += 1
    return OnsetComposition(split=split, horizon=horizon, total=len(records), **counts)


def horizon_sweep(records: Sequence[dict[str, Any]], horizons: Sequence[int],
                  *, band_frames: int = 60) -> list[HorizonPoint]:
    """Re-label ``records`` at each horizon and report base rate, censoring, and the confusable band."""
    points: list[HorizonPoint] = []
    for horizon in horizons:
        usable = [r for r in records if is_usable(r, horizon)]
        positive = sum(1 for r in usable if 0 <= int(r["onset_offset"]) < horizon)
        band = sum(1 for r in usable if horizon <= int(r["onset_offset"]) < horizon + band_frames)
        points.append(HorizonPoint(horizon=horizon, usable=len(usable), positive=positive, band=band,
                                   band_frames=band_frames, censored_out=len(records) - len(usable)))
    return points


def format_composition_table(compositions: Sequence[OnsetComposition]) -> str:
    """Markdown table: one row per split, percentages of all windows."""
    rows = ["| Split | N | positive | genuine | hard-temporal | already-crossed | imbalance |",
            "|---|---|---|---|---|---|---|"]
    for c in compositions:
        cells = " | ".join(f"{c.rate(k) * 100:.1f}% ({getattr(c, k)})"
                           for k in ("positive", "genuine", "hard_temporal", "already_crossed"))
        rows.append(f"| {c.split} | {c.total} | {cells} | {c.imbalance:.1f}:1 |")
    return "\n".join(rows)


def format_sweep_table(points: Sequence[HorizonPoint], *, fps: float = 30.0) -> str:
    """Markdown table: one row per horizon, with the confusable band and the censoring cost."""
    band_s = points[0].band_frames / fps if points else 0.0
    rows = [f"| H (frames) | H (s) | usable | positive | imbalance | band (+{band_s:.0f}s) | band:pos | censored out |",
            "|---|---|---|---|---|---|---|---|"]
    for p in points:
        total = p.usable + p.censored_out
        lost = f"{p.censored_out} ({p.censored_out / total * 100:.0f}%)" if total else str(p.censored_out)
        rows.append(f"| {p.horizon} | {p.horizon / fps:.1f} | {p.usable} | {p.positive} | "
                    f"{p.imbalance:.1f}:1 | {p.band} | {p.band_per_positive:.2f}:1 | {lost} |")
    return "\n".join(rows)

"""Turn the three S1 onset fields into discrete-time hazard supervision.

The label side of the onset-timing method (docs/METHODOLOGY.md prong 2). The binary ``crosses`` label
asks one question — "does a crossing start in the next ``H`` frames?" — and answers it 0 for a window
whose future ran out (a fabricated label) and 0 for a window whose crossing lands at ``H + 5`` (a real
crossing, labelled as if it were a bus-stop loiterer). Discrete-time hazard supervision asks ``K``
smaller questions instead: for each future bin ``k``, *given no crossing has started yet, does one start
in bin ``k``?* Each is an ordinary binary decision, and each can be **masked out** when the answer was
never observed.

Bin ``k`` covers future frames ``[k*w, (k+1)*w)`` measured from the last observed frame, so bin index and
frame offset are related by the bin width ``w`` alone. With ``w = 1`` a bin *is* a frame and the concept
disappears; ``w > 1`` trades timing resolution for a denser positive rate per bin, which is the lever
against the head collapsing to ``h ~ 0`` everywhere (positives are ~2.9% of windows spread over ``K``
bins). :class:`OnsetSpec` requires both ``L`` and ``H`` to divide by ``w`` so no bin ever straddles the
reported horizon.

Four cases, from the three fields (the same four :func:`pedpredict.data.onset_stats.classify_window`
reports on, here turned into gradients):

===================  =========================================  ===========================================
case                 recognised by                              supervision
===================  =========================================  ===========================================
event in range       ``0 <= onset_offset < L``                  bins ``0..e-1`` -> 0, bin ``e`` -> 1,
                                                                bins after ``e`` masked (unobservable:
                                                                the person has started)
event beyond range   ``onset_offset >= L``                      all ``K`` bins -> 0. Honest: the crossing
                                                                was observed, just not within ``L``
censored             ``onset_offset < 0``, enough future to      bins covered by ``future_observed`` -> 0,
                     rule some out                              the rest **masked**. The window says
                                                                "not yet, for this long" and no more
already crossed      ``onset_offset < 0 and track_crosses``      dropped — ``valid = False``. The person
                                                                is not at risk of a *first* crossing, so
                                                                they are neither a negative nor censored
===================  =========================================  ===========================================

The third row is what the binary label cannot express, and the fourth is a bug it cannot avoid: today
both land in ``crosses = 0`` beside genuine non-crossers.

Separately, :func:`readout_targets` builds the label for the horizon-``H`` readout term, which is the
*binary* question again — kept so the reported number keeps something pulling on it directly, and so the
existing baselines stay comparable. At ``H == future_offset + tol`` it reproduces the stored ``crosses``
label exactly (pinned in ``tests/test_onset_target.py``), which is what makes the two formulations
comparable rather than merely adjacent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["OnsetSpec", "OnsetTargets", "hazard_targets", "readout_targets"]


@dataclass(frozen=True)
class OnsetSpec:
    """Geometry of the hazard head: how far ahead, how finely, and where the reported horizon sits.

    ``lookahead`` is a *training* decision and ``horizon`` a *reporting* one; they are allowed to
    differ, and the method depends on ``lookahead > horizon`` — a head only as wide as the horizon
    still labels a crossing at ``H + 5`` a flat negative, which is the failure the method exists to
    remove.
    """

    lookahead: int      # L: future frames the head covers
    bin_width: int      # w: future frames per output bin
    horizon: int        # H: frames the reported P(cross within H) covers; must be <= L

    def __post_init__(self) -> None:
        if self.bin_width < 1:
            raise ValueError(f"OnsetSpec: bin_width must be >= 1, got {self.bin_width}")
        if self.lookahead < 1:
            raise ValueError(f"OnsetSpec: lookahead must be >= 1, got {self.lookahead}")
        if self.horizon < 1 or self.horizon > self.lookahead:
            raise ValueError(
                f"OnsetSpec: horizon must be in [1, lookahead={self.lookahead}], got {self.horizon}"
            )
        for name, value in (("lookahead", self.lookahead), ("horizon", self.horizon)):
            if value % self.bin_width:
                raise ValueError(
                    f"OnsetSpec: {name}={value} is not divisible by bin_width={self.bin_width} — a bin "
                    f"would straddle the boundary, leaving the readout undefined."
                )

    @property
    def num_bins(self) -> int:
        """``K`` — the hazard head's output width."""
        return self.lookahead // self.bin_width

    @property
    def horizon_bins(self) -> int:
        """How many leading bins the reported ``P(cross within horizon)`` multiplies over."""
        return self.horizon // self.bin_width


@dataclass(frozen=True)
class OnsetTargets:
    """Per-window hazard supervision. ``B`` windows, ``K`` bins.

    ``target`` and ``mask`` together are the whole objective: the loss is a per-bin binary
    cross-entropy of ``target`` against the hazard logits, summed where ``mask`` is 1 and *silent*
    everywhere else. Bins past the event or past the censoring point are silent rather than zero —
    the distinction between "they had not started" and "we never saw".
    """

    target: Tensor      # [B, K] float — 1.0 on the event bin, 0.0 elsewhere
    mask: Tensor        # [B, K] float — 1.0 where the bin was observed, 0.0 where it was not
    event_bin: Tensor   # [B] long — index of the event bin, -1 when no event falls inside L
    valid: Tensor       # [B] bool — window is in the risk set AND carries at least one observed bin


def hazard_targets(labels: dict[str, Tensor], spec: OnsetSpec) -> OnsetTargets:
    """Build ``[B, K]`` hazard target + mask from a batch's S1 fields.

    ``labels`` needs ``onset_offset`` / ``future_observed`` / ``track_crosses`` (the collate lifts them
    there from the LMDB meta). Pure and shape-preserving: no config, no device moves beyond following
    the inputs, no gradient.
    """
    onset, observed, ever = _require(labels)
    batch, num_bins = onset.shape[0], spec.num_bins
    device = onset.device

    has_event = (onset >= 0) & (onset < spec.lookahead)
    event_bin = torch.where(has_event, torch.div(onset, spec.bin_width, rounding_mode="floor"),
                            torch.full_like(onset, -1))

    # Observed-bin count. With an event: every bin up to and including it. Without: however many whole
    # bins the observed future covers, capped at K (an event beyond L implies future_observed >= L, so
    # "event beyond range" lands on the full K by the same arithmetic — no special case needed).
    censor_bins = torch.div(observed, spec.bin_width, rounding_mode="floor").clamp(max=num_bins)
    n_observed = torch.where(has_event, event_bin + 1, censor_bins)

    bins = torch.arange(num_bins, device=device).unsqueeze(0)          # [1, K]
    mask = (bins < n_observed.unsqueeze(1)).to(torch.float32)          # [B, K]
    target = torch.zeros(batch, num_bins, device=device, dtype=torch.float32)
    target[has_event, event_bin[has_event]] = 1.0

    # Already crossed: no future crossing seen, but the track crosses — so it happened BEFORE this
    # window (generation drops windows that contain a crossing). Not at risk of a first crossing.
    already_crossed = (onset < 0) & (ever > 0)
    valid = ~already_crossed & (n_observed > 0)
    mask = mask * valid.unsqueeze(1).to(mask.dtype)
    return OnsetTargets(target=target, mask=mask, event_bin=event_bin, valid=valid)


def readout_targets(labels: dict[str, Tensor], spec: OnsetSpec) -> tuple[Tensor, Tensor]:
    """``(label, valid)`` for the horizon-``H`` binary readout term.

    ``label`` is 1 when a crossing starts within ``H`` frames. ``valid`` is the M4 rule applied at ``H``
    (``onset_stats.is_usable``): the answer is knowable only when a crossing was actually seen, or when
    enough future was observed to rule one out. Already-crossed windows are excluded here too, for the
    same reason as in :func:`hazard_targets`.
    """
    onset, observed, ever = _require(labels)
    label = ((onset >= 0) & (onset < spec.horizon)).to(torch.float32)
    knowable = (onset >= 0) | (observed >= spec.horizon)
    already_crossed = (onset < 0) & (ever > 0)
    return label, knowable & ~already_crossed


def _require(labels: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Pull the three S1 fields off a label dict, or say exactly what to run to get them."""
    missing = [f for f in ("onset_offset", "future_observed", "track_crosses") if f not in labels]
    if missing:
        raise KeyError(
            f"Onset supervision needs {missing} in the batch labels, which means the LMDB chunks were "
            f"built before S1. Run scripts/backfill_onset_meta.py over the split's chunk dir (a "
            f"metadata-only pass — image blobs are untouched), or rebuild."
        )
    return labels["onset_offset"], labels["future_observed"], labels["track_crosses"]

"""Discrete-time hazard loss for crossing onset — the objective side of prong 2.

Once the targets are built (:mod:`pedpredict.data.onset_target`), the objective is unremarkable, and
that is the point: **a masked per-bin binary cross-entropy**. Differentiating the right-censored
discrete-time likelihood

    event at bin ``t``:  ``h_t * prod_{j<t} (1 - h_j)``        censored at bin ``c``:  ``prod_{j<=c} (1 - h_j)``

with respect to each hazard logit ``z_k`` gives ``+h_k`` on every bin before the event (push down),
``-(1 - h_t)`` on the event bin (push up), and **exactly zero** past the event or past the censoring
point. That is BCE with targets ``[0, ..., 0, 1]`` and a mask — no bespoke likelihood code, and the
zero-gradient tail is the whole contribution: today those bins are either a fabricated 0 or a dropped
window.

Two terms, weighted independently, because they answer different questions:

* **hazard** — the ``K``-bin NLL above. Sharpens *when*, and is the only term that can reach the bins
  past the reported horizon.
* **readout** — an ordinary CE on ``1 - prod(1 - h_k)`` against the binary horizon label. The hazard
  term never optimises the number that gets reported, and with ``L`` well past ``H`` most of the head's
  capacity is spent on bins nobody reads; this pulls the reported number directly. It is *deliberately
  in tension* with the hazard term on positives — it wants probability smeared across the horizon,
  the hazard term wants it concentrated on one bin — so its weight is a dial between reported F1 and
  timing sharpness, not a free improvement. Default 0.

Reduction is **per window, then mean over valid windows**: a window's loss is the sum over its observed
bins, matching the likelihood (a window observed for 60 bins genuinely carries more information than one
observed for 3), and windows are then averaged so batch loss does not scale with censoring depth.
Invalid windows — already-crossed, or censored before a single whole bin — are excluded from both the
numerator and the denominator.

One consequence of that reduction is worth stating plainly, because it is a footgun rather than a
subtlety: summing over bins puts the hazard term on a **different scale from a cross-entropy**. At
``L = 60`` and initialisation (every hazard ~0.5, so ~0.693 per bin) it starts near 40, against ~0.69 for
each per-task CE, and falls as the hazards saturate low. In the pure-reformulation arm that is fine — it
is the objective. In the auxiliary arm it would swamp the CE heads, so ``train.onset_hazard_weight``
wants to start around 0.02-0.05 there. :class:`OnsetLossOutput` reports the raw unweighted value so the
ratio is visible in the logs rather than inferred.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from pedpredict.config.schema import ModelCfg, TrainCfg
from pedpredict.data.onset_target import OnsetSpec, hazard_targets, readout_targets

__all__ = [
    "HAZARD_OUTPUT_KEY",
    "READOUT_OUTPUT_KEY",
    "OnsetLossOutput",
    "OnsetHazardLoss",
    "build_onset_loss",
    "crosses_metric_keys",
    "onset_spec_from_config",
]

#: Output-dict keys this loss consumes (emitted by ``heads.emit_task_logits`` when the head is on).
HAZARD_OUTPUT_KEY = "crosses_hazard"
READOUT_OUTPUT_KEY = "crosses_readout"


class OnsetLossOutput(NamedTuple):
    """Live ``total`` for backward + detached components for logging."""

    total: Tensor                  # live: hazard_weight * hazard + readout_weight * readout
    hazard: Tensor                 # detached raw per-window mean hazard NLL
    readout: Tensor                # detached raw mean readout CE (0 when the term is off)
    valid_windows: Tensor          # detached count of windows that carried hazard gradient


def crosses_metric_keys(cfg: ModelCfg) -> dict[str, str] | None:
    """Metric routing override for ``MetricAccumulator``, or ``None`` to keep the default contract.

    The single effect of ``model.onset_report_crosses``: score ``crosses`` on the hazard readout
    instead of ``crosses_frame``. Nothing about the loss changes — which is what makes "hazard as an
    auxiliary task" (report from ``crosses_frame``) and "pure reformulation" (report from the readout)
    two runs of the same code rather than two code paths.
    """
    from pedpredict.losses.multitask import TASK_OUTPUT_KEY  # local: multitask imports this module

    if not (cfg.onset_head and cfg.onset_report_crosses):
        return None
    return {**TASK_OUTPUT_KEY, "crosses": READOUT_OUTPUT_KEY}


def onset_spec_from_config(cfg: ModelCfg) -> OnsetSpec:
    """The head's bin geometry, from the model config that built it."""
    return OnsetSpec(
        lookahead=cfg.onset_lookahead, bin_width=cfg.onset_bin_width, horizon=cfg.onset_horizon
    )


class OnsetHazardLoss(nn.Module):
    """Masked per-bin hazard NLL, plus the optional horizon-readout CE."""

    def __init__(self, spec: OnsetSpec, *, hazard_weight: float = 1.0, readout_weight: float = 0.0) -> None:
        super().__init__()
        self.spec = spec
        self.hazard_weight = float(hazard_weight)
        self.readout_weight = float(readout_weight)

    def forward(self, outputs: dict[str, Tensor], labels: dict[str, Tensor]) -> OnsetLossOutput:
        """``outputs`` needs the onset keys; ``labels`` needs the three S1 fields."""
        if HAZARD_OUTPUT_KEY not in outputs:
            raise KeyError(
                f"OnsetHazardLoss needs '{HAZARD_OUTPUT_KEY}' in the model outputs (have: "
                f"{sorted(outputs)}). Set model.onset_head=true so the head is built."
            )
        hazard_logits = outputs[HAZARD_OUTPUT_KEY].float()
        self._check_width(hazard_logits)
        targets = hazard_targets(labels, self.spec)

        # Per-bin BCE, silenced by the mask. `mask` already carries the valid-window gate, so an
        # excluded window contributes nothing without any second masking step.
        per_bin = functional.binary_cross_entropy_with_logits(
            hazard_logits, targets.target.to(hazard_logits.dtype), reduction="none"
        )
        per_window = (per_bin * targets.mask.to(per_bin.dtype)).sum(dim=1)
        n_valid = targets.valid.sum()
        hazard = per_window.sum() / n_valid.clamp(min=1)

        total = self.hazard_weight * hazard
        readout = torch.zeros((), device=hazard.device, dtype=hazard.dtype)
        if self.readout_weight:
            readout = self._readout_term(outputs, labels)
            total = total + self.readout_weight * readout
        return OnsetLossOutput(
            total=total, hazard=hazard.detach(), readout=readout.detach(), valid_windows=n_valid.detach()
        )

    def _readout_term(self, outputs: dict[str, Tensor], labels: dict[str, Tensor]) -> Tensor:
        """CE on the horizon-collapsed readout, over windows whose horizon-``H`` answer is knowable."""
        if READOUT_OUTPUT_KEY not in outputs:
            raise KeyError(
                f"onset_readout_weight is non-zero but '{READOUT_OUTPUT_KEY}' is missing from the model "
                f"outputs (have: {sorted(outputs)})."
            )
        logits = outputs[READOUT_OUTPUT_KEY].float()
        label, valid = readout_targets(labels, self.spec)
        per_sample = functional.cross_entropy(logits, label.long(), reduction="none")
        return (per_sample * valid.to(per_sample.dtype)).sum() / valid.sum().clamp(min=1)

    def _check_width(self, hazard_logits: Tensor) -> None:
        if hazard_logits.shape[-1] != self.spec.num_bins:
            raise ValueError(
                f"OnsetHazardLoss: model emits {hazard_logits.shape[-1]} hazard bins but the loss was "
                f"built for {self.spec.num_bins} (lookahead={self.spec.lookahead}, "
                f"bin_width={self.spec.bin_width}). Model and loss were built from different configs."
            )


def build_onset_loss(model_cfg: ModelCfg, train_cfg: TrainCfg) -> OnsetHazardLoss | None:
    """Wire the onset loss from config, or ``None`` when it contributes nothing.

    ``None`` whenever the head is off or both weights are zero — including the crosses-inactive case,
    which :meth:`TrainCfg.effective_onset_weights` zeroes — so the caller can skip the term entirely
    rather than adding a no-op tensor to every batch.
    """
    if not model_cfg.onset_head:
        return None
    hazard_weight, readout_weight = train_cfg.effective_onset_weights()
    if not hazard_weight and not readout_weight:
        return None
    return OnsetHazardLoss(
        onset_spec_from_config(model_cfg),
        hazard_weight=hazard_weight,
        readout_weight=readout_weight,
    )

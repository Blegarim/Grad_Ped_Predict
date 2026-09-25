"""Backbone BatchNorm diagnostics — checks (a) and (b) of docs/RECIPE_V2_PLAN.md.

Why this exists: ``freeze_vit_backbone`` sets ``requires_grad=False``, but the Trainer puts the whole model
in train mode, so a frozen timm backbone's BatchNorm layers still normalise with per-batch statistics and
overwrite their pretrained running statistics every step. These helpers measure that drift on a saved
checkpoint and swap BN statistics in and out, so a script can see what the drift does to the scores.

Pure helpers over ``nn.Module`` state — no data, no files. ``scripts/diagnose_backbone_bn.py`` owns
loading, evaluation and output.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from pedpredict.utils.amp import autocast_ctx

__all__ = [
    "BNState",
    "BNDrift",
    "batchnorm_layers",
    "bn_state",
    "load_bn_state",
    "bn_drift",
    "summarize_drift",
    "max_param_difference",
    "reestimate_bn",
    "score_summary",
]

#: ``{layer name: {"running_mean": [C], "running_var": [C]}}`` — CPU float32 copies.
BNState = dict[str, dict[str, Tensor]]

_STAT_KEYS = ("running_mean", "running_var")


def batchnorm_layers(module: nn.Module) -> dict[str, _BatchNorm]:
    """Every BatchNorm layer in ``module`` that keeps running statistics, by qualified name."""
    return {
        name: layer
        for name, layer in module.named_modules()
        if isinstance(layer, _BatchNorm) and layer.running_mean is not None and layer.running_var is not None
    }


def bn_state(module: nn.Module) -> BNState:
    """Detached CPU copies of every BN layer's running statistics."""
    return {
        name: {key: getattr(layer, key).detach().float().cpu().clone() for key in _STAT_KEYS}
        for name, layer in batchnorm_layers(module).items()
    }


def load_bn_state(module: nn.Module, state: BNState) -> None:
    """Copy ``state`` into ``module``'s BN running statistics. Layer names must match exactly."""
    layers = batchnorm_layers(module)
    missing, extra = sorted(set(layers) - set(state)), sorted(set(state) - set(layers))
    if missing or extra:
        raise KeyError(f"load_bn_state: layer mismatch — missing {missing[:3]}, unexpected {extra[:3]}")
    with torch.no_grad():
        for name, layer in layers.items():
            for key in _STAT_KEYS:
                target = getattr(layer, key)
                target.copy_(state[name][key].to(device=target.device, dtype=target.dtype))


@dataclass(frozen=True)
class BNDrift:
    """How far one layer's statistics moved from a reference, per channel then summarised.

    ``mean_shift_sd`` is ``|mu - mu_ref| / sd_ref`` — the shift in units of the reference spread, so layers
    of different scale are comparable. ``var_ratio_*`` is ``var / var_ref`` (1.0 = unchanged).
    """

    layer: str
    channels: int
    mean_shift_sd: float
    max_mean_shift_sd: float
    var_ratio_median: float
    var_ratio_min: float
    var_ratio_max: float


def bn_drift(reference: BNState, current: BNState, *, eps: float = 1e-5) -> list[BNDrift]:
    """Per-layer drift of ``current`` from ``reference`` (same layer names required)."""
    if set(reference) != set(current):
        raise KeyError("bn_drift: reference and current cover different BN layers")
    rows = []
    for name in reference:
        mu_ref, var_ref = reference[name]["running_mean"], reference[name]["running_var"]
        mu, var = current[name]["running_mean"], current[name]["running_var"]
        shift = (mu - mu_ref).abs() / (var_ref + eps).sqrt()
        ratio = (var + eps) / (var_ref + eps)
        rows.append(BNDrift(
            layer=name,
            channels=int(mu_ref.numel()),
            mean_shift_sd=float(shift.mean()),
            max_mean_shift_sd=float(shift.max()),
            var_ratio_median=float(ratio.median()),
            var_ratio_min=float(ratio.min()),
            var_ratio_max=float(ratio.max()),
        ))
    return rows


def summarize_drift(rows: Iterable[BNDrift]) -> dict[str, float]:
    """Network-level summary of :func:`bn_drift` rows."""
    rows = list(rows)
    if not rows:
        raise ValueError("summarize_drift: no BN layers")
    shifts = np.array([r.mean_shift_sd for r in rows])
    log_ratios = np.log(np.array([r.var_ratio_median for r in rows]))
    return {
        "layers": float(len(rows)),
        "median_layer_mean_shift_sd": float(np.median(shifts)),
        "max_layer_mean_shift_sd": float(shifts.max()),
        "share_layers_shift_over_0.5sd": float((shifts > 0.5).mean()),
        "median_layer_var_ratio": float(np.exp(np.median(log_ratios))),
        "most_extreme_layer_var_ratio": float(np.exp(log_ratios[np.abs(log_ratios).argmax()])),
    }


def max_param_difference(module: nn.Module, reference: nn.Module) -> float:
    """Largest absolute difference over same-named parameters — ``0.0`` means the weights truly stayed frozen."""
    params, ref = dict(module.named_parameters()), dict(reference.named_parameters())
    if set(params) != set(ref):
        raise KeyError("max_param_difference: modules have different parameter names")
    with torch.no_grad():
        return max(
            (float((params[k].detach().float().cpu() - ref[k].detach().float().cpu()).abs().max()) for k in params),
            default=0.0,
        )


def reestimate_bn(
    module: nn.Module, batches: Iterable[Tensor], *, use_amp: bool = False, device_type: str = "cuda"
) -> int:
    """Forward ``batches`` through ``module`` in train mode without gradients, then switch it to eval.

    This is exactly what every training step does to a frozen-but-train-mode backbone: parameters stay
    put, BN running statistics move toward the batches just seen. Returns the number of batches consumed.
    """
    module.train()
    seen = 0
    try:
        with torch.no_grad():
            for batch in batches:
                with autocast_ctx(use_amp, device_type):
                    module(batch)
                seen += 1
    finally:
        module.eval()
    return seen


def score_summary(y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    """Where the scores sit, not just how they rank: the offset a BN shift would move."""
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=np.float64)
    pos, neg = prob[y_true == 1], prob[y_true == 0]
    return {
        "mean_prob": float(prob.mean()),
        "mean_prob_positives": float(pos.mean()) if pos.size else float("nan"),
        "mean_prob_negatives": float(neg.mean()) if neg.size else float("nan"),
        "share_predicted_positive": float((prob >= threshold).mean()),
    }

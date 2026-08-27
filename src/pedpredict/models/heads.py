"""Task heads + temporal-pooling helpers.

Factored out of OLD ``models/Cross_Attention_Module.py`` (and the copy-pasted equivalents in
``models/AblationModels.py``) so the output contract is testable in isolation and shared by the full
model + every ablation without duplication.

Design constraint — **state_dict key parity**. These are *builder functions* returning bare
``nn.Sequential`` / ``nn.ModuleDict`` / ``nn.Linear``, assigned to the OLD attribute names
(``pool_mlp`` / ``classifier`` / ``crosses_frame_head``) by the caller. A wrapping ``nn.Module`` would
rename keys (``heads.pool_mlp.0`` …) and break ``strict=True`` loading of legacy checkpoints; builders
keep the keys byte-for-byte. The pooling / frame-reduction logic lives here as *stateless* functions.

Resolved band-aid:

* **B4 (dead crosses-pooled head).** ``build_task_classifiers`` still builds ALL three task heads
  (incl. ``crosses``) so legacy param layout is preserved 1:1. Whether the ``crosses`` head is invoked
  (-> the ``crosses_pooled`` output) is the caller's gated, documented decision -- never silent.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

FramePool = Literal["logsumexp", "max", "mean"]
FRAME_POOLS: tuple[str, ...] = ("logsumexp", "max", "mean")

_LOG2 = math.log(2.0)
#: Floor on ``|log survival|`` in the readout — caps the reported onset probability's smallest
#: representable value at ~1e-6 instead of letting an all-zero hazard vector produce ``-inf``.
_LOG1MEXP_EPS = 1e-6


def build_pool_mlp(d_model: int) -> nn.Sequential:
    """Temporal-attention scoring MLP ``[d -> d//2 -> 1]`` (legacy ``pool_mlp``). Keys ``0`` / ``2``."""
    return nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.ReLU(),
        nn.Linear(d_model // 2, 1),
    )


def build_task_classifiers(num_classes: dict[str, int], d_model: int, dropout: float) -> nn.ModuleDict:
    """Per-task classifier MLPs ``[d -> d -> drop -> C]`` (legacy ``classifier`` ModuleDict).

    Builds every task in ``num_classes`` (incl. ``crosses``) for legacy param-layout parity; the keys are
    ``classifier.<task>.0`` / ``.3``. Invocation of the ``crosses`` head is the caller's gated decision (B4).
    """
    return nn.ModuleDict(
        {
            name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n),
            )
            for name, n in num_classes.items()
        }
    )


def build_crosses_frame_head(d_model: int, num_crosses: int) -> nn.Linear:
    """Per-frame crosses head ``Linear(d, C)`` (legacy ``crosses_frame_head``)."""
    return nn.Linear(d_model, num_crosses)


def build_onset_hazard_head(d_model: int, num_bins: int) -> nn.Linear:
    """Onset-timing head ``Linear(d, K)`` — one hazard logit per future bin (``onset_head``).

    Note the axis: ``crosses_frame_head`` runs per *observed* frame and is pooled over ``T`` (the past);
    this one runs on the pooled ``[B, D]`` vector and its ``K`` outputs index the *future*. Two time
    axes, only one of which the model has seen.
    """
    return nn.Linear(d_model, num_bins)


def _log1mexp(x: Tensor) -> Tensor:
    """``log(1 - exp(x))`` for ``x <= 0``, in the stable branch for each magnitude.

    Both branches are evaluated on clamped inputs before :func:`torch.where` selects between them —
    ``where`` computes both sides, and a ``nan`` in the discarded branch still poisons the backward
    pass. ``x`` is clamped just below 0 so an all-zero hazard vector yields a very small probability
    rather than ``log(0) = -inf``.
    """
    x = x.clamp(max=-_LOG1MEXP_EPS)
    near_zero = x > -_LOG2
    safe_near = torch.where(near_zero, x, torch.full_like(x, -_LOG2))
    safe_far = torch.where(near_zero, torch.full_like(x, -_LOG2), x)
    return torch.where(near_zero, torch.log(-torch.expm1(safe_near)), torch.log1p(-torch.exp(safe_far)))


def hazard_to_horizon_logits(hazard_logits: Tensor, horizon_bins: int) -> Tensor:
    """``[B, K]`` hazard logits -> ``[B, 2]`` class logits for *onset within the horizon*.

    The readout that keeps the onset head comparable with the four binary baselines:

    ``P(onset <= H) = 1 - prod_k (1 - h_k)``  over the first ``horizon_bins`` bins.

    Returned as two-class logits (column 1 = onset, matching every other head, since metrics read
    ``softmax(...)[:, 1]``) so the existing loss / accumulator / threshold machinery consumes it with
    no change. The product is computed as a sum of log-survivals, which is why nothing here ever
    forms a probability directly.

    **Deliberate B8 exception**: this upcasts to float32 itself instead of leaving the cast to
    ``MultiTaskLoss`` / ``MetricAccumulator``. Summing up to ``K`` log-survivals and then taking
    ``log(1 - exp(.))`` in fp16 loses the small probabilities this task is entirely about.
    """
    num_bins = hazard_logits.shape[-1]
    if not 1 <= horizon_bins <= num_bins:
        raise ValueError(
            f"hazard_to_horizon_logits: horizon_bins={horizon_bins} outside [1, K={num_bins}]. "
            f"The reported horizon must fit inside the head's lookahead."
        )
    logits = hazard_logits.float()[:, :horizon_bins]
    log_survive = nn.functional.logsigmoid(-logits).sum(dim=1)   # log prod (1 - h_k)
    return torch.stack((log_survive, _log1mexp(log_survive)), dim=1)


def temporal_attention_pool(feats: torch.Tensor, pool_mlp: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax-weighted temporal pool (legacy lines 54-56).

    ``feats [B, T, D]`` -> ``(pooled [B, D], weights [B, T])``. ``weights`` is the per-frame softmax over
    the time axis (already squeezed), reused as the ``temporal_weights`` output.
    """
    scores = pool_mlp(feats)                       # [B, T, 1]
    weights = torch.softmax(scores, dim=1)         # [B, T, 1]
    pooled = (feats * weights).sum(dim=1)          # [B, D]
    return pooled, weights.squeeze(-1)             # [B, D], [B, T]


def frame_pool_reduce(frame_logits: torch.Tensor, mode: FramePool) -> torch.Tensor:
    """Reduce per-frame crosses logits ``[B, T, C]`` over time -> ``[B, C]`` (legacy lines 67-74)."""
    if mode == "logsumexp":
        return torch.logsumexp(frame_logits, dim=1)
    if mode == "max":
        return frame_logits.max(dim=1).values
    if mode == "mean":
        return frame_logits.mean(dim=1)
    raise ValueError(f"Unsupported frame_pool: {mode}")


def emit_task_logits(
    feats: torch.Tensor,
    pool_mlp: nn.Module,
    classifier: nn.ModuleDict,
    crosses_frame_head: nn.Module,
    *,
    frame_pool: FramePool,
    use_frame_crosses: bool,
    emit_crosses_pooled: bool,
    emit_temporal_weights: bool,
    onset_head: nn.Module | None = None,
    onset_horizon_bins: int = 0,
) -> dict[str, torch.Tensor]:
    """Shared output-contract head block: pooled heads + B4 gate + frame reduce.

    The identical tail of ``CrossAttentionModule`` and all three ablations,
    factored here so the output contract lives in ONE place. ``feats [B, T, D]`` are the post-fusion /
    post-encoder features to pool and classify.

    Emission order matches the legacy modules: pooled ``actions`` / ``looks`` -> gated ``crosses_pooled``
    (B4: the legacy-dead ``classifier['crosses']`` head, live-but-unsupervised) -> ``crosses_frame``
    (logsumexp/max/mean over time) -> gated ``crosses_hazard`` / ``crosses_readout`` -> ``temporal_weights``
    (full model only). Gating ``crosses_pooled`` / ``onset_head`` / ``temporal_weights`` never perturbs
    the other keys.

    ``onset_head`` (``None`` = off, the default and the state every existing run was trained in) adds the
    two onset keys: ``crosses_hazard [B, K]`` raw per-future-bin hazard logits, and ``crosses_readout
    [B, 2]``, the horizon-collapsed two-class view of the same numbers that keeps the binary baselines
    comparable. ``onset_horizon_bins`` is how many leading bins the readout multiplies over.
    """
    pooled, weights = temporal_attention_pool(feats, pool_mlp)  # [B, D], [B, T]
    logits: dict[str, torch.Tensor] = {}
    for key, head in classifier.items():
        if key == "crosses":
            if emit_crosses_pooled:
                logits["crosses_pooled"] = head(pooled)
        else:
            logits[key] = head(pooled)
    if use_frame_crosses:
        logits["crosses_frame"] = frame_pool_reduce(crosses_frame_head(feats), frame_pool)
    if onset_head is not None:
        # Onset-timing head (docs/METHODOLOGY.md prong 2). Emitted LAST and only when built, so with
        # `model.onset_head=false` the dict is key-for-key and value-for-value what it has always been
        # — the golden characterization tests pin exactly that.
        hazard = onset_head(pooled)                                    # [B, K] per-future-bin logits
        logits["crosses_hazard"] = hazard
        logits["crosses_readout"] = hazard_to_horizon_logits(hazard, onset_horizon_bins)
    if emit_temporal_weights:
        logits["temporal_weights"] = weights
    return logits

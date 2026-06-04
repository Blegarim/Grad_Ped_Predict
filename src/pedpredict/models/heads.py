"""Task heads + temporal-pooling helpers (Prompt 2.3).

Factored out of OLD ``models/Cross_Attention_Module.py`` (and the copy-pasted equivalents in
``models/AblationModels.py``) so the output contract is testable in isolation and shared by the full
model + every ablation (Prompt 2.5) without duplication.

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

from typing import Literal

import torch
import torch.nn as nn

FramePool = Literal["logsumexp", "max", "mean"]
FRAME_POOLS: tuple[str, ...] = ("logsumexp", "max", "mean")


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
) -> dict[str, torch.Tensor]:
    """Shared output-contract head block (Prompt 2.5): pooled heads + B4 gate + frame reduce.

    The identical tail of ``CrossAttentionModule`` (Prompt 2.3) and all three ablations (Prompt 2.5),
    factored here so the output contract lives in ONE place. ``feats [B, T, D]`` are the post-fusion /
    post-encoder features to pool and classify.

    Emission order matches the legacy modules: pooled ``actions`` / ``looks`` -> gated ``crosses_pooled``
    (B4: the legacy-dead ``classifier['crosses']`` head, live-but-unsupervised) -> ``crosses_frame``
    (logsumexp/max/mean over time) -> ``temporal_weights`` (full model only). Gating ``crosses_pooled`` /
    ``temporal_weights`` never perturbs the other keys.
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
    if emit_temporal_weights:
        logits["temporal_weights"] = weights
    return logits

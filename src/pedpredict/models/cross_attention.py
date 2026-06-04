"""Cross-attention fusion + per-task heads (Prompt 2.3).

Port of OLD ``models/Cross_Attention_Module.CrossAttentionModule``. Behavior-preserving for every key
the legacy module actually emitted (``actions``, ``looks``, ``crosses_frame``, ``temporal_weights``):
numerically equivalent to the legacy module for identical weights+input in eval mode.

Resolved band-aid:

* **B4 (dead/ambiguous crosses head).** The legacy module *allocated* a ``classifier["crosses"]`` MLP but
  its ``forward`` skipped it (``if key != "crosses"``) -- so ``crosses_pooled`` was **never emitted** and
  the head sat as a dead parameter (checkpointed, optimizer-tracked, never run). The docs nonetheless
  advertised a 5-key contract. We resolve the drift by making the head **live but unsupervised**: it is
  computed every forward and emitted as ``crosses_pooled`` (``emit_crosses_pooled=True`` by default), kept
  ready to swap in for ``crosses_frame`` later, but **never routed to the loss** (training/eval supervise
  ``crosses_frame`` only -- see losses 3.1 / metrics 3.2). This is an intentional, flagged ADDITION over
  the legacy output (the 4 legacy keys keep exact golden parity; ``crosses_pooled`` has its own golden
  reference recomputed from the legacy weights). The legacy param layout is preserved 1:1, so an OLD
  ``state_dict`` still loads ``strict=True``.

Other intentional change:

* **``key_padding_mask`` removed.** The legacy ``forward`` accepted it but ``EnsembleModel`` /
  ``model_forward`` always called the module with two positional args (it was permanently ``None``), and
  the data layer emits fixed-length ``seq_len=20`` windows with no padding (Prompt 1.5). Dropping the
  unused parameter is behavior-neutral and removes a dead knob.

The task heads + pooling/reduction helpers live in ``heads.py`` (shared with the ablations, Prompt 2.5).
Attribute names mirror the legacy module verbatim so an OLD ``state_dict`` loads ``strict=True``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pedpredict.config import ModelCfg
from pedpredict.models.heads import (
    FramePool,
    build_crosses_frame_head,
    build_pool_mlp,
    build_task_classifiers,
    emit_task_logits,
)


class CrossAttentionModule(nn.Module):
    """MultiheadAttention (query=motion, key/value=image) -> temporal pool -> per-task heads.

    Inputs ``motion_feats [B, T, D]`` (query) + ``image_feats [B, T, D]`` (key/value) -> logits dict.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("CrossAttentionModule requires num_classes_dict (e.g. ModelCfg.num_classes).")

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        # Heads built via heads.py so the keys match legacy 1:1 (pool_mlp / classifier / crosses_frame_head).
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])

    @classmethod
    def from_config(cls, cfg: ModelCfg) -> CrossAttentionModule:
        """Build from ``ModelCfg`` (``cross_kwargs()`` == OLD ``get_model()`` wiring) + the B4 gate."""
        return cls(**cfg.cross_kwargs(), emit_crosses_pooled=cfg.emit_crosses_pooled)

    def forward(self, motion_feats: torch.Tensor, image_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        """``motion_feats``/``image_feats`` ``[B, T, D]`` -> per-task logits dict (see module docstring)."""
        attn_output, _ = self.cross_attn(
            query=motion_feats,
            key=image_feats,
            value=image_feats,
        )  # [B, T, D]

        # Shared output-contract block (heads.emit_task_logits) — the full model emits temporal_weights.
        return emit_task_logits(
            attn_output,
            self.pool_mlp,
            self.classifier,
            self.crosses_frame_head,
            frame_pool=self.frame_pool,
            use_frame_crosses=self.use_frame_crosses,
            emit_crosses_pooled=self.emit_crosses_pooled,
            emit_temporal_weights=True,
        )


if __name__ == "__main__":  # B6: smoke test driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    _t = 20
    model = CrossAttentionModule.from_config(cfg).eval()
    motion = torch.randn(2, _t, cfg.d_model)
    image = torch.randn(2, _t, cfg.d_model)
    with torch.no_grad():
        out = model(motion, image)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CrossAttentionModule OK | keys {sorted(out)} | params {n_params}")
    assert out["actions"].shape == (2, cfg.num_classes["actions"])
    assert out["crosses_frame"].shape == (2, cfg.num_classes["crosses"])
    assert out["crosses_pooled"].shape == (2, cfg.num_classes["crosses"])
    assert out["temporal_weights"].shape == (2, _t)

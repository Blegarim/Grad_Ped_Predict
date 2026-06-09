"""EnsembleModel — the full multimodal model.

Port of OLD ``models/Unified_Module.EnsembleModel``. Behavior-preserving: numerically equivalent to the
legacy module for identical weights+input in eval mode (the two ``LayerNorm`` fusions + the
cross-attention wiring carry no stochastic ops once Dropout/MHA-dropout are identities).

Wiring (verbatim, LayerNorm-before-fusion preserved)::

    images_context -> vit          -> image_norm  ─┐
                                                    ├─ cross_attention(motion_feats, image_feats)
    (motions, images_tight) -> motion_enc -> motion_norm ┘

The output is the ``cross_attention`` logits dict (keys per the Prompt 2.3 contract:
``actions``, ``looks``, ``crosses_frame``, ``temporal_weights`` + the live-but-unsupervised
``crosses_pooled``). ``return_feats=True`` additionally returns ``(image_feats, motion_feats)`` — the
path the qualitative-viz code consumes.

Attribute names mirror the legacy module verbatim (``motion_enc``, ``vit``, ``cross_attention``,
``image_norm``, ``motion_norm``) so an OLD full-model ``state_dict`` loads ``strict=True``.

Intentional change (behavior-neutral): ``cross_attention`` is called with two positional args — the
legacy ``key_padding_mask`` was permanently ``None`` at every call site and was dropped in Prompt 2.3
(fixed ``seq_len=20`` windows, no padding).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pedpredict.config import ModelCfg
from pedpredict.models.cross_attention import CrossAttentionModule
from pedpredict.models.motion_encoder import MotionEncoder
from pedpredict.models.vit import ViT_Hierarchical

LogitsDict = dict[str, torch.Tensor]


class EnsembleModel(nn.Module):
    """ViT + MotionEncoder + CrossAttention, fused with LayerNorm-before-fusion. Output = logits dict."""

    def __init__(
        self,
        motion_enc: MotionEncoder,
        vit: ViT_Hierarchical,
        cross_attention: CrossAttentionModule,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.motion_enc = motion_enc
        self.vit = vit
        self.cross_attention = cross_attention
        self.image_norm = nn.LayerNorm(d_model)
        self.motion_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> EnsembleModel:
        """Build all three sub-modules from ``cfg`` (== OLD ``get_model('full', ...)`` wiring) and wire them.

        ``img_size`` flows from ``DataCfg.read_context_height`` at the call site (the ViT is
        resolution-bound — see Prompt 2.1); ``cross_attention`` is built via ``CrossAttentionModule.from_config``
        so the ``num_heads=4`` / ``frame_pool`` / ``emit_crosses_pooled`` decisions stay singular (2.3).
        """
        return cls(
            motion_enc=MotionEncoder.from_config(cfg),
            vit=ViT_Hierarchical.from_config(cfg, img_size),
            cross_attention=CrossAttentionModule.from_config(cfg),
            d_model=cfg.d_model,
        )

    def forward(
        self,
        images_tight: torch.Tensor,
        images_context: torch.Tensor,
        motions: torch.Tensor,
        return_feats: bool = False,
    ) -> LogitsDict | tuple[LogitsDict, torch.Tensor, torch.Tensor]:
        """``images_tight/images_context [B, T, 3, H, W]``, ``motions [B, T, motion_dim]`` -> logits dict.

        With ``return_feats=True`` also returns ``(image_feats, motion_feats)`` (both ``[B, T, d_model]``),
        the post-LayerNorm fusion features used by the qualitative-viz path.
        """
        # --- Vision Transformer branch (LayerNorm before fusion) ---
        image_feats = self.vit(images_context)  # [B, T, D]
        image_feats = self.image_norm(image_feats)

        # --- Motion branch (LayerNorm before fusion) ---
        motion_out = self.motion_enc(motions, images_tight)  # [B, T, D]
        motion_feats = self.motion_norm(motion_out)

        # --- Cross-attention fusion (two positional args; key_padding_mask dropped in 2.3) ---
        logits = self.cross_attention(motion_feats, image_feats)

        if return_feats:
            return logits, image_feats, motion_feats
        return logits


if __name__ == "__main__":  # B6: smoke test driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    _b, _t, _img = 2, 3, 224
    model = EnsembleModel.from_config(cfg, img_size=_img).eval()
    tight = torch.randn(_b, _t, cfg.in_channels, 128, 128)
    context = torch.randn(_b, _t, cfg.in_channels, _img, _img)
    motion = torch.randn(_b, _t, cfg.motion_dim)
    with torch.no_grad():
        out = model(tight, context, motion)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EnsembleModel OK | keys {sorted(out)} | params {n_params}")
    assert out["actions"].shape == (_b, cfg.num_classes["actions"])
    assert out["crosses_frame"].shape == (_b, cfg.num_classes["crosses"])
    assert out["temporal_weights"].shape == (_b, _t)

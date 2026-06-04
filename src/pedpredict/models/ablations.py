"""Ablation models (Prompt 2.5) — port of OLD ``models/AblationModels.py``.

``MotionOnlyModel`` (motion branch only), ``VisualOnlyModel`` (ViT branch only) and
``VanillaConcatModel`` (concat-then-fuse instead of cross-attention). Behavior-preserving: numerically
equivalent to the legacy modules for identical weights+input in eval mode (golden parity vs
``tests/fixtures/golden/ensemble.pt``).

Output contract (per CLAUDE.md / Prompt 2.3, kept singular with the full model):

* All three emit ``actions``, ``looks``, ``crosses_frame`` (the ONLY supervised key — logsumexp-pooled
  over frames) and the B4 ``crosses_pooled`` auxiliary (live-but-unsupervised, gated by
  ``emit_crosses_pooled``, default on).
* None emit ``temporal_weights`` — that is structurally full-model-only (the legacy ablation ``forward``
  never returned it; the pooling softmax weights are computed internally but not exposed).

Resolved band-aids:

* **B4 (dead ``crosses_pooled`` head).** Each legacy ablation allocated ``classifier['crosses']`` but its
  ``forward`` skipped it (``if key != 'crosses'``) — a dead parameter, exactly as in the full model. The
  rebuild makes it **live-but-unsupervised** uniformly with ``CrossAttentionModule`` (heads.emit_task_logits
  with ``emit_crosses_pooled``). Param layout is unchanged, so OLD ablation checkpoints still load
  ``strict=True``. Flagged ADDITION over the 3-key legacy output (the auxiliary is never routed to loss).
* **B6 (config drift).** ``from_config(ModelCfg)`` is the single wiring source; ``__main__`` is a smoke test.
* **B10 (stringly dispatch).** Selection/forward routing live in ``registry.py`` (Prompt 2.4); these classes
  expose the uniform ``from_config(cfg, img_size)`` + the per-type ``forward`` signatures it dispatches to.

Intentional, behavior-neutral change:

* The legacy per-call ``frame_pool`` ``forward`` argument is dropped (it was permanently the default at every
  call site, like the 2.3 ``key_padding_mask`` removal). The pooling mode is fixed at construction
  (``self.frame_pool`` from ``cfg.frame_pool``), matching ``CrossAttentionModule`` / ``EnsembleModel``.

Attribute names mirror the legacy modules verbatim (``norm`` / ``motion_norm`` / ``visual_norm`` /
``fusion`` / ``pool_mlp`` / ``classifier`` / ``crosses_frame_head`` + the ``motion_enc`` / ``vit``
sub-encoders) so an OLD ablation ``state_dict`` loads ``strict=True``. The pooled/frame heads are built via
``heads.py`` so their keys stay byte-identical to the full model's.
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
from pedpredict.models.motion_encoder import MotionEncoder
from pedpredict.models.vit import ViT_Hierarchical


def _head_kwargs(cfg: ModelCfg) -> dict:
    """Ablation head wiring from ``cfg`` (== OLD ``get_model(dropout=0.1)`` + the 2.3 contract gates).

    Shared by all three ``from_config`` builders so the head/contract decisions stay in one place.
    """
    return {
        "d_model": cfg.d_model,
        "num_classes_dict": dict(cfg.num_classes),
        "dropout": cfg.head_dropout,
        "use_frame_crosses": cfg.use_frame_crosses,
        "frame_pool": cfg.frame_pool,
        "emit_crosses_pooled": cfg.emit_crosses_pooled,
    }


class MotionOnlyModel(nn.Module):
    """Motion-encoder-only ablation. ``forward(motions, images_tight)`` -> logits dict (no ViT branch)."""

    def __init__(
        self,
        motion_enc: MotionEncoder,
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("MotionOnlyModel requires num_classes_dict (e.g. ModelCfg.num_classes).")
        self.motion_enc = motion_enc
        self.d_model = d_model
        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        self.norm = nn.LayerNorm(d_model)
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> MotionOnlyModel:
        """Build from ``cfg`` (== OLD ``get_model('motion_only', ...)``). ``img_size`` unused (no ViT)."""
        return cls(motion_enc=MotionEncoder.from_config(cfg), **_head_kwargs(cfg))

    def forward(self, motions: torch.Tensor, images_tight: torch.Tensor) -> dict[str, torch.Tensor]:
        """``motions [B, T, motion_dim]`` + ``images_tight [B, T, 3, H, W]`` -> per-task logits dict."""
        feats = self.norm(self.motion_enc(motions, images_tight))  # [B, T, D]
        return emit_task_logits(
            feats,
            self.pool_mlp,
            self.classifier,
            self.crosses_frame_head,
            frame_pool=self.frame_pool,
            use_frame_crosses=self.use_frame_crosses,
            emit_crosses_pooled=self.emit_crosses_pooled,
            emit_temporal_weights=False,
        )


class VisualOnlyModel(nn.Module):
    """ViT-only ablation. ``forward(images_context)`` -> logits dict (no motion branch)."""

    def __init__(
        self,
        vit: ViT_Hierarchical,
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("VisualOnlyModel requires num_classes_dict (e.g. ModelCfg.num_classes).")
        self.vit = vit
        self.d_model = d_model
        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        self.norm = nn.LayerNorm(d_model)
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> VisualOnlyModel:
        """Build from ``cfg`` (== OLD ``get_model('visual_only', ...)``); ``img_size`` sizes the ViT (2.1)."""
        return cls(vit=ViT_Hierarchical.from_config(cfg, img_size), **_head_kwargs(cfg))

    def forward(self, images_context: torch.Tensor) -> dict[str, torch.Tensor]:
        """``images_context [B, T, 3, H, W]`` -> per-task logits dict."""
        feats = self.norm(self.vit(images_context))  # [B, T, D]
        return emit_task_logits(
            feats,
            self.pool_mlp,
            self.classifier,
            self.crosses_frame_head,
            frame_pool=self.frame_pool,
            use_frame_crosses=self.use_frame_crosses,
            emit_crosses_pooled=self.emit_crosses_pooled,
            emit_temporal_weights=False,
        )


class VanillaConcatModel(nn.Module):
    """Concat-fusion ablation: ``[motion ; image]`` -> MLP fusion (no cross-attention).

    ``forward(images_tight, images_context, motions)`` -> logits dict.
    """

    def __init__(
        self,
        motion_enc: MotionEncoder,
        vit: ViT_Hierarchical,
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("VanillaConcatModel requires num_classes_dict (e.g. ModelCfg.num_classes).")
        self.motion_enc = motion_enc
        self.vit = vit
        self.d_model = d_model
        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        self.motion_norm = nn.LayerNorm(d_model)
        self.visual_norm = nn.LayerNorm(d_model)
        # Fusion after concatenation (ablation-specific; legacy keys fusion.0 / fusion.3).
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> VanillaConcatModel:
        """Build from ``cfg`` (== OLD ``get_model('vanilla_concat', ...)``); ``img_size`` sizes the ViT."""
        return cls(
            motion_enc=MotionEncoder.from_config(cfg),
            vit=ViT_Hierarchical.from_config(cfg, img_size),
            **_head_kwargs(cfg),
        )

    def forward(
        self,
        images_tight: torch.Tensor,
        images_context: torch.Tensor,
        motions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Tight+context frames + motions -> per-task logits dict (concat-then-fuse fusion)."""
        image_feats = self.visual_norm(self.vit(images_context))  # [B, T, D]
        motion_feats = self.motion_norm(self.motion_enc(motions, images_tight))  # [B, T, D]
        # Legacy concat order is [motion, image] (OLD AblationModels.py:185) — must not be reversed.
        fused = self.fusion(torch.cat([motion_feats, image_feats], dim=-1))  # [B, T, D]
        return emit_task_logits(
            fused,
            self.pool_mlp,
            self.classifier,
            self.crosses_frame_head,
            frame_pool=self.frame_pool,
            use_frame_crosses=self.use_frame_crosses,
            emit_crosses_pooled=self.emit_crosses_pooled,
            emit_temporal_weights=False,
        )


if __name__ == "__main__":  # B6: smoke tests driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    _b, _t, _ctx = 2, 3, 224
    tight = torch.randn(_b, _t, cfg.in_channels, 128, 128)
    context = torch.randn(_b, _t, cfg.in_channels, _ctx, _ctx)
    motions = torch.randn(_b, _t, cfg.motion_dim)
    expected = {"actions", "looks", "crosses_pooled", "crosses_frame"}

    motion_only = MotionOnlyModel.from_config(cfg, _ctx).eval()
    visual_only = VisualOnlyModel.from_config(cfg, _ctx).eval()
    vanilla = VanillaConcatModel.from_config(cfg, _ctx).eval()
    with torch.no_grad():
        outs = {
            "motion_only": motion_only(motions, tight),
            "visual_only": visual_only(context),
            "vanilla_concat": vanilla(tight, context, motions),
        }
    for name, out in outs.items():
        assert set(out) == expected, f"{name}: {sorted(out)}"
        assert out["crosses_frame"].shape == (_b, cfg.num_classes["crosses"])
        assert "temporal_weights" not in out
        print(f"{name:14s} OK | keys {sorted(out)}")

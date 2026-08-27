"""Ablation models — port of OLD ``models/AblationModels.py`` (+ the M9/A6 hub-ladder additions).

``PedLocalModel`` (pedestrian-local: tight-crop CNN + bbox kinematics, no scene context — the legacy
``motion_only``, renamed per audit A6 because it *does* see pixels), ``VisualOnlyModel`` (ViT branch
only) and ``VanillaConcatModel`` (concat-then-fuse instead of cross-attention). These three are
behavior-preserving: numerically equivalent to the legacy modules for identical weights+input in eval
mode (golden parity vs ``tests/fixtures/golden/ensemble.pt``).

``KinematicsOnlyModel`` (M9.1) is a NEW, pixel-free baseline — bbox kinematics through a small
Conv1d→GRU→attention stack, *no pixels at all* (the literature's embarrassingly-strong baseline that
answers "what do the pixels actually buy?", which ``PedLocalModel`` cannot since it consumes the tight
crop). It has no legacy equivalent, so it carries no golden.

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
* **B10 (stringly dispatch).** Selection/forward routing live in ``registry.py``; these classes
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
from pedpredict.models.cross_attention import CrossAttentionModule
from pedpredict.models.heads import (
    FramePool,
    build_crosses_frame_head,
    build_onset_hazard_head,
    build_pool_mlp,
    build_task_classifiers,
    emit_task_logits,
)
from pedpredict.models.motion_encoder import MotionEncoder
from pedpredict.models.timm_backbone import build_visual_backbone


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
        "onset_bins": cfg.onset_bins,
        "onset_horizon_bins": cfg.onset_horizon_bins,
    }


class PedLocalModel(nn.Module):
    """Pedestrian-local ablation (legacy ``motion_only``, renamed per A6 — it *does* read pixels).

    ``forward(motions, images_tight)`` -> logits dict (tight-crop CNN + bbox kinematics, no ViT branch).
    """

    def __init__(
        self,
        motion_enc: MotionEncoder,
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
        onset_bins: int = 0,
        onset_horizon_bins: int = 0,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("PedLocalModel requires num_classes_dict (e.g. ModelCfg.num_classes).")
        self.motion_enc = motion_enc
        self.d_model = d_model
        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        self.norm = nn.LayerNorm(d_model)
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])
        # Onset head: allocated ONLY when on — default off keeps the parameter layout identical.
        self.onset_horizon_bins = onset_horizon_bins
        self.onset_head = build_onset_hazard_head(d_model, onset_bins) if onset_bins else None

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> PedLocalModel:
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
            onset_head=self.onset_head,
            onset_horizon_bins=self.onset_horizon_bins,
        )


class KinematicsEncoder(nn.Module):
    """Pixel-free temporal encoder over bbox kinematics (M9.1): Conv1d stack -> GRU -> self-attention.

    Structurally the ``MotionEncoder`` with its tight-crop CNN (``img_encoder`` + image-fusion concat)
    removed — the bbox-motion path alone projected to ``d_model``. The A4 ``motion_norm`` choice is shared
    verbatim (fixed image-dimension scale vs legacy per-sequence z-norm) so the kinematics baseline ablates
    on the same axis as the full model; ``motion_norm="none"`` is the pose-arm pass-through (the read path
    already normalized, and ``motion_dim`` may exceed the 9-channel scale contract).
    ``forward(motion [B, T, motion_dim]) -> [B, T, d_model]``.
    """

    def __init__(
        self,
        motion_dim: int = 8,
        hidden_dim: int = 168,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_positions: int = 200,
        motion_norm: str = "image",
        norm_image_size: tuple[int, int] = (1920, 1080),
        ego_speed_scale: float = 50.0,
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        if motion_norm not in {"image", "per_sequence", "none"}:
            raise ValueError(f"motion_norm must be 'image', 'per_sequence' or 'none'; got {motion_norm!r}")
        self.motion_norm = motion_norm
        if motion_norm == "image":
            # Same fixed per-channel scale as MotionEncoder ((cx, cy, dx, dy, w, h, dw, dh, ego) sliced).
            # "none" (the pose arm, normalized at read time) skips the buffer — its motion_dim exceeds
            # the 9-channel scale contract.
            width, height = float(norm_image_size[0]), float(norm_image_size[1])
            scale_full = [width, height, width, height, width, height, width, height, float(ego_speed_scale)]
            if motion_dim > len(scale_full):
                raise ValueError(f"motion_dim={motion_dim} exceeds the {len(scale_full)}-channel motion contract")
            self.register_buffer(
                "motion_scale", torch.tensor(scale_full[:motion_dim]).view(1, 1, motion_dim), persistent=False
            )

        # Bbox-motion Conv1d stack projected up to hidden_dim (no image branch to fuse with).
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pos_encoding = nn.Parameter(torch.randn(1, max_positions, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, d_model) if hidden_dim != d_model else nn.Identity()

    @classmethod
    def from_config(cls, cfg: ModelCfg) -> KinematicsEncoder:
        """Build from ``ModelCfg`` (shares ``motion_kwargs()`` + the A4 norm fields with MotionEncoder)."""
        return cls(
            **cfg.motion_kwargs(),
            motion_norm=cfg.motion_norm,
            norm_image_size=tuple(cfg.motion_norm_image_size),
            ego_speed_scale=cfg.ego_speed_scale,
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """``motion [B, T, motion_dim]`` (raw) -> ``[B, T, d_model]`` (no pixels consumed)."""
        t = motion.shape[1]
        max_pos = self.pos_encoding.shape[1]
        if t > max_pos:
            raise ValueError(
                f"KinematicsEncoder: sequence length T={t} exceeds positional-encoding capacity {max_pos}."
            )
        if self.motion_norm == "image":
            motion_norm = motion / self.motion_scale
        elif self.motion_norm == "per_sequence":
            motion_norm = (motion - motion.mean(dim=1, keepdim=True)) / (motion.std(dim=1, keepdim=True) + 1e-6)
        else:  # "none" — pose arm: PoseMotionTransform already normalized at read time
            motion_norm = motion
        feats = self.motion_encoder(motion_norm.transpose(1, 2))  # [B, hidden_dim, T]
        x = self.fusion(feats.transpose(1, 2))                    # [B, T, hidden_dim]
        x, _ = self.gru(x)
        x = x + self.pos_encoding[:, :t, :]
        residual = x
        x = self.norm(x)
        x, _ = self.temporal_attn(x, x, x)
        return self.proj(residual + self.dropout(x))              # [B, T, d_model]


class KinematicsOnlyModel(nn.Module):
    """Pixel-free baseline (M9.1). ``forward(motions)`` -> logits dict (bbox kinematics only, no pixels)."""

    def __init__(
        self,
        kinematics_enc: KinematicsEncoder,
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
        onset_bins: int = 0,
        onset_horizon_bins: int = 0,
    ) -> None:
        super().__init__()
        if num_classes_dict is None:
            raise ValueError("KinematicsOnlyModel requires num_classes_dict (e.g. ModelCfg.num_classes).")
        self.kinematics_enc = kinematics_enc
        self.d_model = d_model
        self.use_frame_crosses = use_frame_crosses
        self.frame_pool: FramePool = frame_pool
        self.emit_crosses_pooled = emit_crosses_pooled

        self.norm = nn.LayerNorm(d_model)
        self.pool_mlp = build_pool_mlp(d_model)
        self.classifier = build_task_classifiers(num_classes_dict, d_model, dropout)
        self.crosses_frame_head = build_crosses_frame_head(d_model, num_classes_dict["crosses"])
        # Onset head: allocated ONLY when on — default off keeps the parameter layout identical.
        self.onset_horizon_bins = onset_horizon_bins
        self.onset_head = build_onset_hazard_head(d_model, onset_bins) if onset_bins else None

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> KinematicsOnlyModel:
        """Build from ``cfg``. ``img_size`` unused (no ViT, no tight-crop CNN)."""
        return cls(kinematics_enc=KinematicsEncoder.from_config(cfg), **_head_kwargs(cfg))

    def forward(self, motions: torch.Tensor) -> dict[str, torch.Tensor]:
        """``motions [B, T, motion_dim]`` -> per-task logits dict (pixel-free)."""
        feats = self.norm(self.kinematics_enc(motions))  # [B, T, D]
        return emit_task_logits(
            feats,
            self.pool_mlp,
            self.classifier,
            self.crosses_frame_head,
            frame_pool=self.frame_pool,
            use_frame_crosses=self.use_frame_crosses,
            emit_crosses_pooled=self.emit_crosses_pooled,
            emit_temporal_weights=False,
            onset_head=self.onset_head,
            onset_horizon_bins=self.onset_horizon_bins,
        )


class VisualOnlyModel(nn.Module):
    """ViT-only ablation. ``forward(images_context)`` -> logits dict (no motion branch)."""

    def __init__(
        self,
        vit: nn.Module,  # ViT_Hierarchical (legacy) or a TimmBackbone — same [B,T,3,H,W]->[B,T,D] contract
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
        onset_bins: int = 0,
        onset_horizon_bins: int = 0,
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
        # Onset head: allocated ONLY when on — default off keeps the parameter layout identical.
        self.onset_horizon_bins = onset_horizon_bins
        self.onset_head = build_onset_hazard_head(d_model, onset_bins) if onset_bins else None

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> VisualOnlyModel:
        """Build from ``cfg`` (== OLD ``get_model('visual_only', ...)``); ``img_size`` sizes the ViT (2.1)."""
        return cls(vit=build_visual_backbone(cfg, img_size), **_head_kwargs(cfg))

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
            onset_head=self.onset_head,
            onset_horizon_bins=self.onset_horizon_bins,
        )


class VanillaConcatModel(nn.Module):
    """Concat-fusion ablation: ``[motion ; image]`` -> MLP fusion (no cross-attention).

    ``forward(images_tight, images_context, motions)`` -> logits dict.
    """

    def __init__(
        self,
        motion_enc: MotionEncoder,
        vit: nn.Module,  # ViT_Hierarchical (legacy) or a TimmBackbone — same [B,T,3,H,W]->[B,T,D] contract
        d_model: int = 128,
        num_classes_dict: dict[str, int] | None = None,
        dropout: float = 0.1,
        use_frame_crosses: bool = True,
        frame_pool: FramePool = "logsumexp",
        emit_crosses_pooled: bool = True,
        onset_bins: int = 0,
        onset_horizon_bins: int = 0,
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
        # Onset head: allocated ONLY when on — default off keeps the parameter layout identical.
        self.onset_horizon_bins = onset_horizon_bins
        self.onset_head = build_onset_hazard_head(d_model, onset_bins) if onset_bins else None

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> VanillaConcatModel:
        """Build from ``cfg`` (== OLD ``get_model('vanilla_concat', ...)``); ``img_size`` sizes the ViT."""
        return cls(
            motion_enc=MotionEncoder.from_config(cfg),
            vit=build_visual_backbone(cfg, img_size),
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
            onset_head=self.onset_head,
            onset_horizon_bins=self.onset_horizon_bins,
        )


class PoseFullModel(nn.Module):
    """Pose-arm full model (docs/POSE_ENCODER.md): the pose+bbox geometric descriptor as the
    cross-attention query over the ViT context stream — no tight crop.

    ``EnsembleModel`` with ``MotionEncoder`` swapped for :class:`KinematicsEncoder` (LayerNorm-before-
    fusion preserved). ``forward(images_context, motions)`` -> logits dict incl. ``temporal_weights``.
    New module, no legacy equivalent — carries no golden.
    """

    def __init__(
        self,
        kinematics_enc: KinematicsEncoder,
        vit: nn.Module,
        cross_attention: CrossAttentionModule,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.kinematics_enc = kinematics_enc
        self.vit = vit
        self.cross_attention = cross_attention
        self.image_norm = nn.LayerNorm(d_model)
        self.motion_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> PoseFullModel:
        """Build from ``cfg``; ``img_size`` sizes the ViT (same wiring as ``EnsembleModel.from_config``)."""
        return cls(
            kinematics_enc=KinematicsEncoder.from_config(cfg),
            vit=build_visual_backbone(cfg, img_size),
            cross_attention=CrossAttentionModule.from_config(cfg),
            d_model=cfg.d_model,
        )

    def forward(self, images_context: torch.Tensor, motions: torch.Tensor) -> dict[str, torch.Tensor]:
        """``images_context [B, T, 3, H, W]`` + ``motions [B, T, motion_dim]`` -> per-task logits dict."""
        image_feats = self.image_norm(self.vit(images_context))
        motion_feats = self.motion_norm(self.kinematics_enc(motions))
        return self.cross_attention(motion_feats, image_feats)


if __name__ == "__main__":  # B6: smoke tests driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    _b, _t, _ctx = 2, 3, 224
    tight = torch.randn(_b, _t, cfg.in_channels, 128, 128)
    context = torch.randn(_b, _t, cfg.in_channels, _ctx, _ctx)
    motions = torch.randn(_b, _t, cfg.motion_dim)
    expected = {"actions", "looks", "crosses_pooled", "crosses_frame"}

    ped_local = PedLocalModel.from_config(cfg, _ctx).eval()
    kinematics_only = KinematicsOnlyModel.from_config(cfg, _ctx).eval()
    visual_only = VisualOnlyModel.from_config(cfg, _ctx).eval()
    vanilla = VanillaConcatModel.from_config(cfg, _ctx).eval()
    with torch.no_grad():
        outs = {
            "ped_local": ped_local(motions, tight),
            "kinematics_only": kinematics_only(motions),
            "visual_only": visual_only(context),
            "vanilla_concat": vanilla(tight, context, motions),
        }
    for name, out in outs.items():
        assert set(out) == expected, f"{name}: {sorted(out)}"
        assert out["crosses_frame"].shape == (_b, cfg.num_classes["crosses"])
        assert "temporal_weights" not in out
        print(f"{name:14s} OK | keys {sorted(out)}")

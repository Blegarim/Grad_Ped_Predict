"""Pretrained ``timm`` visual backbone drop-in (RQ1 / docs/BACKBONE_STUDY.md WP2).

Slots a timm hierarchical model behind the ``ViT_Hierarchical`` contract — the visual stream is a
pure ``[B, T, 3, H, W]`` context crops -> ``[B, T, d_model]`` map — with a thin wrapper: a
global-avg-pooled timm feature extractor + a ``Linear(feat_dim, d_model)`` frame projection, exactly
mirroring how the legacy ViT projects its final stage to ``d_model``. Selection is by config
(``model.vit_backbone``): a timm model name builds :class:`TimmBackbone`; ``"legacy"`` builds
``ViT_Hierarchical``.

Compatibility (see the design note for the full candidate table):

* **Norm matches.** The read pipeline already applies ImageNet normalization (``data.norm_mean/std``),
  exactly what these ImageNet-pretrained backbones expect — no change.
* **``frame_proj`` absorbs the feature width.** Each backbone has a different pooled ``num_features``
  (320/256/576…); ``Linear(num_features, d_model)`` handles it, identical to the legacy ``36->128``.
* **Pooling caveat (the A1 finding).** ``global_pool='avg'`` re-introduces the context-crop dilution.
  This first pass uses avg-pool (the simplest, honest baseline for the swap); pedestrian-centered /
  attention pooling is a **separate follow-up spoke**, deliberately NOT bundled here.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from pedpredict.config import ModelCfg
from pedpredict.models.vit import ViT_Hierarchical


class TimmBackbone(nn.Module):
    """Pretrained timm backbone as the visual stream: ``[B, T, 3, H, W]`` -> ``[B, T, d_model]``.

    Mirrors ``ViT_Hierarchical``'s public contract (a ``from_config(cfg, img_size)`` builder + the
    per-frame forward) so it drops into ``EnsembleModel`` / the visual ablations behind the same
    ``self.vit`` attribute with no other wiring change.
    """

    def __init__(
        self,
        name: str,
        d_model: int = 128,
        in_channels: int = 3,
        pretrained: bool = True,
        img_size: int = 224,
        keep_eval: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.img_size = img_size
        self.d_model = d_model
        # Recipe v2 (`model.vit_frozen_eval`): the feature extractor ignores train mode — see `train()`.
        self.keep_eval = keep_eval
        # num_classes=0 + global_pool="avg" -> forward returns the pooled feature [B*T, num_features].
        self.net = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg", in_chans=in_channels
        )
        feat_dim = self.net.num_features
        self.frame_proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> TimmBackbone:
        """Build from ``ModelCfg`` (``vit_backbone`` names the timm model; ``vit_pretrained`` gates weights)."""
        return cls(
            name=cfg.vit_backbone,
            d_model=cfg.d_model,
            in_channels=cfg.in_channels,
            pretrained=cfg.vit_pretrained,
            img_size=img_size,
            keep_eval=cfg.vit_frozen_eval,
        )

    def train(self, mode: bool = True) -> TimmBackbone:
        """Standard ``train``, except that with ``keep_eval`` the timm feature extractor stays in eval mode.

        ``requires_grad=False`` stops gradients but not BatchNorm: in train mode a frozen backbone still
        normalises with per-batch statistics and overwrites its pretrained running statistics. Keeping
        ``self.net`` in eval mode makes it truly frozen. ``frame_proj`` still follows ``mode``.
        """
        super().train(mode)
        if self.keep_eval:
            self.net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` context crops -> ``[B, T, d_model]``.

        Also accepts ``[B, T, num_features]`` pooled features from the recipe-v2 feature cache
        (``data.visual_input=cached_features``): the frozen extractor already ran offline, so only
        ``frame_proj`` applies.
        """
        if x.dim() == 3:
            if x.shape[-1] != self.net.num_features:
                raise ValueError(
                    f"TimmBackbone({self.name}): cached features have width {x.shape[-1]}, the backbone "
                    f"emits {self.net.num_features} — cache built for a different backbone?"
                )
            return self.frame_proj(x)
        b, t = x.shape[:2]
        feats = self.net(x.flatten(0, 1))              # [B*T, num_features]
        return self.frame_proj(feats.view(b, t, -1))   # [B, T, d_model]


def build_visual_backbone(cfg: ModelCfg, img_size: int) -> nn.Module:
    """Factory for the visual stream — ``model.vit_backbone`` selects legacy ViT vs a timm drop-in.

    ``"legacy"`` reproduces ``ViT_Hierarchical.from_config(cfg, img_size)`` byte-for-byte (the
    golden-pinned default); any other value is treated as a timm model name and builds
    :class:`TimmBackbone`.
    """
    if cfg.vit_backbone == "legacy":
        return ViT_Hierarchical.from_config(cfg, img_size)
    return TimmBackbone.from_config(cfg, img_size)

"""Temporal motion + tight-crop encoder.

Port of OLD ``models/Motion_Encoder.MotionEncoder``. Behavior-preserving (numerically equivalent to the
legacy module for identical weights+input, eval mode) -- the math, op order, and the in-forward motion
normalization are kept verbatim. Resolved band-aids:

* **B6 (config drift).** ``__main__`` is a smoke test built from ``ModelCfg`` (``motion_kwargs()``), not
  the drifting legacy kwargs (``hidden_dim=224``).
* **B7 (motion-dim contract).** ``motion_dim`` flows from config and equals ``DataCfg.motion_dim`` -- the
  Conv1d input width is the only coupling to the 8-channel motion definition (Prompts 1.2 / 1.4); it uses
  the channel *count*, never the per-channel semantics.

The learned ``pos_encoding`` has a fixed capacity (``max_positions``); the legacy ``pos_encoding[:, :T]``
slice silently corrupts into an opaque broadcast error for ``T > capacity``. A guard surfaces a clear
error instead -- numerically neutral for every valid ``T`` (runtime ``T = seq_len = 20 <= 200``).

Unlike the ViT, this module is **resolution-agnostic**: ``img_encoder`` ends in an adaptive
average pool, so no input resolution is baked into any parameter and there is no B2-style lazy-param trap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pedpredict.config import ModelCfg


class MotionEncoder(nn.Module):
    """Tight-crop CNN + Conv1d motion stack -> fusion -> GRU -> learned pos-enc -> attention.

    Attribute names mirror the legacy module verbatim so an OLD ``state_dict`` loads ``strict=True``.
    Inputs ``motion [B, T, motion_dim]`` (raw) + ``tight [B, T, 3, H, W]`` -> ``[B, T, d_model]``.
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
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model

        # Tight-crop feature extraction -> [B*T, hidden_dim, 1, 1] (adaptive pool == resolution-agnostic).
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Motion feature encoding over the [B, motion_dim, T] sequence.
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        # Fuse image + motion features per frame.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2), hidden_dim),
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
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, max_positions, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, d_model) if hidden_dim != d_model else nn.Identity()

    @classmethod
    def from_config(cls, cfg: ModelCfg) -> MotionEncoder:
        """B6: build from ``ModelCfg`` (``motion_kwargs()`` == OLD ``motion_enc_args_config()``)."""
        return cls(**cfg.motion_kwargs())

    def forward(self, motion: torch.Tensor, tight: torch.Tensor) -> torch.Tensor:
        """``motion [B, T, motion_dim]`` (raw) + ``tight [B, T, 3, H, W]`` -> ``[B, T, d_model]``."""
        b, t = motion.shape[:2]
        max_pos = self.pos_encoding.shape[1]
        if t > max_pos:
            raise ValueError(
                f"MotionEncoder: sequence length T={t} exceeds positional-encoding capacity {max_pos}. "
                f"Increase max_positions or shorten the sequence."
            )

        # Tight crops -> per-frame image features.
        img = tight.flatten(0, 1)                 # [B*T, 3, H, W]
        img_feats = self.img_encoder(img)
        img_feats = img_feats.squeeze(-1).squeeze(-1)  # [B*T, hidden_dim]
        img_feats = img_feats.view(b, t, -1)      # [B, T, hidden_dim]

        # Per-sequence motion normalization (verbatim; std uses the unbiased default).
        motion_norm = (motion - motion.mean(dim=1, keepdim=True)) / (motion.std(dim=1, keepdim=True) + 1e-6)
        motion_feats = self.motion_encoder(motion_norm.transpose(1, 2))  # [B, hidden_dim//2, T]
        motion_feats = motion_feats.transpose(1, 2)                      # [B, T, hidden_dim//2]

        combined = torch.cat([img_feats, motion_feats], dim=-1)  # [B, T, hidden_dim + hidden_dim//2]
        x = self.fusion(combined)                                # [B, T, hidden_dim]

        x, _ = self.gru(x)                                       # [B, T, hidden_dim]
        x = x + self.pos_encoding[:, :t, :]

        residual = x
        x = self.norm(x)
        x, _ = self.temporal_attn(x, x, x)
        return self.proj(residual + self.dropout(x))             # [B, T, d_model]


if __name__ == "__main__":  # B6: smoke test driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    model = MotionEncoder.from_config(cfg).eval()
    motion = torch.randn(2, 20, cfg.motion_dim)
    tight = torch.randn(2, 20, 3, 128, 128)
    with torch.no_grad():
        out = model(motion, tight)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MotionEncoder OK | out {tuple(out.shape)} | params {n_params}")
    assert out.shape == (2, 20, cfg.d_model)

"""Hierarchical windowed-attention ViT on context crops (Prompt 2.1).

Port of OLD ``models/Vision_Transformer.py``. Behavior-preserving (numerically equivalent to the
legacy module for identical weights+input, eval mode) except for the explicitly-resolved band-aids:

* **B2 (lazy relative-position bias).** Legacy built the *global* stage's
  ``relative_position_bias_table`` on first forward (``window_size=None``), which forced the
  dummy-forward hack at ``train.py:311-317`` and broke ``strict=True`` checkpoint load. Here the
  global window is resolved at ``__init__`` from the configured input resolution (``img_size``) — the
  feature-map size of the last stage is deterministic given the stem (conv7 s4) + per-stage conv3-s2
  downsample schedule — so **every parameter exists after construction**. No forward-time mutation;
  ``rebuild_position_bias(img_size)`` is the explicit, opt-in path for a *different* resolution
  (benchmark / export), never silent.
* **B13 (confusing MLP residual).** ``WindowTransformerBlock.forward`` rewritten as an unambiguous
  pre-norm residual; identical math.
* **B6 (config drift).** ``__main__`` is a smoke test built from ``ModelCfg``, not the drifting legacy
  kwargs.

Resolution policy: the model is *built for* one input resolution (a config value, sourced from
``DataCfg.read_context_height/width`` at the call site) and is fixed within a run — train and eval at
that resolution. The relative-position-bias table is a resolution-specific learned weight; a checkpoint
trained at one resolution will not ``strict``-load into a model built for another (by design — the old
lazy path silently reinitialized it instead).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from pedpredict.config import ModelCfg
from pedpredict.models.geometry import feature_map_size
from pedpredict.models.geometry import is_global as _is_global


class MLP(nn.Module):
    """Two-layer GELU MLP with a dropout between the linears (legacy ``MLP``)."""

    def __init__(self, dim: int = 128, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim or dim * 2
        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


def window_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """[B, H, W, C] -> [num_windows*B, wh*ww, C]. Square ``wh==ww`` reproduces the legacy partition."""
    b, h, w, c = x.shape
    x = x.view(b, h // wh, wh, w // ww, ww, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh * ww, c)


def window_reverse(windows: torch.Tensor, wh: int, ww: int, h: int, w: int) -> torch.Tensor:
    """[num_windows*B, wh, ww, C] -> [B, H, W, C] (inverse of :func:`window_partition`)."""
    b = windows.shape[0] // ((h // wh) * (w // ww))
    x = windows.view(b, h // wh, w // ww, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


def _relative_position_index(wh: int, ww: int) -> torch.Tensor:
    """Pairwise relative-position index for a (wh, ww) window -> [wh*ww, wh*ww] long tensor."""
    coords = torch.stack(torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij"))  # 2, wh, ww
    coords_flatten = torch.flatten(coords, 1)  # 2, wh*ww
    relative = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
    relative = relative.permute(1, 2, 0).contiguous()  # N, N, 2
    relative[:, :, 0] += wh - 1
    relative[:, :, 1] += ww - 1
    relative[:, :, 0] *= 2 * ww - 1
    return relative.sum(-1).long().contiguous()  # N, N


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with a relative-position bias.

    Unlike the legacy module, ``window_size`` is ALWAYS a concrete ``(Wh, Ww)`` and the bias table +
    index are built eagerly in ``__init__`` (B2) — there is no deferred/global ``None`` path.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.register_buffer("relative_position_index", _relative_position_index(wh, ww))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)

        wh, ww = self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            wh * ww, wh * ww, -1
        )  # (N, N, heads)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_dropout(self.proj(x))


class WindowTransformerBlock(nn.Module):
    """Pre-norm windowed-attention + MLP block over a ``[B, C, H, W]`` feature map.

    ``window_size`` is a concrete ``(Wh, Ww)`` resolved by the parent ViT (global stages get the
    feature-map size). No lazy bias initialization (B2).
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        window_size: tuple[int, int] = (4, 4),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_dropout=attn_dropout, proj_dropout=proj_dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, C, H, W]``."""
        b, c, h, w = x.shape
        wh, ww = self.window_size
        if (h % wh != 0) or (w % ww != 0):
            raise ValueError(f"Feature map ({h}x{w}) not divisible by window ({wh}x{ww})")

        x_perm = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Block 1: windowed attention (pre-norm residual).
        shortcut = x_perm
        windows = window_partition(self.norm1(x_perm), wh, ww)  # [nW*B, wh*ww, C]
        windows = self.attn(windows)
        attended = window_reverse(windows.view(-1, wh, ww, c), wh, ww, h, w)  # [B, H, W, C]
        x_perm = shortcut + self.drop_path(attended)

        # Block 2: MLP (pre-norm residual). B13: unambiguous shortcut, same math as legacy.
        shortcut_flat = x_perm.reshape(b, h * w, c)
        y = self.dropout(self.mlp(self.norm2(shortcut_flat)))
        x_flat = shortcut_flat + self.drop_path(y)

        return x_flat.view(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]


class ViT_Hierarchical(nn.Module):
    """Hierarchical windowed-attention ViT on context crops -> ``[B, T, d_model]``.

    Args mirror the legacy constructor; ``img_size`` is NEW (B2): the square input resolution used to
    resolve global windows eagerly. Source it from ``DataCfg.read_context_height`` at the call site.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_dims: Sequence[int] = (36, 36, 288, 36),
        layer_nums: Sequence[int] = (2, 4, 5, 7),
        head_nums: Sequence[int] = (2, 2, 16, 2),
        window_size: Sequence[int | str | None] = (8, 4, 2, None),
        mlp_ratio: Sequence[float] = (4, 4, 4, 4),
        d_model: int = 128,
        img_size: int = 224,
        drop_path: float = 0.1,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        num_stages = len(stage_dims)

        def _to_list(x: object, n: int) -> list:
            return list(x) if isinstance(x, list | tuple) else [x] * n

        mlp_ratio = _to_list(mlp_ratio, num_stages)
        window_size = _to_list(window_size, num_stages)
        attn_dropout = _to_list(attn_dropout, num_stages)
        proj_dropout = _to_list(proj_dropout, num_stages)

        self.img_size = img_size
        self.in_channels = in_channels
        self.d_model = d_model
        # normalized per-stage window config (None == global) — kept so rebuild_position_bias can
        # re-resolve windows from a new img_size exactly as __init__ did.
        self._raw_windows: list[int | None] = [None if _is_global(w) else int(w) for w in window_size]

        # progressive stochastic-depth schedule
        dpr = torch.linspace(0, drop_path, sum(layer_nums)).tolist()
        block_idx = 0

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stage_dims[0], kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(stage_dims[0]),
            nn.GELU(),
        )

        self.stages = nn.ModuleList()
        in_dim = stage_dims[0]
        for i in range(num_stages):
            dim, num_layers, num_heads, w_size, mlp_r = (
                stage_dims[i], layer_nums[i], head_nums[i], window_size[i], mlp_ratio[i]
            )
            down_sample = (
                nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False)
                if i != 0 else nn.Identity()
            )
            win = self._resolve_window(w_size, i)  # (Wh, Ww) — global resolved from img_size (B2)
            blocks = []
            for _ in range(num_layers):
                blocks.append(
                    WindowTransformerBlock(
                        dim=dim, num_heads=num_heads, window_size=win, mlp_ratio=mlp_r,
                        dropout=dropout, attn_dropout=attn_dropout[i], proj_dropout=proj_dropout[i],
                        drop_path=dpr[block_idx],
                    )
                )
                block_idx += 1
            self.stages.append(nn.ModuleDict({"down_sample": down_sample, "block": nn.ModuleList(blocks)}))
            in_dim = dim

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.frame_proj = nn.Linear(stage_dims[-1], d_model) if stage_dims[-1] != d_model else nn.Identity()

    def _resolve_window(self, w_size: int | str | None, stage_idx: int) -> tuple[int, int]:
        """Concrete ``(Wh, Ww)`` for a stage; global windows -> the stage's feature-map size (B2)."""
        if _is_global(w_size):
            side = feature_map_size(self.img_size, stage_idx)
            return side, side
        size = feature_map_size(self.img_size, stage_idx)
        if size % int(w_size) != 0:
            raise ValueError(
                f"stage {stage_idx}: feature map {size} not divisible by window {w_size} "
                f"(img_size={self.img_size})"
            )
        return int(w_size), int(w_size)

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> ViT_Hierarchical:
        """Build from ``ModelCfg`` (parity kwargs) + an explicit input resolution."""
        return cls(img_size=img_size, **cfg.vit_kwargs())

    def rebuild_position_bias(self, img_size: int) -> None:
        """Rebuild every window's bias table+index for a NEW resolution (benchmark/export only).

        Explicit and opt-in — never called during ``forward``. Global-window tables change shape (their
        size tracks the feature map), so any previously-loaded bias for those stages is discarded.
        Fixed (non-global) windows keep their size; only their stage's tiling is re-validated.
        """
        self.img_size = img_size
        for i, stage in enumerate(self.stages):
            win = self._resolve_window(self._raw_windows[i], i)
            for block in stage["block"]:
                block.window_size = win
                attn = block.attn
                attn.window_size = win
                wh, ww = win
                device = attn.relative_position_bias_table.device
                attn.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * wh - 1) * (2 * ww - 1), attn.num_heads, device=device)
                )
                nn.init.trunc_normal_(attn.relative_position_bias_table, std=0.02)
                attn.register_buffer("relative_position_index", _relative_position_index(wh, ww).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` context crops -> ``[B, T, d_model]``."""
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.stem(x)
        for stage in self.stages:
            x = stage["down_sample"](x)
            for block in stage["block"]:
                x = block(x)
        x = x.mean([2, 3])           # global average pool -> [B*T, D]
        x = self.norm(x)
        x = x.view(b, t, -1)
        return self.frame_proj(x)    # -> [B, T, d_model]


if __name__ == "__main__":  # B6: smoke test driven by ModelCfg, not drifting legacy kwargs
    cfg = ModelCfg()
    model = ViT_Hierarchical.from_config(cfg, img_size=224).eval()
    dummy = torch.randn(1, 4, cfg.in_channels, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT_Hierarchical OK | out {tuple(out.shape)} | params {n_params}")
    assert out.shape == (1, 4, cfg.d_model)

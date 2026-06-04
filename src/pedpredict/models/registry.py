"""Typed model factory + forward adapter (Prompt 2.4) — replaces OLD ``scripts/model_utils.py`` (B10).

OLD ``model_utils.get_model(model_type: str, ...)`` / ``model_forward(model, model_type: str, ...)``
dispatched on a raw string (a typo was a silent bug, and the ``model_type`` was threaded separately from
the model it described). Here:

* ``ModelType`` (a validated ``str``-``Enum``) replaces the free-form string; ``ModelType.coerce`` raises a
  clear, listing error on anything unknown (preserving the OLD ``ValueError`` contract, now typed).
* ``build_model(cfg, model_type)`` is the single factory. It builds the sub-encoders via their own
  ``from_config`` and stamps ``model.model_type`` so the type is **intrinsic** to the module — no call site
  re-passes a string.
* ``forward_model(model, images_tight, images_context, motions)`` is the single adapter that hides the
  per-type input signatures. It dispatches on ``model.model_type`` and consumes the collate triple directly
  (``forward_model(model, *batch[:3])``).

The three ablation types are wired here but their classes are PLACEHOLDER STUBS until Prompt 2.5 (see
``ablations.py``); ``build_model`` for an ablation type therefore raises ``NotImplementedError`` with a
2.5 pointer today, and flips to a real build with no edit to this file once the stubs are replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn

from pedpredict.config import ModelCfg, RootCfg
from pedpredict.models.ablations import MotionOnlyModel, VanillaConcatModel, VisualOnlyModel
from pedpredict.models.ensemble import EnsembleModel

LogitsDict = dict[str, torch.Tensor]


class ModelType(str, Enum):
    """The four model variants selected via config / CLI (replaces stringly dispatch, B10)."""

    FULL = "full"
    MOTION_ONLY = "motion_only"
    VISUAL_ONLY = "visual_only"
    VANILLA_CONCAT = "vanilla_concat"

    @classmethod
    def coerce(cls, value: ModelTypeLike) -> ModelType:
        """Validate a str/enum into a ``ModelType``; raise listing the valid types on anything unknown."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(f"Unknown model type: {value!r}. Choose from: {valid}") from None


ModelTypeLike = ModelType | Literal["full", "motion_only", "visual_only", "vanilla_concat"]

# Which inputs each type consumes (documentation + the source of truth for forward_model + tests).
MODEL_INPUT_SIGNATURE: dict[ModelType, tuple[str, ...]] = {
    ModelType.FULL: ("images_tight", "images_context", "motions"),
    ModelType.VANILLA_CONCAT: ("images_tight", "images_context", "motions"),
    ModelType.MOTION_ONLY: ("motions", "images_tight"),
    ModelType.VISUAL_ONLY: ("images_context",),
}

# Per-type builder: every ``from_config`` takes the uniform ``(ModelCfg, img_size)`` so the factory is one
# loop. ``img_size`` is ignored by motion_only (resolution-agnostic) but kept for signature regularity.
_BUILDERS: dict[ModelType, Callable[[ModelCfg, int], nn.Module]] = {
    ModelType.FULL: EnsembleModel.from_config,
    ModelType.MOTION_ONLY: MotionOnlyModel.from_config,
    ModelType.VISUAL_ONLY: VisualOnlyModel.from_config,
    ModelType.VANILLA_CONCAT: VanillaConcatModel.from_config,
}

# Fallback type resolution for a model not built through build_model (e.g. loaded from a raw class).
_TYPE_BY_CLASS: dict[type[nn.Module], ModelType] = {
    EnsembleModel: ModelType.FULL,
    MotionOnlyModel: ModelType.MOTION_ONLY,
    VisualOnlyModel: ModelType.VISUAL_ONLY,
    VanillaConcatModel: ModelType.VANILLA_CONCAT,
}


def build_model(cfg: RootCfg, model_type: ModelTypeLike | None = None) -> nn.Module:
    """Factory: build the model for ``model_type`` (defaults to ``cfg.eval.model_type``).

    Reproduces the OLD ``get_model`` wiring per type. ``img_size`` is pulled from
    ``cfg.data.read_context_height`` (the ViT is resolution-bound, 2.1). The returned module carries
    ``model.model_type: ModelType`` so ``forward_model`` needs no separate type argument (B10).
    """
    mt = ModelType.coerce(cfg.eval.model_type if model_type is None else model_type)
    img_size = cfg.data.read_context_height
    model = _BUILDERS[mt](cfg.model, img_size)
    model.model_type = mt
    return model


def _resolve_type(model: nn.Module) -> ModelType:
    """Intrinsic type set by ``build_model``, else fall back to the model's class."""
    mt = getattr(model, "model_type", None)
    if isinstance(mt, ModelType):
        return mt
    for klass, klass_type in _TYPE_BY_CLASS.items():
        if isinstance(model, klass):
            return klass_type
    raise TypeError(f"Cannot resolve ModelType for {type(model).__name__}; build it via registry.build_model().")


def forward_model(
    model: nn.Module,
    images_tight: torch.Tensor,
    images_context: torch.Tensor,
    motions: torch.Tensor,
    *,
    return_feats: bool = False,
) -> LogitsDict | tuple[LogitsDict, torch.Tensor, torch.Tensor]:
    """Single forward adapter hiding per-type input signatures (replaces OLD ``model_forward``).

    Routes inputs per ``MODEL_INPUT_SIGNATURE``; consume the collate triple as
    ``forward_model(model, *batch[:3])``. ``return_feats`` is only valid for the ``full`` model (the viz
    path, 6.2) and raises for any ablation.
    """
    mt = _resolve_type(model)
    if mt is ModelType.FULL:
        return model(images_tight, images_context, motions, return_feats=return_feats)
    if return_feats:
        raise ValueError(f"return_feats=True is only supported for {ModelType.FULL.value!r} (got {mt.value!r}).")
    if mt is ModelType.VANILLA_CONCAT:
        return model(images_tight, images_context, motions)
    if mt is ModelType.MOTION_ONLY:
        return model(motions, images_tight)
    if mt is ModelType.VISUAL_ONLY:
        return model(images_context)
    raise ValueError(f"Unhandled model type: {mt!r}")  # unreachable (coerce validated mt)

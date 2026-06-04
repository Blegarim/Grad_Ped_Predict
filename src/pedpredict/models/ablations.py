"""Ablation models (Prompt 2.5) — PLACEHOLDER STUBS.

Port target: OLD ``models/AblationModels.py`` — ``MotionOnlyModel``, ``VisualOnlyModel``,
``VanillaConcatModel`` (same output-dict contract as the full model; ``VanillaConcatModel`` concatenates
the two branches instead of cross-attending; each reuses the ``heads.py`` builders).

These stubs exist ONLY so ``registry.py`` (Prompt 2.4) can declare the typed factory + forward adapter
over all four ``ModelType`` values NOW. The real classes land in Prompt 2.5 with golden parity reusing
``tests/fixtures/golden/ensemble.pt`` (which already carries the three ablation state_dicts/outputs).

To complete Prompt 2.5: replace each stub with the ported class + a real ``from_config(cfg, img_size)``
(reusing ``build_pool_mlp`` / ``build_task_classifiers`` / ``build_crosses_frame_head`` /
``temporal_attention_pool`` / ``frame_pool_reduce`` from ``heads.py``). ``registry.py`` needs **no edit** —
its dispatch table + input routing are already wired to these names.
"""

from __future__ import annotations

import torch.nn as nn

from pedpredict.config import ModelCfg

_NOT_YET = "Ablation models land in Prompt 2.5 (port OLD/models/AblationModels.py)."


class _AblationStub(nn.Module):
    """Raises until Prompt 2.5 fills in the real ablation classes (kept swiftly replaceable)."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        raise NotImplementedError(_NOT_YET)

    @classmethod
    def from_config(cls, cfg: ModelCfg, img_size: int) -> nn.Module:  # noqa: ARG003 (uniform 2.4 signature)
        raise NotImplementedError(_NOT_YET)


class MotionOnlyModel(_AblationStub):
    """OLD ``MotionOnlyModel`` — motion branch only. ``forward(motions, images_tight)``. (Prompt 2.5)"""


class VisualOnlyModel(_AblationStub):
    """OLD ``VisualOnlyModel`` — ViT branch only. ``forward(images_context)``. (Prompt 2.5)"""


class VanillaConcatModel(_AblationStub):
    """OLD ``VanillaConcatModel`` — concat fusion. ``forward(images_tight, images_context, motions)``. (Prompt 2.5)"""

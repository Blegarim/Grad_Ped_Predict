"""Capture a golden fixture for Prompt 2.4 by running the OLD full model + ablations (provenance, not a test).

Builds the legacy models exactly as ``scripts/model_utils.get_model`` wired them (full +
``motion_only`` / ``visual_only`` / ``vanilla_concat``), runs ONE ``model_forward`` per type in eval mode,
and saves the ``state_dict`` + inputs + outputs keyed by model_type. The new
``pedpredict.models.ensemble.EnsembleModel`` (full) is diffed against this with ``strict=True``; the three
ablation entries are pre-captured here so Prompt 2.5 can reuse this same fixture without re-running.

Not collected by pytest (``capture_*``); rerun manually in a torch environment only if the OLD code or
inputs change::

    python tests/_capture/capture_ensemble_golden.py

Determinism / parity notes:
  * ``.eval()`` makes Dropout / GRU-dropout / MHA-dropout identities and BatchNorm use running stats ->
    the forward is deterministic fp32 math; parity holds at atol=1e-6, rtol=1e-5.
  * The OLD ViT defers its GLOBAL-stage relative-position table to first forward (B2), so we capture each
    ``state_dict`` AFTER one forward (which ``model_forward`` performs). The new model is eager (2.1) and
    loads this dict ``strict=True`` with no forward.

B4 — ``crosses_pooled`` (full only):
  The OLD full forward emitted 4 keys (the ``classifier['crosses']`` head was allocated but skipped). The
  rebuilt ``CrossAttentionModule`` makes it live-but-unsupervised, so we recompute ``crosses_pooled`` here
  from the SAME deterministic post-LayerNorm features (mirroring the 2.3 capture) and store it alongside.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pedpredict.config import ModelCfg

OLD_ROOT = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "ensemble.pt"

_B, _T = 2, 3
_CTX, _TIGHT = 224, 128
_SEED = 0
# Prompt 2.4 captures only the full model (the module it ports). Prompt 2.5 extends this tuple to
# ("full", "motion_only", "visual_only", "vanilla_concat") to add the ablation parity references — each
# ablation carries its own ViT/motion state_dict, so capturing all four ~4x the fixture size; kept lean here.
_MODEL_TYPES = ("full",)


def _dummy_inputs(cfg: ModelCfg) -> dict[str, torch.Tensor]:
    torch.manual_seed(_SEED)
    return {
        "images_tight": torch.randn(_B, _T, cfg.in_channels, _TIGHT, _TIGHT),
        "images_context": torch.randn(_B, _T, cfg.in_channels, _CTX, _CTX),
        "motions": torch.randn(_B, _T, cfg.motion_dim),
    }


def _reconstruct_crosses_pooled(model: object, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Recompute the B4 crosses_pooled head from the legacy full model's deterministic features."""
    image_feats = model.image_norm(model.vit(inputs["images_context"]))
    motion_feats = model.motion_norm(model.motion_enc(inputs["motions"], inputs["images_tight"]))
    ca = model.cross_attention
    attn_output, _ = ca.cross_attn(query=motion_feats, key=image_feats, value=image_feats)
    weights = torch.softmax(ca.pool_mlp(attn_output), dim=1)
    pooled = (attn_output * weights).sum(dim=1)
    return ca.classifier["crosses"](pooled)


def main() -> None:
    sys.path.insert(0, str(OLD_ROOT))
    from models.Motion_Encoder import MotionEncoder as OldMotion  # noqa: E402
    from models.Vision_Transformer import ViT_Hierarchical as OldViT  # noqa: E402
    from scripts.model_utils import get_model, model_forward  # noqa: E402

    cfg = ModelCfg()
    num_classes = dict(cfg.num_classes)
    fixture: dict[str, object] = {}

    for mt in _MODEL_TYPES:
        torch.manual_seed(_SEED)
        vit = OldViT(**cfg.vit_kwargs())  # OLD ViT is lazy: global table materializes at first forward
        motion_enc = OldMotion(**cfg.motion_kwargs())
        model = get_model(mt, motion_enc, vit, d_model=cfg.d_model, num_classes_dict=num_classes, dropout=0.1)
        model.eval()

        inputs = _dummy_inputs(cfg)
        with torch.no_grad():
            out = model_forward(model, mt, inputs["images_tight"], inputs["images_context"], inputs["motions"])
            outputs = {k: v.clone() for k, v in out.items()}
            if mt == "full":  # B4: recompute crosses_pooled from the legacy weights (legacy out had 4 keys)
                outputs["crosses_pooled"] = _reconstruct_crosses_pooled(model, inputs).clone()

        fixture[mt] = {
            "inputs": {k: v.clone() for k, v in inputs.items()},
            "outputs": outputs,
            "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "img_size": _CTX,
            "meta": {
                "src": f"scripts/model_utils.get_model({mt!r}) -> Unified_Module/AblationModels",
                "atol": 1e-6,
                "rtol": 1e-5,
                "legacy_keys": sorted(out.keys()),
                "reconstructed_keys": ["crosses_pooled"] if mt == "full" else [],
                "torch": torch.__version__,
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)
    print(f"wrote {OUT}")
    for mt in _MODEL_TYPES:
        entry = fixture[mt]
        print(f"  {mt:14s} keys {sorted(entry['outputs'])} state_dict {len(entry['state_dict'])}")


if __name__ == "__main__":
    main()

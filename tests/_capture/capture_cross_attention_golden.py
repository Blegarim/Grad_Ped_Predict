"""Capture a golden fixture for Prompt 2.3 by running the OLD CrossAttentionModule (provenance, not a test).

Builds the legacy ``models/Cross_Attention_Module.CrossAttentionModule`` with the training-effective
kwargs (``ModelCfg.cross_kwargs()`` == ``scripts/model_utils.get_model()``'s full-model wiring:
``num_heads=4``, dropout defaulted to 0.1, ``use_frame_crosses=True``, ``frame_pool="logsumexp"``), runs
ONE forward in eval mode, and saves the ``state_dict`` + inputs + outputs. The new
``pedpredict.models.cross_attention.CrossAttentionModule`` is diffed against this with ``strict=True``.
Not collected by pytest (``capture_*``); rerun manually only if the OLD code or inputs change::

    python tests/_capture/capture_cross_attention_golden.py

Determinism / parity notes:
  * ``.eval()`` makes Dropout AND the MultiheadAttention dropout identity -> the forward is pure
    deterministic fp32 math; the ported module uses the same ops/order -> parity holds at atol=1e-6,
    rtol=1e-5.
  * CrossAttentionModule has NO lazy parameters, so the state_dict is captured right after construction
    (no dummy forward) and loads into the new module with strict=True -- INCLUDING the legacy-dead
    ``classifier.crosses`` head (retained for param-layout parity).

B4 — ``crosses_pooled``:
  The OLD ``forward`` NEVER emitted ``crosses_pooled`` (the ``crosses`` classifier head was allocated but
  skipped). The rebuild makes that head live-but-unsupervised. To give the new behavior a *real* golden
  reference derived from the legacy weights, we recompute ``crosses_pooled`` here exactly as the rebuilt
  module does (``classifier["crosses"](pooled)`` over the same deterministic ``attn_output``) and store it
  alongside the 4 genuinely-legacy keys. ``meta.exact`` records which keys are legacy vs reconstructed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pedpredict.config import ModelCfg

OLD_ROOT = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "cross_attention.pt"

_B, _T, _D = 2, 3, 128
_SEED = 0


def main() -> None:
    sys.path.insert(0, str(OLD_ROOT))
    from models.Cross_Attention_Module import CrossAttentionModule as OldCross  # noqa: E402

    cross_kwargs = ModelCfg().cross_kwargs()

    torch.manual_seed(_SEED)
    model = OldCross(**cross_kwargs)
    model.eval()

    torch.manual_seed(_SEED)
    motion_feats = torch.randn(_B, _T, _D)
    image_feats = torch.randn(_B, _T, _D)

    with torch.no_grad():
        out = model(motion_feats, image_feats)  # legacy 4 keys
        # Reconstruct crosses_pooled from the SAME deterministic attn_output (legacy never emitted it).
        attn_output, _ = model.cross_attn(query=motion_feats, key=image_feats, value=image_feats)
        scores = model.pool_mlp(attn_output)
        weights = torch.softmax(scores, dim=1)
        pooled = (attn_output * weights).sum(dim=1)
        crosses_pooled = model.classifier["crosses"](pooled)

    outputs = {k: v.clone() for k, v in out.items()}
    outputs["crosses_pooled"] = crosses_pooled.clone()

    fixture = {
        "inputs": {"motion_feats": motion_feats.clone(), "image_feats": image_feats.clone()},
        "outputs": outputs,
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "cross_kwargs": cross_kwargs,
        "seed": _SEED,
        "meta": {
            "src": "models/Cross_Attention_Module.py::CrossAttentionModule",
            "atol": 1e-6,
            "rtol": 1e-5,
            "exact": "none (float forward)",
            "legacy_keys": ["actions", "looks", "crosses_frame", "temporal_weights"],
            "reconstructed_keys": ["crosses_pooled"],  # B4: not emitted by legacy forward; recomputed here
            "torch": torch.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"wrote {OUT}")
    print(f"  inputs  motion {tuple(motion_feats.shape)}  image {tuple(image_feats.shape)}")
    print(f"  outputs {[f'{k}{tuple(v.shape)}' for k, v in outputs.items()]}")
    print(f"  params  {n_params}")
    print(f"  state_dict keys {len(fixture['state_dict'])}")


if __name__ == "__main__":
    main()

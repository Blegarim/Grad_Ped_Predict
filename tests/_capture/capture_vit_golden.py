"""Capture a golden fixture for Prompt 2.1 by running the OLD ViT_Hierarchical (provenance, not a test).

Builds the legacy ``models/Vision_Transformer.ViT_Hierarchical`` with the training-effective kwargs
(``config.vit_args_config()`` == ``ModelCfg.vit_kwargs()``), runs ONE forward in eval mode to both
materialize the lazily-created global-window ``relative_position_bias_table`` (the B2 band-aid) AND
produce the golden output, then saves the post-forward ``state_dict`` + input + output. The new
``pedpredict.models.vit.ViT_Hierarchical`` is diffed against this with ``strict=True`` weight load.
Not collected by pytest (``capture_*``); rerun manually only if the OLD code or inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_vit_golden.py

Determinism / parity notes:
  * ``.eval()`` makes Dropout / DropPath identity, so the forward is pure deterministic fp32 math; the
    ported module uses the same ops/order -> parity holds at atol=1e-6, rtol=1e-5.
  * The OLD ``state_dict`` is captured AFTER the dummy forward so it CONTAINS the global table; the new
    module creates that table at ``__init__`` (B2 resolved) and loads this dict with ``strict=True``.
  * img_size = 224 (context-crop read size, DataCfg.read_context_height) -> global window resolves to 7x7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pedpredict.config import ModelCfg

OLD_ROOT = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "vit.pt"

_B, _T, _IMG = 2, 3, 224
_SEED = 0


def main() -> None:
    sys.path.insert(0, str(OLD_ROOT))
    from models.Vision_Transformer import ViT_Hierarchical as OldViT  # noqa: E402

    vit_kwargs = ModelCfg().vit_kwargs()

    torch.manual_seed(_SEED)
    model = OldViT(**vit_kwargs)
    model.eval()

    torch.manual_seed(_SEED)
    x = torch.randn(_B, _T, vit_kwargs["in_channels"], _IMG, _IMG)

    with torch.no_grad():
        y = model(x)  # materializes the global-window table AND yields the golden output

    fixture = {
        "inputs": {"x": x.clone()},
        "outputs": {"y": y.clone()},
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "vit_kwargs": vit_kwargs,
        "img_size": _IMG,
        "seed": _SEED,
        "meta": {
            "src": "models/Vision_Transformer.py::ViT_Hierarchical (post dummy-forward state_dict)",
            "atol": 1e-6,
            "rtol": 1e-5,
            "exact": "none (float forward)",
            "torch": torch.__version__,
            "timm": __import__("timm").__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    has_global = any("relative_position_bias_table" in k for k in fixture["state_dict"])
    print(f"wrote {OUT}")
    print(f"  input   x {tuple(x.shape)} {x.dtype}")
    print(f"  output  y {tuple(y.shape)} {y.dtype}")
    print(f"  params  {n_params}")
    print(f"  state_dict keys {len(fixture['state_dict'])}, has rel-pos tables: {has_global}")


if __name__ == "__main__":
    main()

"""Capture a golden fixture for Prompt 2.2 by running the OLD MotionEncoder (provenance, not a test).

Builds the legacy ``models/Motion_Encoder.MotionEncoder`` with the training-effective kwargs
(``config.motion_enc_args_config()`` == ``ModelCfg.motion_kwargs()``), runs ONE forward in eval mode,
and saves the ``state_dict`` + inputs + output. The new ``pedpredict.models.motion_encoder.MotionEncoder``
is diffed against this with ``strict=True`` weight load. Not collected by pytest (``capture_*``); rerun
manually only if the OLD code or inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_motion_golden.py

Determinism / parity notes:
  * ``.eval()`` makes Dropout / GRU-dropout / MultiheadAttention-dropout identity, and BatchNorm uses the
    captured running stats (init mean0/var1) -> the forward is pure deterministic fp32 math; the ported
    module uses the same ops/order -> parity holds at atol=1e-6, rtol=1e-5.
  * MotionEncoder has NO lazy parameters (unlike the ViT's B2 global table), so the state_dict is captured
    right after construction -- no dummy forward needed -- and loads into the new module with strict=True.
  * tight crops are 128x128 (DataCfg.img_height/width); the img_encoder is adaptive-pooled (resolution
    agnostic), so the input resolution is not baked into any parameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from pedpredict.config import ModelCfg

OLD_ROOT = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "motion_encoder.pt"

_B, _T, _IMG = 2, 3, 128
_SEED = 0


def main() -> None:
    sys.path.insert(0, str(OLD_ROOT))
    from models.Motion_Encoder import MotionEncoder as OldMotion  # noqa: E402

    motion_kwargs = ModelCfg().motion_kwargs()

    torch.manual_seed(_SEED)
    model = OldMotion(**motion_kwargs)
    model.eval()

    torch.manual_seed(_SEED)
    motion = torch.randn(_B, _T, motion_kwargs["motion_dim"])
    tight = torch.randn(_B, _T, 3, _IMG, _IMG)

    with torch.no_grad():
        y = model(motion, tight)

    fixture = {
        "inputs": {"motion": motion.clone(), "tight": tight.clone()},
        "outputs": {"y": y.clone()},
        "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "motion_kwargs": motion_kwargs,
        "seed": _SEED,
        "meta": {
            "src": "models/Motion_Encoder.py::MotionEncoder",
            "atol": 1e-6,
            "rtol": 1e-5,
            "exact": "none (float forward)",
            "torch": torch.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"wrote {OUT}")
    print(f"  inputs  motion {tuple(motion.shape)}  tight {tuple(tight.shape)} {tight.dtype}")
    print(f"  output  y {tuple(y.shape)} {y.dtype}")
    print(f"  params  {n_params}")
    print(f"  state_dict keys {len(fixture['state_dict'])}")


if __name__ == "__main__":
    main()

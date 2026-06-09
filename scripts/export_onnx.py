#!/usr/bin/env python3
"""Export a trained model to ONNX and verify onnxruntime parity.

Port of OLD ``onnx/onnx_export.py`` as a thin config-driven CLI.  All export
knobs live in ``ExportCfg`` (configs/export.yaml); the model type is read from
``EvalCfg.model_type`` (configs/eval.yaml) and can be overridden on the CLI.

Usage::

    python scripts/export_onnx.py --checkpoint outputs/runs/<run>/checkpoints/best.pth
    python scripts/export_onnx.py --checkpoint best.pth eval.model_type=motion_only
    python scripts/export_onnx.py --checkpoint best.pth export.output_dir=deploy/onnx

Steps: load config → build model → load weights (strict) → export ONNX →
       onnxruntime parity check (skipped if onnxruntime is absent or --no-parity).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from pedpredict.config import load_config
from pedpredict.export.onnx import check_onnx_parity, export_onnx
from pedpredict.models.registry import ModelType, build_model


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to .pth checkpoint (raw state_dict or {'model_state_dict': ...}).",
    )
    p.add_argument(
        "--config", default="configs", type=Path,
        help="Config directory (default: configs/).",
    )
    p.add_argument(
        "--no-parity", action="store_true",
        help="Skip onnxruntime parity check after export.",
    )
    p.add_argument(
        "overrides", nargs="*",
        help="Dotted config overrides, e.g. eval.model_type=motion_only export.opset=18.",
    )
    return p.parse_args(argv)


def _load_state_dict(path: Path) -> dict:
    """Load a checkpoint, unwrapping common wrapper formats."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    mt = ModelType.coerce(cfg.eval.model_type)

    model = build_model(cfg, mt)
    state = _load_state_dict(args.checkpoint)
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(cfg.export.output_dir)
    out_path = out_dir / f"{mt.value}.onnx"
    exported = export_onnx(model, cfg, out_path, model_type=mt)
    print(f"Exported: {exported}")

    if not args.no_parity:
        try:
            diffs = check_onnx_parity(model, exported, cfg, model_type=mt)
            for key, (abs_d, rel_d) in diffs.items():
                print(f"  parity {key}: max_abs={abs_d:.2e}  max_rel={rel_d:.2e}")
            print("Parity OK.")
        except ImportError as exc:
            print(f"[skip parity] onnxruntime not installed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

# Grad_Ped_Predict

Multimodal pedestrian behavior prediction on the **PIE dataset**. From a sequence of video frames the
model jointly predicts three binary tasks: **actions** (walking/standing), **looks** (looking at traffic),
and **crosses** (will cross soon).

This repository is a **ground-up, behavior-preserving rebuild** of an undergraduate thesis project.
See [REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md) for the master plan, [CLAUDE.md](CLAUDE.md) for the
architecture and conventions, and [MIGRATION.md](MIGRATION.md) for the per-module porting log.

## Status

Phase A — repo scaffold (Prompt 0.1). No model code is implemented yet; this commit ships the package
layout, packaging metadata, hygiene config, and the CI gate.

## Install

Python **3.10–3.12** (matches the legacy interpreter; the pinned `torch`/`numpy` wheels do not yet build
on 3.13+).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Unix: source .venv/bin/activate
pip install -e .[dev]
```

Optional extras:

- `pip install -e .[infer]` — YOLO detection/tracking for inference (`ultralytics`, `lap`).
- `pip install -e .[export]` — ONNX export + onnxruntime parity check.

**System dependency:** `PyTurboJPEG` requires the native `libturbojpeg` library installed at the OS level
(used in the LMDB writer). On CI this path is skipped via the `slow` marker.

**CUDA:** the pinned `torch==2.7.1` resolves to CPU wheels by default. For GPU training install the CUDA
build from the appropriate PyTorch index URL.

## Run the gate

```bash
ruff check .
pytest -m "not slow"
```

Both must pass. This is the baseline safety net; golden-parity, shape, and metric tests land in later
prompts (P8).

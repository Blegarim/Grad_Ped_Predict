# Grad_Ped_Predict

Multimodal **pedestrian behavior prediction** on the **PIE dataset**. From a short sequence of dashcam
video frames the model jointly predicts three binary tasks for each pedestrian:

- **actions** — walking vs standing
- **looks** — looking toward traffic or not
- **crosses** — will cross the road soon

The codebase is clean, tested, and config-driven: typed dataclass configs, a single shared metric path,
golden characterization tests, and a ruff + pytest gate. See [CLAUDE.md](CLAUDE.md) for the full
architecture, the output-dict contract, and the imbalance policy.

> **About this README.** It is a stable, whole-repo overview — the problem, the architecture, the layout,
> and how to set things up and run them. Update it only when the architecture, layout, or setup genuinely
> change.

## Architecture

```
context crop frames → ViT_Hierarchical  ──┐
                                          ├→ CrossAttentionModule → EnsembleModel → {actions, looks, crosses}
tight crop + motion → MotionEncoder    ───┘
```

| Component | Role |
|---|---|
| `ViT_Hierarchical` | Hierarchical windowed-attention ViT over context crops → `[B, T, d_model]`. |
| `MotionEncoder` | Temporal CNN over tight crops + Conv1d motion stack + GRU + attention → `[B, T, d_model]`. |
| `CrossAttentionModule` | Cross-attention (query=motion, key/value=image) → temporal pooling → per-task heads. |
| `EnsembleModel` | Wires the branches (LayerNorm before fusion); ablations swap or drop a branch. |

A unified `d_model = 128` is shared across every module, and models are selected through a typed registry
(`full`, `motion_only`, `visual_only`, `vanilla_concat`). The output-dict contract, the severe `crosses`
class imbalance, and the single imbalance policy are documented in [CLAUDE.md](CLAUDE.md).

## Repository layout

```
src/pedpredict/        # installable package (pip install -e .)
  config/   schema.py loader.py    # yaml → dataclass → argparse override merge
  paths.py
  utils/    seed device amp memory logging
  data/     pie_sequences transforms lmdb_writer lmdb_dataset lmdb_warm
            balance augment collate sampler stats
  models/   vit geometry motion_encoder cross_attention ensemble ablations heads registry
  losses/   multitask.py
  training/ trainer chunk_loader callbacks schedule metrics
  eval/     evaluate benchmark inference
  viz/      plots qualitative
  export/   onnx.py
scripts/    # thin one-job CLIs (make_sequences, build_lmdb, train, evaluate, ...)
configs/    paths.yaml data.yaml model.yaml train.yaml eval.yaml
tests/      # unit + golden characterization tests; fixtures/golden/ pins module numerics
```

## Configuration

Every parameter lives in `configs/*.yaml`, loaded into frozen typed dataclasses
([config/schema.py](src/pedpredict/config/schema.py)) and overridable on the CLI — no hardcoded
hyperparameters or paths in module code:

```bash
python scripts/<job>.py --set train.lr=5e-5 --set data.stride=5
```

The resolved config is dumped per run for reproducibility. Tracking is deliberately minimal: yaml + CSV,
no Hydra, no W&B.

## Data pipeline

Offline → runtime, each stage a thin CLI in `scripts/` over a module in `data/`:

```
PIE → sequence windows → LMDB chunks (JPEG crops + motion/labels) → balance → augment → runtime dataset
```

For example, the first two stages:

```bash
python scripts/make_sequences.py --split all     # PIE → data/sequences/sequences_<split>.pkl
python scripts/build_lmdb.py     --split val     # sequences → preprocessed_<split>/chunk_*.lmdb
```

Crops are stored un-normalized (JPEG); ImageNet normalization is applied at read time. The 8-dim motion
feature and the LMDB key/value schema are documented in
[data/transforms.py](src/pedpredict/data/transforms.py) and
[data/lmdb_writer.py](src/pedpredict/data/lmdb_writer.py).

## Install

Python **3.10–3.12** (the pinned `torch` / `numpy` wheels do not build on 3.13+).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Unix: source .venv/bin/activate
pip install -e .[dev]
```

Optional extras:

- `pip install -e .[infer]` — YOLO detection/tracking for video inference (`ultralytics`, `lap`).
- `pip install -e .[export]` — ONNX export + onnxruntime parity check.

**CUDA:** the pinned `torch==2.7.1` resolves to CPU wheels by default. For GPU training, install the CUDA
build from the appropriate PyTorch index URL.

## Run the gate

```bash
ruff check .
pytest -m "not slow"
```

Both must pass — the lint + test safety net for the codebase. `slow` tests need the PIE dataset or heavy
IO and are excluded from CI.

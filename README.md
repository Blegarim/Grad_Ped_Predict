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
| `ViT_Hierarchical` | Hierarchical windowed-attention ViT over context crops (A1 monotonic stage schedule `[48,96,192,384]`, 7×7 windows) → `[B, T, d_model]`. Swappable for a pretrained `timm` backbone via `model.vit_backbone` (RQ1, see [docs/BACKBONE_STUDY.md](docs/BACKBONE_STUDY.md)); `legacy` is the default. |
| `MotionEncoder` | Temporal CNN over tight crops + Conv1d motion stack + GRU + attention → `[B, T, d_model]`. |
| `CrossAttentionModule` | Cross-attention (query=motion, key/value=image) → temporal pooling → per-task heads. |
| `EnsembleModel` | Wires the branches (LayerNorm before fusion); ablations swap or drop a branch. |

A unified `d_model = 128` is shared across every module, and models are selected through a typed registry
(`full`, `ped_local`, `kinematics_only`, `visual_only`, `vanilla_concat`, `pose_kinematics`, `pose_full`).
The output-dict contract, the severe `crosses` class imbalance, and the single imbalance policy are
documented in [CLAUDE.md](CLAUDE.md).

**Pose-keypoint arm** ([docs/POSE_ENCODER.md](docs/POSE_ENCODER.md)): tests whether 2D pose is a cleaner
geometric substitute for the noisy tight crop. `scripts/extract_pose.py` runs a bbox-conditioned
whole-body extractor (DWPose via `rtmlib`, PIE GT boxes as the person prior) once per unique frame and
caches raw `[23, 3]` keypoints; with `pose.enabled` the LMDB build stores them and the dataset builds a
49-dim bbox-normalized feature block (15 joint coords + confidences + head/body facing angles) at read
time, concatenated onto the image-normalized 9-dim motion vector (`motion_dim=58`,
`model.motion_norm=none`). `pose_kinematics` is the pixel-free arm (`KinematicsOnlyModel` on the 58-dim
input); `pose_full` swaps the full model's motion branch for it (pose as the cross-attention query — no
tight crop). Raw keypoints are stored, features built at read time, so the normalization/angle math stays
iterable without re-extraction; all pose math lives in [data/pose.py](src/pedpredict/data/pose.py).

## Repository layout

```
src/pedpredict/        # installable package (pip install -e .)
  config/   schema.py loader.py    # yaml → dataclass → argparse override merge
  paths.py
  utils/    seed device amp memory logging
  data/     pie_sequences pie_annotations transforms lmdb_writer lmdb_dataset lmdb_warm
            balance augment collate sampler stats pose
            onset_stats onset_target onset_backfill      # onset-timing arm (METHODOLOGY prong 2)
  models/   vit timm_backbone geometry motion_encoder cross_attention ensemble ablations heads registry
  losses/   multitask onset
  training/ trainer chunk_loader callbacks schedule metrics distribution
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
python scripts/make_sequences.py --split all     # PIE → data/sequences/sequences_<split>.pkl (+ stats json)
python scripts/make_sequences.py --benchmark     # TTE-protocol eval windows → sequences_test_benchmark.pkl
python scripts/build_lmdb.py     --split val     # sequences → preprocessed_<split>/chunk_*.lmdb
python scripts/build_lmdb.py     --split test_benchmark  # benchmark eval set → preprocessed_test_benchmark
python scripts/build_lmdb_incremental.py --split train  # disk-bounded + resumable build (see setup.md)
python scripts/balance_dataset.py                       # opt-in offline majority down-sample (default off)
python scripts/augment_dataset.py                       # offline minority-class augmentation
python scripts/extract_pose.py --split all              # pose arm: keypoint cache (--dry-run = synthetic)
```

`build_lmdb` needs every frame a split references staged on disk; `build_lmdb_incremental` extracts only
the frames each chunk uses (cv2, byte-identical to PIE's extractor), builds the chunk, deletes the spent
frames, and resumes from the last completed chunk — for train or any split too large to stage at once.

Crops are stored un-normalized (JPEG); ImageNet normalization is applied at read time. The stored motion
vector is 9-dim — `(cx, cy, dx, dy, w, h, dw, dh, ego_speed)`, frame-0 deltas zero, sliced to
`data.motion_dim` at read time — documented channel-by-channel in
[data/transforms.py](src/pedpredict/data/transforms.py); the LMDB key/value schema (incl. `track_id`)
lives in [data/lmdb_writer.py](src/pedpredict/data/lmdb_writer.py).

## Train, evaluate, export

Config-first — override any field with `--set section.field=value`. The model is chosen by
`eval.model_type` (`full` | `ped_local` | `kinematics_only` | `visual_only` | `vanilla_concat` |
`pose_kinematics` | `pose_full`), **not** `model.*`.

```bash
python scripts/train.py    --set eval.model_type=full --set train.lr=5e-5
python scripts/evaluate.py --split val  --checkpoint <best.pth>   # tune + store val thresholds, then…
python scripts/evaluate.py --split test --checkpoint <best.pth>   # report at the frozen val thresholds
python scripts/run_arm.py  --set eval.model_type=full ...         # full cross-protocol matrix for one arm:
                                                 # trains on BOTH protocols (runner owns data.protocol) and
                                                 # runs val+test × anchored+streaming per leg (10 steps);
                                                 # same --set surface as train.py; --dry-run previews
python scripts/report_distribution.py            # effective per-task sampler-draw distribution
python scripts/count_labels.py                   # dataset-stats drift gate (nonzero exit on drift)
python scripts/report_negative_composition.py    # streaming-negative composition + horizon sweep;
                                                 # --annotations <annotations.zip> reads PIE's XMLs
                                                 # directly, so it runs with no LMDB and no frames
python scripts/backfill_onset_meta.py --dry-run  # write the S1 onset keys into LMDBs built before S1;
                                                 # metadata-only (image blobs untouched), idempotent,
                                                 # verifies track_id+crosses per sample before writing.
                                                 # --split train|val|test|*_benchmark (repeatable).
                                                 # Stop any training job first: Windows blocks a
                                                 # write-open while a chunk is memory-mapped.
python scripts/visualize.py    ...               # plots / qualitative panels
python scripts/infer_video.py  ...               # needs [infer] (YOLO detect/track)
python scripts/export_onnx.py  ...               # needs [export]; runs an onnxruntime parity check
```

Always evaluate `val` **before** `test`: thresholds are tuned on val and frozen for the test report, never
tuned on test. See [CLAUDE.md](CLAUDE.md) for the full experimental-validity rules and imbalance policy.

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
- `pip install rtmlib onnxruntime-gpu` — real pose extraction (`scripts/extract_pose.py` without
  `--dry-run`; lab PC only, deliberately not a packaged extra).

**CUDA:** the pinned `torch==2.7.1` resolves to CPU wheels by default. For GPU training, install the CUDA
build from the appropriate PyTorch index URL.

## Run the gate

```bash
ruff check .
pytest -m "not slow"
```

Both must pass — the lint + test safety net for the codebase. `slow` tests need the PIE dataset or heavy
IO and are excluded from CI.

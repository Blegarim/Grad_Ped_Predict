# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Status: Ground-Up Rebuild (Phase A)

This repo (`Grad_Ped_Predict`, graduate research) is a **complete from-scratch rebuild** of an
undergraduate thesis project. It is **not yet implemented** — currently it holds only
[REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md), the master plan that sequences the rebuild as a DAG of
copy-paste prompts (P0 Foundation → P1 Data → P2 Models → P3 Loss/Metrics → P4 Training → P5 Eval →
P6 Viz → P7 Export → P8 Tests/Docs → P9 Cutover). **Read the schematic before doing structural work.**

**Old repo is read-only reference** at `c:/Users/LENOVO/Desktop/Undergrad_Project/Undergrad_thesis_project`.
Code is ported piece-by-piece into the new layout.

**Rebuild mode — two phases:**
- **Phase A (current): behavior-preserving restructure.** Same model math, same outputs, cleaner code.
  Every ported module must be numerically equivalent (within float tolerance) to the legacy module for
  the same inputs and weights — *unless* a listed band-aid intentionally changes behavior, which must be
  called out and justified. Capture a **golden-output fixture** from the OLD repo *before* porting each module.
- **Phase B (deferred): architecture redesign.** ViT backbone swap, fusion rethink, single unified
  crosses head, online augmentation, standard DataLoader sharding. Out of scope until Phase A parity is locked.

Track progress in a running `MIGRATION.md` (per module: golden fixture path, band-aids resolved, parity result).

## Problem & Architecture (preserved this phase)

Multimodal pedestrian behavior prediction on the **PIE dataset**. From a sequence of video frames the
model jointly predicts three **binary** tasks: **actions** (walking/standing), **looks** (looking at
traffic or not), **crosses** (will cross soon).

```
context crop frames → ViT_Hierarchical  ──┐
                                           ├→ CrossAttentionModule → EnsembleModel → {actions, looks, crosses}
tight crop + motion → MotionEncoder    ───┘
```

| Component | Role |
|---|---|
| `ViT_Hierarchical` | Hierarchical windowed-attention ViT on context crops (stem conv7×7 s4, per-stage downsample s2, global-avg-pool, `frame_proj`). Outputs `[B, T, d_model]`. |
| `MotionEncoder` | Temporal CNN over tight crops + Conv1d motion stack + fusion + GRU + learned pos-encoding + MultiheadAttention. Outputs `[B, T, d_model]`. |
| `CrossAttentionModule` | Cross-attention (query=motion, key/value=image) → pooling MLP → softmax temporal weights → per-task classifier heads. |
| `EnsembleModel` | Wires all components; applies **LayerNorm before fusion**; `return_feats` path used by viz. |
| Ablations | `MotionOnlyModel`, `VisualOnlyModel`, `VanillaConcatModel` (concat instead of cross-attention); same output-dict format. |

- **Unified `d_model = 128`** across ALL modules (`config.get_unified_dim_model()` in old repo). Never
  change one module's dim without the others.
- **Output dict keys**: `actions`, `looks`, `crosses_pooled`, `crosses_frame`, `temporal_weights`.
  Training & eval supervise **ONLY `crosses_frame`** (logsumexp-pooled over frames). `crosses_pooled` is
  computed but **unused** — a known band-aid (B4) to resolve explicitly, not silently keep.
  `temporal_weights` is `[B, T]` softmax from the pooling MLP (full model only).

## Tech Stack

- **Language/DL**: Python + PyTorch. AMP via `torch.amp.autocast('cuda')` + `GradScaler`; `cudnn.benchmark`,
  TF32 / high matmul precision performance flags.
- **Data store**: LMDB chunks (JPEG-encoded crops + pickled metadata). ImageNet normalization.
- **Config + tracking (deliberately minimal)**: `yaml` config files → typed `dataclass` schema →
  `argparse` dotted overrides (e.g. `--train.lr 5e-5`). **No Hydra, no W&B.** Logging stays **CSV**.
  No hardcoded paths in code — everything flows from `configs/paths.yaml`.
- **Packaging**: `pyproject.toml`, src-layout install (`pip install -e .`), `ruff` lint + `pytest`.
- **Export/bench**: ONNX (onnxruntime parity check), `fvcore` for FLOPs.

## Target Repository Layout

```
src/pedpredict/            # installable package
  config/   schema.py loader.py     # dataclass schema + yaml→dataclass→argparse merge
  paths.py
  utils/    seed.py device.py amp.py memory.py logging.py
  data/     pie_sequences.py lmdb_writer.py balance.py augment.py
            lmdb_dataset.py transforms.py collate.py sampler.py stats.py
  models/   vit.py motion_encoder.py cross_attention.py ensemble.py
            ablations.py heads.py registry.py   # typed model factory (replaces stringly dispatch)
  losses/   multitask.py             # unified class-weight + per-task weighting
  training/ trainer.py chunk_loader.py callbacks.py metrics.py
  eval/     evaluate.py benchmark.py inference.py
  viz/      plots.py qualitative.py
  export/   onnx.py
scripts/    # thin CLI wrappers, one job each
  make_sequences.py build_lmdb.py balance_dataset.py augment_dataset.py
  count_labels.py train.py evaluate.py visualize.py export_onnx.py
configs/    paths.yaml data.yaml model.yaml train.yaml eval.yaml
tests/      test_config.py test_data_shapes.py test_lmdb_roundtrip.py
            test_model_shapes.py test_losses.py test_metrics.py test_golden_outputs.py
```

## Data Pipeline

**LMDB schema (written contract)** — keys reset **per chunk** (`<key>` = sample index within the chunk):
- `<key>_meta` (pickle): `motions[T,8]`, `actions`, `looks`, `crosses` (no `bboxes` — dropped in 1.2)
- per-frame `<key>_<t>_tight` and `<key>_<t>_context` JPEG blobs (stored **un-normalized** `[0,1]·255`;
  ImageNet normalize is applied at read time, not by the writer)

Stages (offline → runtime): PIE → sequence generation (sliding windows `seq_len=20`, `stride=3`,
`future_offset=30`, `tol=2`; filter #2 drops windows with any crossing during observation) → crop/motion
extraction + LMDB writer (`context_scale=3.0`, `jpeg_quality=90`, `chunk_size=5000`) → offline balance/split
→ offline augmentation (minority classes) → runtime `LMDBChunkDataset` (per-process env keyed on pid) + collate.

- `crosses` raw labels `{-1,0,1}` are clamped to `{0,1}` (at sequence generation, 1.1; the writer does not re-clamp).
- `context_scale` is a single uniform **3.0** across data + benchmark (kept config-flexible for ablation).
- The **8-dim motion feature** is `(cx, cy, dx, dy, w, h, dw, dh)` from the int-truncated bbox (documented per
  channel in `data/transforms.compute_motion`); `horizontal_flip` augmentation (1.4) negates **index 2 (dx)** —
  the index must match the channel definition or augmented data corrupts silently. ⚠️ Preserved legacy quirk:
  frame-0 `dw`/`dh` (idx 6/7) hold the *raw* `w0`/`h0`, not a delta (Phase-B fix candidate).

### Dataset Statistics (keep current)

Positive-class rates in the generated sequences:

| Split | N | actions=1 | looks=1 | crosses=1 |
|---|---|---|---|---|
| train | 95,684 | 45.3% | 17.1% | 2.6% |
| val   | 22,665 | 41.8% | 11.9% | 2.5% |
| test  | 76,048 | 43.5% | 15.8% | 2.8% |

`crosses` is severely imbalanced (~37:1); `looks` moderately (~5:1); `actions` roughly balanced.
Aggregate accuracy is misleading on `crosses` — rely on F1/AUC/recall. **If sequence generation params or
PIE annotations change, re-run label counting and update this table in the same change.** This table is the
data-layer drift check (`scripts/count_labels.py` should exit nonzero on drift).

## Imbalance Policy (single source of truth)

Three levers exist and must be documented as ONE coherent policy, not three accidents (band-aid B3):
1. **Offline balance** (`data/balance.py`) — constraint-solved `cross=0` down-sampling.
2. **Online sampler** (`data/sampler.py`) — `WeightedRandomSampler`, per-task powers
   (`crosses^1.5 · actions^0.3 · looks^0.7`).
3. **Loss class weights** (`losses/multitask.py`) — inverse-frequency CE weights + per-task scalar
   `loss_weight={actions:0.8, looks:0.8, crosses:1.2}`.

A **single LMDB metadata scan** produces both class frequencies (for loss) and per-sample sampler weights.

**Default (decided in 1.3):** levers **2 + 3 are ON** (both already in `TrainCfg`), layered on offline
**augmentation** (1.4) — this is what legacy training actually ran. Lever **1 (offline balance) is OPT-IN,
`BalanceCfg.enabled=false`** — the majority-downsample *alternative* to augmentation, for ablation; when
enabled, relax 2/3 so the levers don't triple-stack. The single metadata scan (1.6) feeds 2 + 3 only;
offline balance scans the sequence pkls (a separate offline artifact), not the LMDB.

## Band-Aids Being Resolved (legacy → target)

| # | Legacy smell | Resolution in rebuild |
|---|---|---|
| B1 | 635-line `train.py` god-script, hardcoded hyperparams | Split into `trainer`/`chunk_loader`/`callbacks`/`metrics`; all params from config |
| B2 | Lazy ViT `relative_position_bias` → dummy-forward hack before optimizer build | Create ALL params at `__init__` → enables `strict=True` load, clean ONNX export |
| B3 | Three overlapping imbalance mechanisms | Single documented imbalance policy (above) |
| B4 | Dead `crosses_pooled` head (computed, never supervised) | Keep `crosses_frame` supervised; mark/gate `crosses_pooled` explicitly — no silent dead compute |
| B5 | Fragmented 6+ data scripts | Canonical `data/` modules + thin `scripts/` CLIs |
| B6 | Config drift (`config.py` vs model `__main__`) | One typed config; `__main__` becomes a smoke test using `ModelCfg` |
| B7 | Magic constants in collate (`MAX_SEQ_LEN=20`, `motions[...,:8]`) | Move to `DataCfg`; writer emits exactly 8 dims so the slice disappears |
| B8 | Scattered AMP `.float()` casts | Single amp context + `to_float_logits` helper in `utils/amp.py` |
| B9 | Hand-rolled mp prefetch (`mp.Queue`, RAM polling, manual joins) | Encapsulated `ChunkPrefetcher` (start/next/close/`__exit__`), crash-safe |
| B10 | Stringly-typed model dispatch | Typed `registry.py` (`ModelType` enum/Literal + `build_model`/`forward_model`) |
| B11 | `venv/`, ~30 CSVs, root one-offs committed | `.gitignore`s artifacts; one-offs ported/folded-into-tests/dropped |
| B12 | No real tests / CI | Layered `tests/` + golden fixtures + ruff/pytest CI gate |
| B13 | Confusing `WindowTransformerBlock` MLP residual | Clean residual without changing math |

## Evaluation

Report **Accuracy, F1, AUC, Precision, Recall** (per task + macro-F1), logged to CSV. Also report
efficiency: **params, FLOPs (fvcore), latency, FPS, peak VRAM** per `model_type`. A single
`MetricAccumulator` is shared by training-validation and test (no divergence). Degenerate cases use
`zero_division=0`; AUC needs softmax probabilities. Model types: `full`, `motion_only`, `visual_only`,
`vanilla_concat` — selected via the typed registry, not raw strings.

## Working Conventions

- **Config-first**: no hardcoded hyperparameters or paths in module code; add a field to the dataclass schema + yaml.
- **Behavior-preserving**: before porting a module, capture its golden output from the OLD repo; add a
  golden-output test proving parity. Any intentional behavior change must be flagged and justified.
- **Coupled prompts**: imbalance (1.3/1.6/3.1) and output-contract (2.3/2.4/2.5/3.1/3.2/5.1) decisions must
  stay singular — when touching one, honor the decisions made in its siblings.
- **Naming/style**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants, `_` prefix private;
  imports stdlib → third-party → local; type hints on signatures, functions ≤50 lines, lines ≤120 chars.
- **Keep CLAUDE.md in sync**: if a change invalidates anything here (architecture, config defaults, dataset
  stats, output keys, layout), update the relevant section in the same change.

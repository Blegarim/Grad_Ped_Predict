# CLAUDE.md

Guidance for Claude Code when working in this repository.

`Grad_Ped_Predict` (graduate research) is a multimodal **pedestrian behavior prediction** project on the
**PIE dataset**: from a short sequence of dashcam frames it jointly predicts three binary tasks per
pedestrian — **actions** (walking/standing), **looks** (looking at traffic or not), **crosses** (will cross
soon). It began as a behavior-preserving rebuild of an undergraduate thesis; that rebuild has landed
**P0–P8** (foundation → data → models → loss/metrics → training → eval → viz → export → tests/CI) and
**P9 cutover is pending**.

The rebuild scaffolding — the legacy reference repo, the phase rules, and the band-aid inventory — is
quarantined in [Rebuild Context](#rebuild-context-phase-a--retire-at-cutover) at the end of this file and
will be retired at cutover (P9). Everything above that section describes the project as it now stands. For
the live module-by-module porting ledger (golden fixtures, parity results) see [MIGRATION.md](MIGRATION.md);
for the original rebuild plan see [REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md).

## Problem & Architecture

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

- **Unified `d_model = 128`** across ALL modules. Never change one module's dim without the others.
- **Output dict keys**: `actions`, `looks`, `crosses_pooled`, `crosses_frame`, `temporal_weights`.
  Training & eval supervise **ONLY `crosses_frame`** (logsumexp-pooled over frames). `crosses_pooled` is a
  **live-but-unsupervised** auxiliary head (`ModelCfg.emit_crosses_pooled=True` by default) — emitted and
  kept ready to swap in for `crosses_frame`, but **never routed to loss/metrics**; set
  `emit_crosses_pooled=false` to drop it (gating never perturbs the 4 supervised keys). `temporal_weights`
  is `[B, T]` softmax from the pooling MLP (full model only). (Resolves band-aid B4 — see Rebuild Context.)

## Tech Stack

- **Language/DL**: Python + PyTorch. AMP via `torch.amp.autocast('cuda')` + `GradScaler`; `cudnn.benchmark`,
  TF32 / high matmul precision performance flags.
- **Data store**: LMDB chunks (JPEG-encoded crops + pickled metadata). ImageNet normalization.
- **Config + tracking (deliberately minimal)**: `yaml` config files → typed `dataclass` schema → repeatable
  `--set section.field=value` CLI overrides (e.g. `--set train.lr=5e-5`). **No Hydra, no W&B.** Logging
  stays **CSV**. No hardcoded paths in code — everything flows from `configs/paths.yaml`.
  - **Run-dir convention**: one gitignored dir per run under `PathsCfg.runs_dir`
    (`outputs/runs/{run_id}/`, `run_id = {YYYYMMDD_HHMMSS}_{model_type}[_{tag}]`) holding
    `resolved_config.yaml` (config snapshot) + `train_log.csv` (per-epoch train+val) +
    `checkpoints/{best,last}.pth` + `plots/`. Test metrics → `eval_log.csv`. Cross-run comparison =
    `outputs/runs/index.csv` (one row/run, `crosses_f1`-led; `rebuild_index` regenerates it). Schemas are
    composed once: metric columns from `training/metrics.METRIC_COLUMNS`, run/index machinery in
    `utils/logging.py`.
- **Packaging**: `pyproject.toml`, src-layout install (`pip install -e .`), `ruff` lint + `pytest`.
- **Export/bench**: ONNX (onnxruntime parity check), `fvcore` for FLOPs.

## Repository Layout

```
src/pedpredict/            # installable package (pip install -e .)
  config/   schema.py loader.py     # dataclass schema + yaml→dataclass→--set merge
  paths.py
  utils/    seed.py device.py amp.py memory.py logging.py
  data/     pie_sequences.py lmdb_writer.py balance.py augment.py lmdb_warm.py
            lmdb_dataset.py transforms.py collate.py sampler.py stats.py
  models/   vit.py geometry.py motion_encoder.py cross_attention.py ensemble.py
            ablations.py heads.py registry.py   # typed model factory
  losses/   multitask.py             # unified class-weight + per-task weighting
  training/ trainer.py chunk_loader.py callbacks.py schedule.py metrics.py
  eval/     evaluate.py benchmark.py inference.py
  viz/      plots.py qualitative.py
  export/   onnx.py
scripts/    # thin CLI wrappers, one job each
  make_sequences.py build_lmdb.py balance_dataset.py augment_dataset.py count_labels.py
  train.py evaluate.py visualize.py infer_video.py export_onnx.py
configs/    paths.yaml data.yaml model.yaml train.yaml schedule.yaml
            eval.yaml balance.yaml augment.yaml export.yaml infer.yaml
tests/      config · data shapes · lmdb roundtrip · model shapes · losses · metrics · sampler ·
            golden outputs · trainer/eval/onnx · + fixtures/golden/ (captured legacy outputs)
```

## Commands

Setup and the full CLI surface live in [README.md](README.md); the essentials:

```bash
# Setup — core + lint/test; add [infer] for video, [export] for ONNX
pip install -e .[dev]

# Data pipeline (offline → runtime)
python scripts/make_sequences.py  --split all      # PIE → sequences_<split>.pkl
python scripts/build_lmdb.py      --split val      # sequences → LMDB chunks
python scripts/balance_dataset.py                  # opt-in offline balance (default off)
python scripts/augment_dataset.py                  # minority-class augmentation
python scripts/count_labels.py                     # dataset-stats drift gate (nonzero on drift)

# Train / evaluate — config-first; override any field with --set section.field=value
python scripts/train.py    --set model.model_type=full --set train.lr=5e-5
python scripts/evaluate.py --set model.model_type=full # test metrics → run-dir eval_log.csv

# Inference / export / viz
python scripts/infer_video.py  ...                 # needs [infer] (YOLO detect/track)
python scripts/export_onnx.py  ...                 # needs [export]; runs onnxruntime parity
python scripts/visualize.py    ...

# Gate (must pass)
ruff check .
pytest -m "not slow"
```

## Data Pipeline

**LMDB schema (written contract)** — keys reset **per chunk** (`<key>` = sample index within the chunk):
- `<key>_meta` (pickle): `motions[T,8]`, `actions`, `looks`, `crosses` (no `bboxes`)
- per-frame `<key>_<t>_tight` and `<key>_<t>_context` JPEG blobs (stored **un-normalized** `[0,1]·255`;
  ImageNet normalize is applied at read time, not by the writer)

Stages (offline → runtime): PIE → sequence generation (sliding windows `seq_len=20`, `stride=3`,
`future_offset=30`, `tol=2`; filter #2 drops windows with any crossing during observation) → crop/motion
extraction + LMDB writer (`context_scale=3.0`, `jpeg_quality=90`, `chunk_size=5000`) → offline balance/split
→ offline augmentation (minority classes) → runtime `LMDBChunkDataset` (per-process env keyed on pid) + collate.

- `crosses` raw labels `{-1,0,1}` are clamped to `{0,1}` (at sequence generation; the writer does not re-clamp).
- `context_scale` is a single uniform **3.0** across data + benchmark (kept config-flexible for ablation).
- The **8-dim motion feature** is `(cx, cy, dx, dy, w, h, dw, dh)` from the int-truncated bbox (documented per
  channel in `data/transforms.compute_motion`); `horizontal_flip` augmentation negates **index 2 (dx)** —
  the index must match the channel definition or augmented data corrupts silently. ⚠️ Preserved quirk:
  frame-0 `dw`/`dh` (idx 6/7) hold the *raw* `w0`/`h0`, not a delta (improvement candidate).

### Dataset Statistics

Positive-class rates in the generated sequences:

| Split | N | actions=1 | looks=1 | crosses=1 |
|---|---|---|---|---|
| train | 95,684 | 45.3% | 17.1% | 2.6% |
| val   | 22,665 | 41.8% | 11.9% | 2.5% |
| test  | 76,048 | 43.5% | 15.8% | 2.8% |

`crosses` is severely imbalanced (~37:1); `looks` moderately (~5:1); `actions` roughly balanced.
Aggregate accuracy is misleading on `crosses` — rely on F1/AUC/recall. **If sequence generation params or
PIE annotations change, re-run label counting and update this table in the same change.** This table is the
data-layer drift check (`scripts/count_labels.py` exits nonzero on drift).

## Imbalance Policy (single source of truth)

Three levers exist and must be documented as ONE coherent policy, not three accidents:
1. **Offline balance** (`data/balance.py`) — constraint-solved `cross=0` down-sampling.
2. **Online sampler** (`data/sampler.py`) — `WeightedRandomSampler`, per-task powers
   (`crosses^1.5 · actions^0.3 · looks^0.7`).
3. **Loss class weights** (`losses/multitask.py`) — inverse-frequency CE weights + per-task scalar
   `loss_weight={actions:0.8, looks:0.8, crosses:1.2}`.

A **single LMDB metadata scan** produces both class frequencies (for loss) and per-sample sampler weights.

**Default:** levers **2 + 3 are ON** (both in `TrainCfg`), layered on offline **augmentation**. Lever
**1 (offline balance) is OPT-IN, `BalanceCfg.enabled=false`** — the majority-downsample *alternative* to
augmentation, for ablation; when enabled, relax 2/3 so the levers don't triple-stack. The single metadata
scan feeds 2 + 3 only; offline balance scans the sequence pkls (a separate offline artifact), not the LMDB.

## Evaluation

Report **Accuracy, F1, AUC, Precision, Recall** (per task + macro-F1), logged to CSV. Also report
efficiency: **params, FLOPs (fvcore), latency, FPS, peak VRAM** per `model_type`. A single
`MetricAccumulator` is shared by training-validation and test (no divergence). Degenerate cases use
`zero_division=0`; AUC needs softmax probabilities. Model types: `full`, `motion_only`, `visual_only`,
`vanilla_concat` — selected via the typed registry, not raw strings.

## Working Conventions

- **Config-first**: no hardcoded hyperparameters or paths in module code; add a field to the dataclass schema + yaml.
- **Single sources of truth**: the imbalance policy (above) and the output-dict contract are each decided in
  one place — when touching one site, honor the others (loss/sampler/balance move together; head wiring and
  the supervised-keys rule move together).
- **Naming/style**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants, `_` prefix private;
  imports stdlib → third-party → local; type hints on signatures, functions ≤50 lines, lines ≤120 chars.

### Doc-Sync Checklist

When you change… update (in the same change):

| Change | Update |
|---|---|
| Sequence-gen params / PIE annotations | Dataset Statistics table + re-run `count_labels.py` (gate) + `test_stats` fixture |
| Output-dict keys / head wiring | Architecture output-keys note + `heads.py`/`ensemble.py` docstrings + B4 (Rebuild Context) |
| Imbalance levers (balance / sampler / loss weights) | Imbalance Policy section — all three levers together |
| `d_model` / module dims | Architecture table (CLAUDE.md + README) — never one module alone |
| Add / move / remove a `src/` module or `scripts/` CLI | Repository Layout + Commands (CLAUDE.md) + README layout |
| Config schema field / default | `configs/*.yaml` + schema docstrings; Config note if the CLI surface changes |
| New extra / dependency | README Install extras + MIGRATION.md dependency notes |
| Resolve / alter a band-aid | Band-Aids table (Rebuild Context) + MIGRATION.md row |

---

## Rebuild Context (Phase A — retire at cutover)

> **This section is rebuild scaffolding, not standalone project documentation.** It records how the repo
> was rebuilt from the legacy thesis and the legacy smells ("band-aids") that the rebuild resolved. It stays
> load-bearing until the **P9 cutover** (parity gate + legacy retirement), after which it — along with the
> archived [REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md) and [MIGRATION.md](MIGRATION.md) — is removed.

**Legacy reference.** The old undergrad repo is read-only reference at `OLD/Undergrad_thesis_project`
(vendored so remote sandboxes have it; the golden artifact reference — subsampled sequence fixtures — lives
in `OLD/golden/`). Code was ported piece-by-piece into the layout above.

**Two-phase plan.**
- **Phase A (essentially complete, P0–P8): behavior-preserving restructure.** Same model math, same outputs,
  cleaner code. Every ported module is numerically equivalent (within float tolerance) to the legacy module
  for the same inputs and weights — *unless* a listed band-aid intentionally changes behavior, which is
  called out and justified. A **golden-output fixture** was captured from the OLD repo *before* porting each
  module (the behavior-preserving safety net; see [MIGRATION.md](MIGRATION.md) and `tests/_golden.py`).
- **Phase B (deferred): architecture redesign.** ViT backbone swap, fusion rethink, single unified crosses
  head, online augmentation, standard DataLoader sharding. Out of scope until Phase A parity is locked at P9.

**Band-Aids resolved (legacy → rebuild):**

| # | Legacy smell | Resolution in rebuild |
|---|---|---|
| B1 | 635-line `train.py` god-script, hardcoded hyperparams | Split into `trainer`/`chunk_loader`/`callbacks`/`metrics`; all params from config |
| B2 | Lazy ViT `relative_position_bias` → dummy-forward hack before optimizer build | Create ALL params at `__init__` → enables `strict=True` load, clean ONNX export |
| B3 | Three overlapping imbalance mechanisms | Single documented imbalance policy (above) |
| B4 | Dead `crosses_pooled` head (computed, never supervised) | Keep `crosses_frame` supervised; `crosses_pooled` made live-but-unsupervised + gated — no silent dead compute |
| B5 | Fragmented 6+ data scripts | Canonical `data/` modules + thin `scripts/` CLIs |
| B6 | Config drift (`config.py` vs model `__main__`) | One typed config; `__main__` becomes a smoke test using `ModelCfg` |
| B7 | Magic constants in collate (`MAX_SEQ_LEN=20`, `motions[...,:8]`) | Move to `DataCfg`; writer emits exactly 8 dims so the slice disappears |
| B8 | Scattered AMP `.float()` casts | Single amp context + `to_float_logits` helper in `utils/amp.py` |
| B9 | Hand-rolled mp prefetch (`mp.Queue`, RAM polling, manual joins) | Encapsulated `ChunkPrefetcher` (start/next/close/`__exit__`), crash-safe |
| B10 | Stringly-typed model dispatch | Typed `registry.py` (`ModelType` enum/Literal + `build_model`/`forward_model`) |
| B11 | `venv/`, ~30 CSVs, root one-offs committed | `.gitignore`s artifacts; one-offs ported/folded-into-tests/dropped |
| B12 | No real tests / CI | Layered `tests/` + golden fixtures + ruff/pytest CI gate |
| B13 | Confusing `WindowTransformerBlock` MLP residual | Clean residual without changing math |

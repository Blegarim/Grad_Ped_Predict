# CLAUDE.md

Guidance for Claude Code when working in this repository.

> **Read this as broad context, not a changelog.** CLAUDE.md is both an orientation layer and a working
> guideline — it carries enough of the project's shape, contracts, and conventions to brief a capable
> engineer who has never seen the repo, and no more. Keep it live and reactive to *structural* change
> (new contracts, moved modules, changed policy), but don't let it become a running log where every
> incremental edit earns a line. Update stale facts in place; detail that is only true this week belongs
> in a run dir or a `docs/` note, not here.

`Grad_Ped_Predict` (graduate research) is a multimodal **pedestrian behavior prediction** project on the
**PIE dataset**: from a short sequence of dashcam frames it jointly predicts three binary tasks per
pedestrian — **actions** (walking/standing), **looks** (looking at traffic or not), **crosses** (will cross
soon). It is a clean, tested, config-driven PyTorch codebase (v1.0 baseline).

> The project began as a behavior-preserving rebuild of an undergraduate thesis. That history — the legacy
> reference repo, the phase plan, and the resolved band-aid inventory — is archived under
> [`docs/archive/`](docs/archive/) and in the `legacy-archive` git tag; it is no longer load-bearing.

> **Thesis direction (July 2026 pivot — read this before RESEARCH_PLAN).** The thesis spine is now
> **online detection of pedestrian crossing-onset**: the standard PIE protocol is event-anchored (~2.5:1)
> and hides the deployed streaming case (~37:1, full of "will-cross-soon" hard negatives); the
> contribution is to re-evaluate under a streaming protocol and **decompose** the anchored→streaming
> performance gap into a recalibration-fixable part and a residual hard-negative part. Crucially, the
> existing dense-sliding-window pipeline **is** that streaming formulation, so the codebase stands — the
> architecture / imbalance / calibration work (the old RQs, [docs/HOLE_AUDIT.md](docs/HOLE_AUDIT.md),
> [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)) is retained as **supporting** studies, not the thesis
> structure. The brief is [docs/project-context-streaming-crossing-onset.md](docs/project-context-streaming-crossing-onset.md)
> and the execution plan is [docs/streaming-onset-plan.md](docs/streaming-onset-plan.md). HOLE_AUDIT's
> Final attack order still governs the data/engineering steps; docs/PHASE_B_BACKLOG.md is superseded.

## Execution Environment (two machines)

**Training and evaluation run on a separate lab PC that holds the data; ~97% of coding happens on a
personal PC with insufficient hardware for most tasks.** In this working copy, assume **no data or
training trail is visible** — the only run artifacts present are the **CSV logs and JSON/YAML metadata**
under `outputs/runs/{run_id}/`. LMDB stores, sequence pkls, PIE frames, checkpoints, and plots live only
on the lab PC. Do **not** try to scan LMDBs, load checkpoints, or regenerate data here; instead write
code/config the lab PC will run. **Notify the user when a step absolutely requires the data, checkpoints,
or figures** (e.g. the M1 sampler sweep, a training run, an eval pass, a viz pass) so it can be run on
the lab PC.

## Problem & Architecture

```
context crop frames → ViT_Hierarchical  ──┐
                                           ├→ CrossAttentionModule → EnsembleModel → {actions, looks, crosses}
tight crop + motion → MotionEncoder    ───┘
```

| Component | Role |
|---|---|
| `ViT_Hierarchical` | Hierarchical windowed-attention ViT on context crops (stem conv7×7 s4, per-stage downsample s2, global-avg-pool, `frame_proj`). Stage schedule is the **A1 redesign** — monotonic dims `[48,96,192,384]`, real 7×7 windows (last global), ~7–8M params (the collapsed legacy `[36,36,288,36]` + 2×2 windows is golden-pinned in tests, not the default). Outputs `[B, T, d_model]`. The visual stream is **swappable** via `model.vit_backbone` (RQ1): `legacy` (default, this module) \| a `timm` model name (e.g. `tiny_vit_5m_224`) builds `TimmBackbone` (`models/timm_backbone.py`) behind the same `[B,T,3,H,W]→[B,T,d_model]` contract, `model.vit_pretrained` gating ImageNet weights — see [docs/BACKBONE_STUDY.md](docs/BACKBONE_STUDY.md). |
| `MotionEncoder` | Temporal CNN over tight crops + Conv1d motion stack + fusion + GRU + learned pos-encoding + MultiheadAttention. In-forward motion norm is config-gated: `model.motion_norm` = `image` (fixed frame-dim scale, default) \| `per_sequence` (legacy z-norm, A4 ablation arm). Outputs `[B, T, d_model]`. |
| `CrossAttentionModule` | Cross-attention (query=motion, key/value=image) → pooling MLP → softmax temporal weights → per-task classifier heads. `model.fusion_residual` (A3/RQ2, **default on**) adds the motion query back at fusion (`attn_output + motion_feats`) so motion *content* reaches the heads, not just motion-as-attention-mask; `=false` is the no-residual A3 ablation (golden-pinned). |
| `EnsembleModel` | Wires all components; applies **LayerNorm before fusion**; `return_feats` path used by viz. |
| Ablations | `PedLocalModel` (tight-crop CNN + bbox kinematics, no scene context — legacy `motion_only`, renamed per A6), `KinematicsOnlyModel` (NEW, pixel-free bbox-kinematics baseline, M9.1), `VisualOnlyModel`, `VanillaConcatModel` (concat instead of cross-attention); same output-dict format. |
| Pose arm | [docs/POSE_ENCODER.md](docs/POSE_ENCODER.md): `pose_kinematics` (= `KinematicsOnlyModel` fed the 58-dim pose+motion vector, pixel-free) and `pose_full` (`PoseFullModel` — that vector as the cross-attention query over the ViT context, no tight crop; emits `temporal_weights` like `full`). Needs a pose-enabled LMDB build: `pose.enabled` + `data`/`model.motion_dim=58` + `model.motion_norm=none` (validated as a bundle). Pose math + cache reader: `data/pose.py`; extraction: `scripts/extract_pose.py` (`--dry-run` fabricates keypoints so the pipeline runs without frames/extractor). |

- **Unified `d_model = 128`** across ALL modules. Never change one module's dim without the others.
- **Output dict keys**: `actions`, `looks`, `crosses_pooled`, `crosses_frame`, `temporal_weights`.
  Training & eval supervise **ONLY `crosses_frame`** (logsumexp-pooled over frames). `crosses_pooled` is a
  **live-but-unsupervised** auxiliary head (`ModelCfg.emit_crosses_pooled=True`, default on) — kept ready
  to swap in but **never routed to loss/metrics**; `emit_crosses_pooled=false` drops it without perturbing
  the 4 supervised keys. `temporal_weights` is `[B, T]` softmax from the pooling MLP (full model only).

## Tech Stack

Python + PyTorch (AMP, TF32, `cudnn.benchmark`); LMDB chunk store (JPEG crops + pickled meta, ImageNet
norm at read time); typed `dataclass` configs with `--set section.field=value` overrides — **no Hydra, no
W&B**, logging stays **CSV**, no hardcoded paths (all flow from `configs/paths.yaml`). ONNX + `fvcore`
bench. Install, extras, and packaging: [README.md](README.md).

**Run-dir convention** (the only run artifacts visible on the personal PC): one gitignored dir per run at
`PathsCfg.runs_dir/{run_id}/` (`run_id = {YYYYMMDD_HHMMSS}_{model_type}[_{tag}]`) holding
`resolved_config.yaml` (snapshot incl. `train.seed`), `train_log.csv` (per-epoch train+val),
`train_distribution.json` (M1 sampler-draw distribution), `thresholds.json` (M2 val-tuned thresholds),
`checkpoints/{best,last}.pth`, `plots/`; test metrics → `eval_log.csv`; cross-run table →
`outputs/runs/index.csv` (`crosses_f1`-led, `rebuild_index` regenerates). Schema machinery: `utils/logging.py`
+ `training/metrics.METRIC_COLUMNS`.

## Repository Layout

The `src/pedpredict/` package tree, `scripts/` CLIs, and `configs/` are mapped in
[README.md](README.md#repository-layout). What the tree doesn't show: `scripts/` are thin one-job CLI
wrappers; models are built through the typed registry in `models/registry.py`; `tests/` pairs unit tests
with **golden characterization tests** under `tests/fixtures/golden/` (captured legacy numerics — parity
guardrails, not regenerable here).

## Commands

Setup, the data pipeline, training/eval, viz, and export are all in [README.md](README.md) — that is the
full CLI surface. The data-layer contracts and experimental-validity rules that govern *how* to run them
are below. The gate that must pass before any change lands: `ruff check .` and `pytest -m "not slow"`.

## Data Pipeline

Offline → runtime: PIE → sequence windows (`make_sequences`) → LMDB chunks (`build_lmdb`) → offline
balance/augment → runtime `LMDBChunkDataset` + collate. The runtime dataset also supports **config-gated
on-the-fly augmentation** (`augment.runtime`, default off, train split only, seeded per run+epoch+index):
fresh ratio-preserving transforms per sample in the read path (applied pre-ImageNet-norm) — a data-scarcity
regularizer, distinct from the offline minority-oversampling `augment.enabled` and NOT one of the three
imbalance levers below. Sequence-gen params (`seq_len`, `stride`,
`future_offset`, `context_scale`, …) live in `configs/data.yaml`; the **LMDB schema v2** key/value contract
is in [data/lmdb_writer.py](src/pedpredict/data/lmdb_writer.py) and the 9-dim motion channel table in
[data/transforms.py](src/pedpredict/data/transforms.py). Crops are stored un-normalized (ImageNet norm at
read time); consumers slice motions to `data.motion_dim` (8 = no ego, 9 = with ego; 58 = pose arm, where
the read path builds the pose feature block instead of slicing).

**v2 labeling contract — deliberate departures from v1/legacy; do not "fix" these as bugs.** Full rationale
in [docs/HOLE_AUDIT.md](docs/HOLE_AUDIT.md) (M3–M9, A4):
- **M3** — `actions`/`looks` label the **state at the last observed frame**, not the future; only `crosses`
  is a future label (`any()` over the fully-observed future window).
- **M4** — right-censored windows (truncated future) are **dropped, not labeled 0**.
- **M5** — a separate TTE **benchmark** (anchored-protocol) set labels `crosses` by the crossing *event*
  and carries `tte`; built via `make_sequences.py --benchmark --split {train,val,test}` +
  `build_lmdb[_incremental].py --split {train,val,test}_benchmark` → `preprocessed_{split}_benchmark`.
  Which set train/eval read is the runtime switch **`data.protocol`** (`streaming` default = standard v2
  LMDBs, ~37:1 | `anchored` = benchmark LMDBs, ~2.5:1), resolved once in `paths.protocol_lmdb_dirs` and
  honored by train.py / `ChunkPrefetcher` / `evaluate.py` — this is the **cross-protocol-matrix axis**
  (train × test over {anchored, streaming}); see [setup.md §9](setup.md).
- **M6** — every meta carries `track_id` for eval-side track aggregation.
- **M9 / A4** — motion is stored **9-dim** (`MOTION_STORE_DIM`), frame-0 deltas true zeros; normalization is
  a runtime flag `model.motion_norm` (`image` default | `per_sequence` legacy/A4 arm, golden-pinned).
  `horizontal_flip` **must** negate `dx` (idx 2) and reflect `cx` (idx 0) about `data.source_width`, or
  augmented data corrupts silently.
- **Pose (additive meta key)** — when `pose.enabled`, every meta also stores raw keypoints `pose[T,23,3]`
  (COCO-WholeBody body-17+feet-6, absolute px, from the `extract_pose.py` cache); existing consumers
  ignore it. Features are built at **read time** (`data/pose.py`), so `motions` leaves the dataset as
  `[T, 58]` with the 9-dim block already image-normalized — hence pose models run
  `model.motion_norm="none"`. `horizontal_flip` **must** also mirror pose (`flip_pose`: reflect x about
  `data.source_width` + swap left/right joints) — same silent-corruption stakes as the motion-flip rule.
- **S1 (streaming pivot)** — each standard `SequenceRecord` also carries pure per-window onset annotation:
  `onset_offset` (frames from end-of-obs to the first future crossing; `-1` if none), `future_observed`
  (`n − end`), and `track_crosses` (track ever crosses). These do **not** change the `crosses` label or
  which windows are emitted — they let eval re-label at any horizon H and type the streaming negatives
  (genuine / hard-temporal / already-crossed). See [docs/streaming-onset-plan.md](docs/streaming-onset-plan.md).

### Dataset Statistics

(.venv) D:\Grad_Ped_Predict>python scripts/count_labels.py
| Split | N | actions=1 | looks=1 | crosses=1 |
|---|---|---|---|---|
| train | 88214 | 43.4% | 10.5% | 2.9% |
| val | 20490 | 40.3% | 6.7% | 2.8% |
| test | 69875 | 42.2% | 9.8% | 3.1% |

Wrote D:\Grad_Ped_Predict\training_log\label_count.csv
Label-count drift vs documented table:
  [train] N drift: got 88214, expected 95684
  [train] actions=1 drift: got 38320, expected 43310
  [train] looks=1 drift: got 9266, expected 16394
  [val] N drift: got 20490, expected 22665
  [val] actions=1 drift: got 8261, expected 9483
  [val] looks=1 drift: got 1376, expected 2688
  [test] N drift: got 69875, expected 76048
  [test] actions=1 drift: got 29489, expected 33068
  [test] looks=1 drift: got 6815, expected 12013

`crosses` is severely imbalanced (~37:1); `looks` moderately (~5:1); `actions` roughly balanced. Aggregate
accuracy is misleading on `crosses` — rely on F1/AUC/recall. This table is the data-layer drift check
(`scripts/count_labels.py` exits nonzero on drift); re-run it and update the table in the same change as
any sequence-gen / PIE-annotation change.

## Imbalance Policy (single source of truth)

Three levers exist and must be documented as ONE coherent policy, not three accidents:
1. **Offline balance** (`data/balance.py`) — constraint-solved `cross=0` down-sampling.
2. **Online sampler** (`data/sampler.py`) — `WeightedRandomSampler`, per-task powers
   (`crosses^0.5 · actions^0.3 · looks^0.3`, tuned down — run #2 canonical).
3. **Loss class weights** (`losses/multitask.py`) — inverse-frequency CE weights (gated by
   `use_class_weights`) + always-on per-task scalar `loss_weight={actions:0.8, looks:0.8, crosses:1.2}`.

A **single LMDB metadata scan** produces both class frequencies (for loss) and per-sample sampler weights.

**Default (run #2 canonical):** lever **2 (sampler) is ON but tuned down** to `crosses^0.5` and lever
**3's inverse-frequency CE weights are OFF** (`use_class_weights=false`; the per-task scalar `loss_weight`
still applies), layered on offline **augmentation** — this keeps the effective `crosses` training rate at
~26% (vs the 89% the old `crosses^1.5` + class-weights stack produced). Lever
**1 (offline balance) is OPT-IN, `BalanceCfg.enabled=false`** — the majority-downsample *alternative* to
augmentation, for ablation; when enabled, relax 2/3 so the levers don't triple-stack. The single metadata
scan feeds 2 + 3 only; offline balance scans the sequence pkls (a separate offline artifact), not the LMDB.

**Every lever is switchable from config** (M1): `augment.enabled`, `balance.enabled`,
`train.use_weighted_sampler`, `train.use_class_weights` — the lever combination is the RQ3 ablation axis.
**Never toggle blind:** the M1 instrument (`training/distribution.py`, auto-written to every run dir as
`train_distribution.json`; standalone via `scripts/report_distribution.py`) reports the *effective*
per-task positive rate of sampler draws vs. the stored base rate — under the tuned-down default stack the
`crosses` training distribution sits at ~26%, still well above the 2.8% deployment rate (the gap the
instrument exposes), but far below the ~89% the old aggressive stack produced.

## Evaluation

Report **Accuracy, F1, AUC, Precision, Recall** (per task + macro-F1), logged to CSV. Also report
efficiency: **params, FLOPs (fvcore), latency, FPS, peak VRAM** per `model_type`. A single
`MetricAccumulator` is shared by training-validation and test (no divergence). Degenerate cases use
`zero_division=0`; AUC needs softmax probabilities. Model types: `full`, `ped_local` (legacy
`motion_only`, renamed per A6 — it reads the tight crop), `kinematics_only` (pixel-free baseline, M9.1),
`visual_only`, `vanilla_concat`, `pose_kinematics` / `pose_full` (pose arm, need a pose-enabled build) —
selected via the typed registry, not raw strings. **Single-task**
(crosses-only) is not a model type but a config recipe: zero the disabled tasks in *both* imbalance-aware
levers — `--set train.loss_weight.actions=0 train.loss_weight.looks=0 train.sampler_powers.actions=0
train.sampler_powers.looks=0` (the loss honors a 0 weight; the dead heads still print metrics, harmless).

**Experimental-validity rules (M2/M7/M8 — non-negotiable):**
- **Thresholds are tuned on val, never test.** `evaluate.py --split val` sweeps per-task F1-optimal
  thresholds and stores them in the run dir (`thresholds.json`); `--split test` loads and applies them —
  the `tuned_*` columns (incl. `tuned_macro_acc`) are the ONLY reportable threshold-tuned numbers. The
  same-split sweep is logged as `oracle_*` / `oracle_macro_acc` (test-set leakage — diagnosis only,
  **never quote in a results table**). `overall_acc` = pooled micro accuracy; `*_macro_acc` = mean of
  per-task accuracies (Q3 disambiguation).
- **Every run is seeded** (`train.seed`, default 42; in the config snapshot). Multi-seed protocol:
  screen with 1 seed, confirm finalists with 3, report mean±std.
- **Model selection + early stopping read `train.selection_metric`** (default `macro_f1`, maximized;
  options `val_loss`, `crosses_f1`). The LR follows `train.lr_schedule`: default `warmup_cosine`
  (deterministic linear-warmup→cosine to `lr_min`, decoupled from val_loss) or `plateau`
  (`ReduceLROnPlateau` on `val_loss`, the legacy arm). `best_val_loss` in checkpoints/index = the val
  loss at the *selected* best epoch.

## Working Conventions

- **Config-first**: no hardcoded hyperparameters or paths in module code; add a field to the dataclass schema + yaml.
- **Single sources of truth**: the imbalance policy (above) and the output-dict contract are each decided in
  one place — when touching one site, honor the others (loss/sampler/balance move together; head wiring and
  the supervised-keys rule move together).
- **Naming/style**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants, `_` prefix private;
  imports stdlib → third-party → local; type hints on signatures, functions ≤50 lines, lines ≤120 chars.

### Skill Invocation Policy

Skill *descriptions* are always in context (free to scan); only loading a `SKILL.md` *body* costs tokens,
so gate on **invocation**, scored continuously on a low, cost-aware threshold (don't sort into named tiers
— that anchors on heavyweight examples and suppresses loading):

- **Benefit `B`** (0–15) = sum of five 0–3 signals: *match* (how directly the best skill targets the ask),
  *method* (needs a procedure vs plain recall), *surface* (files/decisions in play), *stakes* (read-only 0
  … irreversible/training/prod 3), *doubt* (how unsure I'd be unaided).
- **Cost `C`**: markdown-only repo skill ≈1; bundles scripts ≈2; spawns agents / multi-step
  (`understand-*`, `ship`) ≈3.
- **Load the single best skill when `B ≥ 3 + 2·(C−1)`** — cheap skills trip at `B≥3` (eager), agentic ones
  at `B≥7`. Load **multiple** (one per phase) only when `B ≥ 11` and ≥2 skills clearly apply; below the
  bar, answer directly.
- **Dial:** base `3` is the eagerness knob (lower = more eager). Tie-break to the most specific /
  project-local skill; never reload an active skill. `B` is a graded judgment, not literal arithmetic.

### Doc-Sync Checklist

When you change… update (in the same change):

| Change | Update |
|---|---|
| Sequence-gen params / PIE annotations | Dataset Statistics table + re-run `count_labels.py` (gate) + `test_stats` fixture |
| Output-dict keys / head wiring | Architecture output-keys note + `heads.py`/`ensemble.py` docstrings |
| Imbalance levers (balance / sampler / loss weights) | Imbalance Policy section — all three levers together |
| `d_model` / module dims | Architecture table (CLAUDE.md + README) — never one module alone |
| Add / move / remove a `src/` module or `scripts/` CLI | README layout + README command list (+ the orientation note in CLAUDE.md if a whole subsystem moves) |
| Config schema field / default | `configs/*.yaml` + schema docstrings; Config note if the CLI surface changes |
| New extra / dependency | README Install extras |

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
> reference repo, the phase plan, the resolved engineering audit, and the pre-pivot research plan — is
> archived under [`docs/archive/`](docs/archive/) and in the `legacy-archive` git tag; none of it is
> load-bearing.

## Thesis Direction (current — as of August 2026)

**A methods thesis on streaming crossing-onset detection.** The standard PIE protocol is event-anchored
(~2.5:1) and hides the deployed streaming case (~37:1, full of "will-cross-soon" hard negatives). The
existing dense-sliding-window pipeline **is** that streaming formulation, so the codebase stands.

*The motivation* (measured, done — [outputs/runs/RESULTS_MATRIX.md](outputs/runs/RESULTS_MATRIX.md)): a
model trained on the anchored benchmark loses almost all ranking ability on the stream (AUC 0.88 → 0.53)
and re-tuning thresholds recovers ~nothing (`G_prior ≈ 0`, `G_hardneg ≈ G_total`). On its own that is a
negative result — "the usual benchmark flatters models" — so it became the *justification* rather than the
contribution.

*The contribution* (in progress, spine set 2026-08-20): treat streaming crossing prediction as **onset
timing under censoring** rather than yes/no classification at a fixed horizon. A window whose future ran
out is a censored observation, not a negative; with no fixed cut-off the "will cross, just not yet"
population stops being mislabeled by construction. Two supporting parts: pose *movement* features (the cue
a one-second horizon actually has) and the online-action-detection literature (its metrics as the
instrument, its objectives as comparison baselines). **Hard constraint on any timing model: it must still
emit P(onset within 32 frames), or the four baseline runs stop being comparable.**

| Doc | Role |
|---|---|
| [docs/THESIS_ROADMAP.md](docs/THESIS_ROADMAP.md) | **The tracker** — every stage, what's done, what's left, plus the supporting-study spokes |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | **The method** — onset timing under censoring, the two supporting parts, and how they were chosen (active working reference) |
| [outputs/runs/RESULTS_MATRIX.md](outputs/runs/RESULTS_MATRIX.md) | **The numbers** — the cross-protocol matrix, the baselines, the caveats |
| [docs/project-context-streaming-crossing-onset.md](docs/project-context-streaming-crossing-onset.md) | **The argument** — why streaming evaluation matters (frozen reference) |

**Two consequences that bite when touching anything experimental:** (1) the four existing `pose_full` runs
are now **baselines**, so an accidental config difference between two legs invalidates a comparison rather
than merely footnoting a caveat; (2) the three S1 onset fields are computed but dropped before the trainer
can see them (see the S1 bullet under Data Pipeline), which blocks the method itself — onset timing has
nothing to train against until they arrive.

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
- **Output dict keys**: `actions`, `looks`, `crosses_pooled`, `crosses_frame`, `temporal_weights`, plus
  the gated `crosses_hazard` / `crosses_readout`.
  Training & eval supervise **ONLY `crosses_frame`** (logsumexp-pooled over frames). `crosses_pooled` is a
  **live-but-unsupervised** auxiliary head (`ModelCfg.emit_crosses_pooled=True`, default on) — kept ready
  to swap in but **never routed to loss/metrics**; `emit_crosses_pooled=false` drops it without perturbing
  the 4 supervised keys. `temporal_weights` is `[B, T]` softmax from the pooling MLP (full model only).
- **Onset head** (`ModelCfg.onset_head`, **default off** — golden-pinned): adds `crosses_hazard [B, K]`
  (one hazard logit per future bin) and `crosses_readout [B, 2]` (those bins collapsed to
  `P(onset ≤ onset_horizon) = 1 − Π(1 − h_k)`). Off, there is no extra key and **no extra parameter**, so
  every existing checkpoint still loads `strict=True`. The head hangs off the pooled `[B, D]` vector, so
  its `K` axis indexes the **future** — distinct from `crosses_frame`, whose `T` axis indexes the observed
  past. See the Onset Timing section below.

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

**v2 labeling contract — deliberate departures from v1/legacy; do not "fix" these as bugs.** Each carries
its *why* inline (they were decided in the 2026-06 engineering audit, now archived at
[docs/archive/HOLE_AUDIT.md](docs/archive/HOLE_AUDIT.md) — items M3–M9, A4 — which remains the long-form
record but is no longer required reading):
- **M3** — `actions`/`looks` label the **state at the last observed frame**, not the future; only `crosses`
  is a future label (`any()` over the fully-observed future window). *Why:* `any()` over a 32-frame future
  can only inflate positives, and for a present-state attribute it is simply the wrong question — a single
  one-frame glance anywhere in the future flipped `looks` to 1, which is why its rate fell hardest on the
  v2 regen (17.1% → 10.5%).
- **M4** — right-censored windows (truncated future) are **dropped, not labeled 0**. *Why:* when the future
  was never observed, "did not cross" is a fabricated label, not a missing one. The per-split censored
  count is recorded (`WindowStats` → `sequences_<split>_stats.json`) and is thesis-reportable.
  **Refinement (`data.emit_determined_positives`, default off).** The filter asks only "was the full future
  observed?", so it *also* binned windows whose crossing was plainly **seen** inside the truncated
  remainder — labels that were never in doubt, and positives, the scarcest class. On, those are kept
  (`WindowStats.determined_positive` counts them); truncated windows with no observed crossing are still
  dropped. This is the full knowability rule `data/onset_stats.is_usable` already stated but generation
  never applied. **It changes the window population** — regen + a four-way Dataset Statistics re-pin, and
  it invalidates comparisons against runs built without it.
- **M5** — a separate TTE **benchmark** (anchored-protocol) set labels `crosses` by the crossing *event*
  and carries `tte`; built via `make_sequences.py --benchmark --split {train,val,test}` +
  `build_lmdb[_incremental].py --split {train,val,test}_benchmark` → `preprocessed_{split}_benchmark`.
  Which set train/eval read is the runtime switch **`data.protocol`** (`streaming` default = standard v2
  LMDBs, ~37:1 | `anchored` = benchmark LMDBs, ~2.5:1), resolved once in `paths.protocol_lmdb_dirs` and
  honored by train.py / `ChunkPrefetcher` / `evaluate.py` — this is the **cross-protocol-matrix axis**
  (train × test over {anchored, streaming}); see [setup.md §9](setup.md). *Why an added set rather than a
  reprotocol:* training is untouched — the benchmark set is an *additional evaluation* of the
  already-trained model, so the project's 20-frame/~1 s early-anticipation windowing stays primary (its
  rationale: pedestrians appear suddenly, so the model must commit on short evidence, and the decision
  must hold for the immediate future to be actionable) with the benchmark row as the caveated,
  externally-anchored comparison.
- **M6** — every meta carries `track_id` for eval-side track aggregation. *Why:* at `seq_len=20, stride=3`
  two consecutive windows of the same pedestrian share 17 of 20 frames. A pedestrian tracked 600 frames
  contributes ~200 near-copies, one tracked 25 frames contributes 2 — so long easy tracks dominate the
  score, the ~70k test "samples" behave like far fewer independent ones, and per-*pedestrian* performance
  (what the safety task actually asks) is invisible. Track metrics are reported **alongside** window
  metrics, not instead of them.
- **M9 / A4** — motion is stored **9-dim** (`MOTION_STORE_DIM`), frame-0 deltas true zeros; normalization is
  a runtime flag `model.motion_norm` (`image` default | `per_sequence` legacy/A4 arm, golden-pinned).
  `horizontal_flip` **must** negate `dx` (idx 2) and reflect `cx` (idx 0) about `data.source_width`, or
  augmented data corrupts silently. *Why store wide and slice narrow:* with-ego and without-ego are two
  configs → two trained models (an ablation retrains anyway), so a fixed input width buys nothing and
  zero-padding a dead channel would only inject noise into a param-matched comparison.
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
  (genuine / hard-temporal / already-crossed).
  > ✅ **Plumbed 2026-08-26.** `pack_meta` writes the three fields, the read path tensorises them, and
  > `collate_sequences` lifts them into the `labels` dict so they reach the loss through the existing
  > Trainer path. The keys are **additive and optional** — chunks built before S1 still load, and a batch
  > that mixes vintages fails loudly rather than silently dropping them.
  > **Existing LMDBs still need the one-time backfill** (`scripts/backfill_onset_meta.py`, 🖥️ lab PC):
  > a metadata-only pass that never touches an image blob. It matches samples to records positionally and
  > verifies `track_id` + `crosses` per sample before writing, so a mismatched pkl aborts instead of
  > corrupting. Augmented dirs are not backfillable (oversampling breaks the positional map) — backfill
  > the base dir and re-run `augment_dataset.py`, which now carries the keys through.
  > **The M4-dropped windows do NOT come back from the backfill**: `window_track` skips them at
  > generation, so they were never written to the pkls. Recovering them needs a regen with the M4 filter
  > relaxed — a separate experiment, deliberately not bundled with the objective change.

### Dataset Statistics

**v2 contract**, re-pinned 2026-08-19 from the lab-PC regen (`scripts/count_labels.py`):

| Split | N | actions=1 | looks=1 | crosses=1 |
|---|---|---|---|---|
| train | 88214 | 43.4% (38320) | 10.5% (9266) | 2.9% (2530) |
| val | 20490 | 40.3% (8261) | 6.7% (1376) | 2.8% (569) |
| test | 69875 | 42.2% (29489) | 9.8% (6815) | 3.1% (2140) |

Shifts vs the pre-v2 table are **expected and thesis-reportable**, not drift: M3 relabelled
`actions`/`looks` as state-at-end-of-observation (both rates drop — `looks` most, 17.1% → 10.5%) and M4
dropped right-censored windows (N drops ~8%). `crosses` counts are untouched by both, so the rate *rises*
slightly as the denominator shrinks.

> ⚠️ Pinned with `data.emit_determined_positives=false`. The 2026-08 regen turns it on, which **raises N
> and the `crosses` rate** (it recovers confirmed positives only). Re-pin all four sites from that run.

`crosses` is severely imbalanced (~37:1); `looks` moderately (~9:1); `actions` roughly balanced. Aggregate
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

## Onset Timing (the method — single source of truth for its contracts)

Streaming crossing onset as **timing under censoring** rather than yes/no at a fixed horizon
([docs/METHODOLOGY.md](docs/METHODOLOGY.md) prong 2). Implemented 2026-08-26, **default off**; nothing
below changes any existing run until `model.onset_head=true`.

The binary `crosses` label answers "does a crossing start within `H` frames?" and gets two cases wrong by
construction: a window whose future ran out is labelled 0 (fabricated) or dropped (M4), and a window whose
crossing lands at `H+5` is labelled 0 beside a genuine non-crosser. The hazard head asks `K` smaller
questions instead — *given no crossing has started yet, does one start in bin `k`?* — each of which can be
**masked out** when the answer was never observed.

| Piece | Where |
|---|---|
| Bin geometry, four-case target + mask | [data/onset_target.py](src/pedpredict/data/onset_target.py) (`OnsetSpec`, `hazard_targets`, `readout_targets`) |
| Head + horizon readout | [models/heads.py](src/pedpredict/models/heads.py) (`build_onset_hazard_head`, `hazard_to_horizon_logits`) |
| Loss | [losses/onset.py](src/pedpredict/losses/onset.py) (`OnsetHazardLoss`) — added to `MultiTaskLoss`, not a second call site |
| Backfill for pre-S1 chunks | [data/onset_backfill.py](src/pedpredict/data/onset_backfill.py) + `scripts/backfill_onset_meta.py` |
| Composition / horizon reporting | [data/onset_stats.py](src/pedpredict/data/onset_stats.py) |

**Four cases, from the three S1 fields** — the supervision the binary label cannot express:

| Case | Recognised by | Supervision |
|---|---|---|
| event in range | `0 ≤ onset_offset < L` | bins `0..e-1` → 0, bin `e` → 1, **after `e` masked** |
| event beyond range | `onset_offset ≥ L` | all `K` bins → 0 (honest: the crossing *was* observed) |
| censored | `onset_offset < 0`, some future observed | bins covered by `future_observed` → 0, **rest masked** |
| already crossed | `onset_offset < 0` **and** `track_crosses` | **dropped** — not at risk of a *first* crossing |

Row 3 is what the binary label cannot say; row 4 is a bug it cannot avoid (today both land in
`crosses = 0` beside genuine non-crossers).

**Non-negotiables:**
- **`onset_lookahead > onset_horizon`** (validated). A head only as wide as the reported horizon still
  labels a crossing at `H+5` a flat negative — the exact failure the method exists to remove.
- **`onset_horizon == data.future_offset + data.tol`** (validated). The readout is what the four
  `pose_full` baselines are compared against; a mismatch makes it answer a different question under
  identical metric names. `tests/test_onset_target.py::test_readout_label_reproduces_stored_crosses`
  pins the readout label against the generator's own `crosses` over a real track.
- **Unobserved bins get exactly zero gradient**, never a confident zero. That is the whole contribution;
  asserted with autograd in `tests/test_onset_loss.py::test_no_gradient_past_the_event_or_censor_point`.
- **`onset_report_crosses` affects metrics only**, never loss routing — so "auxiliary" and "pure
  reformulation" are two configs of one code path, with only one gradient path onto any head.

**Three formulations, selected by weights alone** (no code branches):

| `loss_weight.crosses` | `onset_hazard_weight` | `onset_readout_weight` | `onset_report_crosses` | Arm |
|---|---|---|---|---|
| 1.2 | ~0.03 | 0 | false | **auxiliary** — reported number comes from the same head as the baselines. Start here. |
| 0 | 1.0 | 0 | true | pure reformulation — the methodological claim |
| 0 | 1.0 | w | true | the hedge — the hazard term never optimises the number actually reported |

⚠️ **Scale.** The hazard term *sums* over a window's observed bins (likelihood-correct), so at `L=60` it
starts ~40× a per-task CE (`~0.69 × 60` at init) and falls as hazards saturate low. Hence ~0.03 in the
auxiliary arm and 1.0 where it *is* the objective. `OnsetLossOutput.hazard` logs the raw unweighted value
so the ratio is observable rather than inferred.

**Known risk, by design:** with `K` bins the per-bin positive rate is ~`2.9%/K`, so the head can collapse
to `h ≈ 0` everywhere and the readout saturates near zero. The lever is `onset_bin_width` (4 quadruples
the positives per bin, costing timing resolution). Check it on a short run before committing GPU hours.

## Evaluation

Report **Accuracy, F1, AUC, Precision, Recall** (per task + macro-F1), logged to CSV. Also report
efficiency: **params, FLOPs (fvcore), latency, FPS, peak VRAM** per `model_type`. A single
`MetricAccumulator` is shared by training-validation and test (no divergence). Degenerate cases use
`zero_division=0`; AUC needs softmax probabilities. Model types: `full`, `ped_local` (legacy
`motion_only`, renamed per A6 — it reads the tight crop), `kinematics_only` (pixel-free baseline, M9.1),
`visual_only`, `vanilla_concat`, `pose_kinematics` / `pose_full` (pose arm, need a pose-enabled build) —
selected via the typed registry, not raw strings. **Crosses-only** (single-task) is not a model type but
the **head-selection mode**, driven by one field: `--set train.active_tasks=[crosses]
train.selection_metric=crosses_f1`. `active_tasks` (default `[actions,looks,crosses]` = full) is the
single source of truth — the Trainer derives `effective_loss_weight()`/`effective_sampler_powers()`
(inactive tasks → 0 in *both* imbalance levers), builds the metric accumulator + CSV/eval columns over
the active set only (no dead `actions_*`/`looks_*`/`macro_*` columns anywhere), and `macro_f1` averages
only active tasks so a single active task collapses it to that task's F1. **This closes a real bug:** with
the old multi-task `macro_f1` selection, dropped heads still emit predictions, and their transient
epoch-1 F1 (before collapse) spiked `macro_f1` to an unbeatable value — best.pth froze on epoch 1 and
early-stop counted from epoch 2. `active_tasks` makes that impossible; always pair crosses-only with
`selection_metric=crosses_f1` (validated: `crosses_f1` requires `crosses ∈ active_tasks`).

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

### Response Style (how to talk to me, not how to write code)

Write so the reply is *navigable* — something to think and decide with, not a display of reading volume.

- **Plain words first.** Say the thing in ordinary language, then name the technical term once if it earns
  its place. Never use a term or abbreviation the reader hasn't been given: spell out field names,
  metrics, and paper acronyms on first use in a reply (`onset_offset` → "how many frames until the person
  starts crossing"). An unexplained acronym is a dead end for the reader, not a shortcut.
- **Grounded.** Claims come from files actually read, numbers actually in the logs, or sources actually
  fetched — and say which. Distinguish "I verified this" from "I believe this" from "this needs checking."
  Flag when a search was shallow rather than implying it was exhaustive.
- **Comprehensive, not padded.** Cover the whole question, including the parts that complicate it. Length
  should come from substance; cut anything that is just demonstrating effort.
- **No highlight reels.** Don't recount how many files were scanned or how much was read. Report what was
  found and what it means. Findings, not receipts.
- **Answer the question asked.** If the ask is "what are my options," give options — not a plan, a
  schedule, or next steps, unless asked. Recommendations are welcome at the end, briefly.
- **Structure for scanning.** Headers and short blocks so the reader can find their way back in. Tables
  only when comparing things along shared axes.

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
| Sequence-gen params / PIE annotations | Dataset Statistics table + `tests/fixtures/golden/pie_sequences_counts.json` + the expected dict in `test_stats.py::test_reference_fixture_matches_claude_table` + re-run `count_labels.py` (gate) — **all four move together or the gate lies** |
| A run's eval numbers | `outputs/runs/RESULTS_MATRIX.md` (prose ledger, hand-maintained) + `rebuild_index` for `index.csv` (machine table) — never put prose in the CSV |
| Output-dict keys / head wiring | Architecture output-keys note + `heads.py`/`ensemble.py` docstrings |
| Imbalance levers (balance / sampler / loss weights) | Imbalance Policy section — all three levers together |
| `d_model` / module dims | Architecture table (CLAUDE.md + README) — never one module alone |
| Add / move / remove a `src/` module or `scripts/` CLI | README layout + README command list (+ the orientation note in CLAUDE.md if a whole subsystem moves) |
| Onset-timing head / loss / targets | Onset Timing section (the four-case table, the three-arm table, the non-negotiables) — all in one place, never split across docs |
| Config schema field / default | `configs/*.yaml` + schema docstrings; Config note if the CLI surface changes |
| New extra / dependency | README Install extras |
| Thesis direction / stage progress | THESIS_ROADMAP (the tracker) — and the Thesis Direction section here **only** if the spine itself moves |

**Doc layout (consolidated 2026-08-19 — keep it this shape).** Six maintained docs: `CLAUDE.md` (agent
orientation + contracts), [README.md](README.md) (repo overview), [setup.md](setup.md) (runbook),
[docs/THESIS_ROADMAP.md](docs/THESIS_ROADMAP.md) (tracker + RQ spokes),
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) (the method),
[outputs/runs/RESULTS_MATRIX.md](outputs/runs/RESULTS_MATRIX.md) (the numbers). Three frozen references
(the streaming brief, [docs/POSE_ENCODER.md](docs/POSE_ENCODER.md),
[docs/BACKBONE_STUDY.md](docs/BACKBONE_STUDY.md)) — each carries a 🧊 status header stating what is
actually built; correct them if they go stale, don't grow them. Everything else is in
[docs/archive/](docs/archive/) behind a ⛔ RETIRED banner. **When a doc's premise is superseded, rewrite or
retire it — do not prepend another banner.** Three stacked banners is what made the pre-consolidation set
unreadable.

# Ground-Up Rebuild Schematic — Pedestrian Behavior Prediction

> Master plan for rebuilding this thesis/grad-research repo from scratch.
> This document is **not** the implementation. It is a sequenced set of **copy-paste prompts**.
> Each prompt, dropped into a fresh agent session, produces a *detailed* sub-plan for one module.
> You execute the sub-plans one at a time.

---

## 0. Locked decisions (these shape every prompt below)

| Decision | Choice |
|---|---|
| **Rebuild mode** | Fresh repo, port code piece-by-piece. Old repo kept read-only as a reference. |
| **Architecture** | **Restructure now, redesign later.** Phase A (this schematic) is *behavior-preserving*: same model math, same outputs, cleaner code. Architectural redesign (backbone swap, fusion rethink, single crosses head) is a deferred Phase B with its own prompts. |
| **Config + tracking** | Minimal: `yaml` config files + `argparse` overrides + `CSV` logging. No Hydra, no W&B. One typed loader that merges yaml → dataclass → CLI. |
| **Prompt depth** | Self-contained deep briefs. Each prompt embeds its own file list + known band-aids, and references the **Shared Context Block** (§4) which you prepend once. |

**Behavior-preserving contract (applies to every Phase-A prompt):** the rebuilt module must produce numerically equivalent outputs (within float tolerance) to the legacy module for the same inputs and weights, *unless* a listed band-aid explicitly changes behavior — and any such change must be called out and justified in that module's sub-plan. Capture a golden-output fixture from the OLD repo *before* porting.

---

## 1. Current-state diagnosis — band-aid inventory

Grounded in the actual files. Each item names where the rebuilt design must do better.

| # | Band-aid / smell | Location | Why it hurts |
|---|---|---|---|
| B1 | **635-line god-script**: class weights, sampler weights, mp prefetch, train loop, validation, metrics, CSV — all inline; hyperparameters hardcoded mid-function. | `train.py` | Nothing reusable; every experiment edits the file. |
| B2 | **Lazy-param materialization hack**: dummy forward pass purely to create ViT global-window `relative_position_bias` before optimizer build. | `train.py:311-317` ← `models/Vision_Transformer.py:104-146` | Optimizer/checkpoint fragility; `strict=False` hides real mismatches. |
| B3 | **Three overlapping imbalance mechanisms**: CE `weight=` + `WeightedRandomSampler` + `loss_weight` dict, plus standalone `class_imbalance_strategies.py` & `imbalance_config.py`. | `train.py:34-123,292-294`; root files | No single source of truth for imbalance handling. |
| B4 | **Dead/ambiguous crosses head**: model emits `crosses_pooled` *and* `crosses_frame`; train+eval silently use only `crosses_frame`. | `models/Cross_Attention_Module.py:64-77`; `train.py:146-149`; `test.py:53-54` | `crosses_pooled` is computed every step but never supervised. |
| B5 | **Fragmented data pipeline** across 6+ scripts with an older dead variant. | `scripts/generate_sequences.py`, `balance_sequences.py`, `augment_sequences.py`, `split_balance_sequences_all.py`, `preprocess_data_lmdb.py`, `PIE_sequence_Dataset_1.py`, `preprocess_data.py`, `pedestrian_detection.py` | No clear DAG; unclear which script is canonical. |
| B6 | **Two divergent config sources**: `config.py` defaults vs the hardcoded args in `Vision_Transformer.py.__main__`. | `config.py:6-28` vs `models/Vision_Transformer.py:389-401` | Configs drift; reproducibility risk. |
| B7 | **Magic constants in collate**: `MAX_SEQ_LEN=20` cap and `[..., :8]` motion slice silently truncate data. | `scripts/train_utils.py:12-20` | Silent data shaping far from where it's defined. |
| B8 | **Scattered AMP `.float()` casts** as a correctness patch for autocast. | `train.py`, `test.py` (multiple) | Hard to reason about dtype contracts. |
| B9 | **Hand-rolled mp prefetch** (`mp_async_load`, `wait_for_memory`, `mp.Queue`, manual process join, timeout terminate). | `train.py:368-504`, `scripts/train_utils.py:74-98` | Brittle; couples data loading to RAM polling. |
| B10 | **Stringly-typed model dispatch**. | `scripts/model_utils.py` | Easy to pass an invalid `model_type`; no schema. |
| B11 | **Repo hygiene**: `venv/` committed, ~30 CSVs in `training_log/`, root-level one-offs (`ablation_usage_example.py`, `final_ablation_verification.py`, `test_*.py`, `visualize_*.py`, `extract_frames.py`). | repo root | Noise; unclear entrypoints; bloated git. |
| B12 | **No real tests / no CI**; the `test_*.py` files are ad-hoc structure checks. | repo root | No safety net for the rebuild itself. |
| B13 | **WindowTransformerBlock variable reuse** obscures the MLP residual path. | `models/Vision_Transformer.py:243-248` | Confusing; review-time foot-gun. |

---

## 2. Target repository layout (fresh repo)

```
pedpredict/                          # new repo root
  pyproject.toml                     # packaging + pinned deps + tool config (ruff/pytest)
  README.md
  .gitignore                         # venv/, data/, outputs/, logs/, *.lmdb, *.pth
  configs/
    paths.yaml                       # all filesystem locations (no hardcoded paths in code)
    data.yaml                        # seq_len, stride, future_offset, tol, crop scales, chunk_size
    model.yaml                       # d_model, ViT stages, motion enc, cross-attn heads
    train.yaml                       # lr, batch, epochs, loss_weight, sampler powers, amp
    eval.yaml                        # metrics, benchmark settings
  src/pedpredict/
    __init__.py
    config/  __init__.py  schema.py  loader.py     # dataclass schema + yaml->dataclass->argparse merge
    paths.py
    utils/   seed.py  device.py  amp.py  memory.py  logging.py
    data/
      pie_sequences.py     # PIE -> list[dict] (from generate_sequences)
      lmdb_writer.py       # sequences -> LMDB chunks (from preprocess_data_lmdb + PIE_sequence_Dataset_1)
      balance.py           # (from balance_sequences + split_balance_sequences_all)
      augment.py           # (from augment_sequences)
      lmdb_dataset.py      # LMDB -> tensors (from scripts/lmdb_dataset)
      transforms.py        # crop logic + imagenet normalize
      collate.py           # collate_fn + sequence-length policy (replaces magic constants)
      sampler.py           # weighted-sampler weight computation
      stats.py             # label counting (from label_count)
    models/
      vit.py  motion_encoder.py  cross_attention.py  ensemble.py  ablations.py
      heads.py             # task heads + the crosses-head decision
      registry.py          # typed model factory + forward dispatch (replaces model_utils)
    losses/
      multitask.py         # unified class-weight + per-task weighting (replaces B3 sprawl)
    training/
      trainer.py  chunk_loader.py  callbacks.py  metrics.py
    eval/
      evaluate.py  benchmark.py     # accuracy/F1/AUC + FLOPs/latency/FPS
      inference.py                  # video inference (from main.py)
    viz/
      plots.py             # quantitative figures (from scripts/plot_results)
      qualitative.py       # frame overlays / attention viz (from visualize_*.py)
    export/
      onnx.py              # (from onnx/onnx_export.py)
  scripts/                           # thin CLI wrappers, one job each
    make_sequences.py  build_lmdb.py  balance_dataset.py  augment_dataset.py
    count_labels.py  train.py  evaluate.py  visualize.py  export_onnx.py
  tests/
    test_config.py  test_data_shapes.py  test_lmdb_roundtrip.py
    test_model_shapes.py  test_losses.py  test_metrics.py  test_golden_outputs.py
```

---

## 3. Recommended execution order (dependency DAG)

```
P0 Foundation ─► P1 Data ─► P2 Models ─► P3 Loss/Metrics ─► P4 Training ─► P5 Eval ─► P6 Viz ─► P7 Export ─► P8 Tests/Docs ─► P9 Cutover
                  │                                  ▲
                  └──────────── golden fixtures ─────┘ (capture from OLD repo before porting each)
```

- **P0 must come first** (config + utils are imported by everything).
- **P2 models can be ported in parallel** once P0 exists (each model file is independent), but cross-attention (2.3) depends on the heads decision.
- **P3 before P4** (trainer consumes loss+metrics modules).
- **P1.6 sampler/weights** and **P3 loss** jointly resolve B3 — plan them aware of each other.
- Capture **golden outputs** for each module from the OLD repo *before* writing the new one.

---

## 4. SHARED CONTEXT BLOCK  (prepend this to every prompt below)

```
SHARED CONTEXT — Pedestrian Behavior Prediction rebuild (Phase A: behavior-preserving restructure)

PROJECT: Multimodal pedestrian behavior prediction on the PIE dataset. From a sequence of
video frames the model jointly predicts three binary tasks: actions (walking/standing),
looks (looking at traffic or not), crosses (will cross soon).

ARCHITECTURE (must be preserved this phase):
  context crop frames -> ViT_Hierarchical ----\
                                                +-> CrossAttentionModule -> EnsembleModel -> {actions, looks, crosses}
  tight crop + motion -> MotionEncoder -------/
  - Unified d_model = 128 across ALL modules (config.get_unified_dim_model()).
  - Output dict keys: actions, looks, crosses_pooled, crosses_frame, temporal_weights.
    Training & eval currently supervise ONLY crosses_frame (logsumexp pooled). crosses_pooled
    is computed but unused (a known band-aid to resolve, not silently keep).
  - temporal_weights is [B, T] softmax from the pooling MLP (full model only).

DATA: LMDB chunks. Each sample: <key>_meta (pickle: motions[T,8], actions, looks, crosses)
  + per-frame <key>_<t>_tight and <key>_<t>_context JPEG blobs. ImageNet normalization.
  crosses raw labels {-1,0,1} are clamped to {0,1}. Class balance (train): actions ~45% pos,
  looks ~17% pos, crosses ~2.6% pos (severe ~37:1 imbalance on crosses).

REBUILD CONSTRAINTS:
  - Fresh repo with the target layout (src/pedpredict package + thin scripts/ CLIs + configs/*.yaml).
  - Config stack is MINIMAL: yaml files -> typed dataclass schema -> argparse overrides. No Hydra/W&B.
    Logging stays CSV. No hardcoded paths in code; everything flows from configs/paths.yaml.
  - BEHAVIOR-PRESERVING: numerically equivalent outputs vs the legacy module for the same inputs
    & weights, within float tolerance, UNLESS a listed band-aid changes behavior — then call it
    out explicitly and justify. Capture a golden-output fixture from the OLD repo before porting.
  - Python/PyTorch, AMP (torch.amp.autocast('cuda')), GradScaler, cudnn.benchmark.

OLD REPO is read-only reference at (vendored into this repo, golden samples in OLD/golden/):
  OLD/Undergrad_thesis_project

DELIVERABLE OF THIS PROMPT: a DETAILED SUB-PLAN, not final production code. The sub-plan must contain:
  (a) exact target file(s) + public API (function/class signatures with type hints),
  (b) a step-by-step port/refactor procedure referencing the old code,
  (c) which band-aids it removes and how,
  (d) the golden-fixture / test list that proves behavior preservation,
  (e) risks + open questions to confirm before coding.
Skeletons and signatures are welcome; full implementations are not (those come after plan approval).
```

---

## 5. The prompts

Conventions: each prompt is one fenced block — paste the **Shared Context Block** first, then the prompt. `[OLD]` paths are relative to the old repo root.

---

### Phase 0 — Foundation

#### Prompt 0.1 — Repo scaffold, packaging, hygiene
```
ROLE: Plan the fresh-repo scaffold and packaging for the rebuild.

SCOPE:
  - Create target layout from the schematic (src/pedpredict package, scripts/, configs/, tests/).
  - pyproject.toml: package metadata, pinned dependencies (derive from [OLD] requirements.txt),
    ruff + pytest config. Decide src-layout install (pip install -e .).
  - .gitignore that excludes what the OLD repo wrongly committed: venv/, data/, *.lmdb,
    model_outputs/, best_model_outputs/, training_log/*.csv, plots/*.png, qualitative_visualize/.
  - Migration policy for OLD root-level one-offs: ablation_usage_example.py,
    final_ablation_verification.py, test_ablation_*.py, test_imbalance_setup.py,
    visualize_*.py, extract_frames.py, run_env.bat — classify each as PORT / FOLD-INTO-TESTS / DROP.

KNOWN ISSUES TO FIX: B11 (venv & logs committed, root clutter), B12 (no CI/tests baseline).

PRODUCE the sub-plan per the Shared Context deliverable spec, plus:
  - a table mapping every OLD top-level file -> {new location | dropped | merged-into}, with reason;
  - the dependency pin list with justification for any version bumps;
  - a minimal GitHub Actions (or local pre-commit) lint+test gate.
```

#### Prompt 0.2 — Config system (yaml → dataclass → argparse)
```
ROLE: Plan the minimal typed config system that replaces [OLD] config.py and all the
hardcoded hyperparameters scattered through [OLD] train.py / test.py.

SOURCE TO ABSORB:
  - [OLD] config.py (get_unified_dim_model, vit_args_config, motion_enc_args_config)
  - hardcoded values in [OLD] train.py:280-296 (lr=1e-4, batch_size=4, num_epochs=30,
    num_workers=4, loss_weight={0.8,0.8,1.2}, sampler_powers={crosses:1.5,actions:0.3,looks:0.7})
  - [OLD] scripts/train_utils.py:12 (MAX_SEQ_LEN=20) and :18 (motion [...,:8] slice)

DESIGN:
  - configs/{paths,data,model,train,eval}.yaml as the source of truth.
  - src/pedpredict/config/schema.py: frozen dataclasses (PathsCfg, DataCfg, ModelCfg, TrainCfg,
    EvalCfg, RootCfg) with full type hints + defaults matching today's effective values.
  - config/loader.py: load yaml -> instantiate dataclasses -> apply argparse dotted overrides
    (e.g. --train.lr 5e-5). Validation (e.g. d_model divisible by heads). Dump resolved config
    to the run's CSV/log dir for reproducibility.

KNOWN ISSUES TO FIX: B1 (hardcoded hyperparams), B6 (config drift vs Vision_Transformer.__main__),
  B7 (magic constants move into DataCfg).

PRODUCE the sub-plan per spec, including the exact dataclass fields with current default values,
the yaml file contents, the override-merge algorithm, and tests in tests/test_config.py.
```

#### Prompt 0.3 — Utilities (seed / device / amp / memory / logging) + paths
```
ROLE: Plan src/pedpredict/utils/* and paths.py — the small shared helpers everything imports.

SOURCE TO ABSORB / CONSOLIDATE:
  - device + perf flags from [OLD] train.py:244-255 (cuda, cudnn.benchmark, tf32, matmul precision)
  - AMP usage + scattered .float() casts (B8) -> a single amp context + a dtype-safe logits helper
  - RAM-pressure polling [OLD] scripts/train_utils.py:74-77 wait_for_memory
  - gc.collect()/empty_cache() cleanup idioms repeated across train.py
  - seeding (currently absent — add deterministic seed util)

DELIVER a utils API:
  seed.py: set_seed(seed, deterministic: bool)
  device.py: get_device(), enable_perf_flags()
  amp.py: autocast_ctx(enabled), to_float_logits(outputs: dict) -> dict
  memory.py: wait_for_memory(threshold, interval), free_cuda()
  logging.py: get_csv_logger(...), structured run-dir creation
  paths.py: resolve all locations from PathsCfg (no os.path.join literals in module code)

KNOWN ISSUES TO FIX: B8 (centralize dtype handling), part of B1/B9 (extract memory polling).

PRODUCE the sub-plan per spec with each helper's signature, the run-directory convention
(run_id = timestamp + model_type + tag), and tests/test_utils minimal coverage.
```

---

### Phase 1 — Data layer (preprocessing split into focused sections)

#### Prompt 1.1 — PIE → sequence generation
```
ROLE: Plan src/pedpredict/data/pie_sequences.py, porting [OLD] scripts/generate_sequences.py.

SOURCE: [OLD] scripts/generate_sequences.py (generate_sequences: PIE.generate_data_trajectory_sequence
  -> sliding windows seq_len=20 stride=3 future_offset=30 tol=2; filter #2 drops windows with any
  crossing during observation; labels via any(...) over future window; clamp_to_binary; has_onset unused).
  Reads PIE via [OLD] PIE/utilities/pie_data.py.

REQUIREMENTS:
  - All window params come from DataCfg (no literals). Keep the EXACT current filtering/labeling
    logic (this drives the documented dataset statistics — preserve them).
  - Pure function: imdb + DataCfg -> list[dict]; pickle I/O isolated from logic.
  - Note has_onset() is dead code — decide keep (for future onset-based labeling) or drop.
  - Re-derive and verify the dataset stat table (train 95,684 / val 22,665 / test 76,048;
    crosses pos-rate ~2.6/2.5/2.8%). If new code changes counts, that is a behavior break — flag it.

KNOWN ISSUES TO FIX: B5 (canonicalize the first pipeline stage).

PRODUCE the sub-plan per spec, including a verification step that regenerates one split and
diffs label counts against the documented table, and tests/test_data_shapes coverage for window math.
```

#### Prompt 1.2 — Crop/motion extraction + LMDB writer
```
ROLE: Plan src/pedpredict/data/lmdb_writer.py + transforms.py, porting [OLD]
  scripts/preprocess_data_lmdb.py and [OLD] scripts/PIE_sequence_Dataset_1.py.

SOURCE:
  - [OLD] scripts/preprocess_data_lmdb.py (save_dataset_in_chunks_lmdb: chunked LMDB writer,
    JPEG-encodes tight+context crops, computes map_size, num_workers DataLoader, context_scale=2.0,
    jpeg_quality=90, chunk_size=5000).
  - [OLD] scripts/PIE_sequence_Dataset_1.py (PIESequenceDataset, load_sequences_from_pkl: does the
    actual cropping — tight bbox crop + context crop at context_scale — and motion-feature construction).
  - Output schema consumed later by lmdb_dataset.py: <key>_meta pickle{motions[T,8],actions,looks,
    crosses} + <key>_<t>_tight / _context JPEG blobs.

REQUIREMENTS:
  - Separate concerns: transforms.py owns crop geometry + ImageNet normalize; lmdb_writer.py owns
    serialization/chunking only.
  - Pin down the exact 8-dim motion feature definition (document each channel) — it is currently
    implicit; make it explicit and tested. Note collate later slices motions[...,:8] (B7); confirm
    writer emits exactly 8 so the slice becomes a no-op we can delete.
  - map_size heuristic ([OLD] line 52-54) -> documented, configurable.

KNOWN ISSUES TO FIX: B5, and the upstream half of B7 (motion dimensionality).

PRODUCE the sub-plan per spec, the LMDB key/value schema as a written contract, a
tests/test_lmdb_roundtrip plan (write a tiny chunk, read it back, assert tensor shapes/dtypes), and
the motion-channel documentation table.
```

#### Prompt 1.3 — Balancing & splitting
```
ROLE: Plan src/pedpredict/data/balance.py, consolidating [OLD] scripts/balance_sequences.py and
  [OLD] scripts/split_balance_sequences_all.py.

SOURCE:
  - [OLD] scripts/balance_sequences.py (groups by (actions,looks,crosses); a constraint solver
    _solve_cross0_counts that target-balances cross=0 group to match cross=1 counts).
  - [OLD] scripts/split_balance_sequences_all.py (split + balance across all splits).

REQUIREMENTS:
  - Clarify how balancing INTERACTS with the runtime WeightedRandomSampler + loss class weights
    (B3). Three imbalance levers exist; the rebuilt design should make explicit which lever runs at
    which stage (offline balance vs online sampler vs loss weight) and default to ONE coherent policy.
    Document the recommended default; keep the others as opt-in config flags.
  - Deterministic given a seed; pure functions; pickle I/O isolated.

KNOWN ISSUES TO FIX: B3 (offline portion), B5.

PRODUCE the sub-plan per spec, including a written decision on the imbalance-handling policy
(coordinated with Prompt 1.6 and Prompt 3.1) and tests covering the count-solver invariants.
```

#### Prompt 1.4 — Sequence augmentation
```
ROLE: Plan src/pedpredict/data/augment.py, porting [OLD] scripts/augment_sequences.py.

SOURCE: [OLD] scripts/augment_sequences.py (SequenceAugmenter: horizontal_flip — note it negates
  motions[:,2]; color_jitter; motion_noise; random_erase). Targets minority classes (crosses=1, looks=1).

REQUIREMENTS:
  - Preserve label-validity guarantees of each transform. CRITICAL: horizontal_flip flips images AND
    negates a motion channel (index 2) — verify this index matches the motion-channel definition from
    Prompt 1.2; a mismatch silently corrupts augmented data.
  - All probabilities/params from DataCfg. Make augmentation composable and seedable.
  - Decide: augment offline (produces preprocessed_train_aug LMDB, as today) vs online in Dataset.
    Keep offline to preserve current behavior; note online as a redesign-phase option.

KNOWN ISSUES TO FIX: B5; hidden coupling between flip and motion-channel semantics.

PRODUCE the sub-plan per spec, an explicit per-transform "label/feature-invariance" justification
table, and tests asserting flip(flip(x)) == x and motion-channel sign handling.
```

#### Prompt 1.5 — Runtime dataset + collate + transforms
```
ROLE: Plan src/pedpredict/data/lmdb_dataset.py + collate.py, porting [OLD] scripts/lmdb_dataset.py
  and the collate from [OLD] scripts/train_utils.py.

SOURCE:
  - [OLD] scripts/lmdb_dataset.py (LMDBChunkDataset: per-process env via _get_env keyed on pid,
    __getstate__ drops env for pickling, decodes JPEG per frame, raises on frame-count mismatch).
  - [OLD] scripts/train_utils.py:15-20 collate_fn (stacks tight/context/motions, MAX_SEQ_LEN=20 cap,
    motions[...,:8] slice, builds labels dict).

REQUIREMENTS:
  - Keep the worker-safe LMDB env handling (multiprocessing correctness) — port carefully, test under
    num_workers>0.
  - Replace magic constants (B7): MAX_SEQ_LEN -> DataCfg.max_seq_len; the [...,:8] slice should become
    unnecessary once the writer (1.2) emits exactly 8 dims — confirm and remove, or keep with a guard.
  - Sequence-length policy made explicit (truncate vs pad vs variable) and documented.

KNOWN ISSUES TO FIX: B7.

PRODUCE the sub-plan per spec, the Dataset/collate signatures, the worker-safety test plan, and a
note on how this couples to Prompt 4.2 (chunk loader).
```

#### Prompt 1.6 — Sampler + class-weight computation (unify imbalance levers)
```
ROLE: Plan src/pedpredict/data/sampler.py — the online imbalance lever — porting and UNIFYING
  the duplicated weight code in [OLD] train.py.

SOURCE (note the duplication):
  - [OLD] train.py:34-72 compute_class_weights_from_lmdb (inverse-freq class weights for CE loss)
  - [OLD] train.py:74-123 build_sampler_weights + _inverse_class_weights (per-sample WeightedRandomSampler
    weights with per-task powers crosses^1.5 * actions^0.3 * looks^0.7)
  These two scan the SAME LMDB metadata for overlapping purposes.

REQUIREMENTS:
  - Single metadata-scan pass that produces BOTH: per-task class frequencies (for loss weights, used by
    Prompt 3.1) and per-sample sampler weights. Cache results per chunk.
  - sampler powers + min_weight come from TrainCfg.
  - Coordinate with Prompt 1.3 (offline balance) and 3.1 (loss weights) so the three levers are
    documented as one policy, not three accidents (B3).

KNOWN ISSUES TO FIX: B3 (online portion + dedup of the two scan functions).

PRODUCE the sub-plan per spec, the unified weights API, and tests asserting weight monotonicity
(rarer class -> higher weight) on a synthetic label distribution.
```

#### Prompt 1.7 — Dataset statistics / label counting
```
ROLE: Plan src/pedpredict/data/stats.py + scripts/count_labels.py, porting [OLD] label_count.py.

SOURCE: [OLD] label_count.py (scans LMDB metadata, writes training_log/label_count.csv).

REQUIREMENTS:
  - Reuse the single metadata scan from Prompt 1.6 (don't add a third scanner).
  - Output the canonical stats table (per split: N, actions/looks/crosses pos-rate) as CSV + printed
    table, in the exact form documented in CLAUDE.md so it can be diffed for drift.
  - This is the verification tool the whole data layer is checked against.

PRODUCE the sub-plan per spec and a check that compares fresh counts to the documented table,
exiting nonzero on drift (CI-friendly).
```

---

### Phase 2 — Model architecture (one prompt per component)

#### Prompt 2.1 — ViT_Hierarchical (and the lazy-param band-aid)
```
ROLE: Plan src/pedpredict/models/vit.py, porting [OLD] models/Vision_Transformer.py and RESOLVING
  the dynamic relative-position-bias band-aid that forces the dummy-forward hack in training.

SOURCE: [OLD] models/Vision_Transformer.py
  - MLP, window_partition/window_reverse, WindowAttention (relative_position_bias_table built eagerly
    for fixed windows; for window_size=None it defers and rebuilds via init_relative_position_bias,
    mutating _buffers — lines 104-146), WindowTransformerBlock (forwarder init_relative_position_bias;
    note the confusing MLP residual at lines 243-248 — B13), ViT_Hierarchical (stem conv7x7 s4,
    per-stage downsample s2, global-avg-pool, frame_proj to d_model). Outputs [B,T,d_model].

CRITICAL — B2: the global-window blocks lazily create relative_position_bias_table on first forward,
  which is why [OLD] train.py:311-317 runs a dummy forward before building the optimizer/loading
  checkpoint. Plan a fix so ALL parameters exist at __init__ (e.g. compute global window size from a
  configured input resolution / feature-map size, or precompute bias tables up front). This REMOVES
  the dummy-forward hack and lets strict=True checkpoint loading work.

ALSO: clean WindowTransformerBlock.forward residual (B13) without changing the math; reconcile config
  defaults with the old __main__ (B6) — the __main__ becomes a smoke test using ModelCfg.

BEHAVIOR: outputs must match legacy ViT for identical weights+input within tolerance. Capture a
  golden [B,T,d_model] fixture from the OLD model first.

PRODUCE the sub-plan per spec, the eager-parameter design (with the exact shape derivation for global
  windows), a golden-output test, and a strict-load test proving B2 is gone.
```

#### Prompt 2.2 — MotionEncoder
```
ROLE: Plan src/pedpredict/models/motion_encoder.py, porting [OLD] models/Motion_Encoder.py.

SOURCE: [OLD] models/Motion_Encoder.py (img_encoder CNN over tight crops -> [B,T,hidden];
  motion_encoder Conv1d stack over motion[B,T,8]; per-sequence motion normalization at line 82;
  fusion Linear+LN; GRU(num_layers); learned pos_encoding[1,200,hidden]; MultiheadAttention; residual
  proj to d_model). Outputs [B,T,d_model].

REQUIREMENTS:
  - Preserve the math exactly (incl. the in-forward motion normalization and the pos_encoding[:, :T]
    slice — verify T<=200 invariant; surface a clear error otherwise).
  - All dims from ModelCfg (motion_dim=8, hidden_dim, num_layers, num_heads, dropout). Confirm motion_dim
    is consistent with the data-layer motion-channel definition (Prompts 1.2/1.4).
  - Keep the __main__ smoke test as a real shape test.

BEHAVIOR: golden-output preserving.

PRODUCE the sub-plan per spec, signatures, the T<=200 guard design, and a golden-output test.
```

#### Prompt 2.3 — CrossAttentionModule + task heads + the crosses-head decision
```
ROLE: Plan src/pedpredict/models/cross_attention.py + heads.py, porting [OLD]
  models/Cross_Attention_Module.py and RESOLVING the dual crosses-head ambiguity (B4).

SOURCE: [OLD] models/Cross_Attention_Module.py (MultiheadAttention query=motion key/value=image;
  pool_mlp -> softmax temporal weights -> pooled [B,D]; per-task classifier ModuleDict; emits
  actions, looks, crosses_pooled (from pooled), crosses_frame (separate crosses_frame_head over
  per-frame attn_output, pooled by logsumexp/max/mean), temporal_weights).

CRITICAL — B4: crosses_pooled is produced but NEVER supervised (train+eval use crosses_frame only).
  For a behavior-preserving phase, you must KEEP crosses_frame as the supervised output, but make the
  status explicit: either (a) keep crosses_pooled as an auxiliary diagnostic clearly marked unused, or
  (b) gate it behind a config flag default-off. Do NOT silently keep dead compute. Document the chosen
  output contract; it feeds the eval/training code exactly as today.

REQUIREMENTS:
  - Factor task heads into heads.py (classifier MLPs + crosses_frame_head + pooling-MLP) so the output
    contract is testable in isolation. frame_pool in {logsumexp,max,mean} stays configurable
    (default logsumexp).
  - Output dict keys unchanged: actions, looks, crosses_pooled, crosses_frame, temporal_weights.

BEHAVIOR: golden-output preserving for all emitted keys.

PRODUCE the sub-plan per spec, the explicit output-contract table (key -> shape -> supervised?),
the crosses_pooled decision with rationale, and a golden-output test over every key.
```

#### Prompt 2.4 — EnsembleModel + typed model registry/factory
```
ROLE: Plan src/pedpredict/models/ensemble.py + registry.py, porting [OLD] models/Unified_Module.py
  and replacing [OLD] scripts/model_utils.py (stringly-typed dispatch, B10).

SOURCE:
  - [OLD] models/Unified_Module.py (EnsembleModel: vit -> image_norm; motion_enc -> motion_norm;
    cross_attention fusion; return_feats option).
  - [OLD] scripts/model_utils.py (get_model + model_forward dispatch by model_type string).

REQUIREMENTS:
  - registry.py: an enum/Literal ModelType + a typed factory build_model(cfg) -> nn.Module and a
    single forward adapter forward_model(model, batch) -> dict that hides the per-type input signature
    differences (full/vanilla take tight+context+motion; motion_only takes motion+tight; visual_only
    takes context). Replace string typos-as-bugs (B10) with validated types.
  - Preserve EnsembleModel LayerNorm-before-fusion behavior and the return_feats path (used by viz).

BEHAVIOR: golden-output preserving; factory must reproduce the exact module wiring of the old
  get_model for every model_type.

PRODUCE the sub-plan per spec, the ModelType + factory signatures, the unified forward-adapter design,
and tests that every model_type builds and runs a forward on a dummy batch.
```

#### Prompt 2.5 — Ablation models
```
ROLE: Plan src/pedpredict/models/ablations.py, porting [OLD] models/AblationModels.py
  (MotionOnlyModel, VisualOnlyModel, VanillaConcatModel).

SOURCE: [OLD] models/AblationModels.py (same output-dict format; VanillaConcatModel concatenates
  branches instead of cross-attention; some implement crosses_frame, some don't).

REQUIREMENTS:
  - Conform each to the output contract defined in Prompt 2.3 (document which keys each ablation emits;
    crosses_frame presence differs — make it explicit so eval/train branch correctly).
  - Wire through the registry (Prompt 2.4). Preserve math.
  - Fold the OLD root-level ablation scripts (ablation_usage_example.py, final_ablation_verification.py,
    test_ablation_*.py) into proper tests/ rather than top-level scripts (B11).

BEHAVIOR: golden-output preserving per ablation.

PRODUCE the sub-plan per spec, a per-ablation output-key table, and a consolidated
tests/test_model_shapes covering all four model types.
```

---

### Phase 3 — Losses & metrics

#### Prompt 3.1 — Unified multitask loss
```
ROLE: Plan src/pedpredict/losses/multitask.py — the single place imbalance is applied in the loss,
  consolidating the loss logic currently inline in [OLD] train.py.

SOURCE:
  - [OLD] train.py:341-345 criterion = per-task CrossEntropyLoss(weight=class_weights[task])
  - [OLD] train.py:144-153 the per-head loss accumulation: crosses uses outputs["crosses_frame"];
    actions/looks use their keys; weighted by loss_weight={actions:0.8,looks:0.8,crosses:1.2}.
  - class weights come from Prompt 1.6's unified scan (inverse-freq).

REQUIREMENTS:
  - One MultiTaskLoss(nn.Module): takes outputs dict + labels dict, applies per-task CE with
    class weights AND per-task scalar loss_weight, returns total + per-task breakdown (for logging).
  - Encodes the crosses->crosses_frame routing as the explicit output contract (Prompt 2.3), not a
    magic if-branch buried in the loop.
  - Coordinate the imbalance policy with Prompts 1.3 + 1.6 (offline balance / sampler / loss weight):
    document the default combination and which are active.

KNOWN ISSUES TO FIX: B3 (loss portion), part of B1.

PRODUCE the sub-plan per spec, the MultiTaskLoss API, the imbalance-policy summary table, and
tests/test_losses (known-input loss values, weight effect, reduction correctness).
```

#### Prompt 3.2 — Metrics module
```
ROLE: Plan src/pedpredict/training/metrics.py, consolidating the metric code duplicated across
  [OLD] train.py (validation) and [OLD] test.py (evaluate).

SOURCE:
  - [OLD] train.py:177-234 validate_one_epoch (accuracy via argmax, accumulates preds/targets,
    sum-of-per-sample loss) and :578-601 F1/precision/recall/macro-F1 computation.
  - [OLD] test.py:31-110 evaluate (accuracy, F1, AUC via softmax probs, precision, recall,
    binary vs macro avg, temporal-weight collection).

REQUIREMENTS:
  - A MetricAccumulator that ingests per-batch (logits, targets) per task and yields the required
    metric set: Accuracy, F1, AUC, Precision, Recall (per task + macro-F1). Single implementation used
    by BOTH training-validation and test (no divergence).
  - Handle the crosses->crosses_frame routing once. Handle degenerate cases (single-class targets ->
    zero_division=0) and AUC needing probabilities.
  - Keep the CSV column schema compatible with the OLD logs where reasonable, or define a clean new
    schema and a one-time migration note.

KNOWN ISSUES TO FIX: B1 (metric duplication between train.py and test.py).

PRODUCE the sub-plan per spec, the accumulator API, the metric/CSV schema, and tests with a tiny
hand-checkable confusion matrix.
```

---

### Phase 4 — Training

#### Prompt 4.1 — Trainer core
```
ROLE: Plan src/pedpredict/training/trainer.py — the clean training loop replacing the [OLD] train.py
  god-script (B1).

SOURCE: [OLD] train.py:125-175 train_one_chunk, :236-632 main (epoch loop, per-chunk dataloader build,
  AMP+GradScaler, grad-clip max_norm=1.0, ReduceLROnPlateau, best/last checkpointing, CSV logging,
  EarlyStopping).

REQUIREMENTS:
  - A Trainer class consuming: model (from registry 2.4), MultiTaskLoss (3.1), MetricAccumulator (3.2),
    chunk loader (4.2), callbacks (4.3), TrainCfg. Methods: fit(), train_chunk(), validate().
  - AMP + GradScaler + grad-clip preserved. Optimizer/scheduler built from config. No hardcoded
    hyperparameters (all from TrainCfg).
  - Depends on B2 being fixed (Prompt 2.1) so NO dummy-forward materialization is needed and the
    optimizer can be built normally over all params.
  - Keep model_type as a config field routed through the registry's forward adapter.

KNOWN ISSUES TO FIX: B1, B2 (consumer side), B8 (use utils amp/dtype helpers).

PRODUCE the sub-plan per spec, the Trainer API, the epoch/chunk control flow, the
checkpoint-without-dummy-forward verification, and a fast end-to-end smoke test on 1 tiny chunk.
```

#### Prompt 4.2 — Chunk prefetch loader
```
ROLE: Plan src/pedpredict/training/chunk_loader.py, replacing the hand-rolled multiprocessing
  prefetch in [OLD] train.py (B9).

SOURCE: [OLD] train.py:368-504 (mp.Queue(maxsize=3), preload window of 3, mp_async_load warms LMDB,
  wait_for_memory RAM polling, manual process join/terminate-on-timeout, results dict bookkeeping) +
  [OLD] scripts/train_utils.py:80-98 mp_async_load.

REQUIREMENTS:
  - An iterator that yields ready LMDBChunkDataset DataLoaders in shuffled order while prefetching the
    next N chunks, bounded by RAM threshold (utils.memory). Encapsulate ALL the queue/process
    bookkeeping behind a clean ChunkPrefetcher API; the Trainer just iterates.
  - Preserve current effective behavior (preload depth, timeout handling, per-epoch reshuffle) but make
    it testable and crash-safe (no leaked processes).
  - Decide: keep custom prefetch vs lean on torch DataLoader workers. Recommend, but default to
    preserving behavior this phase.

KNOWN ISSUES TO FIX: B9.

PRODUCE the sub-plan per spec, the ChunkPrefetcher API + lifecycle (start/next/close/__exit__),
a process-leak test, and a note on coupling to the Dataset (1.5).
```

#### Prompt 4.3 — Checkpointing, resume, callbacks
```
ROLE: Plan src/pedpredict/training/callbacks.py — checkpointing, early stopping, LR scheduling,
  consolidating logic scattered in [OLD] train.py.

SOURCE: [OLD] train.py best/last/final torch.save calls (model-type suffix naming), checkpoint LOAD
  at :319-328 (strict=False with missing/unexpected print — a B2 symptom), EarlyStopping
  ([OLD] scripts/train_utils.py:23-37), ReduceLROnPlateau (:348-350), scheduler.step(val_loss).

REQUIREMENTS:
  - CheckpointManager: save best/last with run-id naming, save FULL training state (model, optimizer,
    scaler, scheduler, epoch, best metric) for true resume — not just model.state_dict(). Load with
    strict=True now that B2 is fixed (Prompt 2.1); strict=False becomes an explicit opt-in.
  - EarlyStopping + LR scheduler as callbacks driven by val metric from TrainCfg.
  - Checkpoint directory + retention policy via PathsCfg (stop committing weights — B11).

KNOWN ISSUES TO FIX: B2 (load side), B11 (artifact hygiene), B1.

PRODUCE the sub-plan per spec, the checkpoint payload schema, the resume procedure, and a
save->resume->continue equivalence test.
```

#### Prompt 4.4 — Two-phase training strategy
```
ROLE: Plan how to express [OLD] train_two_phase.py in the new Trainer abstraction.

SOURCE: [OLD] train_two_phase.py (balanced-subset warmup -> full fine-tune -> decouple classifiers).

REQUIREMENTS:
  - Reframe as a configurable training schedule / phase list on top of Trainer (4.1), not a second
    god-script. Each phase = (data source, frozen/unfrozen params, lr, epochs). Driven by train.yaml.
  - Identify what "decouple classifiers" does to parameter groups and express it as a phase transition
    (param-group freezing/reinit) the Trainer can apply.
  - Preserve the documented two-phase behavior.

KNOWN ISSUES TO FIX: B1 (the second god-script).

PRODUCE the sub-plan per spec, the phase-schedule config schema, the param-group transition design,
and a smoke test running 1 epoch per phase on a tiny chunk.
```

#### Prompt 4.5 — CSV logging & experiment-tracking conventions
```
ROLE: Plan src/pedpredict/utils/logging.py CSV conventions + run-dir layout (the minimal tracking
  stack), replacing the ad-hoc CSVs in [OLD] training_log/.

SOURCE: [OLD] train.py CSV writer (:261-275 header, :604-618 rows), [OLD] test.py test_log CSV,
  the ~30 uncommitted-worthy CSVs in [OLD] training_log/.

REQUIREMENTS:
  - One run-dir per run: configs snapshot (resolved yaml), train CSV, val/test CSV, checkpoints ptr,
    plots. run_id = timestamp + model_type + tag (from TrainCfg).
  - Stable CSV schemas shared with the metrics module (3.2); document columns once.
  - Define how runs are compared (a small index.csv across runs) — minimal, CSV-only, no W&B.
  - Existing experiment-tracking SKILL conventions in the OLD repo (.claude/skills/experiment-tracking)
    should inform the schema — read and align.

KNOWN ISSUES TO FIX: B11 (log hygiene), B1.

PRODUCE the sub-plan per spec, the run-dir spec, the CSV schemas, and the cross-run index design.
```

---

### Phase 5 — Evaluation & inference

#### Prompt 5.1 — Evaluation pipeline
```
ROLE: Plan src/pedpredict/eval/evaluate.py + scripts/evaluate.py, porting [OLD] test.py.

SOURCE: [OLD] test.py (evaluate() metrics over LMDB test chunks; supports --model_type
  full/motion_only/visual_only/vanilla_concat; uses crosses_frame; collects temporal_weights;
  writes test_log CSV).

REQUIREMENTS:
  - Reuse MetricAccumulator (3.2), registry+forward adapter (2.4), Dataset (1.5), config (0.2).
    Strip duplicated metric/dataloader code.
  - Per-model-type evaluation via config; export per-sample predictions optionally (for viz 6.2).
  - Save temporal_weights for the attention visualization phase.

KNOWN ISSUES TO FIX: B1 (eval duplication), B10 (typed model selection).

PRODUCE the sub-plan per spec, the evaluate() API, the output artifacts (metrics CSV + optional
preds NPZ), and a smoke test on a tiny chunk.
```

#### Prompt 5.2 — Benchmarking (FLOPs / latency / FPS)
```
ROLE: Plan src/pedpredict/eval/benchmark.py — the efficiency metrics required by the thesis.

SOURCE: [OLD] models/Vision_Transformer.py __main__ uses fvcore FlopCountAnalysis; CLAUDE.md requires
  reporting FLOPs, latency, FPS. (No consolidated benchmark module exists today.)

REQUIREMENTS:
  - Measure params, FLOPs (fvcore), latency (warmup + timed runs, CUDA sync), FPS, peak VRAM — per
    model_type, at the configured input shapes. Output a CSV row per model.
  - Deterministic, documented methodology (batch size, seq len, warmup count) from EvalCfg.

PRODUCE the sub-plan per spec, the benchmark API, the methodology doc, and the output schema.
```

#### Prompt 5.3 — Video inference
```
ROLE: Plan src/pedpredict/eval/inference.py + scripts/... , porting [OLD] main.py.

SOURCE: [OLD] main.py (inference on a video: frame extraction, pedestrian crop, motion build, forward,
  output). Related: [OLD] extract_frames.py, [OLD] scripts/pedestrian_detection.py.

REQUIREMENTS:
  - Reuse the data-layer crop/motion/transform code (Phase 1) rather than re-implementing preprocessing.
  - Clear separation: detection/tracking -> sequence assembly -> model forward -> overlay output.
  - Decide whether pedestrian_detection.py is in-scope (porting) or an external dependency.

PRODUCE the sub-plan per spec, the inference pipeline stages, the reuse map to Phase-1 modules,
and a manual smoke-test procedure on the existing qualitative_visualize frames.
```

---

### Phase 6 — Visualization

#### Prompt 6.1 — Quantitative plots
```
ROLE: Plan src/pedpredict/viz/plots.py + scripts/visualize.py, porting [OLD] scripts/plot_results.py.

SOURCE: [OLD] scripts/plot_results.py (reads CSV/NPZ artifacts, writes PNGs: Phase 1-4 = loss curves,
  PR curves, ablation bars, temporal attention). Existing outputs: plots/loss_curves.png,
  plots/per_head_f1_curves.png.

REQUIREMENTS:
  - Consume the NEW run-dir CSV schemas (4.5) + eval artifacts (5.1). Each figure = a pure function
    (data in -> fig out) so figures are testable/regenerable.
  - Config-driven input/output paths (PathsCfg); no hardcoded filenames.
  - Keep the 4 figure families; note any schema changes needed from 3.2/4.5.

PRODUCE the sub-plan per spec, the per-figure function list + inputs, and a regeneration smoke test
on sample CSVs.
```

#### Prompt 6.2 — Qualitative visualization (frame overlays / attention)
```
ROLE: Plan src/pedpredict/viz/qualitative.py, consolidating [OLD] visualize_comparison.py,
  [OLD] visualize_gt.py, and the temporal-attention overlay use of temporal_weights.

SOURCE: [OLD] visualize_comparison.py, [OLD] visualize_gt.py (root-level one-offs, B11),
  qualitative_visualize/ frames, temporal_weights from the model (per-frame attention).

REQUIREMENTS:
  - Overlay predictions vs ground truth on frame sequences; visualize temporal_weights as per-frame
    importance. Reuse eval preds (5.1) and dataset (1.5).
  - Promote the two root scripts into one clean module + scripts/visualize.py subcommand (B11).

PRODUCE the sub-plan per spec, the overlay API, the inputs from eval artifacts, and a sample-output
procedure.
```

---

### Phase 7 — Export

#### Prompt 7.1 — ONNX export
```
ROLE: Plan src/pedpredict/export/onnx.py + scripts/export_onnx.py, porting [OLD] onnx/onnx_export.py.

SOURCE: [OLD] onnx/onnx_export.py.

REQUIREMENTS:
  - Export each model_type via the registry (2.4). Handle dynamic axes (batch, seq len). CRITICAL:
    verify the B2 fix (Prompt 2.1) makes export clean — no lazy params created mid-trace; all params
    exist at __init__ so tracing is deterministic.
  - Validate exported model outputs vs PyTorch (onnxruntime parity check) within tolerance.

KNOWN ISSUES TO FIX: B2 (export benefits directly).

PRODUCE the sub-plan per spec, the export API, dynamic-axis spec, and the parity-check test.
```

---

### Phase 8 — Tests & docs

#### Prompt 8.1 — Test suite & CI
```
ROLE: Plan the tests/ suite + CI gate that protects the rebuild, absorbing the OLD ad-hoc
  test_*.py scripts.

SOURCE: [OLD] test_ablation_models.py, test_ablation_structure_clean.py, test_imbalance_setup.py,
  final_ablation_verification.py (structure checks, not real tests, B12).

REQUIREMENTS:
  - Layered tests: config, data shapes, LMDB roundtrip, model shapes (all 4 types), golden outputs
    (every ported module vs captured legacy fixtures), losses, metrics, sampler weights, checkpoint
    resume, ONNX parity. Plus fast smoke tests (1 tiny chunk) for trainer/eval.
  - Golden fixtures captured from the OLD repo (the behavior-preserving safety net).
  - CI runs lint (ruff) + pytest on push.

KNOWN ISSUES TO FIX: B12.

PRODUCE the sub-plan per spec, the full test inventory mapped to the modules they guard, the
golden-fixture capture procedure, and the CI config.
```

#### Prompt 8.2 — Docs (CLAUDE.md / README) regeneration
```
ROLE: Plan the rebuilt-repo CLAUDE.md + README + module docstrings.

SOURCE: [OLD] CLAUDE.md (architecture table, config, dataset-statistics table, output-dict keys,
  commands, shared-utility map), [OLD] README.md, [OLD] GUIDELINE.md.

REQUIREMENTS:
  - New CLAUDE.md reflecting the target layout, the resolved band-aids (esp. B2, B3, B4), the new
    config/CLI surface, and the SAME dataset-statistics table (kept in sync per the existing rule).
  - Document the imbalance policy decided across 1.3/1.6/3.1 in ONE place.
  - Commands section pointing at the new scripts/ CLIs.

PRODUCE the sub-plan per spec, the CLAUDE.md outline, and the doc-sync checklist (what to update when
data/model/config change).
```

---

### Phase 9 — Cutover & cleanup

#### Prompt 9.1 — Migration cutover & legacy retirement
```
ROLE: Plan the final cutover from OLD repo to the new repo and the retirement of legacy artifacts.

SCOPE:
  - Parity gate: new repo reproduces OLD metrics on the test set per model_type (within tolerance),
    using ported weights, before legacy is retired.
  - Data migration: confirm LMDB chunks are reused as-is (schema unchanged) or document a re-gen.
  - Retire: remove venv/ from history (git filter / fresh repo means it's simply excluded), drop dead
    scripts (preprocess_data.py, has_onset, crosses_pooled if dropped), archive OLD training_log CSVs.
  - Tag a v1.0 "clean baseline" release; write the Phase-B (architecture redesign) backlog seeded from
    deferred items.

PRODUCE the sub-plan per spec, the parity checklist, the legacy-retirement list with reasons, and the
seed backlog for the future architecture-redesign phase.
```

---

## 6. How to use this document

1. Start at **Prompt 0.1**. For each prompt: open a fresh agent session, paste the **Shared Context Block (§4)**, then the prompt. Review the sub-plan it returns, adjust, approve, then implement.
2. Respect the DAG (§3): finish P0 before P1, etc. Within a phase, independent prompts (e.g. the model files 2.1/2.2/2.5) can run in parallel.
3. Before porting any module, **capture its golden output** from the OLD repo — that fixture is what makes "behavior-preserving" verifiable.
4. The three imbalance prompts (**1.3, 1.6, 3.1**) and the output-contract prompts (**2.3, 2.4, 2.5, 3.1, 3.2, 5.1**) are coupled — when you run one, mention the decisions made in its siblings so the policy/contract stays singular.
5. Keep a running `MIGRATION.md` in the new repo recording, per module: golden fixture path, band-aids resolved, parity result.

---

*Deferred to Phase B (architecture redesign, out of scope here): ViT backbone replacement, fusion
redesign, single unified crosses head, online vs offline augmentation, replacing custom chunk prefetch
with standard DataLoader sharding. Each will get its own schematic when Phase A parity is locked.*
```

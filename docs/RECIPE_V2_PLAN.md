# Recipe v2 — truly frozen backbone, cached features, 3 seeds per arm

**Status: approved 2026-09-17 (decisions below); Phases 0–3 done, Phase 4 RUNNING on the research PC since
18:05 UTC.** Phase 0 ✅ · Phase 1 ✅ (all three checks ran; both scripts reproduced R1's stored eval rows
first) · Phase 2 ✅ (`model.vit_frozen_eval`, `model.train_frame_proj`, both default off) · Phase 3 ✅ (cache
module + read path + builder with a real-data parity check) · Phase 4 ▶ tmux `queue2` → `queue_v2.sh`:
cache build → 5 arms × 3 seeds, automatic gates G_parity / G_smoke / G_head_r3 / G_head_r4 in place of the
manual Checkpoints 3–4. Laptop gate green (687 → 713 tests on the research PC); 33 files deployed, md5-verified.

### Phase 4 incident — G_parity tripped twice on its own check (2026-09-17, fixed 20:56 UTC)

**The caches were correct both times; the check was not.** Verified at the only place that matters: the
build path's inputs and the image path's inputs are **bit-identical** (`max|diff| = 0.000e+00`) on both
`preprocessed_train` and `preprocessed_train_benchmark`.

**Round 1 — fp32 vs AMP (19:51 UTC).** The build finished all 27 train chunks (6362 s), then failed at
`2.20e-02` against a `1e-2` tolerance. `verify_chunk_cache` recomputed in **fp32** and compared against a
cache built under **autocast** (the cache's own `meta.json` records `autocast: true`), so it measured
AMP-vs-fp32 drift on TinyViT-5M — against features whose entire range is only ±1.8. Measured on
`train/chunk_000000`: fp32 recompute 1.885e-02, **AMP recompute 6.275e-03**. fp16 storage is not the cause
(its step there is 9.8e-4). Fix: the recompute now mirrors `meta["autocast"]`.

**Round 2 — `max` is not a stable statistic (20:41 UTC).** With that fixed, train/val/test passed and
`train_benchmark` failed at `3.89e-02`. The distributions showed why: the *typical* error is the same
across splits (mean 7.7e-4 train vs 8.5e-4 benchmark, 11% apart) while the **max** moved 5.4×
(6.3e-3 vs 3.4e-2). The gate was taking the worst of ~4×10⁵ AMP-noisy values. Sharper still: `val`'s max
read **1.73e-2 and 6.18e-4 in two runs of the same cache with the same code** — the max is unstable
run-to-run, not merely split-to-split (AMP reduction order; the build batches 512 frames, the check one
window).

**Fix:** `verify_chunk_cache` now returns `ParityStats(max, p999, mean, count)` and the gate reads
**`p99.9 ≤ 2e-2`**, printing all three. Rationale: a real preprocessing mismatch shifts the *whole*
distribution to O(1), so a percentile is both more sensitive and more stable than a `max` threshold set
loose enough not to false-alarm. Post-fix, all four built splits pass with ≥2.1× headroom:

| split | p99.9 (gated) | max (diagnostic) | mean |
|---|---|---|---|
| train | 3.50e-03 | 7.18e-03 | 7.14e-04 |
| train_benchmark | 9.42e-03 | 3.87e-02 | 8.49e-04 |
| val | 4.74e-04 | 6.18e-04 | 6.76e-05 |
| test | 3.36e-03 | 6.15e-03 | 6.77e-04 |

**No data lost across either round** — every built chunk re-read as up to date, and the gate exits before
any dependent job is marked `.skipped`. Queue restarted 20:56 UTC; `val_benchmark` and `test_benchmark`
were the only caches still unbuilt.

### PROBE VERDICT — the cache is exonerated (2026-09-18 10:45 UTC)

`20260918_084507_pose_full_probe_imgmode_frozeneval` — v2 BASE with `data.visual_input=images`, seed 42,
stopped at the pre-registered decision point (epoch 3) rather than running to 5.

| epoch | v1 baseline | v2 (cached) | **probe (images)** |
|---|---|---|---|
| 1 | 0.5673 | 0.4858 | 0.4860 |
| 2 | 0.4473 | 0.3344 | 0.3346 |
| **3** | **0.3934** | **0.2448** | **0.2449** |

**The probe tracks cached-mode v2 to four decimals and is nowhere near v1.** Swapping cached features for
real JPEG decode — which restores fresh per-epoch colour jitter AND frame-erase — changed *nothing*.

**Consequences.** (1) The augmentation-diversity hypothesis is dead, and the K=6 cache rebuild was chasing
the wrong variable — an honest cost of ~4.2 h plus 13 GB. (2) **The feature cache and its 3.4x speed-up are
not the problem and can be kept.** (3) The damage comes from the two model flags.

**What the probe canNOT tell us, stated because I originally framed it as if it could:** it differs from v1
in **two** flags, not one — `vit_frozen_eval=true` *and* `train_frame_proj=true`. So "the BN drift was the
regulariser" is only one of two readings. The other is `train_frame_proj`: in v1 that 320->128 projection is
a **frozen random matrix** sitting on the sole path between the backbone and every downstream module, which
is a severe capacity bottleneck. Unfreezing it adds 41k parameters exactly where they most increase fitting
capacity, and that explains "fits much faster" at least as naturally as BN statistics do.

**The separating experiment (cheap, ~1 h, run it after R4):** cached mode + `vit_frozen_eval=true` +
`train_frame_proj=FALSE`, 5 epochs. Legal because the cache stores features *before* `frame_proj`, so
whether that layer trains is independent of caching. Epoch-3 train_loss near 0.39 indicts
`train_frame_proj`; near 0.24 indicts `vit_frozen_eval`. If it is `train_frame_proj`, the payoff is large:
cached mode + the BN fix + a frozen projection would differ from v1 in only the genuine bug fix, keeping the
speed-up and the 3 seeds it buys.

### v2 arm 1 measured: materially WORSE than v1 (2026-09-18)

`20260918_050956_pose_full_v2_base_trainstreaming_s42`, `best.pth` = epoch 13, streaming test (n=69,875),
against the v1 baseline `20260714_134253`:

| streaming test | v1 | v2 arm 1 |
|---|---|---|
| crosses AUC | **0.784** | 0.7209 |
| tuned crosses F1 | **0.225** | 0.1126 |
| oracle F1 (leakage ceiling) | — | 0.1178 |

**Half the F1 and 0.063 less AUC.** AUC is threshold-free, so this is NOT the calibration artifact that makes
val F1@0.5 unreadable — v2 ranks genuinely worse. The oracle figure (tuned on test, never reportable) shows
no threshold choice recovers it.

Caveats, stated rather than buried: `best.pth` is epoch 13, which `selection_metric=crosses_f1` picked off a
noisy 0.5-threshold F1 over the healthier epoch 3 (val AUC 0.774 vs 0.819), and epoch 3's weights were
overwritten so that cannot be re-tested; the run was stopped at epoch 14 of 30; and it is one seed. None of
those plausibly account for 0.063 of AUC.

**Open question this forces:** v2 changed three things at once, and the two candidates for the damage are
`vit_frozen_eval=true` (the drift may have been acting as a regulariser) and `visual_input=cached_features`
(augmentation reduced to discrete variants, frame-erase gone). The 5-epoch probe
`probe_imgmode_frozeneval` — BASE with `visual_input=images`, one flag different — separates them: train
loss tracking v1 (~0.39 at epoch 3) indicts the cache; collapsing like v2 (~0.24) indicts the BN fix.

### K=6 is a clean NEGATIVE — colour jitter is not the lever (2026-09-18)

Raising `cache_color_variants` 2 -> 6 changed nothing. Matched epochs, same seed, streaming val:

| ep | K=2 train / val | K=6 train / val | K=2 AUC | K=6 AUC |
|---|---|---|---|---|
| 3 | 0.2458 / 0.2192 | 0.2448 / 0.2148 | 0.816 | 0.819 |
| 5 | 0.2034 / 0.3091 | 0.1935 / 0.3393 | 0.790 | 0.772 |
| 7 | 0.1085 / 0.4912 | 0.1145 / 0.5059 | 0.811 | 0.816 |
| 9 | 0.0697 / 0.5350 | 0.0788 / 0.5757 | 0.811 | 0.772 |

**The gap that matters is v2 vs v1, and it is in TRAIN loss, which is not calibration-sensitive.** Against
the v1 baseline `20260714_134253` at the same epochs: v1 train 0.393 / 0.407 / 0.369 / 0.336 (ep 3/5/7/9)
against v2's 0.245 / 0.194 / 0.115 / 0.079, while v1 val stayed flat at 0.148-0.180 and v2 climbed
0.215 -> 0.576. v1 is barely fitting where v2 memorises.

**Mechanism (hypothesis, consistent with the evidence): effective feature diversity, not jitter strength.**
v1 augmented *images* and re-ran the backbone, so a window produced a effectively continuous set of feature
vectors across epochs — fresh colour jitter, frame erase, plus BatchNorm statistics that drifted every batch
(the "frozen backbone" bug, which was doing unintended regularisation work). v2 serves **14 fixed feature
vectors** per window. Going 6 -> 14 is still discrete, which is why it changed nothing: the axis is
continuous-vs-discrete, not 2-vs-6.

**Therefore do NOT raise K further.** The candidate interventions are feature-space augmentation (noise or
mixup on the cached vectors, cheap and preserves the speed-up, but new code), or reverting to image-mode
augmentation with `vit_frozen_eval=true` (keeps the BN fix, forfeits the 3.4x speed-up).

**Not yet known, and it gates everything:** whether v2 is actually worse on the *reportable* metric. AUC is
comparable (v2 peaks 0.819 vs v1's 0.858) and val F1@0.5 is calibration-sensitive, so the decisive number is
the val-tuned test F1 from this arm's own eval cells, against v1's 0.225. Read that before spending the
remaining ~3.5 days of queue.

### Trap: a recipe change under an existing run dir gets silently RESUMED (hit 2026-09-18 05:09 UTC)

`train_run2.sh::run_dir()` globs `outputs/runs/*_<TAG>` for the newest dir with a `checkpoints/` subdir and
passes `--resume <that>/checkpoints/last.pth`. That is correct crash recovery, and it is exactly wrong after
a recipe change: when the queue restarted following the K=6 cache rebuild, it **resumed the K=2 smoke run**
instead of starting fresh. Epochs 1-9 had trained on the K=2 cache, 10-16 on the K=6 cache — a run that
answers no question and is not comparable to the seed 43/44 runs.

Caught ~1.5 h in by noticing epochs 1-9 were byte-identical (same losses *and* same epoch times) to the
abandoned run. **Clearing the cache markers was not enough; the run state has to be cleared too.**

**Rule when changing a recipe or a cache mid-queue:** stop the queue, then make the old run dir unmatchable
by the glob (rename with a suffix, e.g. `.abandoned_k2cache`) or delete its `checkpoints/`. Verify with
`for d in $(ls -dt outputs/runs/*_<TAG>); do echo "$d"; done` printing nothing before restarting.

**Structural follow-up (not done — `train_run2.sh` is being executed by a live job and must not be
overwritten):** have the launcher diff the candidate run's `resolved_config.yaml` against the recipe flags
and refuse to resume on a mismatch, rather than relying on the operator remembering.

### Phase 1 outcome — one cause measured, one hypothesis refuted

Full numbers in [RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md) § "R1 results and what they cost to
interpret". In short:

* **(a)+(b) BatchNorm.** Weights genuinely frozen (max param diff 0.0); statistics drift (median layer
  mean-shift 0.06 sd, stem conv variance → ~8% of pretrained). Re-estimating the statistics alone moves val
  AUC 0.788–0.809 and val F1@0.5 0.208–0.234 — **the size of the R1-vs-baseline difference**, which is the
  evidence for both the fix and the 3 seeds. Restoring pretrained statistics drops val AUC to 0.702 (the
  trained head depends on the drifted features; not evidence about the fix, since v2 trains from scratch).
* **Refuted:** none of the three re-estimations produced an "everything is a crossing" epoch, so **BN drift
  does not explain those collapses** (tested at the epoch-8 weights only — an interaction at early-epoch
  weights is not excluded). Consequence: **G_smoke was rewritten** before launch. It no longer requires the
  collapses to disappear; it requires the run to learn through cached features (some epoch in 1–5 with val
  AUC ≥ 0.70 and not collapsed) and reports the collapse count without gating on it.
* **(c) Timing.** The onset head orders windows by time-to-onset and its readout slightly edges the reported
  head it was auxiliary to. The instrument (`eval/onset_timing.py` + two scripts) is now what the v2 onset
  arms are judged with. A transient execution note — when the work
lands, fold the outcomes into THESIS_ROADMAP / RESULTS_MATRIX / CLAUDE.md and move this file to
`docs/archive/`.

## Overview

Every `pose_full` run so far (the four baselines, R1, R2) was trained on a "frozen" TinyViT that is not
actually frozen, and single-seed runs cannot resolve the small differences the onset method is expected to
make. This plan (1) measures what the backbone problem cost, on R1's existing checkpoint; (2) fixes the
recipe behind a config flag; (3) caches the frozen backbone's features so training stops being bound by JPEG
decoding; and (4) re-runs every arm under the fixed recipe with 3 seeds each.

## Why (findings, 2026-09-17 — verified in code unless marked)

- **BatchNorm drift.** TinyViT-5M has 27 BatchNorm layers. `freeze_vit_backbone` only sets
  `requires_grad=False`; `Trainer.train_chunk` calls `model.train()`, so the frozen layers normalise with
  per-batch statistics (4 clips x 20 near-identical frames) and overwrite the pretrained running statistics.
  Tested: running stats change under `requires_grad=False` + `train()`.
- **Frozen random projection.** `freeze_vit_backbone` freezes every `vit.*` parameter, including
  `vit.frame_proj` (randomly initialised `Linear(320 -> 128)`). Counts match the research-PC log exactly
  (215 frozen tensors, 721,725 trainable).
- **Symptom (hypothesis, not proven).** "Everything is a crossing" validation epochs in R1 (7/28), the
  baseline `20260714_134253` (5/19) and R2-streaming (epochs 1, 2, 4 — reproducing the baseline epoch for
  epoch) with ranking intact (AUC 0.65–0.81) and the onset head flipping on the same epochs: a shared
  upstream shift, consistent with BN drift.
- **Wasted decode.** `LMDBChunkDataset.__getitem__` decodes tight AND context JPEGs per frame; `pose_full`
  never uses the tight crop.

## Architecture decisions (proposed — see Open questions)

1. **Everything new is config-gated, default = v1 behaviour.** Existing checkpoints, the golden tests and the
   paused v1 queue (R2-streaming resumable from `last.pth`) stay reproducible. v2 is a recipe (a set of
   `--set` flags), not a new default. The default can flip once v2 baselines exist.
2. **v2 = eval-mode frozen backbone + trainable `frame_proj` + cached features.** v2 re-runs every arm, so
   bundling does not confound any v2-vs-v2 comparison; check (b) attributes the BN part on its own.
3. **Cache the 320-d pooled features, before `frame_proj`**, so the projection stays trainable.
4. **Augmentation in cached mode:** horizontal flip (cached flipped features + the existing pose/motion flip),
   **K pre-built color-jitter variants** (jitter drawn once per window at build time, deterministic seed;
   each costs one extra GPU pass), and motion noise at read time. Frame erase is dropped (it cannot be
   pre-built). Training picks among the cached variants each epoch instead of drawing fresh jitter.
   **K = 6 since 2026-09-18** (was 2): the first smoke run overfit hard at K=2 — train loss 0.486 -> 0.070
   by epoch 9 with val loss rising every epoch from 3, where v1's fresh-per-epoch jitter took 20+ epochs to
   reach 0.29. Measured build cost, contrary to the original guess that decode dominates: **~41 s per
   variant per 5,000 windows**, i.e. essentially linear in V (V=1 chunk 30 s, V=6 235 s, V=14 571 s), so
   the rebuild cost 4.2 h and the train cache went 11 GB -> 24 GB. Note K=6 still gives 6 fixed jitters
   against v1 drawing a fresh one every epoch, so v2 remains the less-augmented recipe. The offline augmented LMDB
   (`preprocessed_train_aug`) is cached as-is, so its pre-baked copies survive.
5. **Nothing else changes in v2** (batch 4 x accum 8, lr schedule, sampler, loss weights, `crosses_f1`
   selection) so the only moving parts are the ones above.
6. **The v1 queue is left untouched on HOLD.** v2 gets its own `recipes_v2.sh` / `queue_v2.sh`.

## Phase 0 — Pause R2-streaming ✅ (2026-09-17 12:07 UTC)

`queue/HOLD` set, queue stopped, training stopped after epoch 4 (`last.pth` kept). Resume as-was:
`rm queue/HOLD; bash boot_resume.sh`. Decide later whether to resume (~20 h) or drop it.

## Phase 1 — Three checks on R1's checkpoint (no retraining; research PC while idle)

### Task 1: Backbone BN diagnostics — checks (a) and (b)

**Description:** A script that (a) tabulates how far the BN running statistics in R1's `best.pth` and
`last.pth` drifted from the pretrained TinyViT weights, and (b) re-scores R1 on streaming val three ways:
as saved, with the pretrained BN statistics restored, and with BN re-estimated from 3 different draws of
~50 sampler-drawn training batches (how much the scores jitter from the statistics alone). Writes to a
diagnostics dir — never to the run dir (no `eval_log.csv` / thresholds side effects).

**Acceptance criteria:**
- [ ] (b-i) "as saved" reproduces R1's stored streaming-val row: crosses F1 0.2442, AUC 0.8051 (±0.001)
- [ ] Per-layer drift table (relative mean shift, variance ratio) + one JSON summary
- [ ] (b-ii)/(b-iii) report AUC, F1@0.5, val-tuned F1, mean predicted P(cross), share predicted positive

**Verification:** unit tests on a `pretrained=False` TinyViT (the swap touches exactly the `vit.net` BN
buffers; drift stats on synthetic buffers); `ruff check .`; `pytest -m "not slow"`; the parity criterion above.

**Dependencies:** None. **Files:** `scripts/diagnose_backbone_bn.py`, `src/pedpredict/eval/diagnostics.py`,
`tests/test_bn_diagnostics.py`. **Scope:** M. **Runtime (est.):** ~30 min (a val pass took ~3.5 min for R1).

### Task 2: Onset-timing instrument — check (c)

**Description:** (1) A dump script that runs an onset-head checkpoint over a split and saves per window: the
`crosses_frame` probability, the readout probability, the K hazard logits, and the labels (`crosses`,
`onset_offset`, `future_observed`, `track_crosses`, `track_id`). (2) An analysis module that reads the dump
and reports: readout vs `crosses_frame` (AUC, val-tuned F1 at 32 frames); mean score by true onset group
(0–15, 16–31, 32–63, 64–95 frames, never within 96, censored, already crossed); per-horizon AUC at
16/32/64/96 frames over windows whose answer is knowable (the M4 rule at that horizon); AUC of
"crosses within 32" against "crosses in 33–96" and against "never". This becomes the timing measure for the
v2 onset arms.

**Acceptance criteria:**
- [ ] Dump for R1 on streaming val + test; `crosses_frame` AUC reproduces the stored rows (val 0.8051,
      test 0.7716 ±0.001)
- [ ] Report with every metric listed above, for both heads where it applies
- [ ] Analysis runs on the laptop from the dump alone (no data, no checkpoint)

**Verification:** synthetic tests (a perfectly timed hazard gives per-horizon AUC 1.0; unknowable windows are
excluded at each horizon, matching `onset_stats.is_usable`); gate green; parity criterion.

**Dependencies:** None. **Files:** `scripts/dump_onset_predictions.py`, `src/pedpredict/eval/onset_timing.py`,
`scripts/report_onset_timing.py`, `tests/test_onset_timing.py`. **Scope:** M. **Runtime (est.):** ~20 min.

### Task 3: Run the checks and record findings

**Description:** Upload + run Tasks 1–2 on the research PC (queue on HOLD), pull outputs to the laptop,
write the findings into RESULTS_MATRIX (R1 section) and memory.

**Acceptance criteria:**
- [ ] Both parity criteria pass on real data (else stop: the scripts are wrong, not the model)
- [ ] Findings stated plainly: how much BN drift moves val scores; whether R1's onset head orders windows
      by time-to-onset

**Dependencies:** Tasks 1, 2. **Scope:** S.

### Checkpoint 1 — review with you

- [ ] Gate green; parity passed; findings reviewed
- [ ] Check (c) sets arm priority: no timing order at all -> run R4 before R3 in Phase 4

## Phase 2 — The v2 recipe fix (laptop)

### Task 4: Keep a frozen backbone in eval mode

**Description:** New flag (name TBD, e.g. `model.vit_frozen_eval_mode`, default `false`). When on,
`TimmBackbone.train()` keeps `self.net` in eval mode, so BN uses and keeps the pretrained statistics.
Validated: requires `freeze_vit_backbone=true` and a timm backbone.

**Acceptance criteria:**
- [ ] Flag off: a train-mode forward still changes BN running stats (pins v1 behaviour)
- [ ] Flag on: running stats unchanged after train-mode forwards; backbone output equals its eval output
- [ ] No parameter-layout change (old checkpoints load `strict=True`); goldens untouched

**Verification:** new tests in `tests/test_timm_backbone.py`; config validation test; gate green.
**Dependencies:** None. **Files:** `models/timm_backbone.py`, `config/schema.py`, `config/loader.py`,
`configs/model.yaml` (or equivalent), tests. **Scope:** S.

### Task 5: Optionally trainable `frame_proj`

**Description:** New flag (e.g. `model.train_frame_proj`, default `false`); `freeze_vit_backbone` skips
`vit.frame_proj.*` when on.

**Acceptance criteria:**
- [ ] Flag off: 215 frozen tensors / 721,725 trainable (current, pinned)
- [ ] Flag on: 213 frozen / 762,813 trainable (+ 320x128 + 128)

**Verification:** param-count tests; gate green. **Dependencies:** None. **Files:** `training/schedule.py`,
`config/schema.py`, tests. **Scope:** XS.

### Checkpoint 2

- [ ] Gate green; CLAUDE.md architecture row (visual stream) + config docs updated (doc-sync table)

## Phase 3 — Feature cache (laptop code, research-PC build)

### Task 6: Cache format + builder

**Description:** `scripts/build_feature_cache.py` walks each LMDB chunk, decodes **context crops only**,
runs the eval-mode backbone on the GPU, and writes per chunk: `feats.npy` float16 `[N, V, T, 320]` and a
metadata JSON (seq-id order = `LMDBChunkDataset.seq_ids`, the variant list, backbone name, weight hash, timm
version, context size, dtype, jitter seed). Variants for train dirs: clean, flip, and K color-jitter
variants each with and without flip (`V = 2 + 2K`); val/test dirs: clean only (`V = 1`). Location from a new
`paths` field.

**Acceptance criteria:**
- [ ] Order and metadata match the chunk; flipped features equal backbone(flipped images)
- [ ] Builder refuses to overwrite a cache made by a different backbone/weights

**Verification:** tests on a tiny synthetic LMDB with `pretrained=False`; gate green.
**Dependencies:** Task 4. **Files:** `src/pedpredict/data/feature_cache.py`, `scripts/build_feature_cache.py`,
`config/schema.py`, tests. **Scope:** M. **Size (est.):** ~2.5 GB for all splits incl. flips.

### Task 7a: Dataset + augmentation read path for cached features

**Description:** `LMDBChunkDataset` in cached mode reads meta (motions, pose, labels, S1 fields) from the
LMDB and features from the cache — no JPEG decode. Feature-mode augmentor: flip (selects the flipped row,
applies the existing motion/pose flip) + motion noise.

**Acceptance criteria:**
- [ ] Same labels/motions/pose as the image path for every sample; flip consistent between features and pose
- [ ] Fails loudly on a missing or mismatched cache

**Verification:** tests on synthetic LMDB + cache; gate green. **Dependencies:** Task 6.
**Files:** `data/lmdb_dataset.py`, `data/augment.py`, `data/collate.py`, tests. **Scope:** M.

### Task 7b: Model + loaders + eval wiring

**Description:** A `data.visual_input: images | cached_features` switch honoured by `ChunkPrefetcher`,
`evaluate.py` and `forward_model`; `TimmBackbone` accepts `[B, T, 320]` and applies only `frame_proj`.
Validation: cached mode requires the eval-mode frozen backbone (Task 4).

**Acceptance criteria:**
- [ ] Parity: eval-mode model on images == model on cached features (fp32, tolerance ~1e-4) on synthetic data
- [ ] Train + eval run end-to-end in cached mode on synthetic data

**Verification:** parity test; e2e smoke test; gate green. **Dependencies:** Tasks 4, 7a.
**Files:** `models/timm_backbone.py`, `models/registry.py`, `training/chunk_loader.py`, `eval/evaluate.py`,
`config/*`, tests. **Scope:** M.

### Task 8: Build caches + measure on the research PC

**Description:** Build caches for streaming train (base + aug) / val / test and anchored train / val / test.
Spot-check parity on real data (200 val windows: image-path vs cached features). Time one streaming training
epoch and one val pass in cached mode.

**Acceptance criteria:**
- [ ] Real-data parity within tolerance
- [ ] Measured epoch time and eval time recorded; compute budget for Phase 4 recalculated

**Dependencies:** Tasks 6, 7a, 7b. **Scope:** S (mostly wall-clock).

### Checkpoint 3 — review with you

- [ ] Parity on real data; measured speed-up; go / no-go on the seed x arm budget below

## Phase 4 — v2 runs (research PC)

### Task 9: v2 smoke run (fail fast on the hypothesis)

**Description:** Baseline-v2 on streaming, seed 42, full v2 recipe. Stop after epoch 5 if anything is off.

**Acceptance criteria:**
- [ ] No "everything is a crossing" epochs in epochs 1–5 (v1 had 3 of 4 in epochs 1–4)
- [ ] Epoch time matches Task 8's measurement

**If collapses persist:** stop — the BN hypothesis was wrong or partial. Next suspects: the pose encoder's
BatchNorm1d at batch 4, logsumexp pooling under the sampler's 26% training prior. Do not spend seeds.

**Dependencies:** Checkpoint 3. **Scope:** S.

### Task 10: v2 queue — 5 arms x 3 seeds

**Description:** `recipes_v2.sh` + `queue_v2.sh` (+ its verify script), same safety rules as the v1 queue.
Order: baseline-v2 anchored-trained x3 -> baseline-v2 streaming-trained x3 -> R3-v2 x3 -> R4-v2 x3 ->
R1-v2 aux x3 (last, optional). Seeds 42 / 43 / 44. Every run scored val -> test on both protocols; onset
arms also get the Task 2 dump + timing report. First seed of R3 and R4: stop at epoch 2 if
`onset_readout_p95 - p05` has collapsed (dead-head rule).

**Acceptance criteria:**
- [ ] verify script passes (every recipe resolves; only the intended fields differ between arms)
- [ ] All runs + eval cells complete, or failures marked and reported

**Dependencies:** Task 9. **Scope:** M (code) + wall-clock.

### Task 11: Results

**Description:** RESULTS_MATRIX v2 section (mean ± std over seeds, both protocols, timing metrics for onset
arms); `rebuild_index`; THESIS_ROADMAP; CLAUDE.md (recipe contract, whether the default flips to v2).

**Dependencies:** Task 10. **Scope:** S–M.

### Checkpoint 4 — final review with you

## Compute budget (estimates until Task 8 measures)

| Item | Today (v1) | v2 guess | Count |
|---|---|---|---|
| Streaming epoch | ~47 min (JPEG-bound) | **~13.9 min measured** (guess was 3–8) | up to 30 per run |
| Streaming training run | ~15–22 h | **~6.9 h measured** (guess was 2–4) | 12 (4 streaming arms x 3) |
| Anchored training run | ~1 h | minutes | 3 |
| Eval pass (val or test, per protocol) | 3–15 min | ~1–3 min | ~60 |

Measured 2026-09-17 on the K=2 smoke run (712–973 s/epoch, mean 832 s): the speed-up is **3.4x, not the
6–15x guessed**, which puts Phase 4 at **~3.5–4 days serial**, not 1.5–2.5. Original estimate follows. If Task 8 shows a smaller speed-up, trim in this
order: drop R1-v2 aux, then 2 seeds for the anchored baseline. With CPU no longer the bottleneck, two runs
side by side on the 5090 is also an option.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BN fix does not remove the collapses | High | Task 9 fails fast before any seeds; named next suspects |
| Cache parity drift (fp16 storage) | Med | Synthetic parity test (Task 7b) + real-data spot check (Task 8) |
| Losing color jitter / erase increases overfitting | Med | Watch early-stop epochs vs v1; option to cache a few color-jittered variants later |
| Diagnostic scripts misreport | Med | Hard parity gates against stored eval rows (Tasks 1–3) |
| v2 numbers are not comparable to v1 | Med | By design: v1 stays as the motivation evidence; v2 re-derives the anchored-vs-streaming pair itself |
| Research-PC outages (owner VPN, gateway laptop) | Low | Existing queue safety rules carried into `queue_v2.sh` |
| Personal-PC disk nearly full breaks tests | Low | Clear stale pytest temp dirs before running the gate |

## Decisions (resolved 2026-09-17)

1. **Flag defaults:** v1 stays the default; v2 is selected by recipe flags.
2. **`frame_proj`:** trainable in v2 (features are cached before it, so caching is unaffected).
3. **Augmentation in cached mode:** flip + K pre-built color-jitter variants + motion noise; frame erase
   dropped (see Architecture decision 4). **K = 6** (raised from 2 on 2026-09-18 after the K=2 smoke run
   overfit; the abandoned run keeps its evidence in its own `NOTE_ABANDONED.md`).
4. **Arms:** all five x 3 seeds, R1-aux included (still scheduled last).
5. **Selection metric:** keep `crosses_f1`.

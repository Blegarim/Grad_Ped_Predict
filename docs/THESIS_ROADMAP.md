# Thesis Roadmap — Streaming Crossing-Onset Detection

**Renamed from** `streaming-onset-plan.md` (obsolete name). **Prepared:** July 2026 · **Last swept:** 2026-07-21.

> **What this document is.** The single, findable, end-to-end checklist for the whole thesis — from the
> pre-migration prototype (~2 months ago) through to defending the paper. It supersedes the old
> `streaming-onset-plan.md` (which only covered the pivot) and is the *tracker*; the *why* lives in the
> two companions and is not repeated here:
> - **The research brief** — [`project-context-streaming-crossing-onset.md`](project-context-streaming-crossing-onset.md) (read for the thesis argument, the dead ends, the reviewer objections).
> - **The supporting-studies plan** — [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) (the RQ ablations, re-slotted as support).
> - **The resolved engineering audit** — [`HOLE_AUDIT.md`](HOLE_AUDIT.md) (the data-contract decisions + Final attack order).
>
> **The one-line thesis:** the standard PIE protocol is event-anchored (~2.5:1) and hides the deployed
> streaming case (~37:1, full of "will-cross-soon" hard negatives). The contribution is to re-evaluate
> under a streaming protocol and **decompose** the anchored→streaming performance gap into a
> recalibration-fixable part (**G_prior**) and a residual hard-negative part (**G_hardneg**).

**Legend:** `[x]` done · `[~]` in progress / partial · `[ ]` not started · 💻 personal PC (code, no data) ·
🖥️ lab PC (needs PIE data / GPU). Sub-items are granular on purpose — each is meant to be individually
checkable.

---

## Progress at a glance

| Stage | Theme | State |
|---|---|---|
| **0** | Prototype → clean rebuild (behavior-preserving port) | ✅ done |
| **1** | Engineering audit + v2 data-contract *code* | ✅ done |
| **2** | v2 data regeneration + baseline runs | ✅ done (streaming + anchored builds exist, trained) |
| **3** | Streaming pivot: onset metadata + protocol switch + pose arm | ✅ code done · 🖥️ pose extraction done |
| **4** | **The decomposition (G_prior / G_hardneg)** — the headline | 🟢 **NEXT — computable now, no GPU** |
| **5** | Streaming-leg convergence + matched-config matrix | 🟡 partial (streaming trains but unstable) |
| **6** | ODAS/OAD streaming metrics + negative-composition report | ⬜ net-new, mostly 💻 |
| **7** | Constructive close (curated hard-neg recipe) + supporting studies | ⬜ not started |
| **8** | Write-up, defense, release | ⬜ not started |

The critical path to a *defensible thesis* runs **4 → 5 → 6 → 8**. Stage 7 and the RESEARCH_PLAN RQ
spokes are the "full paper vs. cautionary note" upgrade — land them if time allows, but 4–6 alone
support a thesis (see brief §4, the demote-to-short-paper branch).

---

## Stage 0 — Prototype → clean rebuild ✅

The behavior-preserving port of the undergrad prototype into `src/pedpredict/`. Archived under
[`docs/archive/`](archive/) + the `legacy-archive` git tag; no longer load-bearing. Recorded here only so
the arc is complete.

- [x] 💻 Typed config system (yaml → dataclass → argparse, `--set` overrides; no Hydra/W&B)
- [x] 💻 Port ViT_Hierarchical, MotionEncoder, CrossAttentionModule + heads, EnsembleModel
- [x] 💻 Typed model registry (`models/registry.py`); ablation models (`ped_local`, `visual_only`, `vanilla_concat`)
- [x] 💻 Unified multitask loss + imbalance stack (balance / sampler / class-weights)
- [x] 💻 Trainer, CheckpointManager, two-phase schedule, run-dir layout + cross-run CSV index
- [x] 💻 Eval pipeline (shared MetricAccumulator) + efficiency benchmark (fvcore)
- [x] 💻 Quantitative + qualitative viz; ONNX export + parity checks
- [x] 💻 Golden characterization tests captured (`tests/fixtures/golden/`); `ruff` + `pytest` gate green
- [x] 💻 **P9 cutover** — retire rebuild scaffolding, stand up v1.0 clean baseline

---

## Stage 1 — Engineering audit + v2 data-contract *code* ✅

Two-pass audit ([`HOLE_AUDIT.md`](HOLE_AUDIT.md)) catalogued every validity/design/correctness issue and
resolved each with a fixed **Final attack order**. Two batches merged to `main`.

**Batch 1 — data-independent code fixes (M1, M7, M8)**
- [x] 💻 Reproducible seeding (`train.seed`, in config snapshot)
- [x] 💻 F1-based model selection (`train.selection_metric`); early-stop reads it
- [x] 💻 **Val-tuned decision thresholds** — test-set leakage removed; same-split sweep renamed `oracle_*`, never reported
- [x] 💻 Effective-distribution **instrument** (`training/distribution.py` → `train_distribution.json` every run)
- [x] 💻 `train.use_class_weights` switch + latent-bug fixes

**Batch 2 — v2 data-contract code (M3–M6, M9, A4)**
- [x] 💻 M3 — `actions`/`looks` relabeled as state-at-end-of-observation (`crosses` stays future-any)
- [x] 💻 M4 — right-censored windows dropped (not silently labeled 0), counted per split
- [x] 💻 M5 — benchmark/anchored eval mode (`make_sequences.py --benchmark --split {train,val,test}`)
- [x] 💻 M6 — `track_id` threaded `SequenceRecord` → LMDB meta → dataset items (enables track aggregation)
- [x] 💻 M9/A4 — motion stored 9-dim; frame-0 deltas true zeros; `model.motion_norm` runtime flag (`image` default | `per_sequence`); flip negates `dx` + reflects `cx`

---

## Stage 2 — v2 data regeneration + baseline runs ✅

The one-time lab-PC regen the code above enabled, plus the first real training.

- [x] 🖥️ Regenerate sequences + build LMDBs for all three splits (streaming protocol)
- [x] 🖥️ Build the anchored/benchmark LMDBs (`*_benchmark` dirs)
- [x] 🖥️ Re-pin Dataset Statistics table + `count_labels` drift gate + `test_stats` fixture
- [x] 🖥️ First v2 training run (`20260616_153511_full`) → **the imbalance-collapse finding** (AUC ~0.57, crosses F1 0.081, effective-crosses ~89%). Reframed by the pivot as *the 37:1-collapses-a-from-scratch-ViT finding*, not a bug.
- [x] 🖥️ Retuned imbalance stack (run #2 canonical: sampler `crosses^0.5`, class-weights off) → effective crosses ~26%
- [x] 🖥️ warmup→cosine LR strategy; gradient accumulation; fusion-residual default-on
- [x] 🖥️ A1 ViT redesign (monotonic dims `[48,96,192,384]`, real 7×7 windows)

---

## Stage 3 — Streaming pivot: onset metadata + protocol switch + pose arm

The July reframe. Code is landed; the pose extraction pass ran on the lab PC.

**S1 — per-window onset annotations (the H-sweep + negative-typing enabler)**
- [x] 💻 `onset_offset` (frames end-of-obs → first future crossing; `-1` if none) on `SequenceRecord`
- [x] 💻 `future_observed` (`n − end`) — makes the H-sweep rigorous against right-censoring
- [x] 💻 `track_crosses` (track ever crosses) — separates genuine non-crosser from will-/already-crossed
- [x] 💻 Tests: genuine-non-crosser, in-horizon positive, hard-temporal-negative, already-crossed
- [ ] 🖥️ Regenerate standard pkls carrying S1 fields **← still needs the final lab-PC pkl re-gen** (rides pkls, no JPEG re-encode; joins to test preds by index)

**Protocol switch (`data.protocol`)**
- [x] 💻 `data.protocol={streaming,anchored}` resolved once in `paths.protocol_lmdb_dirs`
- [x] 💻 Honored by `train.py` / `ChunkPrefetcher` / `evaluate.py`
- [x] 💻 `eval_log.csv` carries a `protocol` column
- [x] 💻 `thresholds.json` keyed by protocol (`thresholds_{streaming,anchored}.json`) — fixes the second-val-pass overwrite

**Pose arm** (steps per [`POSE_ENCODER.md`](POSE_ENCODER.md))
- [x] 💻 Steps 0–4: PoseCfg, `data/pose.py` feature math, `PoseFullModel` + `pose_kinematics`/`pose_full` registry, `extract_pose.py` (`--dry-run`)
- [x] 🖥️ Step 5: pose extraction pass over PIE → `pose_cache/` (streams frames in-memory from clips)
- [x] 🖥️ `pose_full` frozen-TinyViT runs trained (anchored + streaming) — see Stage 5

---

## Stage 4 — 🟢 THE DECOMPOSITION (the headline) — computable NOW, no GPU

**This is the next action and the single most valuable thing on the board.** The four macro_f1 matrix
cells already exist in eval logs; the G_prior/G_hardneg number has *not* been written down. This stage is
pure 💻 analysis.

**Reference — the completed macro_f1 matrix** (crosses, test split, tuned thresholds):

|  | test = anchored | test = streaming |
|---|---|---|
| **train = anchored** (`20260710_122947`) | AUC 0.873 · F1 0.694 | AUC **0.540** · F1 **0.064** |
| **train = streaming** (`20260710_152517`) | AUC 0.514 · F1 0.322 | AUC 0.742 · F1 **0.190** |

**Metric discipline (settled 2026-07-21):** G_prior and G_hardneg are **operating-point** quantities —
measure them in **F1 (or precision-at-fixed-recall)**, the currency recalibration acts on. **AUC is the
*diagnostic*, not the measurement** — it explains *why* recalibration recovers nothing (a discrimination
collapse, not a mis-set dial), but it must not be quoted *as* G_prior.

- [ ] 💻 Compute **G_total** = (streaming-trained, test-streaming F1) − (anchored-trained, test-streaming raw F1) = 0.190 − 0.064 = **0.126**
- [ ] 💻 Compute **G_prior** = (anchored-trained, test-streaming, *recalibrated* F1) − (raw F1) = 0.064 − 0.064 = **≈0.00** *(threshold re-tuning recovered nothing; precision stuck ~3.3%)*
- [ ] 💻 Compute **G_hardneg** = (streaming-trained, test-streaming F1) − (anchored-trained, recalibrated F1) = 0.190 − 0.064 = **≈0.126** (the whole recoverable gap)
- [ ] 💻 Report the AUC diagnostic (0.873 → 0.540) as the *explanation* of why G_prior ≈ 0 — a discrimination collapse recalibration can't touch
- [ ] 💻 Write the decomposition up as a `docs/` results note (table + arithmetic + the two caveats below)
- [ ] 💻 **Caveat to state, not bury:** the 0.126 conflates "trained streaming" with "trained on 18× more data" (anchored train ≈ 4.9k vs streaming ≈ 88k). The matched-config sweep (Stage 5) is what bulletproofs it.
- [ ] 💻 **Caveat 2:** F1 at 37:1 is jumpy; the sharper version uses precision-at-fixed-recall / AP (Stage 6), which tightens this conclusion, not changes it.

*Deliverable:* the sentence "of the total anchored→streaming deployment gap, ~0% is recalibration-fixable
and ~100% is a genuine hard-negative skill gap the standard benchmark never trains" — with the table
behind it. **This is the thesis headline.**

---

## Stage 5 — Streaming-leg convergence + matched-config matrix 🟡

The decomposition's 0.126 is only as trustworthy as the streaming-trained model and the config match.
Right now the streaming leg trains but wobbles, and the crosses-only matrix is half-built.

**Stabilize the streaming-trained model** (Phase C of the old plan; the G_hardneg ceiling)
- [ ] 🖥️ Diagnose the val_loss instability in `20260714_134253` (spikes to 4–8 on epochs 1/2/4/13/17; recall collapses to 1.0 = all-positive; best epoch 8 is a lucky trough, not a plateau)
- [ ] 🖥️ Stabilize: revisit LR / warmup / focal-or-class-balanced loss / sampler power under raw 37:1 (this *is* the "curated streaming training" recipe — see Stage 7)
- [ ] 🖥️ Confirm a streaming-trained model that plateaus cleanly (not a single lucky epoch)

**Complete the crosses-only 2×2** (the multi-task-confound-free matrix)
- [x] 🖥️ train=streaming × test=streaming — `20260714_134253` (AUC 0.78 / tuned-F1 0.22)
- [x] 🖥️ train=streaming × test=anchored — `20260714_134253` (AUC 0.68 / tuned-F1 0.26)
- [ ] 🖥️ **train=anchored × test={anchored,streaming}** — *missing row.* One run: `pose_full`, `data.protocol=anchored`, `train.active_tasks=[crosses] train.selection_metric=crosses_f1`, config-matched to `134253`, then eval both protocols
- [ ] 💻 Recompute the Stage-4 decomposition on the crosses-only matrix; compare to the macro_f1 numbers

**Matched-config rigor (the "it was just more data" rebuttal)**
- [ ] 🖥️ Re-run anchored-train + streaming-train under **one identical config** (same seed, selection, arch, augment), so the only difference is the training protocol
- [ ] 🖥️ (validity) Multi-seed the headline cells — screen with 1 seed, confirm finalists with 3, report mean±std

---

## Stage 6 — ODAS/OAD streaming metrics + negative composition ⬜

Net-new, and mostly 💻 — unit-testable on synthetic streams before any lab-PC apply. This is what makes
the evaluation *correct* for a rare-event stream (F1 is a blunt instrument at 37:1) and delivers two
unpublished sub-contributions.

**Metric suite (import from OAD/ODAS — brief §2.4)**
- [ ] 💻 Per-frame mAP over the stream
- [ ] 💻 **Calibrated AP (cAP / mcAP)** — the OAD metric built for heavy-background imbalance
- [ ] 💻 **Point-level AP (p-AP)** with temporal-offset tolerance (onset-timing quality; needs S1 onset ground truth)
- [ ] 💻 Precision-at-fixed-recall / AP as the stable G_prior/G_hardneg currency (feeds back into Stage 4)
- [ ] 💻 Unit tests on synthetic streams (no data)
- [ ] 🖥️ Apply the suite across the matrix cells

**Negative-composition report (brief §2.3, from S1)**
- [ ] 💻 Classify every streaming negative into: genuine non-crosser / hard-temporal / already-crossed (from `track_crosses` + `onset_offset`)
- [ ] 💻 Pick + implement a **junk** signal (bbox height below threshold / occlusion / low `future_observed`)
- [ ] 🖥️ Report the composition (fraction hard-temporal vs junk vs genuine) — the "what the benchmark omits" figure
- [ ] 🖥️ **Confirm residual failures concentrate in the will-cross-soon windows** (validates the hard-negative story — brief §5 supporting analysis a)

---

## Stage 7 — Constructive close + supporting studies ⬜

Turns the diagnosis into a fix (the "full paper" upgrade), and re-slots the RESEARCH_PLAN RQs as support.

**Curated streaming recipe (narrows G_hardneg — brief Phase 5)**
- [ ] 🖥️ Pretrained backbone (RQ1) + focal/class-balanced loss + hard-negative emphasis + junk filtering
- [ ] 🖥️ Show the recipe measurably narrows G_hardneg vs the naive streaming-trained model
- [ ] 🖥️ **H-sweep** (near-free via S1) — base rate + difficulty move with horizon H → rebuts objection #2 ("just the TTE sweep")
- [ ] 🖥️ Ego-speed-under-streaming leakage probe (RQ4) — does streaming lean *harder* on the shortcut?

**Supporting RQ spokes** (from [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — hub-and-spoke, single-axis, vs the frozen hub)
- [~] RQ1 backbone — TinyViT-5M drop-in landed + trained (frozen); PVTv2-B0 / other candidates optional
- [~] RQ3 imbalance levers — the lever combination *is* the curated recipe above
- [x] RQ2 fusion residual — `fusion_residual` default-on; no-residual arm golden-pinned
- [x] RQ4 motion-norm — `image` vs `per_sequence` ablatable from one dataset
- [ ] RQ4 ego-speed on/off — accuracy effect + leakage reading
- [ ] RQ5 efficiency — params/FLOPs/latency/FPS/peak-VRAM per model type (harness exists)
- [ ] RQ6 calibration — reliability diagrams + temperature scaling on val (calibration script scaffolded)
- [ ] 🖥️ Kinematics-only pixel-free baseline (`kinematics_only`) trained as the floor
- [ ] (stretch) Published baselines (PCPA / SF-GRU / a transformer) retrained — **de-scopable v2**, the schedule killer (brief §7.1)
- [ ] (stretch) Replicate the core matrix on JAAD — single-dataset-validity hedge (brief §7.7)

---

## Stage 8 — Write-up, defense, release ⬜

- [ ] 💻 Reconcile the anchor docs to the streaming spine (RESEARCH_PLAN spine line, CLAUDE.md S1 line, HOLE_AUDIT imbalance-is-now-the-base-rate-finding note) — *deferred; do after the plan locks*
- [ ] 💻 Related-work survey (brief §9 reading list; know the nearest neighbors cold — Coupling-Intent, Diving-Deeper, LIM, GTransPDM)
- [ ] 💻 Draft: intro + the two-formulation background + the base-rate-fallacy primer
- [ ] 💻 Draft: methods (both samplers, the protocol, the metric suite, the decomposition math)
- [ ] 💻 Draft: results (the matrix, the G_prior/G_hardneg headline, the negative-composition figure, the H-sweep)
- [ ] 💻 Draft: pre-empt the reviewer objections (brief §7) — especially "just recalibrate" (answered by §4) and "just the TTE sweep" (answered by H-sweep)
- [ ] 💻 Discussion + limitations + the constructive close (protocol + recipe as community artifacts)
- [ ] 🖥️ Regenerate final figures/tables from the frozen final runs
- [ ] 💻 Release-quality git tag of the final codebase
- [ ] Defense slides + practice talk
- [ ] (optional) Workshop/conference submission — realistic venues: IV, ITSC, WACV, AD/safe-ML workshop (brief §7)

---

## Execution risks (thesis-level — see brief §7 for the full list)

1. **Published-baseline retraining is the schedule killer.** The decomposition on *our own model alone*
   is a defensible thesis + workshop paper; the multi-model matrix is the reach, explicitly de-scoped to
   Stage 7 stretch.
2. **G_hardneg depends on a streaming model that converges.** Mitigated by ordering: Stage 4 (G_prior)
   is already done and independent of a well-trained streaming model. If Stage 5 stays hard, the thesis
   reports "G is almost all hard-negative and the streaming leg is itself hard to train" — which is
   *still the finding*, not a collapse.
3. **Two-machine latency.** ~97% of coding is 💻; every 🖥️ item is a lab-PC batch. Front-load all the
   💻 analysis (Stages 4 + 6 metric code) so lab-PC runs are never the bottleneck for *thinking*.

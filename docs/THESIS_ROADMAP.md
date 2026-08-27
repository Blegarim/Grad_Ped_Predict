# Thesis Roadmap — Streaming Crossing-Onset Detection

**Prepared:** July 2026 · **Last swept:** 2026-08-20.

> **What this document is.** The single, findable, end-to-end checklist for the whole thesis — from the
> pre-migration prototype through to the defense — plus the supporting-study spokes at the bottom. It is
> the **tracker**. The *why* lives in two companions and is not repeated here:
> - **The argument** — [`project-context-streaming-crossing-onset.md`](project-context-streaming-crossing-onset.md): the thesis case, the dead ends already ruled out, the reviewer objections. Frozen reference.
> - **The method** — [`METHODOLOGY.md`](METHODOLOGY.md): onset timing under censoring, plus the two supporting parts and how they were chosen. The active working reference.
>
> The numbers live in [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md), which is authoritative for
> every figure — this file never restates a metric it does not own.

**Where the thesis stands (August 2026).** It is a **methods** thesis. The measurement — take a model
trained the standard event-anchored way, test it on a realistic continuous stream, watch its ability to
rank pedestrians by risk collapse to near-chance while threshold re-tuning recovers essentially nothing —
is *done* (Stage 4). But on its own that is a negative result: "the usual benchmark flatters models." So it
became the **motivation**, and the contribution is what follows from it: *a way to train for streaming
crossing-onset that the standard setup cannot produce.*

**The method has a spine (2026-08-20).** It is no longer three parallel prongs: the contribution is to
treat streaming crossing prediction as **onset timing under censoring** rather than yes/no classification
at a fixed horizon. What forced the change was an objection with no good answer inside the binary framing
— a pedestrian who walks normally and then turns abruptly is *unpredictable* a few seconds out, so part
of the confusing population is not hard but impossible, and any claim to "handle the hard negatives"
overclaims. Dropping the fixed cut-off removes that population by construction instead of fighting it.
Pose-movement features and the online-action-detection material stay, in support. Full reasoning, and the
measurements that back it, in [`METHODOLOGY.md`](METHODOLOGY.md).

Three things follow, and they govern how to read the stages below:

- **The centre of gravity is Stage 7**, not Stage 4. Critical path: **4 (done) → 6 → 7 → 8**.
- **The four existing runs are now baselines.** An accidental config difference between two legs used to be
  a footnote on a caveat; it now silently invalidates a method comparison. One such difference exists — see
  Stage 5.
- **The matched-size control is demoted** from headline-critical to an acknowledged caveat, because a
  motivating measurement is allowed to carry a stated confound.

---

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
| **3** | Streaming pivot: onset metadata + protocol switch + pose arm | 🟡 code done incl. onset plumbing + head + loss (2026-08-26) · awaiting the lab-PC backfill run |
| **4** | The decomposition (G_prior / G_hardneg) — **now the motivation, not the headline** | ✅ measured + written up ([RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md)) |
| **5** | Streaming-leg convergence + baseline hygiene | 🟡 demoted from headline — one real config mismatch found, must be fixed |
| **6** | Rare-event metrics + negative-composition report | 🟡 composition + horizon sweep ✅ done 2026-08-20 · metric suite still 💻 **NEXT** |
| **7** | **The method** — onset timing under censoring + supporting studies | 🟢 **the thesis now lives here** → [METHODOLOGY.md](METHODOLOGY.md) |
| **8** | Write-up, defense, release | ⬜ not started |

The critical path is now **6 → 7 → 8** (Stage 4 is done). Stage 6 comes first because it is entirely
laptop work and because it supplies the measuring instrument: F1 at a 1-in-37 base rate is too unstable to
tell whether a new method actually helped, so building the method before the metric means not being able to
read the result. Stage 5 no longer gates anything as a *finding*, but its baseline-hygiene half is now
load-bearing — the four existing runs are the comparison baselines, and one of them has a configuration
mismatch that would corrupt any comparison built on it.

**The two dependencies worth knowing before planning any lab visit:**
- The three onset fields never reach the database the trainer reads (Stage 3, last unchecked item; the gap
  is described in CLAUDE.md § Data Pipeline, S1 bullet). Since 2026-08-20 this blocks **the method itself**,
  not just two of four candidate directions — onset timing has nothing to train against until the fields
  arrive. The fix is small and mostly 💻. *(It does not block the negative-composition report, which is
  done: that reads the sequence pkls or PIE's annotation XMLs, both upstream of the packing step.)*
- Nothing in Stages 6–7 needs a *training* run to make progress. Metrics, features, and the negative
  census are all written and tested on the laptop; the lab machine only executes.

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

Two-pass audit ([`archive/HOLE_AUDIT.md`](archive/HOLE_AUDIT.md) — retired 2026-08-19, its attack order
fully executed) catalogued every validity/design/correctness issue and
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
- [x] 🖥️💻 Re-pin Dataset Statistics table + `count_labels` drift gate + `test_stats` fixture *(table re-pinned at regen; the golden fixture + its doc-sync test were only re-pinned 2026-08-19 — until then the gate failed on every run)*
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
- [x] 💻+🖥️ **Get the three onset fields into the trainer's reach.** Done 2026-08-26 except the lab-PC run.
      - [x] 💻 (a) `pack_meta` + read path + collate (the fields ride in `labels`, so no new Trainer wiring)
      - [x] 💻 (b) backfill script — `scripts/backfill_onset_meta.py`, verifies `track_id` + `crosses`
            per sample before writing, aborts on a mismatched pkl
      - [ ] 🖥️ (c) **run it** (metadata-only pass — images untouched). Stop any training job first:
            Windows refuses a write-open while a chunk is memory-mapped. `--dry-run` verifies safely.
      - [ ] 🖥️ (d) re-run `augment_dataset.py` so `preprocessed_train_aug` inherits the keys
            (augmented dirs are not backfillable — oversampling breaks the positional sample→record map)

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

## Stage 4 — ✅ THE DECOMPOSITION — done (and reclassified as the motivation)

**Closed 2026-07-22, audited 2026-08-18.**

> **The numbers live in [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md) and only there** — both
> models' 2×2 matrices, all three G-figures, the arithmetic, and the four caveats that must ship with them.
> This section used to carry a second copy; it was removed 2026-08-19 because two copies of a number drift.
> Update the matrix file after every eval pass; update this section only when a *checkbox* changes.

*Result, in one sentence:* **of the total anchored→streaming deployment gap, ~0% is recalibration-fixable
and ~100% is a genuine hard-negative skill gap the standard benchmark never trains.**

**Metric discipline (settled 2026-07-21, and the reason the arithmetic is shaped the way it is):** G_prior
and G_hardneg are **operating-point** quantities — measure them in **F1 (or precision-at-fixed-recall)**,
the currency recalibration acts on. **AUC is the *diagnostic*, not the measurement** — it explains *why*
recalibration recovers nothing (a discrimination collapse, not a mis-set dial), but it must never be quoted
*as* G_prior.

- [x] 💻 Compute all three G-figures on the 3-head matrix (Model A) under that discipline
- [x] 💻 Report the AUC diagnostic as the *explanation* of why G_prior ≈ 0 — a discrimination collapse recalibration can't touch
- [x] 💻 Write the decomposition up next to the runs → [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md), not in `docs/`
- [x] 💻 Second model added (crosses-only) — same pattern, larger G_hardneg
- [x] 💻 All four caveats stated in the matrix file, not buried: data-volume confound · F1 instability at 37:1 · Model B's sampler mismatch · single seed

*Why it is motivation rather than result:* it establishes that a real problem exists and that the cheap fix
(re-tuning the threshold) does not work. It does not offer a way forward. The thesis contribution is the way
forward — Stage 7.

---

## Stage 5 — Streaming-leg convergence + baseline hygiene 🟡

**Reclassified 2026-08-18.** This stage used to exist to bulletproof the gap's magnitude. Now that the
figure is motivation rather than result, the "prove the magnitude" half is optional. The other half got
*more* important, not less: **these four runs are the baselines every new method will be measured
against.** A baseline with an accidental configuration difference doesn't weaken a caveat — it makes the
comparison meaningless.

**One mismatch is open** — Model B's two halves trained with different weighted-sampler settings. It is
**described once, in [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md)** (register footnote † and
caveat 3), including which direction it might push the numbers and why that is genuinely unclear. Not
restated here; the actionable half is the checkbox below. Model A is unaffected and is the headline matrix.

Separately: the streaming leg still does not converge cleanly, which caps how good any streaming-trained
model can look — including new methods.

**Stabilize the streaming-trained model** (Phase C of the old plan; the G_hardneg ceiling)
- [ ] 🖥️ Diagnose the val_loss instability in `20260714_134253` (spikes to 4–8 on epochs 1/2/4/13/17; recall collapses to 1.0 = all-positive; best epoch 8 is a lucky trough, not a plateau)
- [ ] 🖥️ Stabilize: revisit LR / warmup / focal-or-class-balanced loss / sampler power under raw 37:1 (this *is* the "curated streaming training" recipe — see Stage 7)
- [ ] 🖥️ Confirm a streaming-trained model that plateaus cleanly (not a single lucky epoch)

**Complete the crosses-only 2×2** (removes the multi-task confound, adds a sampler one)
- [x] 🖥️ train=streaming × test=streaming — `20260714_134253` (AUC 0.78 / tuned-F1 0.22)
- [x] 🖥️ train=streaming × test=anchored — `20260714_134253` (AUC 0.68 / tuned-F1 0.26)
- [x] 🖥️ train=anchored × test={anchored,streaming} — `20260721_123011` (AUC 0.88 / 0.53), filled 2026-07-21
- [x] 💻 Recompute the decomposition on the crosses-only matrix → G_hardneg 0.162 vs 0.126, in [RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md)
- [ ] 🖥️ **Re-run the anchored half with the sampler ON** to match `134253` — the mismatch above. ~1.5 h, cheapest outstanding lab item

**Baseline hygiene (now load-bearing, since these are the comparison baselines)**
- [x] 💻 Audit all four `resolved_config.yaml` snapshots for unintended differences (done 2026-08-18 — found the sampler mismatch; everything else matches)
- [x] 💻 Add a config-diff step to the results-logging procedure so this is caught at fill time, not a month later ([RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md), "Adding a new run", step 3)
- [ ] 🖥️ Re-evaluate all four existing checkpoints with `--save-predictions --save-temporal-weights`. **Evaluation only, no training.** Unlocks a large amount of laptop-only follow-up: combining models offline, breaking errors down by how soon the crossing happens, and seeing which frame the model actually reacted to. None of the four runs saved these.
- [ ] 🖥️ (validity) Multi-seed the cells that end up in the write-up — screen with 1 seed, confirm with 3, report mean±spread

**Optional now (was headline-critical, demoted by the reframe)**
- [ ] 🖥️ Matched-*size* control: a streaming run subsampled to the anchored set's ≈4.9k windows, so training protocol is the only difference. Needs a small config knob (no cap on training-set size exists today). Cheap to run (~1/18th the epoch cost) if the knob is added.

---

## Stage 6 — 🟢 NEXT — rare-event metrics + negative composition

Net-new, and almost entirely 💻: the metrics can be written and tested on made-up streams, with no data and
no GPU, before anything is applied on the lab machine.

**Why this comes before building the method.** F1 at a 1-in-37 base rate moves a lot when only a handful of
samples cross the threshold. If the method is built first, there is no reliable way to tell whether it
helped — the measurement noise is comparable to the effect being looked for. Building the instrument first
makes every later comparison readable. The online-action-detection literature made the same move for the
same reason, which is also where these metrics come from.

The negative-composition half answers a question that sizes the whole method effort: **how much of the
1-in-37 imbalance is actually the hard case?** "Not crossing" currently lumps together someone who never
crosses, someone who crosses in four seconds, and someone who has already finished crossing. Counting them
is cheap and decides how much is on the table.

**Metric suite (import from OAD/ODAS — brief §2.4)**
- [ ] 💻 Per-frame mAP over the stream
- [ ] 💻 **Calibrated AP (cAP / mcAP)** — the OAD metric built for heavy-background imbalance
- [ ] 💻 **Point-level AP (p-AP)** with temporal-offset tolerance (onset-timing quality; needs S1 onset ground truth)
- [ ] 💻 Precision-at-fixed-recall / AP as the stable G_prior/G_hardneg currency (feeds back into Stage 4)
- [ ] 💻 Unit tests on synthetic streams (no data)
- [ ] 🖥️ Apply the suite across the matrix cells

**Negative-composition report (brief §2.3, from S1)** — ✅ **done 2026-08-20**, and it landed on the
laptop rather than the lab PC: PIE's behaviour tags are a ~25 MB XML set, so the whole label contract can
be re-derived without frames, LMDBs or a GPU. Numbers in [METHODOLOGY.md](METHODOLOGY.md) § What the
negatives are actually made of.
- [x] 💻 Classify every streaming negative into: genuine non-crosser / hard-temporal / already-crossed (from `track_crosses` + `onset_offset`) — `data/onset_stats.py`
- [x] 💻 Report the composition — **hard-temporal is 25.7% of train / 39.2% of test windows**, i.e. 8.9× and 12.8× the positives; the direction is confirmed
- [x] 💻 Bucket it by time-to-onset — two-thirds of that mass sits >5 s out; the genuinely confusable 1–3 s band is ~1.7× the positive set
- [x] 💻 **Horizon sweep** (was filed under Stage 7) — what a longer horizon buys and costs; justifies keeping ~1 s
- [x] 💻 CLI + unit tests + a laptop-side drift gate on the golden counts (`scripts/report_negative_composition.py`, `tests/test_onset_stats.py`)
- [ ] 💻 Pick + implement a **junk** signal (bbox height below threshold / occlusion) — the one composition axis still unmeasured
- [ ] 🖥️ **Confirm residual failures concentrate in the will-cross-soon windows** (validates the hard-negative story — brief §5 supporting analysis a)
- [ ] 🖥️ **The ceiling measurement** — separability stratified by time-to-onset; the distance at which prediction decays to chance. New, and it is what keeps the claim honest.

**Train/test asymmetry found while doing this.** The two splits differ structurally in what their
negatives are made of (25.7% vs 39.2% hard-temporal; 64.3% vs 53.4% never-crossers), so a train-to-test
difference is not purely a generalisation gap. Caveat recorded in [RESULTS_MATRIX.md](../outputs/runs/RESULTS_MATRIX.md).

---

## Stage 7 — 🟢 THE METHOD (was "constructive close") + supporting studies

**This is where the thesis contribution now lives.** It used to be the optional upgrade; after the August
reframe it is the centre. The detailed direction — the method and its two supporting parts, what has been tried in the literature, what
transfers, the open decision points — is in **[`METHODOLOGY.md`](METHODOLOGY.md)**, which is the working
reference. Kept here in summary so the tracker stays complete:

1. **Onset timing under censoring — the method.** Get the three onset fields into the trainer's reach,
   then predict *when* the crossing starts, treating a window whose future ran out as a censored
   observation rather than a negative. This removes the confusing case by construction, recovers the
   windows M4 currently discards, and lets any horizon be read off afterwards. 💻 plumbing + a small lab
   pass. **Hard constraint: it must still emit the probability of onset within 32 frames**, or the four
   baseline runs stop being comparable.
2. **Pose-motion features** — turn the raw keypoints into features that describe *movement*, not just
   posture. All current pose features are single-frame. 💻, no re-extraction needed. This is what gives a
   timing model something to read at a one-second horizon.
3. **Online-action-detection material** — that community named this exact failure in 2018, and solved the
   measurement problem for it. Its metrics are the instrument (Stage 6); its objectives are available
   where they fit; its obvious alternatives (focal, class-balanced) are the comparison baselines a methods
   thesis has to run rather than dismiss.

**Onset-timing build order** (supersedes the pre-reframe recipe list below)
- [x] 💻 S1 fields into `pack_meta` + the read path + collate
- [x] 💻 Backfill script over the existing LMDBs — 🖥️ **still to run**
- [x] 💻 Timing output + a censoring-aware loss (`model.onset_head`, `losses/onset.py`; default off).
      Contracts and the three-arm weight table live in **[CLAUDE.md](../CLAUDE.md) § Onset Timing**.
- [x] 💻 Conversion back to the baseline question — `hazard_to_horizon_logits`, pinned against the
      generator's own `crosses` label so the four baselines stay comparable
- [ ] 🖥️ Short smoke run to check the head does not collapse to `h ~ 0` (the predicted failure mode)
- [ ] 🖥️ Auxiliary-arm run first (reported number still from `crosses_frame` = baselines untouched),
      then the pure-reformulation arm
- [ ] 💻+🖥️ Censored windows restored — needs a **regen** with the M4 filter relaxed, NOT the backfill
      (`window_track` skips them at generation, so they were never in the pkls). Separate experiment.
- [ ] 🖥️ Restore censored windows to training — a data-quantity change, measured separately from the objective change
- [ ] 💻 Two literature checks before any novelty wording: onset time as a training signal, and censoring-aware modelling, both specifically for crossing prediction

**Curated streaming recipe (narrows G_hardneg — brief Phase 5).** Pre-reframe; still valid work, now
comparison baselines rather than the plan.
- [ ] 🖥️ Pretrained backbone (RQ1) + focal/class-balanced loss + hard-negative emphasis + junk filtering
- [ ] 🖥️ Show the recipe measurably narrows G_hardneg vs the naive streaming-trained model
- [x] 💻 **H-sweep** — done 2026-08-20 as a label-side study (no training needed); rebuts objection #2 and justifies keeping ~1 s. The *trained* version (does difficulty actually move with H?) is still open.
- [ ] 🖥️ Ego-speed-under-streaming leakage probe (RQ4) — does streaming lean *harder* on the shortcut?

---

## Supporting studies (the RQ spokes)

*Absorbed from the retired `RESEARCH_PLAN.md` (2026-08-19, now at
[`archive/RESEARCH_PLAN.md`](archive/RESEARCH_PLAN.md)) — that document's engineering foundation is
complete and its thesis narrative was superseded twice, so only this list survives.*

These were the thesis before the July pivot; they are now **single-axis side studies** that support the
method rather than structure the argument. **Experimental design: hub-and-spoke, never factorial.** One
frozen hub baseline; every spoke changes exactly **one** axis and is compared straight back to the hub,
never against each other. Screen at 1 seed, confirm finalists at 3, report mean±spread.

| RQ | Question | State |
|---|---|---|
| **RQ1** Visual backbone | Does a modern pretrained hierarchical backbone beat the from-scratch ViT at a matched param/FLOP budget? Motivation: the v1 ViT collapsed 288→36 dims before `frame_proj`, spent its attention FLOPs on 2×2 windows, and had never seen data outside PIE. | [~] TinyViT-5M drop-in landed + trained (frozen); PVTv2-B0 and the rest optional. Desk study: [BACKBONE_STUDY.md](BACKBONE_STUDY.md) |
| **RQ2** Fusion | Is motion-as-saliency the right design? The cross-attention originally had **no residual**, so motion *content* never reached the heads — only motion-shaped attention over image values. | [x] `fusion_residual` default-on; the no-residual arm is golden-pinned and ablatable |
| **RQ3** Imbalance levers | Which of the three stacked levers (offline aug, weighted sampler, class-weighted loss) actually earns its place, measured against the effective-distribution instrument? | [~] Folded into the curated streaming recipe above — the lever combination *is* that recipe |
| **RQ4** Input representation | Do corrected, image-normalized motion features help over per-sequence z-norm? And does the **ego-speed** channel help — or is it a shortcut? (The driver brakes *because* the pedestrian will cross, so its on/off ablation doubles as a leakage probe. Cheapest scientific finding available.) | [x] motion-norm ablatable from one dataset · [ ] ego-speed on/off untested |
| **RQ5** Efficiency | Is the full multimodal model worth its cost — params, FLOPs, latency, FPS, peak VRAM per model type? | [ ] Harness exists (fvcore + ONNX parity), never run across the ladder |
| **RQ6** Calibration | Are the crossing probabilities *honest*, and what is the single canonical operating point? Reliability diagrams → temperature scaling on val → one threshold shared by eval **and** `infer_video`. The bridge to a controller-consumable probability. | [ ] Calibration script scaffolded; **note the pivot reframed this** — val-tuned recalibration is now `G_prior`, and it measured ≈0 |

**Remaining spoke work**
- [ ] 🖥️ RQ4 ego-speed on/off — accuracy effect + leakage reading
- [ ] 🖥️ RQ5 efficiency sweep — params/FLOPs/latency/FPS/peak-VRAM per model type
- [ ] 🖥️ RQ6 calibration — reliability diagrams + temperature scaling on val
- [ ] 🖥️ Kinematics-only pixel-free baseline (`kinematics_only`) trained as the floor
- [ ] 💻 Single unified crosses head — collapse `crosses_frame` + the live-but-unsupervised `crosses_pooled` into one supervised head, retiring the dual-head contract *(absorbed from the retired PHASE_B_BACKLOG)*
- [ ] 💻 Variable-length sequences — drop the fixed `seq_len=20` truncate-no-pad policy; add padding + masking *(same source; not an audit hole, still live)*
- [ ] (stretch) Published baselines (PCPA / SF-GRU / a transformer) retrained — **de-scopable v2**, the schedule killer (brief §7.1)
- [ ] (stretch) Replicate the core matrix on JAAD — single-dataset-validity hedge (brief §7.7)

---

## Stage 8 — Write-up, defense, release ⬜

- [x] 💻 Reconcile the anchor docs to the streaming spine — done 2026-08-19 in the doc-consolidation pass (RESEARCH_PLAN / CHANGELOG / PROPOSAL / PHASE_B_BACKLOG / HOLE_AUDIT retired to `archive/`; their live content absorbed here and in CLAUDE.md)
- [ ] 💻 Related-work survey (brief §9 reading list; know the nearest neighbors cold — Coupling-Intent, Diving-Deeper, LIM, GTransPDM)
- [ ] 💻 Draft: intro + the two-formulation background + the base-rate-fallacy primer
- [ ] 💻 Draft: **motivation** — the two protocols, the decomposition, and why re-tuning the threshold cannot fix it (this is the old "results headline", moved forward to justify the method)
- [ ] 💻 Draft: methods (both samplers, the protocol, the metric suite, the decomposition math, **and the onset-timing method from [METHODOLOGY.md](METHODOLOGY.md)**)
- [ ] 💻 Draft: results (the method measured against the four baseline runs, the negative-composition figure, the H-sweep)
- [ ] 💻 Draft: pre-empt the reviewer objections (brief §7) — especially "just recalibrate" (answered by §4) and "just the TTE sweep" (answered by H-sweep)
- [ ] 💻 Discussion + limitations + the constructive close (protocol + recipe as community artifacts)
- [ ] 🖥️ Regenerate final figures/tables from the frozen final runs
- [ ] 💻 Release-quality git tag of the final codebase
- [ ] Defense slides + practice talk
- [ ] (optional) Workshop/conference submission — realistic venues: IV, ITSC, WACV, AD/safe-ML workshop (brief §7)

---

## Execution risks (thesis-level — see brief §7 for the full list)

*(Rewritten 2026-08-18 for the reframe — the old list assumed the decomposition was the result.)*

1. **The method might not work.** This is the real risk now, and it is new. Before the reframe, the thesis
   rested on a measurement that was already taken; now it rests on making something better. Mitigation is in
   the ordering: build the measuring instrument (Stage 6) before the method, so a null result is *readable*
   and reportable — "these three interventions, measured properly, did not close the gap, and here is what
   that rules out" is a defensible thesis, but only if the measurements are trustworthy. Also prefer the
   prongs with independent value: better pose features and the negative-composition census are useful
   findings whether or not the headline method succeeds.
2. **The streaming leg does not converge cleanly.** It caps how good *any* streaming-trained model can look,
   new methods included, and it costs ~21 hours per attempt (67 min/epoch versus 3 for anchored). This is
   both a risk and, framed differently, part of the contribution — "training under a realistic base rate is
   itself unstable" is a finding worth reporting, and the stabilisation recipe is a deliverable.
3. **Published-baseline retraining remains the schedule killer.** Comparing against other papers' models
   requires retraining them. Still explicitly de-scoped; the method is measured against our own four runs.
4. **Two-machine latency.** ~97% of coding is 💻; every 🖥️ item is a lab batch. Front-load laptop work so the
   lab machine only ever executes and never blocks *thinking*. Concretely: write the check scripts, the
   metrics, and the feature math before travelling, and bundle the lab passes into one ordered visit.
5. **Baseline drift.** With the four runs now serving as baselines, any un-audited configuration difference
   silently invalidates a comparison. Mitigated by the config-diff step now written into the results
   procedure — but it only works if it is actually run at fill time.

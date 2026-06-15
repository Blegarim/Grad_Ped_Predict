# Research Plan — Multimodal Pedestrian Behavior Prediction

**Student:** Nguyen Bao Viet
**Program:** Master's (graduate research)
**Date:** June 2026
**Repository:** `Grad_Ped_Predict` (private, v1.0 baseline)

> This plan is the thesis-level companion to the resolved engineering audit,
> [`HOLE_AUDIT.md`](HOLE_AUDIT.md). The audit catalogues every methodological, architectural, and
> correctness issue in the v1.0 codebase and records a final decision for each; this document turns
> those decisions into a research narrative, work packages, and a schedule. Where the two touch, the
> audit's **Final attack order** is the authoritative execution sequence and this plan follows it.

---

## 1. Project Summary

Pedestrian behavior prediction is a safety-critical capability for autonomous and assisted driving:
a vehicle must anticipate, seconds in advance, whether a pedestrian at the curb will step into the
road. This project develops and studies a **multimodal deep-learning model** that, from a short
sequence of dashcam frames, jointly predicts three binary behaviors per pedestrian:

- **actions** — walking vs. standing,
- **looks** — looking at traffic or not,
- **crosses** — will the pedestrian cross in the near future (the primary, safety-critical task).

The work is grounded in the **PIE dataset** (Pedestrian Intention Estimation; Rasouli et al.,
ICCV 2019), the standard large-scale benchmark for this task. The observation/prediction protocol is
a deliberate **early-anticipation** choice — 20-frame (~0.67 s) observation windows with a ~1 s
prediction horizon — motivated by two safety arguments: pedestrians can appear suddenly, so the model
must commit on short evidence, and a crossing decision must hold for the immediate future to be
actionable. This protocol differs from the published PIE/JAAD time-to-event benchmark; the difference
is made explicit and an externally-anchored benchmark comparison is added (see RQ-P and WP1).

The model is a two-stream architecture fused by cross-attention:

```
context crop frames → hierarchical ViT (visual stream)  ──┐
                                                           ├→ cross-attention fusion → {actions, looks, crosses}
tight crop + bbox motion → temporal CNN+GRU (motion stream)┘
```

A visual stream (hierarchical windowed-attention Vision Transformer over wide "context" crops)
captures scene appearance; a motion stream (temporal CNN + GRU over tight pedestrian crops and
bounding-box kinematics) captures dynamics; a cross-attention module (motion as query, vision as
key/value) fuses them, with learned temporal pooling and per-task classifier heads. Modality and
fusion ablations (`ped_local`, `visual_only`, `vanilla_concat`) plus a planned pixel-free
kinematics-only baseline isolate the contribution of each component.

### Starting point

The project extends an undergraduate thesis prototype. **Phase A — a behavior-preserving rebuild —
is complete**: the prototype has been re-engineered into a clean, tested, config-driven PyTorch
codebase (v1.0), with a reproducible data pipeline (PIE → sequence windows → LMDB store), golden-output
tests proving numerical equivalence with the original at module level, a unified multi-task loss and
imbalance stack, CSV-based experiment tracking, and ONNX export with parity checks.

**Phase B is underway.** A full two-pass code audit ([`HOLE_AUDIT.md`](HOLE_AUDIT.md)) catalogued and
resolved every validity-, design-, and correctness-level issue in v1.0, with a fixed execution order.
The first two batches are **implemented and merged to `main`**:

- *Batch 1 — data-independent code fixes:* reproducible seeding, F1-based model selection, val-tuned
  decision thresholds (test-set leakage removed), an imbalance-distribution instrument, and a set of
  latent-bug fixes.
- *Batch 2 — the v2 data-contract code:* the relabeling, censoring, metadata, motion-representation,
  ego-speed, and benchmark-protocol changes that the dataset rebuild depends on (see WP0). This is the
  pipeline *code*; the one-time data regeneration and statistics re-pin it enables is the remaining
  WP0 gate.

This engineering foundation is what makes the research phase feasible: every proposed change can be
ablated against a trusted, reproducible, leakage-free baseline.

---

## 2. Problem Statement and Research Questions

The audit surfaced one finding that frames the whole research program. **Three issues form a chain:**
the default imbalance stack trains the model on a `crosses` distribution of roughly 50–85% positive
(versus 2.8% at deployment), guaranteeing severe miscalibration and over-prediction; the evaluation
pipeline then *tuned the decision threshold on the test set*, silently compensating for that
over-correction. The consequence is that the prototype's headline numbers were both optimistic and
self-obscuring. Fixing this chain honestly — measure the training distribution, decide the imbalance
policy on evidence, tune thresholds on validation, and calibrate the probabilities — is the spine of
the thesis, and it connects directly to a downstream control-systems framing in which a *calibrated*
crossing probability is the quantity a planner consumes.

The research questions treat the prototype's weaknesses as open questions rather than mere defects:

- **RQ1 (Visual backbone).** The from-scratch hierarchical ViT exhibits a self-defeating stage
  schedule (features pass through 288 dims, then are crushed to 36 before projection to `d_model`) and
  spends most of its attention FLOPs on 2×2 windows, while never having seen data outside ~96k PIE
  crops. Does a modern, pretrained, hierarchical/windowed backbone (TinyViT-5M / PVTv2-B0 /
  MobileViT-S / FastViT-T8 / DeiT-Tiny) improve accuracy at a matched parameter/FLOP budget, as a
  pooled-features → `frame_proj` → 128 drop-in?
- **RQ2 (Multimodal fusion).** The cross-attention "fusion" has **no residual**, so the classifier
  sees only image-derived values reweighted by motion–image affinity — motion content never reaches
  the heads (only `vanilla_concat` passes motion content directly). Is motion-as-saliency the right
  design? How do a motion residual, query/key role swaps, bidirectional/concat variants, and fusion
  depth compare?
- **RQ3 (Severe class imbalance).** `crosses` is ~37:1 imbalanced. The pipeline stacks three levers
  (offline augmentation, weighted online sampling, class-weighted loss) into an extreme, previously
  *unmeasured* training distribution. Which lever combination is actually effective and principled,
  measured against the now-instrumented effective distribution and evaluated without test-set leakage?
- **RQ4 (Input representation).** Per-sequence z-normalization of motion erases absolute geometry
  (curb proximity, box-size-as-distance), amplifies pixel-quantization jitter, and a frame-0 quirk
  effectively deletes two of eight channels. Do corrected, image-dimension-normalized motion features —
  plus an **ego-vehicle speed** channel from PIE's OBD data — measurably help? (Ego-speed is also a
  known causal confound: the driver brakes *because* the pedestrian will cross — making its on/off
  ablation the cheapest scientific finding available, a leakage probe.)
- **RQ5 (Efficiency).** What accuracy/efficiency trade-off (parameters, FLOPs, latency, FPS, peak
  VRAM) does each design choice impose, and is the full multimodal model justified over its ablations
  for deployment-style settings?
- **RQ6 (Calibration & uncertainty).** Are the model's crossing probabilities *honest*, and what is
  the single canonical operating-point policy? This covers reliability diagrams, temperature scaling
  on validation, a val-tuned decision threshold consumed by **both** eval and video inference, and —
  if the control framing proceeds — conformal prediction. This is the bridge from classification
  metrics to a quantity a downstream controller can trust.

A cross-cutting **protocol-hygiene question (RQ-P)** underlies all of the above: are the labels,
windows, metrics, and evaluation split defined correctly and comparably? It is not a contribution in
itself but a precondition for every comparison being meaningful, and is resolved once, before any
baseline, in the v2 dataset rebuild.

---

## 3. Scope

**In scope**
- Training and evaluating the v1.0 baseline and all redesigned variants on PIE (published
  train/val/test splits), for all three tasks, with `crosses` as the primary task.
- A single corrected **v2 dataset rebuild** (sequences + LMDBs, all three splits) that fixes
  labeling, censoring, motion representation, and metadata in one pass.
- Architecture research: visual-backbone swap (RQ1) and fusion redesign (RQ2).
- Data-layer research: corrected motion features, ego-speed, normalization choice (RQ4); imbalance
  policy decided by ablation (RQ3).
- Honest evaluation under imbalance (F1/AUC/recall-led), calibration, and a canonical operating
  point (RQ6); efficiency benchmark per variant (RQ5).
- An externally-anchored benchmark-protocol evaluation row (test-split only) for literature
  comparability, alongside the project's own early-anticipation protocol.
- Engineering hygiene that protects validity: reproducible seeding, F1-based model selection,
  val-tuned thresholds, golden re-baselining per change, track-aggregated metrics, and a cross-run
  index.

**Out of scope (for this thesis)**
- Trajectory forecasting (continuous future positions) — classification only.
- Datasets beyond PIE (e.g., JAAD) — stretch-goal generalization check only.
- Real-time in-vehicle deployment; ONNX export + latency serve as a deployment proxy.
- Detection/tracking research — raw-video inference uses an off-the-shelf detector (YOLO) as a fixed
  front-end.
- Resuming the v1 training-LMDB build — explicitly **abandoned** (see WP0); the v2 rebuild supersedes
  all v1 dataset artifacts.

---

## 4. Work Plan

Five work packages. WP0 establishes a trustworthy baseline; WP1–WP2 carry the research
contributions as single-axis ablations; WP3 is calibration and consolidated evaluation; WP4 is
writing. The execution order follows the audit's **Final attack order**.

The governing experimental design is **hub-and-spoke, never factorial.** The hub is the WP0 `full`
baseline under a frozen protocol (v2 dataset, fixed seed, val-tuned thresholds, F1 selection metric).
Every ablation changes exactly **one** axis and is compared back to the hub — axes are never
cross-compared. Budget: a ~6-run WP0 ladder, then ~10–12 single-axis spokes, with one seed for
screening and three seeds only for the 3–4 headline comparisons — **~20–25 trainings total**, feasible
on the available A4500. If two single-axis wins look compoundable, exactly one combined v2 candidate
is tested; the grid is never searched.

### WP0 — Clean baseline: the hub (months 1–3)

1. **Data-independent code fixes — DONE (merged).** Reproducible seeding (`train.seed`), F1-based
   model selection (`train.selection_metric`), **val-tuned decision thresholds** (test-set leakage
   removed; test-swept numbers renamed `oracle_*` and never reported), an effective-distribution
   **instrument** logged into every run, the `train.use_class_weights` switch, and latent-bug fixes.
2. **v2 data-contract code — DONE (merged).** The pipeline changes that the rebuild depends on, batched
   so the dataset is touched exactly once (the v1 build is *abandoned*, the ~20k v1 train chunks
   discarded, and the runtime dataset hard-errors on v1 chunks):
   - relabel `actions`/`looks` as state-at-end-of-observation (`crosses` stays future-any);
   - drop right-censored windows (unobserved futures, previously silently labeled 0), counted per split;
   - carry a `track_id` through `SequenceRecord` → LMDB meta → dataset items (enables track aggregation);
   - motion-v2 (9-dim: frame-0 deltas = 0, flip reflects absolute cx, **ego-speed** channel, stored
     wide and sliced to `motion_dim`; `model.motion_norm` is a runtime flag — `image` default vs.
     legacy `per_sequence`, so old-vs-new normalization ablates from one dataset);
   - a benchmark-protocol eval mode (`make_sequences.py --benchmark`, test-split only) for literature
     comparison.
3. **Run the rebuild + re-pin — the remaining WP0 gate.** Execute the regeneration on real PIE data
   (per [`V2_REBUILD_RUNBOOK.md`](V2_REBUILD_RUNBOOK.md)): regenerate sequences + build LMDBs for all
   three splits + the benchmark set; then doc-sync the now-STALE numbers — re-pin the Dataset
   Statistics table (recording the M3 class-rate shift and the M4 censored-window count as
   thesis-reportable figures), restore the `count_labels` drift gate to exact counts, update the
   `test_stats` fixture, and re-baseline goldens for the changed label/motion behavior.
4. **The hub ladder (~6 runs).** Registry/rename wave first (add a pixel-free **kinematics-only**
   model; rename `motion_only` → `ped_local`; single-task config via zeroed loss weights *and* sampler
   powers), then train under the frozen protocol with the distribution instrument logging into every
   run: kinematics-only, `ped_local`, `visual_only`, `vanilla_concat`, `full`, and crosses-only.
- *Deliverable:* the frozen reference table + checkpoints every later result is measured against.

### WP1 — Data, imbalance, and protocol spokes (months 3–6, RQ3, RQ4, RQ-P)
- **Imbalance lever ablation (RQ3):** none / weights-only / sampler-only / aug-only / current stack —
  each read against the effective-distribution instrument and against the hub.
- **Motion normalization (RQ4):** per-sequence z-norm vs. fixed image-dimension norm, ablatable from
  the same v2 data (the choice is a runtime flag).
- **Ego-speed on/off (RQ4):** accuracy effect *and* the leakage-probe reading.
- **Benchmark-protocol row (RQ-P):** report the trained model on the TTE-sampled benchmark set,
  caveated, as the one externally-anchored comparison.
- *Deliverable:* ablation report quantifying each data-layer change on `crosses` F1/AUC, plus the
  imbalance-policy recommendation.

### WP2 — Architecture spokes (months 5–8, RQ1, RQ2)
- **Backbone study (RQ1):** a desk study ranking TinyViT-5M / PVTv2-B0 / MobileViT-S / FastViT-T8 /
  DeiT-Tiny on timm pretrained availability, resolution fit, hierarchical/windowed structure,
  params/FLOPs at matched budget, A4500 latency, and drop-in cleanliness → a design note naming a
  primary + fallback → the swap, compared to the hub. The v1 ViT's 36→288→36 collapse and 2×2 windows
  are the written motivation.
- **Fusion grid (RQ2):** the flagged motion residual (`model.fusion_residual`), query/key swap,
  bidirectional + concat, against `vanilla_concat` — each a single-axis spoke. If `vanilla_concat`
  matches or beats `full`, the no-residual finding is the explanation, a publishable observation.
- *Deliverable:* redesigned v2 architecture with a full comparison table against the hub.

### WP3 — Calibration and consolidated evaluation (months 7–10, RQ5, RQ6)
- **Calibration workstream (RQ6):** reliability diagrams from the predictions NPZ → temperature
  scaling on val → the single canonical operating-point policy (val-tuned threshold stored in the run
  dir, consumed by eval **and** `infer_video`) → conformal prediction only if the control framing
  proceeds.
- **Consolidation:** final runs of the selected v2 configuration and headline ablations at three
  seeds (mean±std); efficiency benchmark per variant (params, FLOPs, latency, FPS, peak VRAM) with the
  ONNX parity check as deployment proxy; track-aggregated metrics alongside window metrics.
- **Qualitative:** temporal-attention visualizations; `crosses` false-negative failure study (the
  safety-critical error mode).
- *Deliverable:* complete experimental-chapter material.

### WP4 — Thesis writing and dissemination (months 10–12)
- Thesis writing, integrating the design notes and ablation reports produced per work package.
- Optional workshop/conference paper submission if results warrant.
- Release-quality tag of the final codebase.
- *Deliverable:* thesis manuscript; optional paper draft.

### Timeline overview

| Month → | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WP0 Baseline (rebuild + hub) | ■ | ■ | ■ | | | | | | | | | |
| WP1 Data / imbalance / protocol | | | ■ | ■ | ■ | ■ | | | | | | |
| WP2 Architecture | | | | | ■ | ■ | ■ | ■ | | | | |
| WP3 Calibration / evaluation | | | | | | | ■ | ■ | ■ | ■ | | |
| WP4 Writing | | | | | | | | | | ■ | ■ | ■ |

---

## 5. Evaluation Methodology

- **Metrics:** Accuracy, F1, AUC, Precision, Recall per task plus macro-F1. Because `crosses` is
  ~37:1 imbalanced, aggregate accuracy is explicitly uninformative for it; F1/AUC/recall lead all
  comparisons. A single shared metric accumulator guarantees train-validation and test metrics cannot
  diverge in implementation. **Track-aggregated** metrics (predictions grouped by `track_id`, mean
  probability per track) are reported alongside window metrics, because stride-3 windows of one
  pedestrian overlap ~85% and the nominal 76k test "samples" behave like far fewer independent ones.
- **Decision thresholds (non-negotiable):** tuned on **val**, frozen, applied to test. The
  `tuned_*` columns are the only reportable threshold-tuned numbers; the same-split sweep is logged as
  `oracle_*` (test-set leakage, diagnosis only, never quoted).
- **Imbalance transparency:** the effective per-task sampler-draw distribution is measured and logged
  into every run, so no lever combination is ever toggled blind.
- **Calibration:** reliability diagrams and temperature scaling on val report whether probabilities
  are honest, independent of any threshold choice.
- **Efficiency:** parameters, FLOPs (fvcore), latency, FPS, peak VRAM per variant.
- **Protocol:** PIE's published splits; the project's early-anticipation windowing as the primary
  protocol, with a caveated benchmark-protocol row for external comparability; fixed seeds with
  three-seed final runs (mean±std); config-snapshot per run; CSV logs + cross-run index; a data-drift
  gate (`count_labels`) pinning dataset statistics across pipeline changes.
- **Validity safeguards:** every behavioral change lands behind a config flag with a golden-output
  re-baseline, so regressions are attributable to a single change rather than accumulated drift; the
  hub-and-spoke design keeps every comparison single-axis.

## 6. Expected Contributions

1. A systematic, ablation-backed redesign of a multimodal pedestrian-behavior model on PIE, with each
   design decision (backbone, fusion, input representation) empirically justified against a frozen,
   leakage-free baseline.
2. An **evidence-based imbalance-handling recipe** for severely skewed crossing prediction (~37:1),
   clarifying which of the commonly stacked levers actually contribute — backed by a measured training
   distribution and honest, val-tuned evaluation.
3. A **calibration and operating-point study** for crossing prediction: reliability analysis,
   temperature scaling, and a single canonical threshold policy shared by evaluation and inference —
   the bridge toward a controller-consumable crossing probability.
4. An accuracy/efficiency characterization across modality and architecture ablations, informing
   whether multimodal fusion (and each of its parts) earns its cost.
5. A reproducible, tested, open-style research codebase (config-driven, golden-tested,
   ONNX-exportable, with an effective-distribution instrument) usable by subsequent students.

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPU/compute limits multi-seed runs | Medium | Hub-and-spoke (~20–25 runs, not factorial); single-seed screening, three-seed only for 3–4 finalists; `crosses` prioritized |
| Redesigned components fail to beat the hub | Medium | Negative results are reported as findings; v1.0 remains a complete fallback baseline |
| v2 dataset rebuild costs (disk/IO) | Low–Medium | One batched rebuild (not iterative); resume-guard + map_size fixes already landed; v1 build abandoned rather than finished |
| Imbalance/calibration findings prove inconclusive | Low | The instrument + leakage fix make even a null result a clean, reportable methodological contribution |
| Scope creep into trajectory prediction / new datasets | Medium | Section 3 is the contract; JAAD is stretch-only |
| Timeline slip in WP2 | Medium | WP0+WP1 (clean baseline + data/imbalance/calibration) alone support a defensible thesis; WP2 items are independently land-able |

## 8. Key References

- Rasouli, A., Kotseruba, I., Kunic, T., Tsotsos, J. K. *PIE: A Large-Scale Dataset and Models for
  Pedestrian Intention Estimation and Trajectory Prediction.* ICCV 2019.
- Kotseruba, I., Rasouli, A., Tsotsos, J. K. *Benchmark for Evaluating Pedestrian Action
  Prediction.* WACV 2021.
- Rasouli, A., Kotseruba, I., Tsotsos, J. K. *Are They Going to Cross? A Benchmark Dataset and
  Baseline for Pedestrian Crosswalk Behavior.* ICCV Workshops 2017 (JAAD).
- Dosovitskiy, A., et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at
  Scale.* ICLR 2021.
- Liu, Z., et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*
  ICCV 2021.
- Wu, K., et al. *TinyViT: Fast Pretraining Distillation for Small Vision Transformers.* ECCV 2022.
- Lin, T.-Y., et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks.*
  ICML 2017.
- Angelopoulos, A. N., Bates, S. *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification.* 2021.

*(Reference list to be expanded with a full related-work survey during WP1.)*

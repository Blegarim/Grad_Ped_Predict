# Research Plan — Multimodal Pedestrian Behavior Prediction

**Student:** Nguyen Bao Viet
**Program:** Master's (graduate research)
**Date:** June 2026
**Repository:** `Grad_Ped_Predict` (private, v1.0 baseline)

---

## 1. Project Summary

Pedestrian behavior prediction is a safety-critical capability for autonomous and assisted driving:
a vehicle must anticipate, seconds in advance, whether a pedestrian at the curb will step into the
road. This project develops and studies a **multimodal deep-learning model** that, from a short
sequence of dashcam frames, jointly predicts three binary behaviors per pedestrian:

- **actions** — walking vs. standing,
- **looks** — looking at traffic or not,
- **crosses** — will the pedestrian cross in the near future.

The work is grounded in the **PIE dataset** (Pedestrian Intention Estimation; Rasouli et al.,
ICCV 2019), the standard large-scale benchmark for this task, with ~194k labelled sequence windows
after preprocessing.

The model is a two-stream architecture fused by cross-attention:

```
context crop frames → hierarchical ViT (visual stream)  ──┐
                                                           ├→ cross-attention fusion → {actions, looks, crosses}
tight crop + bbox motion → temporal CNN+GRU (motion stream)┘
```

A visual stream (hierarchical windowed-attention Vision Transformer over wide "context" crops)
captures scene appearance; a motion stream (temporal CNN + GRU over tight pedestrian crops and
8-dimensional bounding-box kinematics) captures dynamics; a cross-attention module (motion as
query, vision as key/value) fuses them, with learned temporal pooling and per-task classifier
heads. Three ablation variants (motion-only, visual-only, and concatenation-fusion) are built into
the codebase to isolate the contribution of each component.

### Starting point

The project extends an undergraduate thesis prototype. **Phase A — a behavior-preserving rebuild —
is complete**: the prototype has been re-engineered into a clean, tested, config-driven PyTorch
codebase (v1.0), with a reproducible data pipeline (PIE → sequence windows → LMDB store), unit and
golden-output tests proving numerical equivalence with the original at module level, unified
multi-task loss and imbalance handling, CSV-based experiment tracking, and ONNX export with parity
checks. This engineering foundation is what makes the research phase below feasible: every proposed
change can be ablated against a trusted, reproducible baseline.

---

## 2. Problem Statement and Research Questions

The prototype demonstrated feasibility but carries known architectural and methodological
weaknesses, catalogued during the rebuild. The master's research treats these as open questions
rather than mere defects:

- **RQ1 (Visual backbone).** The current hierarchical ViT uses resolution-bound relative-position
  encodings, locking the model to one input resolution. Does a modern, resolution-agnostic backbone
  (or rel-pos scheme) improve accuracy and/or flexibility at comparable cost?
- **RQ2 (Multimodal fusion).** Is the current arrangement — LayerNorm-then-cross-attention with
  motion as query and vision as key/value — the right fusion design? How do alternative query/key
  assignments, fusion depths, or fusion-free baselines compare?
- **RQ3 (Severe class imbalance).** The key task, `crosses`, is imbalanced ~37:1. The current
  pipeline stacks three levers (offline resampling, weighted online sampling, class-weighted loss).
  What is the most effective and principled imbalance treatment (e.g., focal loss, global sampler
  statistics, calibrated thresholds), and how should it be evaluated honestly?
- **RQ4 (Input representation).** The motion features carry documented quirks (frame-0 deltas
  holding raw values; flip augmentation not reflecting absolute coordinates), and sequences are
  fixed-length (20 frames, truncate-no-pad). Do corrected features, online augmentation, and
  variable-length sequences with masking measurably help?
- **RQ5 (Efficiency).** What accuracy/efficiency trade-off (parameters, FLOPs, latency, VRAM) does
  each design choice impose, and is the full multimodal model justified over its ablations for
  deployment-style settings?

---

## 3. Scope

**In scope**
- Training and evaluating the v1.0 baseline and all redesigned variants on PIE (train/val/test
  splits as published), for all three tasks, with `crosses` as the primary task.
- Architecture redesign of the visual backbone, the fusion module, and the `crosses` head
  (collapsing the current dual frame-level/pooled head into one supervised head).
- Data-layer research: motion-feature corrections, online augmentation, variable-length sequences.
- Imbalance-policy redesign and honest evaluation under imbalance (F1/AUC/recall-led reporting).
- Systematic ablations and an efficiency benchmark per model variant.
- Engineering hygiene that protects validity: golden re-baselining per change, CI coverage floor,
  reproducible run tracking.

**Out of scope (for this thesis)**
- Trajectory forecasting (continuous future positions) — the project is classification-only.
- Additional datasets beyond PIE (e.g., JAAD) — listed as a stretch goal only, for generalization
  checks if time permits.
- Real-time in-vehicle deployment; ONNX export and latency benchmarks serve as a deployment proxy.
- Detection/tracking research — inference on raw video uses an off-the-shelf detector (YOLO) as a
  fixed front-end.

---

## 4. Work Plan

The work is organized into five work packages (WP). WP0 is housekeeping already in progress; WP1–WP3
carry the research contributions; WP4 is consolidation and writing. Each architecture or data change
follows a fixed protocol: design note → implementation behind config flags → golden re-baseline →
ablation run → logged comparison against the standing baseline.

### WP0 — Clean baselines (months 1–2)
- Finish the disk-bounded incremental LMDB build for the training split (val/test complete).
- Train the v1.0 baseline end-to-end for all four model types (`full`, `motion_only`,
  `visual_only`, `vanilla_concat`); record per-task Accuracy/F1/AUC/Precision/Recall and efficiency
  metrics as the reference table all later work is measured against.
- *Deliverable:* baseline results table + frozen reference checkpoints.

### WP1 — Data and input representation (months 2–5, RQ3, RQ4)
- Correct the documented motion-feature quirks (frame-0 delta semantics; flip-consistency of
  absolute coordinates; augmentation noise on absolute channels); ablate corrected vs. preserved.
- Replace offline write-time augmentation with online augmentation.
- Add a padding + masking path for variable-length sequences; ablate against fixed 20-frame windows.
- Imbalance policy v2: global (corpus-level) sampler statistics, focal-loss variant, and a
  principled decision on which levers stack; re-examine scheduler/early-stopping driving signal
  (macro-F1 vs. validation loss).
- *Deliverable:* ablation report quantifying each data-layer change on `crosses` F1/AUC.

### WP2 — Architecture redesign (months 4–8, RQ1, RQ2)
- Evaluate a modern, resolution-agnostic visual backbone as a drop-in replacement for the
  hierarchical ViT; compare at matched parameter/FLOP budgets.
- Fusion study: cross-attention query/key role swap, fusion depth, normalization placement, and the
  concatenation baseline; select the best design on validation metrics.
- Unify the `crosses` head: collapse the current frame-level + pooled dual-head contract into a
  single supervised head; verify no regression.
- *Deliverable:* redesigned v2 architecture with a full comparison table against v1.0.

### WP3 — Consolidated evaluation (months 8–10, RQ5)
- Final training runs of the selected v2 configuration and all ablations with multiple seeds.
- Efficiency benchmark per variant: parameters, FLOPs, latency, FPS, peak VRAM; ONNX-exported
  parity check as a deployment proxy.
- Qualitative analysis: temporal-attention visualizations, failure-case study on `crosses`
  false negatives (the safety-critical error mode).
- Stretch: cross-dataset sanity check on JAAD if time permits.
- *Deliverable:* complete experimental chapter material.

### WP4 — Thesis writing and dissemination (months 10–12)
- Thesis writing, integrating the design notes and ablation reports produced per work package.
- Prepare a workshop/conference paper submission if results warrant.
- Release-quality tag of the final codebase.
- *Deliverable:* thesis manuscript; optional paper draft.

### Timeline overview

| Month → | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WP0 Baselines | ■ | ■ | | | | | | | | | | |
| WP1 Data/imbalance | | ■ | ■ | ■ | ■ | | | | | | | |
| WP2 Architecture | | | | ■ | ■ | ■ | ■ | ■ | | | | |
| WP3 Evaluation | | | | | | | | ■ | ■ | ■ | | |
| WP4 Writing | | | | | | | | | | ■ | ■ | ■ |

---

## 5. Evaluation Methodology

- **Metrics:** Accuracy, F1, AUC, Precision, Recall per task plus macro-F1. Because `crosses` is
  ~37:1 imbalanced, aggregate accuracy is explicitly treated as uninformative for it; F1/AUC/recall
  lead all comparisons. A single shared metric accumulator guarantees train-validation and test
  metrics cannot diverge in implementation.
- **Efficiency:** parameters, FLOPs (fvcore), latency, FPS, and peak VRAM per model variant.
- **Protocol:** PIE's published train/val/test split; fixed seeds with multi-seed final runs;
  config-snapshot per run; CSV logs and a cross-run index for traceability; data-drift gate
  (`count_labels`) ensuring dataset statistics stay pinned across pipeline changes.
- **Validity safeguards:** every behavioral change lands behind a config flag with a golden-output
  re-baseline, so regressions are attributable to a single change rather than accumulated drift.

## 6. Expected Contributions

1. A systematic, ablation-backed redesign of a multimodal pedestrian-behavior model on PIE, with
   each design decision (backbone, fusion, head, input representation) empirically justified.
2. An evidence-based imbalance-handling recipe for severely skewed crossing prediction (~37:1),
   clarifying which of the commonly stacked levers actually contribute.
3. An accuracy/efficiency characterization across modality ablations, informing whether multimodal
   fusion earns its cost for this task.
4. A reproducible, tested, open-style research codebase (config-driven, golden-tested, ONNX-exportable)
   usable by subsequent students.

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPU/compute availability limits multi-seed runs | Medium | Efficiency-first variants; prioritize `crosses`; single-seed screening, multi-seed only for finalists |
| Redesigned components fail to beat baseline | Medium | Negative results are reported as findings; v1.0 remains a complete fallback thesis baseline |
| Data-pipeline rebuild costs (disk/IO) recur | Low | Incremental, resumable LMDB builder already implemented; artifacts cached |
| Scope creep into trajectory prediction / new datasets | Medium | Scope section above is the contract; JAAD is stretch-only |
| Timeline slip in WP2 | Medium | WP1 results alone support a defensible thesis; WP2 items are independently land-able |

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
- Lin, T.-Y., et al. *Focal Loss for Dense Object Detection.* ICCV 2017.

*(Reference list to be expanded with a full related-work survey during WP1.)*

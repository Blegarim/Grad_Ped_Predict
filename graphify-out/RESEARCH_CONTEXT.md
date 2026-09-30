# Research context audit — working tree, 2026-09-29

This report interprets the 25 documentation files in `semantic_files_1.json`, plus the working copies of `paper/main.tex`, `paper/refs.bib`, and `paper/project_context.md`. It does not independently rerun training, reproduce predictions, or verify external literature claims. Existing project files were not modified. The manuscript has substantial uncommitted changes; conclusions below refer to those current working files, not only HEAD.

## The research question and its evolution

The project began as a behavior-preserving rebuild of an undergraduate multimodal pedestrian predictor: context crops, pedestrian crops/motion, fusion, and three tasks (walking, looking, future crossing). The rebuild isolated modules, configuration, metrics and data contracts and pinned old numerical behavior with golden fixtures. The archive is historical evidence, not a current task list (`docs/archive/MIGRATION.md:12`, `docs/archive/legacy_baselines.md:10`, `docs/archive/README.md:1`). The subsequent engineering audit deliberately changed scientifically invalid behavior, including test-set threshold tuning, present-state labels taken from future frames, negative labels for unobserved futures, uncontrolled imbalance, and correlated-window evaluation (`docs/archive/HOLE_AUDIT.md:59`).

The substantive pivot is that event-anchored prediction and continuous deployment ask different questions. Anchored sampling takes a few windows relative to a known crossing event. Streaming sampling covers tracks continuously; someone who eventually crosses is often a negative because onset is not imminent. Those hard temporal negatives do not occur in the anchored benchmark (`docs/project-context-streaming-crossing-onset.md:49`, `paper/main.tex:173`). At length 20, stride 3, observation is about 0.67 seconds; the reported horizon remains 32 frames, about 1.07 seconds. The motivation is acting on short evidence when pedestrians appear suddenly, not achieving a favorable class balance by widening the horizon (`docs/archive/HOLE_AUDIT.md:192`, `paper/main.tex:476`).

The thesis originally promoted protocol-gap measurement, then demoted it to motivation for a methods thesis: predict onset timing under right-censoring. The latest evidence does not establish a detection improvement from that method. The current manuscript therefore supports a measurement and evaluation contribution, plus a censoring-aware formulation with an inconclusive objective comparison (`docs/THESIS_ROADMAP.md:14`, `paper/main.tex:211`, `paper/main.tex:1063`).

## Current model versus historical architecture

The general codebase remains multimodal, with TinyViT context embeddings, motion/pose temporal encoding, residual cross-attention and three optional tasks. That is not the model used in the current paper campaign. All v4 paper runs use pixel-free `pose_kinematics`: 58 channels per frame (9 bounding-box/ego-motion channels plus 49 pose features), training-statistics standardization clipped at ±5 SD, temporal convolutions, two-layer GRU and multi-head attention, dimension 128. Binary prediction pools frame logits by log-sum-exp; size is about 0.66M parameters (`CLAUDE.md:66`, `paper/main.tex:490`).

Pose features normalize joint positions by bbox center and height, separating articulation from absolute location/scale already present in motion. Confidence accompanies coordinates, and orientation uses sin/cos pairs. Existing temporal machinery was reused instead of introducing an ST-GCN; pose travels in the existing `motions` tensor to preserve shared data/model contracts. The proposed separate `PoseMotionEncoder` was never needed: the implementation extended `KinematicsEncoder` and uses `KinematicsOnlyModel` (`docs/POSE_ENCODER.md:125`, `docs/POSE_ENCODER.md:165`, `docs/POSE_ENCODER.md:211`, `docs/POSE_ENCODER.md:343`).

The old TinyViT choice addressed a collapsing legacy ViT channel schedule, tiny attention windows and weak pretraining/data efficiency. A later recipe tried truly frozen BatchNorm, trainable projection and cached features. Image-mode parity probes exonerated caching; the recipe itself underperformed, so the image campaign remained parked. Those results are historical supporting studies and must not be described as the canonical current model (`docs/BACKBONE_STUDY.md:22`, `docs/RECIPE_V2_PLAN.md:47`, `docs/AUTONOMY_PLAN_2026-09-18.md:22`, `docs/RERUN_PLAN_2026-09-26.md:219`).

## Timing objective and valid comparisons

The onset head predicts conditional hazard over 24 future bins (96-frame look-ahead, four frames per bin). Its comparable readout is `P(onset within 32 frames) = 1 - product(1-h_k)` across the first eight bins. The 96-frame look-ahead covers the 64-frame confusable band beyond the reporting horizon; four-frame bins trade 133 ms resolution for denser supervision. Only observed at-risk bins contribute; already-crossed windows leave the risk set (`paper/main.tex:663`, `paper/main.tex:755`).

The experimental arms are R2 binary, R1 auxiliary hazard (binary remains reported), R3 pure hazard, R4 hazard plus direct horizon-readout loss, and R3C pure hazard plus recovered censored training windows. Censored binary placeholder labels must never be consumed by a binary loss; the repository enforces this as a configuration guard (`docs/RERUN_PLAN_2026-09-26.md:105`, `CLAUDE.md:165`).

The primary outcome is detection rate at fixed false-alarm budgets, with lead time reported alongside, under both `per_window` and `per_track` counting. Threshold fitting must be distinguished from diagnostic same-split sweeps. The preregistered headline requires hazard's mean detection to exceed binary by more than the sum of their sample SDs at at least three of five budgets; this is a specified heuristic decision rule, not a conventional significance test (`docs/SEED_PLAN_2026-09-21.md:47`, `docs/RERUN_PLAN_2026-09-26.md:200`, `paper/main.tex:793`).

## Latest campaign and findings

The 20-run v4 campaign uses seeds 42/43/44 for each primary configuration; the three-task Model A has only seed42 under each of two training protocols. Every run is new because checkpoint selection changed from fixed-cut F1 to validation AUC, invalidating reuse of earlier ladder checkpoints (`docs/RERUN_PLAN_2026-09-26.md:13`, `docs/RERUN_PLAN_2026-09-26.md:140`). The recipe was launched by explicit user override of the memorization NO-GO. Input scaling, globally shuffled chunks and actual batch32 improve stability, but do not remove overfitting (`docs/RERUN_PLAN_2026-09-26.md:8`, `paper/main.tex:647`).

Current working-manuscript results, whose stated provenance is `outputs/diagnostics/v4_report/tables/`:

| Finding | Evidence |
|---|---|
| Anchored-trained Model B AUC drops 0.889 ± 0.007 to 0.516 ± 0.010 on streaming | `paper/main.tex:542` |
| Anchored model still ranks who eventually crosses (AUC 0.799 ± 0.010), but ranks imminent onset below distant onset (0.414 ± 0.012) | `paper/main.tex:597` |
| Pre-registered post-hoc anchored/streaming score combination does not materially improve baseline detection | `paper/main.tex:620` |
| At 205 alarms/hour, binary detects 42.3 ± 3.2%; pure hazard 44.1 ± 4.7%, `per_window` | `paper/main.tex:921` |
| At 41 alarms/hour, binary detects 22.9 ± 3.5%; pure hazard 17.4 ± 9.1% | `paper/main.tex:923` |
| No budget clears the preregistered criterion; comparison is inconclusive | `paper/main.tex:928` |
| Binary's mean lead time is earlier at every primary budget, e.g. 4.7 versus 3.4 seconds at 205/hour | `paper/main.tex:941` |
| R3C reaches 47.8 ± 3.0% at 205/hour, still below the required margin versus binary | `paper/main.tex:972` |
| 19/20 runs peak in validation AUC within eight epochs and subsequently overfit | `paper/main.tex:647` |

The single-seed 2–6× hazard advantage is retracted. It must not be presented as the project's present outcome. Inconclusive evidence also does not establish equivalence or prove that hazard objectives can never help.

## Documentation conflicts and manuscript issues

1. `docs/METHODOLOGY.md:29` still says the timing method is decisively better and detects 2–6× more crossings. Its reasoning about censoring remains useful, but its empirical headline is stale.
2. `docs/THESIS_ROADMAP.md:58` retracts that headline while `docs/THESIS_ROADMAP.md:78` still promotes it. The tracker also says the v4 campaign is armed, although current manuscript tables already contain completed results.
3. `paper/RESULTS_TO_FILL.md:20` contains v1 image-model, single-seed values and old writing tasks. It is historical, not a source for current table cells.
4. `paper/project_context.md:12` records the v4 replacement and retraction, while its identity, claims, baseline lock, timeline and many open questions retain old facts. Its claim that introduction/conclusion still need replacing is partly superseded by the working `main.tex`, whose abstract and conclusion already report the null comparison (`paper/main.tex:138`, `paper/main.tex:1060`).
5. `CLAUDE.md:49` claims onset metadata still does not reach the trainer; `docs/THESIS_ROADMAP.md:54` says that plumbing is complete. The former is stale.
6. Current `paper/main.tex:233` says the F1 SDs exceed the mean gap. Its macros give 0.008 and 0.025 versus gap 0.034, and its own comment says gap 0.0340 slightly exceeds the SD sum 0.0331 (`paper/main.tex:83`, `paper/main.tex:106`). That introduction sentence is numerically inconsistent.
7. `paper/main.tex:505` says every configuration has three seeds, while `paper/main.tex:518` correctly identifies Model A as seed42 only.
8. `paper/main.tex:1035` still attributes collapse to data ordering despite the canonical recipe's globally shuffled storage and the updated threats paragraph emphasizing early memorization (`paper/main.tex:502`, `paper/main.tex:647`). This appears to be leftover v1 interpretation and needs reconciliation against actual v4 logs.

## Remaining scientific limitations

The matched-size streaming control is still unrun and explicitly pending. Therefore the residual deployment gap mixes protocol and training-set-size differences; it is not clean causal attribution to hard temporal negatives (`paper/main.tex:632`). Only PIE is evaluated, no published baselines were retrained, and cross-dataset generalization remains open (`paper/main.tex:1032`, `paper/project_context.md:189`). Look-ahead/bin-width, focal/class-balanced loss and pooling comparisons remain planned (`paper/main.tex:986`).

The primary metric was adopted after window-metric results were inspected; the later replicate criterion was preregistered. Both facts must remain visible (`paper/main.tex:656`). Absolute alarm rates remain high, so this is research evidence rather than a demonstrated deployable safety system. Three seeds delimit what can be claimed; the chosen spread rule does not turn an inconclusive result into equality.

The bibliography connects PIE/JAAD and crossing benchmarks to online action/start detection (OADTR, LSTR, TeSTra, MAT, ODAS, StartNet), survival modeling (Cox, Kaplan–Meier, DeepHit, DeepWait), rare-event losses, whole-body pose and Page's CUSUM. Bibliographic presence and source-level citations were extracted; external factual verification was outside this read-only context pass (`paper/refs.bib:14`, `paper/refs.bib:70`, `paper/refs.bib:136`, `paper/refs.bib:212`).

## Graph artifact

`.graphify_chunk_1.json` contains 425 nodes, 543 edges and two hyperedges. Nodes distinguish historical, stale, current and unresolved statements where curated; document-section nodes are navigation anchors rather than independent validated facts. Explicit document links and manuscript bibliography citations use EXTRACTED confidence 1.0. Cross-file conceptual relationships use discrete INFERRED confidence values. Token usage is unavailable in this extraction runtime; JSON `input_tokens=0` and `output_tokens=0` are sentinels, not a claim of zero work.

# Project Context Brief
## Online Detection of Pedestrian Crossing-Onset: Bridging Online Action Detection into the PIE/JAAD Intention Subfield, and Decomposing the Anchored→Streaming Performance Gap

> **Purpose of this document.** Self-contained context read for a new standalone research project, meant to be dropped into a fresh project thread so none of the reasoning below has to be re-derived. It is fully technical; nothing is dumbed down. It records not only the proposed angle but the *dead ends already ruled out*, so future-you does not re-tread them. It assumes familiarity with the author's existing pipeline (multimodal ViT + Motion Encoder + Cross-Attention + Ensemble, per-frame multi-task heads, LMDB data, PIE dataset).

**Prepared:** July 2026 · **Status: 🧊 frozen reference** (as of 2026-08-19) — the argument here is still
correct and is not maintained further. Two updates apply on top of it and are **not** folded back in:

1. **§4–§5 Phase 4 is done.** The decomposition was measured: `G_prior ≈ 0`, `G_hardneg ≈ G_total`. The
   §4 fork ("if G ≈ G_prior … demote to a cautionary short paper") resolved to the *other* branch —
   G_hardneg is the whole gap. Numbers: [`../outputs/runs/RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md).
2. **The result was reclassified as the motivation** (August 2026). This document treats the decomposition
   as the headline contribution; the thesis now treats it as the *justification* for a method. Phase 5's
   "constructive close" is therefore the centre, not the epilogue — see
   [`METHODOLOGY.md`](METHODOLOGY.md) and [`THESIS_ROADMAP.md`](THESIS_ROADMAP.md).

Read it for: the two task formulations, why the ratios differ, the hard-temporal-negative definition, the
base-rate-fallacy primer, the **dead ends already ruled out** (§1 — do not re-tread them), the reviewer
objections (§7), the reading list (§9), and the glossary (§10).

**Working title (one line):** *The pedestrian-intention benchmark never learned to run in a stream — and the case it hides is the one that matters.*

---

## 0. TL;DR

The dominant PIE/JAAD crossing-prediction protocol is **event-anchored**: for each pedestrian track it clips observation at the crossing onset and places a short observation window 1–2 s before a *known* event, producing a per-track binary label and a mild (~2.5:1) class ratio. This is not the condition a deployed model faces. A deployed model is a **streaming detector** monitoring every moment, where the true rate of "a crossing is about to start in the next ~1 s" is far rarer (the author's own dense sliding-window pipeline yields ~37:1) and where the dominant negatives are **hard temporal negatives** — windows of a pedestrian who *will* cross but not yet — which the anchored protocol never generates for training *or* testing.

The contribution is a **bridge + diagnosis**: import the mature **Online Action Detection (OAD) / Online Detection of Action Start (ODAS)** task definition and metrics into the PIE/JAAD intention subfield (which never adopted them despite being ODAS's canonical motivating example), re-evaluate anchored-trained SOTA models under a streaming protocol, and **decompose** the resulting performance gap into (a) a *prior-shift* component that threshold recalibration fixes cheaply and (b) a *residual hard-temporal-negative* component that recalibration cannot fix because the model was never trained or tested on those windows. The decomposition is the rigorous core that elevates this above "we applied ODAS to PIE."

---

## 1. Origin, and the dead ends already ruled out

This angle is the survivor of a long pruning process. Record of what was killed and why, so it is not revisited:

1. **VRU-safety digital twin under imperfect sync (latency safety-cliff + fail-safe controller).** Rejected: the "find the latency threshold past which the twin is untrustworthy" framing is a calibration exercise, not a research question; there is no single latency number (danger depends on how fast the scene is changing, not on data age); and the corrected version (risk-aware control under delayed/dropped state) walks into mature Networked-Control-Systems / estimation territory.
2. **"Benchmark saturation via aleatoric ceiling."** Weakened: uncertainty (epistemic/aleatoric) is already modeled in crossing prediction (evidential DL, KL/Mahalanobis heads, threshold networks); "nobody has framed the residual as irreducible" is false.
3. **"Ego-motion shortcut inflates the leaderboard."** Occupied: **LIM ("Less Is More")** makes exactly this its central thesis (ego-vehicle speed causally confounds crossing; *removing* it helps; skeleton-only model with adversarial speed removal). IntFormer notes "most crossing cases in PIE share a similar pattern" (ego-speed). Azarmi's **CAPFI** review quantifies the driver-side bias.
4. **"Safety-weighted / risk-stratified re-scoring of PIE."** Occupied on multiple axes: **CAPFI** stratifies PIE by pedestrian–ego distance; **"Diving Deeper" (2024)** proposes a per-sample TTE-weighted metric (and found the reweighting *marginal*); **"A Novel Benchmarking Paradigm" (2023)** stratifies PIE *by ego-motion* (speed, yaw, acceleration) — but for *trajectory regression*, not intention classification. The remaining sliver ("crossing-classification × ego-kinematic-danger axis × leaderboard ranking inverts") was judged too thin and one arm's reach from published work.
5. **"Streaming evaluation is a new idea."** False. OAD/ODAS is a mature subfield (see §3). This is the key correction that *reshaped* the present angle from "I discovered streaming" into "the intention subfield never imported streaming."

**What survived and why.** The present angle is the first that (i) originates from the author's own pipeline friction (the 37:1 imbalance from dense sliding-window sampling), (ii) has a rigorous decomposable core rather than a vibe, (iii) is a *bridge* that inherits ready-made machinery, and (iv) has a clean, honest "why nobody did this" (subfield path-dependence on the 2019 benchmark).

---

## 2. Background primer (concepts, defined)

### 2.1 The two task formulations

**Event-anchored per-track classification (the standard PIE/JAAD protocol; Rasouli/Kotseruba benchmark).**
- One label per pedestrian *track* (instance).
- For crossing pedestrians, observation is **clipped at the first crossing frame** (the `crossing_point` tag); for non-crossers, at the last visible frame.
- The observation window (commonly 16 frames ≈ 0.5 s) is sampled so its **last frame sits 1–2 s (TTE 30–60 frames) before the event**, with overlap (0.5–0.8) for augmentation.
- Class ratio ≈ 2.5:1 non-crossing:crossing (a *track-level* ratio, reflecting how many pedestrians eventually cross).
- Train/val/test are produced by splitting the resulting samples (PIE: sets 01/02/06 train, 04/05 val, 03 test). **The test set carries the same anchored distribution and the same structural omission** (see §2.3).

**Streaming online detection of crossing-onset (the proposed / author's formulation).**
- Dense sliding window across the entire track.
- Label per window: does a crossing **onset** fall within the next H frames (e.g., observe 1–20, predict onset in 21–50)?
- Windows where crossing already began during observation are discarded (correct: you cannot "anticipate" an onset already underway).
- Class ratio ≈ 37:1 (a *temporal-density* ratio, reflecting how rare an imminent onset is per observed moment).

### 2.2 Why the ratios differ (mechanism — state this precisely, it is the crux)
The anchored 2.5:1 is a **sampling artifact**: taking ~one labeled window per track, anchored right before the event, never samples the many "nothing imminent yet" windows earlier in each track. The streaming 37:1 reflects the **true base rate** of imminent onsets in continuous observation. They are not two estimates of the same quantity; they measure different things (track-level class balance vs. per-window onset density).

### 2.3 The hard temporal negative (the safety-critical case the protocol omits)
Streaming negatives are of three kinds:
1. **Genuine non-crossers** (waiting, walking along) — also present anchored.
2. **Hard temporal negatives** — windows of a pedestrian who *will* cross, but not for another few seconds. Same person, same appearance, same scene, labeled negative *now* only because the onset is beyond horizon H. **The anchored protocol, by clipping at onset, generates these for neither training nor testing.** These are exactly the windows a deployed detector must not false-alarm on, and they are a genuine *likelihood* (discrimination) problem, not a *prior* problem.
3. **Junk** — occlusion, sensor noise, far/low-information windows. This is the fraction that drowns naive training on raw 37:1.

**Why this matters for the author's past failure:** feeding raw, unmanaged 37:1 to a from-scratch, data-hungry ViT collapses training — the positive gradient is negligible before crossing features are learned. The fix is *curated* streaming training (emphasize hard temporal negatives, down-weight/filter junk, pretrained backbone, focal/weighted objective), not abandoning the real distribution.

### 2.4 Online Action Detection (OAD) and Online Detection of Action Start (ODAS)
Mature action-recognition subfields for streaming, per-frame settings:
- **OAD**: per-frame labeling of a streaming video using only current + past frames (no future access). Datasets: TVSeries, THUMOS14, HDD. Methods: TRN, OadTR, RED, FATSnet. Metric: per-frame mAP; **calibrated AP (cAP/mcAP)** to handle heavy background.
- **ODAS**: detect the *onset* of an action instance as early as possible in an untrimmed stream with large background. Methods: StartNet. Metric: **point-level AP (p-AP)** with temporal-offset tolerance.
- **Critical fact:** ODAS/OAD papers repeatedly cite *pedestrian crossing* as the canonical safety-critical motivating example, yet evaluate on TV/sports/ego-driver-maneuver data (HDD's "crosswalk passing" is the *ego-vehicle driver's* action, not the pedestrian's). The rigor exists; it was never applied to pedestrian-crossing *intention*.

### 2.5 The base-rate fallacy (why "good discrimination" ≠ "deployable")
Discrimination (AUC) and operating-point usefulness (precision at a threshold) diverge under base-rate shift. Worked example, same model (recall 0.9, FPR 0.1):
- At 2.5:1 (crossing ≈ 28.6%): precision ≈ 0.9·0.286 / (0.9·0.286 + 0.1·0.714) ≈ **0.78**.
- At 37:1 (crossing ≈ 2.6%): precision ≈ 0.9·0.026 / (0.9·0.026 + 0.1·0.974) ≈ **0.19**.
Four of five alarms become false purely from the prior. Part of this is *fixable by recalibration* (shift threshold / prior-correct); the hard-temporal-negative part is *not*.

---

## 3. The precise thesis

> The pedestrian-crossing-intention subfield inherited a 2019 event-anchored classification protocol and never adopted the streaming/ODAS evaluation rigor developed — in a sibling subfield — for exactly this safety-critical action-start problem. Reformulating PIE crossing prediction as online detection of crossing-onset (a) exposes a true base rate (~37:1) that the anchored protocol hides, (b) reveals a class of hard temporal negatives absent from both anchored training and anchored *testing*, and (c) causes anchored-trained SOTA models to degrade. That degradation decomposes into a prior-shift component (recalibration fixes it) and a residual hard-temporal-negative component (recalibration cannot). The deliverable is a streaming evaluation protocol for the subfield plus a quantified decomposition of what the standard protocol has been systematically failing to measure.

---

## 4. The decomposition (the rigorous core)

For a model trained anchored and evaluated streaming, total deployment gap G = G_prior + G_hardneg, where:
- **G_prior** = the portion recovered by applying the optimal prior/threshold correction to the true streaming base rate (Bayesian prior correction or validation-set threshold re-selection at the true prior).
- **G_hardneg** = the residual after correction, measured against a model *trained* streaming (curated) and evaluated streaming. This residual is attributable to the hard-temporal-negative discrimination problem the anchored model never learned.

Headline number = the fraction of G explained by each. Interpretation:
- If G ≈ G_prior (recalibration recovers most): the author's own intuition wins, contribution shrinks to "remember to recalibrate" (a known trick → demote to a cautionary short paper).
- If G_hardneg is large: the protocol systematically under-trains and under-tests the safety-critical case → full paper.

This decomposition is also the pre-emptive answer to the strongest reviewer objection ("just recalibrate the threshold"): the paper *measures* whether that is true.

---

## 5. Plan of attack (phased)

**Phase 0 — Final kill-check (do first, cheap).** Search specifically for ODAS / online action detection already applied to PIE/JAAD *pedestrian-crossing intention* (not ego-vehicle maneuvers). Surfaced so far: ODAS-on-driving via HDD = ego-driver actions (different task). If a direct PIE-crossing-as-ODAS paper exists, re-scope. If not, runway is clear.

**Phase 1 — Day-one empirical de-risk.** Take 2–3 published models with released code (e.g., PCPA, SF-GRU, one recent transformer). Train anchored (standard). Evaluate on both anchored and streaming test sets (start with the "eventual crossing" label to isolate prior effects). Check: does precision collapse at the true prior, and does simple threshold recalibration recover it? Result tells go/no-go in days. (The author's own per-frame multi-task heads make the streaming harness cheap.)

**Phase 2 — Formalize both samplers + characterize negatives.** Implement the anchored sampler (benchmark-faithful) and the streaming sampler (dense sliding window, onset-in-horizon label). **Tag streaming negatives into the three types (§2.3) and report the composition** (fraction hard-temporal vs. junk vs. genuine non-crosser). This breakdown is itself unpublished and is a sub-contribution.

**Phase 3 — Cross-protocol matrix (the spine table).** For each of N models (RNN, GRU, transformer, the author's ViT+Motion+Cross-Attn ensemble), run {train anchored, train streaming} × {test anchored, test streaming}. Report **discrimination** (AUC, average precision) *separately from* **operating-point** metrics (precision/recall/F1 at the true-prior-recalibrated threshold) *and* **ODAS metrics** (per-frame mAP / calibrated AP, point-level AP with offset tolerance).

**Phase 4 — Decomposition (§4).** In the train-anchored/test-streaming cell, apply optimal prior correction, then measure residual vs. train-streaming/test-streaming. Report G_prior vs. G_hardneg fractions. This is the headline.

**Phase 5 — Constructive close (turns critique into a tool).** Ship (i) the streaming ODAS-style evaluation protocol for the subfield and (ii) the curated hard-negative streaming training recipe; show it narrows G_hardneg. Benchmark-critique reviewers want a fix, not just a diagnosis.

**Supporting analyses.** (a) Confirm residual failures concentrate in the "will-cross-soon" windows (validates the hard-negative story). (b) Sweep horizon H — base rate and difficulty both move with H; use this to distinguish streaming from a mere large-TTE sweep (see §7). (c) Check whether models lean *harder* on the ego-speed shortcut under streaming (quietly reconnects the LIM/CAPFI confound thread as a secondary finding).

---

## 6. Methods & tools

| Tool / concept | Role | Priority |
|---|---|---|
| Author's existing PIE pipeline (ViT + Motion + Cross-Attn + Ensemble, per-frame heads, LMDB) | Streaming harness + one of the N models | Essential, in hand |
| Anchored sampler (Rasouli/Kotseruba protocol) | Baseline formulation; must be benchmark-faithful | Essential |
| Streaming/dense sampler (onset-in-horizon) | The proposed formulation | Essential, in hand |
| Published baselines w/ code (PCPA, SF-GRU, a transformer) | Multi-model evidence for a *protocol* claim | Essential |
| ODAS/OAD metrics: per-frame mAP, calibrated AP (cAP/mcAP), point-level AP | Import from OAD; the correct streaming metrics | Essential (read + implement) |
| Prior correction / threshold recalibration (Bayesian prior shift, val-set threshold selection) | Isolates G_prior | Essential |
| Focal loss / class-balanced loss / curated hard-negative mining | Curated streaming training (§2.3) | Essential |
| TRN / OadTR as reference streaming architectures | Optional strong streaming baselines | Optional v2 |

---

## 7. Risks & reviewer objections (pre-empt in the writing)

1. **"Streaming eval is old (OAD/ODAS)."** True — do not claim to invent it. Position as *importing* mature rigor into a siloed subfield that skipped it due to path dependence on the 2019 benchmark. Novelty = the bridge + the decomposition + the field-wide blind-spot demonstration, not the streaming idea.
2. **"You just renamed the TTE sweep."** GTransPDM shows PIE accuracy → 99%+ as TTE→0; a reviewer will equate streaming with large-TTE. **Rebuttal (must show, not assert):** large-TTE is still *one anchored window per track, placed a known distance before a known event*; streaming is *dense windows, no event anchor, flooded with hard temporal negatives*. The difference is the **negative distribution**, not the horizon. Demonstrate via the negative-composition analysis (Phase 2) and the H-sweep (Phase 5b).
3. **"Just recalibrate the threshold."** Answered structurally by the decomposition (§4) — the paper measures exactly how much recalibration recovers.
4. **Nearest neighbor — Coupling-Intent's "two sampling settings."** Setting 1 ("all original data") includes windows long before the event and notes they are easy; that is the closest prior observation. Cite and differentiate: they used whole-track windows with an *eventual-crossing* label and dismissed early windows as boring; this work uses dense *onset-in-horizon* labeling and argues the early windows of *eventual crossers* are the hard, safety-critical negatives.
5. **"Someone already bridged ODAS→PIE."** Residual risk; resolve in Phase 0.
6. **Labor/reproducibility.** Retraining several published models is real work and depends on their code cooperating. Budget for it; prefer models with maintained repos.
7. **Single-dataset validity.** PIE is Toronto/daylight/clear only. Consider replicating the core matrix on JAAD (note: JAAD lacks numeric ego-speed) to show the effect is not PIE-specific.

**Realistic venues:** IV, ITSC, WACV, or an autonomous-driving / safe-ML workshop. A strong, clean decomposition could reach a main CV or robotics track.

---

## 8. Open sub-questions / natural sequels
- Formalize crossing-onset detection with a proper ODAS metric suite as a *new public evaluation track* on PIE/JAAD (community artifact).
- Time-to-cross **regression** framing as a complement that sidesteps the binary prior entirely — compare against the detection framing.
- Cost-sensitive / decision-theoretic operating points: pick thresholds by a braking-cost vs. miss-cost model rather than F1.
- Curriculum over the hard-temporal-negative horizon (train from easy far-from-onset to hard near-onset negatives).
- Interaction with the ego-speed shortcut (LIM/CAPFI): does streaming make the shortcut more or less load-bearing?
- Correlated/bursty degradation (occlusion runs) as a distinct hard-negative subtype.

---

## 9. Key papers to read first (grouped)

**The protocol you are critiquing**
- Rasouli et al., *PIE: A Large-Scale Dataset and Models…* (ICCV 2019) — dataset + original per-track protocol, ~2.5:1.
- Kotseruba et al., *Benchmark for Evaluating Pedestrian Action Prediction* (WACV 2021) — the standard sampling/splits everyone follows; TTE-vs-accuracy effect; easy/medium/hard sample analysis.

**Nearest neighbors (know cold; differentiate)**
- *Coupling Intent and Action for Pedestrian Crossing Behavior Prediction* — the "two sampling settings" observation (closest prior thought); Naive-baseline-beats-SOTA note.
- *Diving Deeper Into Pedestrian Behavior Understanding* (2024) — per-sample TTE-weighted metric (found marginal); balanced-accuracy/mAP.
- Azarmi et al., *Feature Importance… CAPFI* (2024) and *…via Vision-Language Foundation Models* (2025) — context/distance stratification; ego-speed driver-side bias.
- *A Novel Benchmarking Paradigm…* (2023) — ego-motion scenario stratification, but for trajectory regression.
- *Causal Confusion… The Role of Ego-Vehicle Speed* (LIM, "Less Is More") — the ego-speed shortcut, claimed and "fixed."
- GTransPDM — TTE→0 gives ~99% accuracy (the loaded gun for objection #2).

**The rigor to import (OAD/ODAS)**
- Xu et al., *Temporal Recurrent Network (TRN)* — online detection + anticipation.
- Wang et al., *OadTR: Online Action Detection with Transformers* — per-frame mAP / calibrated AP.
- Shou et al., *Online Detection of Action Start (ODAS/StartNet)* — point-level AP; cites pedestrian crossing as motivation.
- De Geest et al. (TVSeries) — origin of OAD + calibrated AP.
- A survey on online action detection / action anticipation — for metric definitions and framing.

---

## 10. Glossary (plain definitions)
- **Event-anchored sampling:** placing the observation window a fixed time before a *known* crossing event; clips at onset; yields ~2.5:1.
- **Streaming / online detection:** per-window (or per-frame) decision using only current+past frames, over the whole track; yields the true ~37:1 onset rate.
- **Crossing onset:** the first frame the pedestrian begins crossing (`crossing_point`).
- **Hard temporal negative:** a window of a pedestrian who will cross, but not within horizon H — labeled negative now; absent from anchored data.
- **Base rate / prior:** the fraction of positive windows in the evaluated distribution.
- **Prior shift / base-rate fallacy:** good discrimination (AUC) can coexist with poor precision when the prior is small; part fixable by recalibration.
- **Prior correction / recalibration:** adjusting the decision threshold (or posterior) to the true deployment prior without retraining.
- **TTE (time-to-event):** frames between last observation and the event; the anchored protocol fixes it to 1–2 s.
- **Horizon H:** the anticipation window in the streaming formulation ("onset within next H frames").
- **OAD (Online Action Detection):** per-frame labeling of a stream; metric per-frame mAP / calibrated AP.
- **ODAS (Online Detection of Action Start):** detect the onset as early as possible in an untrimmed stream; metric point-level AP.
- **Calibrated AP (cAP):** AP variant that compensates for heavy background/negative imbalance (from TVSeries/OAD).
- **G_prior / G_hardneg:** decomposition of the anchored→streaming gap into the recalibration-fixable part and the residual hard-negative part.

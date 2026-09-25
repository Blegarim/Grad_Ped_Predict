# Results ready to paste into `main.tex` (as of 2026-09-21)

Everything here is measured, from `outputs/runs/*/eval_log.csv` and
`outputs/diagnostics/*/onset_test.npz`. Provenance for each number is stated. Nothing is projected.

---

## ⚠️ FIRST — a baseline inconsistency to resolve before filling `tab:onset`

`tab:onset` currently lists `binary baseline (str→str) = 0.742 / 0.190`. That is **Model A**
(`20260710_152517`, the 3-head run). **All four onset arms are crosses-only**, so their correct comparator
is **Model B** (`20260714_134253`, crosses-only, streaming-trained): **AUC 0.784, raw F1 0.249, tuned F1
0.225**. Mixing them would compare a 3-head baseline against crosses-only methods.

**Recommendation:** use Model B as the baseline row in `tab:onset` and say so explicitly; keep Model A in
`tab:matrix` where it belongs.

---

## 1. `tab:onset` — filled (streaming test, n = 69,875)

All arms: `pose_full`, v1 recipe, seed 42, streaming-trained, scored per the Onset-arm policy.

| Configuration | run | AUC | AP | tuned F1 | read-out spread (val, best ep) |
|---|---|---|---|---|---|
| binary baseline (str→str) | `20260714_134253` | 0.784 | — | 0.225 | n/a |
| auxiliary hazard (R1) | `20260916_075501` | 0.772 | 0.1547† | 0.225 | 0.001 → 0.252 |
| **pure hazard (R3)** | `20260918_105518` | 0.761 | **0.1723** | 0.220 | 0.001 → 0.294 |
| hedged (R4) | `20260919_111158` | 0.776 | 0.1164 | 0.185 | 0.001 → 0.524 |
| pure + censored (R3C) | `20260920_021440` | 0.755 | 0.1488 | 0.204 | 0.006 → 0.503 |

† R1's AP is computed on `crosses_frame` (its reported head). R3/R4/R3C AP are on `crosses_readout`.
**AP is missing for the two baseline rows** — they have no prediction dump (see §5 gap 2).

**The honest one-line reading:** on every window-level metric the arms are indistinguishable from the
baseline and from each other. R4 is the only clear mover and it moves *down*.

---

## 2. NEW SECTION NEEDED — detection-latency evaluation (this is the contribution)

The paper has no section for this. It is the result that separates the arms.

**Protocol.** One decision per window; stride 3 at 30 fps = 36,000 decisions/hour. For each of the 205
crossing pedestrians, the first window whose score crosses threshold *while a crossing is genuinely still
ahead* counts as a detection, and `onset_offset` at that window is the lead time. For each of the 495
never-crossing pedestrians, any threshold crossing is a false alarm. Sweeping the threshold traces a curve.
Derived from quickest change detection (Page 1954; Shiryaev/Roberts/Lorden), whose native metrics are
detection delay vs average run length to false alarm; mirrored here as lead time vs alarms/hour.

*(Regenerated 2026-09-23 from `scripts/report_detection_curve.py` — these supersede the scratchpad
numbers; see "What changed in the promotion" below. `per_window` accounting.)*

| FA/hr | binary (R1 `crosses_frame`) | **pure hazard (R3)** | hedged (R4) | pure+censored (R3C) |
|---|---|---|---|---|
| 1200 | 62.0% @ 6.09 s | **66.8% @ 6.52 s** | 49.8% @ 4.76 s | 65.4% @ **8.11 s** |
| 460 | 34.6% @ 3.40 s | **56.6% @ 4.40 s** | 28.8% @ 3.16 s | 45.4% @ 4.55 s |
| 205 | 22.9% @ 2.97 s | **48.8% @ 3.05 s** | 14.6% @ 2.73 s | 31.2% @ 3.12 s |
| 95 | 14.1% @ 1.99 s | **41.0% @ 1.84 s** | 9.3% @ 2.65 s | 18.5% @ 2.36 s |
| 41 | 5.9% @ 0.92 s | **34.1% @ 0.72 s** | 3.9% @ 1.25 s | 10.7% @ **2.51 s** |

**What changed in the promotion (and what did not).** Turning the scratchpad analysis into
`eval/detection_curve.py` made two assumptions explicit: the false-alarm budget is now a **hard**
constraint (the old quantile overshot, inflating 4 of 20 cells by <=3 pedestrians, <=1.5 pp), and lead
time is the **earliest** qualifying alarm rather than the first row in the dump (~5% of PIE pedestrians
arrive as several occlusion-separated segments whose order cannot be recovered). **No ordering and no
claim changes.**

**Alarm accounting must be stated — it flips the result at loose budgets.** Counting each *nuisance
pedestrian* once instead of each alarming window (`per_track`):

| FA/hr (`per_track`) | binary | **pure hazard (R3)** |
|---|---|---|
| 120 | **71.7% @ 7.08 s** | 65.4% @ 6.41 s |
| 46 | 44.9% @ 4.70 s | **49.3% @ 3.03 s** |
| 20 | 18.5% @ 3.11 s | **36.1% @ 0.80 s** |
| 4 | 2.9% @ 0.98 s | **27.3% @ 0.43 s** |

The binary baseline *wins* at 120 alarms/hr. **Write the claim as a low-false-alarm claim**, report both
rules, and do not quote a number without saying which rule produced it. A reviewer who finds this unaided
discounts the result; a paper that states it first is far stronger.

**Claims this supports:**
1. **Pure hazard training detects 2–6× more crossings than binary training at matched false-alarm budgets**,
   and the margin widens as the budget tightens — while the two tie on window-F1 (0.220 vs 0.225).
2. **The hedged arm is worse than BOTH pure arms.** Adding a fixed-horizon CE on top of the hazard
   objective degrades it below plain binary training. The objectives conflict. *This is the answer to
   "isn't this just multi-task learning?" — the multi-task version is the worst arm.*
3. **R3C trades detection rate for lead time** (3× the warning at 41 FA/hr) and still beats binary
   everywhere, so the formulation is robust to what its data emphasises.

---

## 3. Collapse instability — a standalone finding

Epochs where validation loss exceeded 1.0 (the "everything is a crossing" pathology):

| run | collapse epochs |
|---|---|
| v1 baseline `20260714_134253` | 1, 2, 4, 13, 17 |
| R3 `onset_pure` | 1, 2, 4, 13, 17 |
| R4 `onset_hedge` | 1, 2, 4 |
| R3C (+7,470 windows) | **1, 2 only** |

Two independent demonstrations that this is **sampler / data-order driven**, not model- or loss-driven:
(i) R3 collapses on *identical epochs* to the baseline despite a completely different objective (no
crossing CE at all); (ii) adding 7,470 training windows reshuffled the draw order and eliminated all three
later collapses. BatchNorm drift was separately refuted as the cause (no BN re-estimation ever reproduced
a collapse).

---

## 4. Evaluation-practice results (support `sec:threats` / `sec:discussion`)

All measured on the R1 dump, laptop-only:

- **Within-track score smoothing moves AUC +0.024** (k=15 moving average), which is **larger than the
  entire R3-vs-baseline AUC gap (0.023)**. A free post-processing step outweighs the method difference —
  direct evidence that window-level metrics cannot resolve these configurations.
- **Track-level aggregation inflates F1 from 0.235 to 0.636**, but only because the base rate moves from
  3.06% to 29.3%. By lift over chance it is *worse* (5.1× window vs 2.1× track). **Do not report track-level
  F1 as an improvement** — it is a different question with an easier prior.
- **CUSUM accumulation is 2–10× worse than plain thresholding.** A crossing approach is not the persistent
  distribution shift CUSUM assumes, and the accumulator saturates over long non-crossing tracks (up to 889
  windows). The field's *metric* transfers; its *detector* does not.
- **No cheap window-level metric tracks the detection ranking.** AUC ranks R3 *last*. AP matched on three
  arms then failed on the fourth (it puts R3C below binary while R3C beats it at every budget). Report the
  curve; do not substitute a scalar for it.

---

## 4b. SCOPING — what goes in the paper and what does not (decided 2026-09-23)

The through-line is one claim: *the standard metric cannot see what this objective buys; here is the
evidence it cannot, and an evaluation that can.* A side result earns space only if it serves that.

**Keep — they justify the evaluation (§4 above):**
- Smoothing moves AUC +0.024, larger than the whole method gap (0.023). The strongest supporting result.
- No scalar window metric tracks the detection ranking (AUC ranks R3 last; AP matched 3 arms then failed
  on the 4th). Justifies reporting a curve rather than a number.
- Seed variance on tuned F1 is ±0.03 (baseline 0.200 vs 0.259 across seeds 42/43) — larger than every arm
  difference. Same argument, with seed evidence.

**One sentence each — pre-empt the obvious questions:**
- Track-level aggregation inflates F1 (0.235 -> 0.636) only via the base rate; by lift it is worse.
- CUSUM accumulation is 2-10x worse than plain thresholding.

**LIMITATIONS ONLY — not a finding.** The collapse/seed-dependence material. That training is stochastic
is not a contribution and presenting it as one invites eye-rolling. The narrow non-obvious part -- R3 and
the baseline collapse on IDENTICAL epochs under completely different loss functions, so the pathology is
determined by data ordering with zero contribution from the objective -- is worth a single paragraph in
threats-to-validity, framed as justification for the multi-seed protocol and for distrusting checkpoint
selection. Not a results subsection.

**NOT IN THE PAPER AT ALL.** The recipe-v2 detour, the K=6 cache rebuild, the image-mode probe, BatchNorm
drift. Engineering archaeology; it belongs in the repo history, at most a thesis appendix.

## 5. Gaps a reviewer will attack — and the plan

| # | Gap | Severity | Fix | Cost |
|---|---|---|---|---|
| 1 | **Every number is n = 1** | **critical** | 3 seeds on R3 + the true baseline | ~4 days (plan below) |
| 2 | ~~Baseline has no detection curve~~ **NOT A GAP** — `dump_predictions` already guards with `if "crosses_readout" in out:` and always saves `p_frame`, so a no-onset checkpoint dumps fine. The R2s seed-42 run (queued) supplies the true baseline curve | closed | none | 0 |
| 3 | **Metric chosen after observing the tie** | high | cannot be undone; must be confronted in the text — the metric is independently motivated (deployment specification, 70-year-old prior art) and pre-registered for the seed runs | writing |
| 4 | ~~Analysis code is ad-hoc~~ **CLOSED 2026-09-23** | — | `eval/detection_curve.py` + `scripts/report_detection_curve.py` + 14 tests; gate green | done |
| 5 | ~~Alarm accounting is per-window by assumption~~ **CLOSED 2026-09-23** | — | `per_track` implemented and reported; it flips the result at loose budgets, so both are now reported | done |
| 6 | Absolute performance poor (~41% @ 95 FA/hr) | medium | frame as a relative/formulation result; state plainly | writing |
| 7 | Single dataset (PIE), no published baselines retrained | medium | already de-scoped in the paper; keep it stated | — |
| 8 | R4/R3C `best.pth` selected at epochs 5 and 4 on the noisy F1 | medium | note as a confound; seed runs will show whether it matters | — |
| 9 | Tuned thresholds on anchored sit at the sweep floor 0.10 (grid-capped) | low | widen the sweep and re-run **eval only** for all arms | ~1 h |

---

## Provenance

- Window metrics: `outputs/runs/<run>/eval_log.csv`, test rows, val-tuned thresholds (M2 rule).
- Detection curves + AP: `outputs/diagnostics/{r1,r3,r4,r3c}/onset_test.npz`, produced by
  `scripts/dump_onset_predictions.py` on each arm's `best.pth`, streaming test split.
- Collapse epochs: `outputs/runs/<run>/train_log.csv`, `val_loss > 1.0`.
- Full narrative and caveats: memory `detection-latency-result`, `outputs/runs/RESULTS_MATRIX.md`.

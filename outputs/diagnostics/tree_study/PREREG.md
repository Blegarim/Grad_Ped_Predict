# Pre-registration: tree-model study + loss-side deep arms

**Written 2026-10-04, before any number from this study exists.** Executable form: `src/pedpredict/eval/tree_study.py`
(arms, fitting, selection) and `src/pedpredict/eval/tree_study_report.py` (verdicts). Driver:
`scripts/run_tree_study.py`. Deep arms: `outputs/diagnostics/v5_box/queue_v5.sh` (section D). If the code and this
file disagree, this file states the intent and the disagreement is a bug to report, not a licence to pick.

## 1. Why this study exists

The v4 deep comparison of objectives (pure hazard R3 vs binary R2) was **inconclusive at 0 of 5 budgets**. Measured
2026-10-04 on the local v4 dumps, before this file:

- **Test-set noise.** The streaming test split has 205 crossing and 495 never-crossing pedestrians. Resampling them
  moves one model's detection rate by about 4.3 points at 205 alarms/hour and 8.6 at 41.
- **Deep models do not pair.** On the same resampled pedestrians, the gap between two deep runs has an sd of 6.3 /
  12 points (R2 s42 vs R3 s42, at 205 / 41). Their alarms are barely correlated, so a paired test does not help.
- **Deep models memorize.** 15 of 20 v4 runs peak on validation AUC at epoch 1–3, then decline while train loss
  falls to 0.03–0.06.
- **A tree on the same input is better and stable.** The 2026-09-25 gradient-boosted tree probe, on per-window
  summaries of the identical 58-dim input, detects 56.7% / 38.9% of crossers at 205 / 41 alarms/hour. The deep
  binary model detects 42.3% / 22.9%. Two tree seeds on the same resampled pedestrians differ by about 2.3 points.

The paper's claim is about the training objective, not the architecture (decision 2026-09-26). So the comparison
moves to the learner that can resolve a gap of a few points.

## 2. What is already known (not blind)

- The **binary tree's level.** The probe numbers above are known, so arm `B` (the probe's successor with fixed
  settings) is not a blind measurement. Everything else is unseen: the hazard arms, the censored windows, the horizon
  sweep, the cue ablation, the learning curve, the matched-size control, and the anchored-trained tree.
- **Deep seed ensembles** were computed 2026-10-04 while planning (3-seed rank average, `per_window`
  @1200/460/205/95/41): R2 69.8/56.6/46.3/35.1/26.8; R3 72.2/61.0/48.8/39.5/23.4; R3C 72.2/60.5/49.3/36.6/30.2.
  They are **post hoc** and are reported only as such (`post_hoc_deep_ensembles` in the report).
- The tree probe's separability by time to onset (s42) is 0.835 within 1 s. Beyond 1 s it is *below* the deep R2
  (1–2 s: 0.757 vs 0.768).

## 3. Data and gates

Exports from the research PC (`scripts/export_window_features.py`): base streaming train `preprocessed_train`
(88,214 windows), the censored set `preprocessed_train_censored` (7,470), val, test, and the three benchmark sets.
The read path is `pose_kinematics`, 58 channels, **not standardized**. Before any fit, `run_tree_study.py verify`
must report:

1. **Counts** equal the pinned Dataset Statistics: train 88,214 / 2,530 positives, val 20,490 / 569, test
   69,875 / 2,140, censored 7,470.
2. **Row alignment:** the exported test split equals every local v4 test dump and the probe dumps, row for row
   (`crosses`, the onset fields, `track_id`).
3. **Deep replay:** the `c4_r2s_s42` checkpoint, re-scored from the exported test inputs with its own recorded
   standardization, reproduces its stored dump (max abs diff < 1e-3). This proves the export is exactly what the
   deep models consumed.
4. **Probe refit**, informative only: sklearn defaults with `class_weight="balanced"`. Expected within AUC ±0.01
   and ±3 points at 205/hr of the stored probe. The probe's own script was lost, so a miss is reported, not fatal.

Gates 1–3 failing stops the study.

## 4. Learner and features (fixed for every arm)

- **Features.** Per channel, over the 20 frames: last value, mean, last − first, sd. That gives 232 features (a
  channel subset for the cue arms).
- **Learner.** `HistGradientBoostingClassifier` with learning rate 0.1, max 300 iterations, 31 leaves, min 50 samples
  per leaf, L2 1.0, `max_features` 0.5, no early stopping, `random_state` = seed. Feature subsampling is the only
  source of seed randomness.
- **Binary objective.** One row per window.
  - `stored` = the stored `crosses` label: the tree twin of the deep R2.
  - `drop` at horizon H keeps only windows whose answer is knowable at H.
  - `zero` keeps every window and calls an unseen future "no crossing".
  - Already-crossed windows stay negatives, as in the stored label.
- **Hazard objective.** Person-period form of discrete-time survival: one row per *observed* bin, built by
  `onset_target.hazard_targets`. These are the same four cases as the deep loss: unobserved bins have no row, and
  already-crossed windows leave the risk set. The bin index is an extra feature. The readout is
  P(onset ≤ H) = 1 − Π_{k<H/w} (1 − h_k).
- **Selection reads validation only.**
  - Each fit is cut at the boosting iteration with the best validation AUC of its reported score.
  - Each arm picks its weighting, `none` or `balanced` (sklearn's rule as row weights), once, on the first seed's
    best validation AUC (tie → `none`).
  - Arms marked "inherits" reuse their parent's weighting, so they differ from the parent in exactly one thing.
  - The selection target is the stored validation `crosses` at H = 32. At other horizons it is the label at H on
    validation windows whose answer is knowable.
  - The anchored arm is selected on the benchmark validation set.
  - The test split is read only to score a finished model.

## 5. Arms

| arm | objective | training data | other | seeds |
|---|---|---|---|---|
| **B** | binary, stored | train | — | 42–46 |
| **Hz** | hazard, L 96, w 4 | train | — | 42–46 |
| **HzC** | hazard, L 96, w 4 | train + censored | also the H = 32 hazard of Q2 | 42–46 |
| Bd32 / Bz32 | binary, drop / zero, H 32 | train + censored | — | 42–46 |
| Bd64 / Bz64 | binary, drop / zero, H 64 | train + censored | — | 42–46 |
| Hz64 | hazard, L 128, w 8, H 64 | train + censored | — | 42–44 |
| Bd160 / Bz160 | binary, drop / zero, H 160 | train + censored | — | 42–46 |
| Hz160 | hazard, L 224, w 16, H 160 | train + censored | — | 42–44 |
| B_no_ego, B_box_ego, B_box, B_pose, B_ego | binary, stored | train | channel subset; inherits B | 42–46 |
| Hz_no_ego | hazard, L 96, w 4 | train | no ego channel; inherits Hz | 42–44 |
| B_f / Hz_f, f ∈ {0.125, 0.25, 0.5} | as B / Hz | stratified share f of training pedestrians | inherits B / Hz; seed = draw | 42–44 |
| A | binary, stored | benchmark (anchored) train | selected on benchmark val | 42–46 |
| B_matched | binary, stored | random len(anchored train) streaming windows | inherits B; seed = draw | 42–46 |

Horizon geometry: L = H + 64 frames (the 2 s confusable band beyond the horizon, as in the primary geometry), with
w = 4 / 8 / 16 so each horizon spans 8–10 bins.

## 6. Metric, test and decision rule (one rule for every question)

- **Metric.** Detection rate at {1200, 460, 205, 95, 41} alarms/hour (`eval/detection_curve.py`). `per_window` is
  primary; `per_track` is reported beside it. Each arm's rate is the mean over its seeds.
- **Test.** Paired pedestrian bootstrap (`eval/paired_bootstrap.py`) with 2,000 resamples. Crossing and
  never-crossing pedestrians are drawn separately with replacement, every arm and seed is scored on the same draw,
  and seeds are also drawn with replacement within each arm. The interval is the 95% percentile interval of the
  contrast.
- **Verdict per contrast** (`per_window`), among the five budgets:
  - **"X better"** if the interval excludes zero in X's favour at **3 or more** budgets;
  - else **"equivalent within 5 pp"** if the interval lies inside ±5 points at 3 or more budgets;
  - else **"inconclusive"**.
  - The `per_track` verdict is stated beside it. If the two disagree, both are reported, as in the paper.
- **Context, not verdicts.** Window AUC and F1 at the validation-tuned threshold (grid 0.001–0.999).

## 7. Questions and the reading fixed in advance

- **Q1 (PRIMARY): Hz − B.** The hazard objective against the binary one, same learner, same data.
  - *Hz better* → the reformulation improves detection once the learner can use it.
  - *Equivalent* → a powered null: at a 1 s horizon the label fix does not change detection by 5 points or more.
  - *B better* → the reformulation hurts.
  - *Inconclusive* → reported as such.
- **Q1b: HzC − B and Q1c: HzC − Hz.** Does adding the censored windows help?
- **Q2: horizon.** Prediction: the hazard gain over `drop` binary grows with H, because censoring only bites past the
  32 future frames that generation guarantees. Test: the difference of gaps (Hz160 − Bd160) − (HzC − Bd32) under the
  rule. The per-horizon gaps against `drop` and against `zero` are reported for H = 32, 64 and 160.
- **Q3: ego speed.** B_no_ego − B. If "B better", part of the detection on PIE comes from ego-vehicle speed, the
  driver's own reaction. A self-driving car cannot use that cue as evidence about the pedestrian. The cue table
  (box, box + ego, pose, ego alone) is descriptive.
- **Q3b: Hz_no_ego − B_no_ego.** Does Q1's reading hold without ego speed? Hazard arm at 3 seeds.
- **Q4: learning curve.** Detection at 205 / 41 alarms/hour and AUC against the fraction of training pedestrians, for
  B and Hz. Descriptive, plus the Hz − B rule per fraction (exploratory, 3 draws). Reading: if the curve still
  rises at full data, the task on this input is short of data, not at a ceiling.
- **Q5: matched size.** B_matched − A on the streaming test. Prediction: B_matched better, so the anchored model's
  collapse comes from the protocol, not the 18× smaller training set (the paper's first threat to validity). Also
  reported: A's streaming AUC (the cross-protocol collapse in a second model family) and the gap split in tree form.

## D. Loss-side deep arms (research PC, `queue_v5.sh`)

**Arms.** Four arms on the v4 recipe, 3 seeds each (42–44):

- anchored-trained **focal** (γ = 2, no class weights);
- anchored-trained **class-balanced** (effective-number weights, β = 0.9999, Cui et al. 2019);
- the same two trained on streaming.

**Design.**
- In all four the weighted sampler is **off**: the imbalance lever moves from the sampler into the loss. Everything
  else equals R2.
- Each arm is config-checked against `pf_fix_s42` before it trains.
- Selection is validation `crosses_auc`.

**Questions.**
- **D1, the paper's stated prediction.** Re-weighting moves the operating point, not the ranking, so it cannot
  rescue an anchored-trained model on the stream. **Holds** if the mean streaming-test AUC of *both* anchored loss
  arms is below 0.60. Fails otherwise; v4 anchored R2 was 0.516 ± 0.010. G_prior is recomputed for each.
- **D2.** Streaming focal / class-balanced against v4 R2 (sampler). Deep comparisons are under-powered (section 1),
  so this uses the v4 unpaired rule: mean gap greater than the sum of sds at 3 or more budgets. It is expected to be
  inconclusive and is reported as context.

## 8. Commitments

- Every verdict above is reported in the paper, whatever it is.
- A new arm or contrast added after any result is labelled post hoc.
- No selection ever reads the test split.
- Restarts resume finished seeds and never refit them.

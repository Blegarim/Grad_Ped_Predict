# Pre-specification: does the deep network show the scene audit's who/when pattern? (deep scene arm)

**Written 2026-10-09, before any scene-input network has been trained or any of its numbers seen.**

## Known beforehand

- **Trees (scene audit, `scene_audit/report.md`, 5 seeds).** Lights + traffic + layout (RTS) raised *who* from
  0.755 to 0.853 and *when* from 0.805 to 0.845. Per pedestrian, detection rose by +6.3 / +10.7 / +10.3 points at
  205 / 95 / 41 alarms/hr (verdict "better"); per window, +1.3 to +6.2 (inconclusive).
- **Deep base (v4 R2 streaming, `c4_r2s_s42/43/44`, 3 seeds).** *who* 0.766 ± 0.034, *when* 0.753 ± 0.008, window
  AUC 0.783, detection at 205/hr 42.3 ± 3.2 % per window and 86.3 ± 1.3 % per pedestrian.
- The network learns the ~240 training crossings within a few epochs and gets less out of the 58-dim input than the
  trees (`RESULTS_MATRIX.md`). Layout is constant per location, so the network may memorise locations instead.

We expect the tree pattern to replicate. The open risk is that the network does not use the scene input at all.

## Question

The scene audit used trees only. If the deep network, given the same scene information, also gains in *who* but
much less in *when*, the paper's central reading ("who responds to the information we can add; when does not")
holds in a second model family. A gain in *who* also shows that the network used the new input, so a flat *when*
can be read. Without that gain, nothing can be said about *when*.

## Arms

- **base:** the existing `c4_r2s_s42/43/44` runs and their streaming test dumps. Not retrained.
- **scene:** `c6_r2s_scene_s42/43/44`. The R2-streaming recipe (`recipes_v4.sh` R2S: `pose_kinematics`, pixel-free
  read, weighted sampler, AUC selection, warmup-cosine, batch 32, runtime augmentation, train-set input standardization)
  plus one change: the window's scene features (groups R, T and S of `scene_audit/PREREG.md`, the trees' 21 raw
  features), appended as constant channels on every frame of the 58-dim input. The network is otherwise unchanged;
  only its input width grows.
- **Encoding** (`data/scene_lookup.encode_scene`, fitted on the training windows only): numeric features
  standardized and clipped to ±5, a missing value set to 0 with a 0/1 missing column, categories (light states,
  intersection type, signalization, traffic direction) one-hot. The encoded width K is set by the sidecar build.
- The config check against `pf_fix_s42` may differ only in `train.seed`, `train.selection_metric` (as for every v4
  arm), `data.scene_features`, `data.scene_dim`, `data.motion_dim` and `model.motion_dim`.

## Gates (all must pass before the first scene arm trains; a failure is fixed and recorded as an amendment)

1. **Sidecar build:** every scene file has its export's row count, and every key shared by several windows (a PIE
   pedestrian split into segments) can be told apart by box, flipped copies included.
2. **Coverage:** every window of the arm's train dir (`preprocessed_train_shuffled`, offline copies included), val
   and test dirs resolves to a scene row through the production read path, at width 58 + K.
3. **Reproduction:** with the scene input off, the new code re-dumps `c4_r2s_s42`'s best checkpoint on streaming
   test with every window's score within 1e-5 of its stored dump.

## Questions and how each answer will be read

Each statistic is computed per seed; the rule is the paper's rule for networks: a difference counts when the mean
gap exceeds the sum of the two arms' standard deviations over seeds. *who* and *when* are
`eval/tree_study_report.intent_timing` (the paper's definitions).

- **Q1 (usage check; gates the reading of Q2): who(scene) − who(base).**
  - **Used** if the gap is positive and exceeds the sum of the standard deviations.
  - Otherwise **not shown to be used**: reported as "the network did not use the scene input"; Q2 is then
    reported but not read as evidence about *when*.
- **Q2 (primary, read only if Q1 = used): the *when* gain against the *who* gain.**
  - **Replicates** the tree pattern if Δwhen < ½ Δwho.
  - **Contradicts** it if Δwhen ≥ Δwho and Δwhen exceeds the sum of the standard deviations.
  - Otherwise **intermediate**.
  - Reported beside it, descriptive: *when* averaged within pedestrians, and *when* against hard temporal negatives
    in the confusable band (the paper's post hoc versions).
- **Q3 (secondary): detection.** Per pedestrian at 460 / 205 / 95 / 41 alarms/hr (1200 excluded: every arm detects
  ~100 %) and per window at all five budgets; "better" or "worse" at 3 or more budgets by the rule above, otherwise
  inconclusive. Per window we expect inconclusive, as for the trees.
- **Q4 (descriptive):** window AUC; separability by time to onset.

Everything above is reported whatever it shows, including a failed usage check. Anything added after the numbers is
labelled post hoc.

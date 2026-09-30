# Experiment context ? verified 2026-09-29

This records the locally available evidence, distinguishes current results from historical claims and plans, and does not certify unseen remote data or checkpoints. No training or checkpoint loading was performed.

## Current result set and independent verification

The canonical result set is the v4 pixel-free campaign: 20 completed runs, six arms with seeds 42/43/44 plus two Model A protocol runs at seed 42. Run configuration snapshots, 20 raw evaluation CSVs, generated table JSON and campaign completion log agree. All 20 streaming-test AUC values in the generated JSON match raw evaluation rows. `outputs/runs/RESULTS_MATRIX.md:19`; `outputs/diagnostics/v4_campaign/queue_v4.log:967`.

Independent array recomputation in `verify_experiments.py` reads nine stored NPZ files with `allow_pickle=False`, checks alignment on track_id, crosses, onset_offset, future_observed and track_crosses, and implements grouping, budget thresholds, detections, lead times and seed aggregation without importing report functions. All 360 detection/count/lead comparisons and nine AUC comparisons agree with the existing combination JSON. Results: `EXPERIMENT_VERIFICATION.json`. This verifies derived metrics from stored predictions, not model inference or the upstream dataset.

## What the canonical campaign measures

All v4 models use `pose_kinematics`, 58 pose/motion input dimensions, no image input, 20-frame windows, stride 3 and 30 fps. Pixel-free input is standardized from train statistics and clipped at 5; batch size is 32. Checkpoints are selected by validation `crosses_auc`. R2 uses binary crossing supervision; R3 uses hazard likelihood alone (hazard weight 1, binary weight 0, readout BCE weight 0); R1 adds hazard weight 0.1 to binary crossing; R4 adds readout BCE weight 0.5 to pure hazard; R3C uses the censored training store. Model A retains three tasks; Model B/crossing arms train only crosses. `outputs/runs/20260927_044252_pose_kinematics_c4_r3_s42/resolved_config.yaml:134`; `outputs/diagnostics/v4_report/tables/README.md:9`.

Do not confuse the historical experiment-tracking skill?s F1-primary instruction and generic CSV schema with current logged behavior: actual v4 configs select AUC and current CSV schemas contain protocol, split, tuned metrics and oracle fields. The latter are not deployable validation-selected numbers.

## Primary finding

The prespecified R3-versus-R2 detection comparison is **INCONCLUSIVE**, not a demonstrated hazard advantage and not proof of equivalence. The criterion requires R3 mean detection to exceed R2 by more than the sum of sample standard deviations at at least three of five budgets; it exceeds at zero. The budgets are correlated operating points, and this heuristic is not a formal equivalence or significance test. `outputs/diagnostics/v4_report/headline.md:11`.

| Alarms/hour | R2 detection % | R3 detection % | Gap in percentage points |
|---|---:|---:|---:|
|1200|69.6 ? 6.0|72.0 ? 2.8|+2.4|
|460|55.0 ? 6.2|59.2 ? 4.9|+4.2|
|205|42.3 ? 3.2|44.1 ? 4.7|+1.8|
|95|32.0 ? 4.2|31.4 ? 5.9|?0.7|
|41|22.9 ? 3.5|17.4 ? 9.1|?5.5|

Streaming test AUC / validation-tuned F1: R2 **0.783 ? 0.010 / 0.243 ? 0.008**; R3 **0.793 ? 0.010 / 0.277 ? 0.025**; R1 **0.797 ? 0.010 / 0.255 ? 0.032**; R4 **0.789 ? 0.008 / 0.245 ? 0.059**; R3C **0.795 ? 0.009 / 0.271 ? 0.028**. R1/R4/R3C secondary comparisons also fail the criterion under both accountings. GBM reference detection is **56.7 ? 1.2% at 205/hr** and **38.9 ? 1.0% at 41/hr**, higher than every neural arm?s means at those budgets. `outputs/diagnostics/v4_report/tables/window_metrics.md:7`; `outputs/diagnostics/v4_report/tables/detection_per_window.md:7`.

## Protocol mismatch is the stronger measured finding

Model B anchored?anchored test AUC is **0.889 ? 0.007**, versus anchored?streaming **0.516 ? 0.010**; validation-tuned streaming F1 of the anchored-trained arm is **0.062 ? 0.002**. Streaming?streaming AUC is **0.783 ? 0.010**, tuned F1 **0.243 ? 0.008**. Model A (one seed) reproduces the pattern: anchored?streaming AUC **0.511**, streaming?streaming **0.785**. `outputs/diagnostics/v4_report/tables/matrix.md:5`.

The recorded arithmetic attributes Model B F1 gap **0.180 ? 0.006** almost entirely to the residual named G_hardneg, with G_prior **?0.000 ? 0.001**. This establishes that threshold retuning recovers essentially none of the observed gap; the residual name alone is not causal identification of hard negatives. `outputs/diagnostics/v4_report/tables/gap.md:8`.

Who/when split: anchored-trained B ranks eventual crossers at pedestrian AUC **0.799 ? 0.010** but within-crossers timing AUC **0.414 ? 0.012**; streaming B has **0.766 ? 0.034 / 0.753 ? 0.008**. Trailing causal smoothing k=15 lowers streaming AUC (R2 ?0.004, R3 ?0.013); centered smoothing gains require look-ahead. `outputs/diagnostics/v4_report/tables/analyses.md:14`.

## Latest follow-up: September 29 combination test

A preregistered, training-free product combines the anchored model?s p_frame with the paired streaming R2 p_frame. Primary AUC is **0.757 ? 0.021**, versus R2 **0.783 ? 0.010**. Detection means at 1200/460/205/95/41 per hour are **69.8/54.3/42.3/31.2/23.1%** (R2 **69.6/55.0/42.3/32.0/22.9%**). Primary verdict is **INCONCLUSIVE**: zero of five per-window budgets exceed the sd-sum criterion; two of five per-track budgets, also insufficient. `outputs/diagnostics/combo_test/PREREG.md:11`; `outputs/diagnostics/combo_test/results.md:10`.

Secondary variants are causal running-mean anchored intent ? R2, validation-fitted logistic stacking of anchored/R2 logits and product, and anchored ? R3. All are inconclusive; selecting a secondary after seeing test results is prohibited by preregistration. This latest experiment does not support the complementary-who/when combination hypothesis under these rules.

## Selection and metric caveats

Window tuned F1 uses a validation threshold sweep 0.01?0.99 at 0.01 increments, applied to test. Detection curves instead choose thresholds from negative scores **in the same supplied test dump**, maximizing detections subject to a matched false-alarm budget. They are descriptive test-distribution operating points, not demonstrated operating rates under validation-fixed deployment thresholds. `outputs/diagnostics/v4_report/tables/README.md:19`; `src/pedpredict/eval/detection_curve.py:204`.

Detection denominator is 205 crossing pedestrians. Never-crossing exposure sums windows at 36,000 decisions/hour (30fps/stride3). False alarms exclude post-onset windows on crossing pedestrians. Per-window counts every alarming never-crossing window; per-track counts any alarm per never-crossing pedestrian with the same exposure denominator. At 1200/hr per-track, all models trivially reach 100% because only 495 negative pedestrians yield maximum rate ~477.5/hr. The long 14.34s mean lead there should not be presented as a difficult operating point. `src/pedpredict/eval/detection_curve.py:23`; `outputs/diagnostics/v4_report/per_track/curves.md:17`.

Only three training seeds and one underlying test split support the primary result; Model A is n=1. The original metric was chosen after observing F1 ties; the Sept 21 plan explicitly admits this and preregisters later seed confirmation. `docs/SEED_PLAN_2026-09-21.md:70`.

## Training stability and chronology

- July baseline/configured pose models and the old cross-protocol matrix are historical. The August 18 audit discovered unequal weighted-sampler settings between old Model B protocol legs; do not reuse that matrix as a confound-free comparison. `outputs/runs/RESULTS_MATRIX.md:10`
- September 18?20 single-seed image-model hazard results motivated the hypothesis; old R1 frame head was an auxiliary-trained proxy, not a clean binary baseline. Sept 21 plan demanded R2s and R3 across three seeds. `docs/SEED_PLAN_2026-09-21.md:21`
- September 25?26 v3 image-model rerun ended with problems; its paused run was evaluated, but queue records include a failed training job. `outputs/diagnostics/v4_campaign/queue_v3.log:1408`
- September 26 pixel-free recipe ladder achieved no-collapse runs but **NO-GO** on late validation decline, epoch-1 peaks for seeds 43/44 and validation-AUC range 0.0297. Mean test AUC 0.7851 and 41/hr detection 20.2% passed their gates. `outputs/diagnostics/rerun_gate/verdict.md:1`
- September 27 02:05:25 UTC: campaign launched by explicit **manual override**. Recipe was pf_fix plus validation AUC selection. Memorization ladder A completed, B stopped mid-run, C did not run; no memorization fix was adopted. `outputs/diagnostics/v4_campaign/OVERRIDE.md:1`
- September 28 10:45:25 queue log: v4 complete, every job succeeded; Sept 29 results ledger declares v4 canonical and combination follow-up is available. `outputs/diagnostics/v4_campaign/queue_v4.log:967`

Selected v4 epochs remain early: R2 seeds 42/43/44 choose **8/1/1**, R3 **2/2/2**, R1 **1/1/3**, R4 **5/1/3**, R3C **2/2/2**. Most validation AUCs deteriorate while training loss drops. No collapse epochs by the recorded detector does not mean the model is free of memorization or late decline. `outputs/diagnostics/v4_report/tables/training.md:5`.

## Availability and remaining plans

Local inventory: **39** training CSVs, **36** evaluation CSVs, **57** NPZ prediction/probe dumps, **zero** checkpoint .pth files under outputs/runs/*/checkpoints. All 20 v4 configs/logs are available; stored predictions enable analysis without checkpoints. `training_log/` contains only label_count.csv: original train/val/test windows 88,214/20,490/69,875 and crossing positives 2,530/569/2,140 (2.868%/2.777%/3.0626%). This original count table is not the augmented/shuffled sampler distribution. `training_log/label_count.csv:2`.

Planned/de-scoped matters are not measurements: JAAD replication, retraining published PCPA/SF-GRU baselines, onset look-ahead/bin-width sweep, and the uncompleted memorization ladder must remain labeled accordingly. The Sept 21 plan?s decision to leave R4/R3C n=1 was superseded by the v4 measured n=3 results, illustrating why plans cannot substitute for completed logs. `docs/SEED_PLAN_2026-09-21.md:84`; `outputs/diagnostics/v4_campaign/OVERRIDE.md:5`.

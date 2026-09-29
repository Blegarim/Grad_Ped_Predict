# Pre-registration: anchored "who" × streaming "when" (written 2026-09-29, before any combined number)

**Question.** The anchored-trained model ranks eventual crossers well (per pedestrian AUC 0.799) but its
timing is inverted (0.414); streaming-trained models time well (0.753–0.766). Does combining the two, with no
retraining, detect crossings better than the streaming model alone?

**Inputs.** Existing v4 campaign checkpoints only (no training). Test scores: the streaming-test dumps already
logged. Validation scores: new streaming-val dumps of the same checkpoints (inference only), used only by the
fitted variant. Seeds are paired: anchored `c4_r2a_sN` with streaming `c4_r2s_sN` (and `c4_r3_sN`).

**Primary (fixed now, no fitted parameters):** per window, `p_combo = p_anc × p_str`, where `p_anc` is the
anchored model's `p_frame` and `p_str` the streaming binary baseline's `p_frame` (R2).

**Secondary (reported, not the claim):**
- S1 — causal intent: `mean(p_anc over this pedestrian's windows so far) × p_str` (deployable, no look-ahead).
- S2 — fitted: logistic regression on streaming **val** with features `logit p_anc`, `logit p_str`, and their
  product; fitted per seed, applied unchanged to test.
- S3 — the primary product with R3's hazard read-out as `p_str` instead of R2.

**Metric and criterion (same as the campaign's).** Detection rate at {1200, 460, 205, 95, 41} alarms/hr,
`per_window` accounting, streaming test, mean ± sd over the 3 paired seeds. **The combination wins if its mean
beats R2 alone by more than the sum of the two sds at ≥ 3 of the 5 budgets.** Anything less is inconclusive.
`per_track`, window AUC and lead time are reported alongside.

**Not allowed after this point:** changing the combination rule, the pairing, the metric, the budgets or the
criterion; picking among S1–S3 by their test results.

# Hole Audit — v1.0 baseline

**Date:** 2026-06-11 · **Scope:** all of `src/pedpredict/`, `scripts/`, `configs/`, tests at coverage level.
**Method:** full two-pass read (read + catalogue, then verify + triage). Nothing below is speculation;
every item was traced to code. Where a claim rests on arithmetic rather than a traced execution path it
is marked *(derived)*.

**How to use this document:** each hole has an ID, a severity, an explanation, and either a patch or a
thinking direction. Answer inline under **Your call** — `accept` (it's fine, here's why), `fix` (patch
it), `investigate` (it becomes an experiment), or `defer`. The answered version of this file replaces
`PHASE_B_BACKLOG.md` as the working setlist.

> **Status (2026-06-11): ANSWERED & RESOLVED.** Every hole now carries a **Resolution** line recording
> the final decision and answering the inline questions. This file is now the working setlist;
> [PHASE_B_BACKLOG.md](PHASE_B_BACKLOG.md) is superseded (its unabsorbed items are noted there). The
> executable ordering is the **Final attack order** at the bottom — note in particular: **do not resume
> the v1 train LMDB build**; all dataset-touching decisions are batched into ONE v2 rebuild.

**Severity legend**
- **[M] Methodology** — threatens the *validity* of thesis numbers. These decide whether results survive an examiner.
- **[A] Architecture** — design choices that are unjustified, self-defeating, or block a research question.
- **[P1] Latent code bug** — won't crash today, silently corrupts results under specific conditions.
- **[P2/P3] Quality/perf** — debt, waste, or confusion; no validity risk.

**Headline finding (read this first).** Three holes form a chain: the default imbalance stack trains the
model on a `crosses` distribution of roughly 50–85% positive (vs. 2.8% at test) → the model is
guaranteed to be badly miscalibrated and over-predict → the eval pipeline then *tunes the decision
threshold on the test set* (M2), which silently compensates. The reported `opt_*` numbers are therefore
not just optimistic — they hide the imbalance-stack over-correction from you. M1 + M2 + M10 must be
resolved together, and they map directly onto the calibration/uncertainty research scope.

---

## Section M — Methodology (research validity)

### M1 · The imbalance levers triple-stack into an extreme, unmeasured training distribution
**Severity: M (highest).**
**Where:** [sampler.py:197-232](../src/pedpredict/data/sampler.py#L197-L232), [trainer.py:219-223](../src/pedpredict/training/trainer.py#L219-L223), [augment.py:186-195](../src/pedpredict/data/augment.py#L186-L195), defaults in `configs/train.yaml` + `configs/augment.yaml`.

The documented policy ("levers 2+3 ON, layered on augmentation") sounds mild. The arithmetic is not:

1. **Offline augmentation** multiplies every `crosses=1` record ×6 (≈2.5k → ≈15k items, many of them
   *byte-identical* copies — see C5), raising the union train rate from 2.6% to ≈10%.
2. **The online sampler** applies `(c0/c1)^1.5` relative weight. In a *base* (non-aug) chunk at 2.6%
   positive, that ratio is ≈37^1.5 ≈ **225×**. Expected positive fraction of drawn samples per base
   chunk ≈ 0.026·225 / (0.026·225 + 0.974) ≈ **86%** *(derived; actions^0.3·looks^0.7 modulate this
   somewhat)*. Each of a base chunk's ~130 positives is drawn ~30× per epoch pass.
3. **The loss class weights** are computed from the post-augmentation, *pre-sampler* label frequencies
   (`_build_loss` scans `train_lmdb_paths`, which includes aug chunks) — ≈4–5× extra weight on
   positives. But the sampler has already changed what the batches contain, so this inverse-frequency
   correction is applied **on top of** an already-rebalanced stream. The weights correct an imbalance
   that no longer exists at batch level.

Consequences: the model trains on a `crosses` prior wildly different from deployment; probability
outputs will be severely inflated; precision will collapse at the default threshold; and **no number
anywhere in the repo tells you the effective training distribution** — it has never been measured.

Additionally: **lever 3 has no off switch.** `TrainCfg` has `use_weighted_sampler` but no flag to
disable the inverse-frequency CE class weights — `Trainer._build_loss` computes them unconditionally.
The policy doc says "relax 2/3 when enabling lever 1," but the config surface cannot express relaxing 3.

**Patch direction:** (a) add `train.use_class_weights: bool`; (b) write a tiny script that *simulates*
one epoch of sampler draws over the real chunk scans and reports the effective per-task batch
distribution — make that number part of every run's log; (c) make the lever combination an explicit
ablation axis (none / sampler-only / weights-only / aug-only / current stack) — this is the M1
experiment, and it is precisely RQ3 in the research plan.
**Your call:** Pretty severe overcompensation for crosses, but rather straightforward fix is it not? a switch on each method and simple on-off argparse call during training would solve it?
**Resolution: fix now (flag + instrument); lever ablation in WP1.** Yes — the switch is the easy half,
with two corrections. (1) Per repo convention it is a config field, not argparse: add
`train.use_class_weights: bool`, driven by `--set train.use_class_weights=false`. With that one field
all four levers become switchable (`augment.enabled`, `balance.enabled`, `train.use_weighted_sampler`
already exist). (2) Switches alone leave you toggling blind — nothing reports what distribution a given
combination actually produces. The other half of the fix is the *instrument*: a small script that
simulates one epoch of sampler draws over the real chunk scans and logs the effective per-task batch
distribution into every run dir. Which lever combination is *right* is not a fix at all — that is RQ3,
answered by the WP1 lever ablation (none / weights-only / sampler-only / aug-only / current stack).

### M2 · Decision thresholds are tuned on the test set
**Severity: M (highest).**
**Where:** [metrics.py:197-221](../src/pedpredict/training/metrics.py#L197-L221) (`optimal_threshold_metrics`), called in [evaluate.py:187](../src/pedpredict/eval/evaluate.py#L187) on whatever split is being evaluated (default: test).

`run_evaluation` sweeps thresholds 0.10–0.90 and reports per-task metrics at the F1-maximizing
threshold — computed **on the same test data it reports on**. The `opt_*` columns in `eval_log.csv`
are textbook test-set leakage. If any `opt_*` number reaches a thesis table, an examiner who notices
will discount the entire results chapter. This is inherited OLD behavior, but the rebuild made it a
first-class, documented output, which makes it *more* likely to be quoted.

It interacts with M1: because the model over-predicts positives, the swept threshold will be far from
0.5, and the gap between `crosses_f1` and `opt_crosses_f1` will be large — the leakage is doing real
work, not cosmetic work.

**Patch (obvious):** tune thresholds on **val**, freeze them, apply to test. Concretely: run the sweep
in an eval pass over val, store per-task thresholds in the run dir, and have the test pass *load* them.
Keep the test-swept numbers if you like, but rename the columns (`oracle_*`) and never report them.
**Your call:** i dont know how this got through. test should have been complete unseen, separated, and non training/tuning set. this is sloppy work on my part
**Resolution: fix before the first baseline run.** (It was inherited from OLD, not introduced — the
rebuild's sin was promoting it to a documented output.) Mechanical patch as written: sweep on **val**,
store per-task thresholds in the run dir, test pass *loads* them; test-swept columns renamed `oracle_*`
and never reported. Q3's `overall_acc` rename rides this same change.

### M3 · All three labels are "any() over the future window" — the docs describe something else
**Severity: M.**
**Where:** [pie_sequences.py:101-119](../src/pedpredict/data/pie_sequences.py#L101-L119) (`_label_future_window`).

`actions`, `looks`, **and** `crosses` are all labeled as `any(signal[end : end+future_offset+tol])` —
i.e., every task is a *future* prediction: "will walk at least one frame in the next ~1s", "will look
at least once in the next ~1s". But CLAUDE.md/README describe `actions` as "walking/standing" and
`looks` as "looking at traffic or not" — present-tense state descriptions. Two problems:

1. **Semantic drift between docs and data.** An examiner asking "what exactly does actions=1 mean?"
   currently gets two different answers from the repo.
2. **Is future-any the *right* label for actions/looks?** The literature treats action/look as
   per-frame observation attributes (auxiliary tasks describing the *current* state). A one-frame
   glance anywhere in a 32-frame window makes `looks=1` — that's a noisy, threshold-free OR over a
   second of behavior. This was probably an accident of reusing the crosses labeling code, not a choice.

**Direction:** decide deliberately. Either (a) keep future-any and fix every doc to say so, or
(b) relabel actions/looks as state-at-end-of-observation (one-line change in `_label_future_window`,
requires sequence regen + stats table update + golden re-baseline). (b) matches the literature and
makes the auxiliary tasks cleaner; it changes ~nothing downstream because only label values move.
**Your call:** we can go with (b) for now, but will need to observe carefully changes in class imbalance, cause this is different logistic entirely
**Resolution: fix — (b), in the ONE v2 rebuild.** `actions`/`looks` become state-at-end-of-observation;
`crosses` stays future-any. Imbalance expectation: both positive rates should **drop** — `any()` over a
32-frame window can only inflate positives, and `looks` (a one-frame glance currently flips the label)
will fall hardest from its 12–17%. Procedure: regen sequences → run `count_labels` before/after and
diff → update the Dataset Statistics table + `test_stats` fixture in the same change (doc-sync). Loss
weights and sampler weights adapt automatically (both computed from the metadata scan), but re-check
`sampler_powers` for `looks` if its rate falls far.

### M4 · Right-censored windows: unobserved futures are silently labeled "0"
**Severity: M.**
**Where:** [pie_sequences.py:114](../src/pedpredict/data/pie_sequences.py#L114) — `future_end = min(end + future_offset + tol, n)`.

A window ending near the end of a track gets a *truncated* future window — down to zero frames — and
an empty/short future yields labels of 0 by construction. A pedestrian whose track ends 3 frames after
the observation (left frame, occluded, video ended) is labeled "will not cross" even though the future
is simply *unobserved*. This is the classic censoring problem, and at 2.6% positive rate, a small
number of censored false-negatives meaningfully pollutes the minority class — these are exactly the
"hard negatives" that teach the model wrong things.

**Patch (obvious):** require a full future window: `if end + future_offset + tol > n: continue` (one
line in `window_track`). Costs some windows; changes the dataset → stats table + `count_labels`
fixture + golden re-baseline in the same change (doc-sync checklist applies). Measure how many windows
are censored first — if it's >a few %, this also shifts the published positive rates.
**Your call:** another slop that got through. patch immediately, tho would affect sample count
**Resolution: fix — in the ONE v2 rebuild** (it changes the dataset, so it cannot land "immediately" in
isolation without forcing two rebuilds). First run a quick pkl scan to *measure* the censored-window
count and record it — the thesis needs the sentence "N windows excluded as right-censored."

### M5 · The evaluation protocol is not comparable to the published PIE benchmark
**Severity: M — gates WP0's "where do we stand" question.**
**Where:** windowing params in `configs/data.yaml` (`seq_len=20, stride=3, future_offset=30, tol=2`, filter #2).

The standard benchmark (Kotseruba et al., WACV 2021) samples ~16-frame observations at fixed
time-to-event (30–60 frames before the crossing point) and labels by the crossing *event*. This
pipeline slides 20-frame windows at stride 3 over *all* tracks, drops windows with observation-time
crossing, and labels by any-cross-within-32-frames. Different sampling population, different label
definition, different horizon (1.0–1.07s vs. 1–2s TTE). **Your `crosses_f1` and a published PCPA
`crosses_f1` are not the same quantity.** Comparing them in a table without a caveat would be wrong;
with the caveat, the comparison is weak evidence at best.

**Direction:** decide what "comparison to literature" means for the thesis. Options: (a) implement a
benchmark-protocol evaluation mode (TTE-sampled windows from the same sequence pkls — an eval-side
change only, no retraining needed); (b) keep your protocol, position results as self-contained, and
compare only *within* the repo (ablation thesis framing); (c) both. (a) costs maybe a week and buys
you the only externally-anchored row in the thesis. Recommend (c) with (a) scoped early.
**Your call:** this is confusing. ive read a number of literature and they have all been using various benchmark. but i guess (c) is applicable - theres a defensible reason for me to choose the observation and prediction window, though its a bit arbitrary ill admit, 20 frame aka 2/3rd of a second observation cause pedestrians may appear suddenly and abruptly, and 1 second prediction window because decision must be made NOW and within the very near future for integrity. but i guess we should add additional benchmark - but would that not mean re running sequence generation and data preprocessing all over again?
**Resolution: (c), and no — no full redo.** Two reasons. (1) Training is untouched: the model stays
trained on your protocol; the benchmark protocol is an *additional evaluation* of the already-trained
model. (2) The benchmark artifact is **test-split-only and small**: TTE-sampled windows exist only for
pedestrians with crossing-event annotations — a few thousand windows vs. 76k — generated from the same
PIE annotations by a sequence-gen mode flag and built once with the same writer. And since M3/M4/A4/M9
already force one full regen+rebuild, the benchmark eval set is produced *in that same pass* at near-zero
marginal cost. Your 20-frame/1-second rationale (sudden appearance; the decision must hold for the
immediate future) is defensible — write it into the thesis protocol section as a deliberate
early-anticipation choice, with the benchmark row as the externally-anchored comparison (caveated).

### M6 · Window-level metrics over heavily correlated samples
**Severity: M.**
**Where:** [evaluate.py](../src/pedpredict/eval/evaluate.py) — metrics are computed over all windows.

Stride-3 windows from one track overlap by 17/20 frames; consecutive test samples are near-duplicates.
Effects: (1) long, easy tracks dominate the metric; (2) the effective sample size is far below the
nominal 76k, so any future significance claim ("model A beats B") computed over windows is
overconfident; (3) per-window accuracy says nothing about per-pedestrian performance, which is what
the task actually demands.

**Patch direction:** keep window metrics (comparable to your own past runs), *add* track-aggregated
metrics: group predictions by (track), aggregate (e.g., mean prob or majority), report both. Requires
carrying a track/video id through the LMDB meta — currently **discarded at sequence generation**, so
add a `track_id` field to `SequenceRecord` + meta (writer change → rebuild or sidecar-map it from the
pkls, which are ordered and recoverable without rebuilding).
**Your call:** huh??
**Resolution: fix (track_id in the v2 rebuild; aggregation in eval).** Plain-language version: with
`seq_len=20, stride=3`, two consecutive windows of the *same pedestrian* share 17 of their 20 frames —
they are near-identical inputs with (almost always) the same label. A pedestrian tracked for 600 frames
therefore contributes ~200 of these near-copies to the test set; one tracked for 25 frames contributes 2.
Three consequences: (1) long, easy tracks dominate the score — getting one easy pedestrian right is
rewarded 200×; (2) the 76k "samples" behave statistically like far fewer independent ones, so any
"model A beats model B" comparison over windows looks more certain than it is; (3) you never see
per-*pedestrian* performance, which is what the safety task actually asks for. The fix keeps window
metrics (continuity with past runs) and **adds** track-level ones: carry a `track_id` field into the
LMDB meta (one line in `SequenceRecord` + writer — rides the v2 rebuild), then in eval group test
predictions by track, aggregate (mean prob per track), and report per-track metrics alongside.

### M7 · Training is completely unseeded
**Severity: M + P1.**
**Where:** `utils/seed.py` defines `set_seed`; **no script or the Trainer ever calls it** (verified by
grep — the only `manual_seed` calls are in offline augmentation, benchmark, and ONNX dummies). There is
no `train.seed` config field.

Weight init, WeightedRandomSampler draws, chunk shuffle order, dropout masks, DataLoader worker
ordering: all unseeded. Two consequences: (1) no run is reproducible, contradicting the repo's central
"reproducible, config-driven" claim; (2) you cannot distinguish a real improvement from seed noise —
fatal for an ablation-style thesis where effect sizes will sometimes be small.

**Patch (obvious):** add `train.seed: int` to `TrainCfg` + yaml; call
`set_seed(cfg.train.seed)` at the top of `scripts/train.py::main` (and evaluate.py for symmetry); log
the seed in the run-dir snapshot (free — it's in the config). Decide the multi-seed protocol now:
screen with 1 seed, confirm finalists with 3; report mean±std.
**Your call:** yeah this one is indeed obvious.
**Resolution: fix immediately** as written — `train.seed` field, `set_seed` at the top of
`train.py`/`evaluate.py`, seed lands in the run snapshot for free. Multi-seed protocol adopted: screen
with 1 seed, confirm finalists/headline comparisons with 3, report mean±std.

### M8 · Model selection, early stopping, and LR schedule all hang off class-weighted val loss
**Severity: M.**
**Where:** [trainer.py:290-322,376-392](../src/pedpredict/training/trainer.py#L290-L322).

`val_loss` is the *class-weighted* multitask loss. With crosses positives weighted ~5–19× and only
~570 positives in val, a handful of hard positive windows dominate the scalar that picks `best.pth`,
steps the scheduler, and stops training. Selection is therefore high-variance and skewed toward
whatever the crosses head does on a tiny subpopulation — while the thesis metric is F1. (Known as
Phase B item 9; flagged here because it interacts with M7: noisy selection + no seed = irreproducible
"best" checkpoints.)

**Direction:** small, contained experiment — add `train.selection_metric: {val_loss, macro_f1,
crosses_f1}` and switch best-checkpoint + early-stop to it (scheduler can stay on val_loss). One
config field, ~15 lines in Trainer. Do it *before* WP0 baselines, or the baselines are selected by the
noisy criterion and everything after inherits it.
**Your call:** hmm alright but, i feel like we can do something else with what we have, or even trimming of it rather than some more overhead
**Resolution: fix before WP0 — and it IS "with what we have", zero overhead.** The val pass *already*
computes `macro_f1` and `crosses_f1` every epoch (the shared `MetricAccumulator` runs at validation;
the columns are already in `train_log.csv`). The change is only *which scalar* the best-checkpoint
comparison and early-stop counter read — a one-line comparison swap (plus a sign flip: F1 is
maximized). The `train.selection_metric` field adds no computation; it just names the choice so the run
snapshot records it (config-first convention). Default `macro_f1` — selecting on `crosses_f1` alone is
itself high-variance with ~570 val positives. Scheduler stays on `val_loss`.

### M9 · The ablation ladder is missing its most important rungs
**Severity: M (it's the thesis).**
**Where:** [registry.py](../src/pedpredict/models/registry.py) — four model types only.

Missing, in order of importance:
1. **Kinematics-only baseline** (bbox features → small GRU/MLP; *no pixels at all*). The literature's
   embarrassingly-strong baseline. Without it, "what do pixels buy?" is unanswerable — note that
   `motion_only` does NOT answer it (see A6). An afternoon of work.
2. **Single-task baselines** (`crosses` alone vs. joint). Multi-task with `loss_weight={0.8,0.8,1.2}`
   is assumed beneficial, never tested; negative transfer is common. Zero new code if implemented as
   `loss_weight={actions:0, looks:0, crosses:1}` — check that a 0 weight is honored end-to-end (it is
   in the loss; metrics will still report the dead heads, which is fine).
3. **Ego-vehicle speed** — PIE ships OBD; the 8-dim motion vector ignores it; the literature says it's
   one of the strongest cues *and* a known causal confound (the driver brakes because the pedestrian
   will cross). Adding it (motion_dim 8→9/10, flip-aware) plus an on/off ablation is both a likely
   accuracy jump and the cheapest *scientific* finding available (the leakage probe).

**Patch direction:** all three are registry/config additions that ride existing infrastructure. The
kinematics-only model needs a writer-side nothing (motions already stored); ego-speed needs a writer
change + rebuild (bundle it with whatever M3/M4 relabeling decision forces anyway — **batch the
dataset-touching decisions into ONE rebuild**).
**Your call:** alright, this is gonna be quite a bit, so drink up. Firstly - the kinematic stuff: Somehow incorporate ego-speed but preserve model structure for both with and without, would we need padding for the additional dimension that would otherwise be blank without ego-speed?; ablation for each of the components of the motion encoder line against each other, although i think this is a bit redundant and too much - the branch is about 1M params count, and we have not even established that it affects the model extensively.; the "no pixel, pure motion" line is concrete and straightforward tho. Secondly - single task base line: setting weight to 0 0 1 is ridiculous but sound so right that i have never even think about it. Thirdly - ablation on the ViT is done through drop in model thats fine, but question is we cannot perform ablation on every variables of the model - about 6 7 from motion encoder and 3 4 ViT spawn upward 30 models - training takes time as i mentioned limited hardware, and its just not feasible, so what do we serve as our benchmark?
**Resolution — point by point:**
1. **Ego-speed: no padding.** Store wide, slice narrow: the v2 writer always stores the full motion
   vector (8 corrected channels + ego-speed → 9), and the dataset slices to `data.motion_dim` at load
   time. The model's input layer is sized from `model.motion_dim` (already a config field,
   cross-checked against `DataCfg` in `validate_config`), so with-ego and without-ego are simply two
   configs → two trained models — an ablation retrains anyway, so a fixed input width buys nothing.
   Zero-padding a dead channel would only inject noise and muddy the param-matched comparison.
2. **Motion-encoder internal ablation: agreed, deprioritized** — ~1M params and no evidence yet that
   the branch's internals matter; A5 stays "time permitting." **Kinematics-only: proceed** (registry
   entry; motions already in the meta, so no writer work).
3. **Single-task: proceed** via `loss_weight={actions:0, looks:0, crosses:1}`. Two footnotes: zero the
   `sampler_powers` for the disabled tasks in those runs too (or the sampler still rebalances on dead
   tasks), and verify the 0-weight is honored end-to-end (it is in the loss; metrics will still print
   the dead heads, which is harmless).
4. **What serves as benchmark: hub-and-spoke, never factorial.** The hub is the WP0 `full` baseline
   under the frozen protocol (v2 dataset, fixed seed, val-tuned thresholds, selection metric). Every
   ablation changes exactly **one** axis and is compared back to the hub — never cross-compared between
   axes. Budget: WP0 ladder ≈ 6 runs (kinematics-only, ped_local, visual_only, vanilla_concat, full,
   crosses-only); then per-axis spokes: backbone swap (1–2), fusion grid (3), motion v1-vs-v2 (1),
   ego on/off (1), imbalance levers (4) ≈ 10–12 more. One seed for screening, 3 seeds only for the 3–4
   headline comparisons → **~20–25 trainings total spread over WP1–WP3**, feasible on the A4500. If two
   single-axis wins look compoundable, test that ONE combination as the v2 candidate — don't search the
   grid.

### M10 · There is no coherent threshold/calibration policy anywhere in the system
**Severity: M — and it is the bridge to the control-systems research scope.**
**Where:** [inference.py:311-332](../src/pedpredict/eval/inference.py#L311-L332) (argmax = implicit 0.5), [metrics.py:197](../src/pedpredict/training/metrics.py#L197) (test-swept), training (no calibration).

Three different implicit answers to "when do we say *crossing*": training optimizes a weighted loss
(M1), eval reports both 0.5-threshold and test-swept-threshold numbers (M2), and video inference
hard-argmaxes at 0.5. Given M1, the 0.5 operating point is meaningless and the deployed inference path
will flag pedestrians constantly. Nobody has ever looked at a reliability diagram of the crosses head.

**Direction:** this is not a patch, it's the calibration workstream: (1) after WP0, plot reliability
diagrams (predictions NPZ already has the probs — `viz` work only); (2) temperature-scale on val;
(3) define the single canonical operating-point policy (val-tuned threshold, stored in the run dir,
consumed by eval AND `infer_video`); (4) conformal prediction on top if the control framing proceeds.
**Your call:** i dont get it. what calibration/threshold are we discussing here?
**Resolution: investigate (post-WP0 workstream). Explanation — two distinct things:**
- **Threshold** = the cutoff that turns the model's p(cross) into a yes/no decision. The repo currently
  answers "what cutoff?" three different ways: eval reports metrics at 0.5 (argmax) *and* at the
  test-swept optimum (the M2 leakage); `infer_video` hard-codes argmax (= 0.5); training never defines
  one at all. No single place owns "the operating point."
- **Calibration** = whether p itself is honest: of all pedestrians the model assigns p≈0.8, do ~80%
  actually cross? Because M1 trains on batches that are ~50–85% positive while reality is 2.8%, the
  model's learned prior is wrong by an order of magnitude — its probabilities are hugely inflated, so
  model-p = 0.9 may correspond to a true frequency of ~10%. A **reliability diagram** (bin predictions
  by p, plot observed positive frequency per bin) makes this visible; **temperature scaling** (one
  scalar fitted on val that rescales the logits) is the standard cheap correction.
- Why it matters: F1-at-0.5 is meaningless if 0.5 is nowhere near a sensible operating point; and a
  *calibrated probability* is exactly the quantity a downstream planner/controller would consume — this
  is the bridge to the control-systems framing. Policy adopted: calibrate on val → pick threshold on
  val → store both in the run dir → eval **and** `infer_video` load them. M2's val-tuned threshold
  mechanism is the first rung; reliability diagrams + temperature scaling follow WP0; conformal only if
  the control framing proceeds.

---

## Section A — Architecture

### A1 · The ViT's stage schedule fights itself: 36→36→288→36 dims, and the 288-dim stage attends over 2×2 windows
**Severity: A.**
**Where:** `configs/model.yaml` (`stage_dims`, `window_size`), [vit.py](../src/pedpredict/models/vit.py).

Three compounding oddities, inherited from the undergrad design:
- **Dimension collapse:** the deepest features pass through 288 dims (stage 3, 5 layers, 16 heads) and
  are then *crushed to 36 dims* in stage 4 before being projected back up to `d_model=128`. The
  per-frame visual representation bottlenecks at 36 floats.
- **Tiny attention windows where it matters:** stage 3 (the expensive one) uses 2×2 windows — its
  attention mixes exactly 4 tokens at a time. Most of the model's attention FLOPs buy almost no
  receptive field.
- **No cross-window mixing within a stage** (no shifted windows à la Swin, no overlap): windows only
  ever communicate through the stride-2 downsample between stages.

No examiner question here has a good answer except "legacy parity." This is RQ1 territory.
**Direction:** don't hand-tune this; it's the strongest argument for the pretrained-backbone swap
(TinyViT-5M / PVTv2-B0 — hierarchical, windowed, pretrained, ~4–5M params). The drop-in contract is
just pooled-features→`frame_proj`→128. Benchmark old-vs-new at matched params; expect a large gap.
**Your call:** the architecture was adopted from Swin-Transformer and Hiera - ViT without bells and whistles, but i'll admit 2x2 windows is just funny. maybe no hand tweaking but i think we definitely should do something about it
**Resolution: fold into the A2 backbone swap** — one experiment answers A1+A2; no hand-tuning of the
stage schedule. The 36→288→36 collapse and the 2×2-window observation go into the thesis as the
*motivation* for the swap (they make the "why replace it" argument for you).

### A2 · From-scratch ViT on ~96k small crops
**Severity: A.** (Stated in earlier discussion; recorded here for completeness.)
ViTs are data-hungry; every competitive PIE method uses pretrained backbones. The current visual
stream has never seen anything but PIE crops. Same resolution as A1 — one experiment answers both.
**Your call:** like you have mentioned "Pretrained lightweight hierarchical ViT — these exist, and one is almost exactly what you described. TinyViT-5M (Microsoft, ECCV 2022) is hierarchical, window-attention based — genuinely Swin-style — at ~5.4M params, pretrained on ImageNet-21k with distillation, and available in timm. Other candidates in the same weight class: PVTv2-B0 (~3.4M, hierarchical pyramid but spatial-reduction attention instead of windows), MobileViT-S (~5.6M, hybrid CNN-ViT), FastViT-T8 (~4M), and DeiT-Tiny (~5.7M, plain non-hierarchical ViT but trivially resolution-flexible)." we should investigate these models explicitly and select which is appropriate to slap in
**Resolution: investigate — backbone candidate study, scoped early in WP2** (~2–3 days desk work + a
day of harness). Rank TinyViT-5M / PVTv2-B0 / MobileViT-S / FastViT-T8 / DeiT-Tiny on: (a) timm
availability with pretrained (ideally 21k) weights; (b) accepts the actual context-crop resolution
without surgery; (c) hierarchical/windowed — keeps the factorized-space-time story intact; (d)
params/FLOPs at a budget matched to the current ViT; (e) measured A4500 latency; (f) clean
pooled-features→`frame_proj`→128 drop-in. Deliverable: one design note naming a primary + a fallback.
Prior favorite: TinyViT-5M (genuinely Swin-style, so the architecture narrative survives the swap).

### A3 · The cross-attention "fusion" contains no motion content — only motion-shaped attention
**Severity: A (analysis finding, not a bug).**
**Where:** [cross_attention.py:87-105](../src/pedpredict/models/cross_attention.py#L87-L105).

`attn_output = MHA(query=motion, key=image, value=image)` and there is **no residual connection** —
the heads consume `attn_output` alone. Since MHA's output is a weighted sum of *value* vectors, the
classifier sees only image-derived features, reweighted by motion-image affinity. Motion information
enters solely through *where the attention looks*, never through *what is represented*. Whether this
is a feature (motion as a saliency/gating signal) or a defect (motion content discarded at fusion) is
exactly the fusion research question (RQ2) — but the current architecture description ("fuses both
modalities") oversells what happens.

**Direction:** the fusion study has a natural, cheap grid: + residual from motion (`attn_output +
motion_feats`), query/key swap, bidirectional + concat, vs. existing `vanilla_concat`. Note
`vanilla_concat` is currently the *only* variant where motion content reaches the heads directly —
if it matches or beats `full` in WP0, this hole is the explanation, and that's a publishable
observation about the architecture.
**Your call:** this is a migration bug. i added residual explicitly in OLD, but thats alright. straightforward fix no investigation
**Resolution: ⚠️ re-audited — this is NOT a migration bug; the recollection is wrong.** Verified against
the `legacy-archive` tag (`OLD/Undergrad_thesis_project/models/Cross_Attention_Module.py`): OLD's
forward is `attn_output, _ = self.cross_attn(...)` and everything downstream (pool MLP, heads,
frame head) consumes `attn_output` alone — **no residual at the fusion**. The residual you remember is
*inside* `Motion_Encoder.py` (`self.proj(residual + self.dropout(x))`, around its own self-attention),
which the rebuild preserves. Independent confirmation: the `cross_attention.pt` golden parity test
passes — it could not if the rebuild had dropped a residual. Consequence: adding
`attn_output + motion_feats` is a **behavior change, not a parity fix** → land it behind a
`model.fusion_residual` config flag with a golden re-baseline, and it becomes the *first rung of the
RQ2 fusion grid* (compared against the hub like every other spoke), not a silent bug fix.

### A4 · Per-sequence motion normalization destroys geometry and amplifies quantization noise — and the frame-0 quirk corrupts two channels
**Severity: A.**
**Where:** [motion_encoder.py:123](../src/pedpredict/models/motion_encoder.py#L123), [transforms.py:102-140](../src/pedpredict/data/transforms.py#L102-L140).

`(motion - mean_T) / (std_T + 1e-6)` per sequence, per channel. Three consequences *(derived)*:
1. **Absolute geometry is erased.** Position in frame (curb proximity!) and absolute box size
   (distance proxy) survive only as temporal *patterns*, not values. For crossing prediction, where
   the pedestrian *is* matters enormously; the model never cleanly sees it.
2. **Quantization-jitter amplification.** Boxes are int-truncated; a slow/far pedestrian's dx channel
   is mostly ±1-px jitter. Z-normalization rescales that jitter to full-scale signal —
   indistinguishable from real motion.
3. **The frame-0 `dw`/`dh` quirk is worse than documented.** `dw[0] = w0` (raw width, ~50–300) among
   deltas of ~±2 means the channel's std is dominated by the t=0 spike; post-norm, `dw`/`dh` are
   ≈ a one-hot spike at t=0 encoding initial size, with the actual size-change signal crushed to
   near-zero. The "preserved quirk" doesn't just bias one value — it effectively deletes two of the
   eight channels and re-purposes them as a redundant initial-size feature.

**Direction:** motion representation v2 (Phase B item 4, now with mechanism): normalize by *image
dimensions* (fixed, global) instead of per-sequence stats; fix frame-0 deltas to 0; keep absolute
(normalized) cx, cy, w, h as real features; make flip reflect cx. One writer rebuild (batch with M9's
ego-speed). Ablate old-vs-new representation — likely one of the larger single wins available.
**Your call:** holy shit this is lowkey the biggest production breaking slop. please fix that yes
**Resolution: fix — motion v2, in the ONE rebuild, with a useful split.** Only the *data-side* fixes
need the writer (frame-0 deltas = 0; flip reflects cx in augmentation; ego-speed channel from M9). The
*normalization choice* (per-sequence z-norm vs. fixed image-dimension norm) lives at runtime in
`motion_encoder.py:123` — make it a model/config flag, so old-vs-new normalization is ablatable from
the **same** v2 data with no extra build. Keep absolute (image-normalized) cx, cy, w, h as real features.

### A5 · The motion branch stacks three temporal mechanisms; the visual stream has none
**Severity: A.**
**Where:** [motion_encoder.py](../src/pedpredict/models/motion_encoder.py) (Conv1d + GRU + MHA, plus
positional encoding added *after* the GRU), [vit.py:308-320](../src/pedpredict/models/vit.py#L308-L320)
(strictly per-frame).

Asymmetric and unexamined: temporal reasoning is CNN→GRU→self-attention piled in one branch (never
ablated against each other; the pos-encoding-after-GRU ordering is also odd — the GRU is already
order-aware), while context frames never exchange information before fusion. The "factorized
space-time" defense (cite your 2022 source paper) covers the *visual* side; it does not justify the
triple stack on the motion side.
**Direction:** secondary ablation if time permits (drop GRU / drop attention / drop CNN — three runs).
Low priority vs. A1/A4.
**Your call:** this has been explained per M9, we're adding ablations. im gonna stretch the factorization justification as far as we can go before we crash.
**Resolution: defer (time permitting), per M9.2** — the internal triple-stack ablation is a low-priority
spoke behind backbone/fusion/motion-v2. The factorization citation covers the visual side in the thesis.

### A6 · `motion_only` is not motion-only — it sees the pedestrian's pixels
**Severity: A (interpretation hazard).**
**Where:** [ablations.py:71-115](../src/pedpredict/models/ablations.py#L71-L115) — `MotionOnlyModel`
consumes `images_tight` through the MotionEncoder's CNN.

The ablation suite's naming will cause a wrong conclusion: `motion_only` is really *pedestrian-local*
(appearance + kinematics, no scene context). If it matches `full`, you cannot tell whether context is
useless or whether kinematics alone suffice. The missing kinematics-only rung (M9.1) resolves this;
also rename or re-document the variant (`ped_local`?) before any results table is written —
table semantics are nearly impossible to fix retroactively in readers' heads.
**Your call:** its just a rename i guess. a whole wave of ablation is also coming in so a whole wave of relabel should come as well
**Resolution: fix — `motion_only` → `ped_local`, in the same change that adds the M9 registry entries**
(one rename wave: registry key, configs, README/CLAUDE.md tables, run-naming in `index.csv`). Do it
before any results table exists.

---

## Section C — Code correctness (latent)

### C1 · Chunks are silently skipped on warm timeout — including validation chunks
**Severity: P1.**
**Where:** [chunk_loader.py:125-138, 181-198](../src/pedpredict/training/chunk_loader.py#L125-L198).

If a warm worker doesn't report within `chunk_queue_timeout=300s`, `_await_chunk` returns `None` and
`__next__` silently `continue`s — **no log line whatsoever** (verified: no print/raise on that path).
On your HDD, a cold 300s warm is plausible. Consequences: a training epoch silently trains on fewer
chunks; worse, **`val_loaders` uses the same iterator**, so `val_loss` — which selects best.pth and
stops training (M8) — can be computed on a silently varying *subset* of validation. Metrics become
non-comparable across epochs without any visible sign.

**Patch (obvious):** at minimum, log loudly (`print`/`warnings.warn`) with chunk path on every skip;
for validation, make a skip a hard error (`raise`) — a partial val set is never acceptable. Two small
edits in `_await_chunk`/callers + a config flag if you want skip-tolerant training.
**Your call:**

### C2 · Incremental-build resume trusts a partial final chunk — silent data loss
**Severity: P1 (live risk for *your next action* — finishing the train LMDB).**
**Where:** [incremental.py:56-58](../src/pedpredict/data/incremental.py#L56-L58) (`next_chunk_start`),
[build_lmdb_incremental.py:57-61](../scripts/build_lmdb_incremental.py#L57-L61).

`next_chunk_start` resumes one past the *highest existing* `chunk_*.lmdb` dir. If the previous run
died mid-write (your disk-full crash is exactly this), the highest chunk exists but is **incomplete**,
and auto-resume skips past it forever. The script's guard (`if partial exists at start_idx → exit`)
checks the *new* start index, which by construction doesn't exist — it protects only against an
explicit `--start-idx` collision, not the crash case. The docstring says "delete the partial final
chunk first," but nothing enforces it; the failure is silent and surfaces months later as a `count_labels`
drift or a mysteriously easier training set.

**Patch (obvious):** before resuming, open the highest chunk and count `_meta` keys; expected count is
`min(chunk_size, n_records - start)`. If short → refuse with a clear message (or auto-delete and
rebuild that chunk). ~15 lines; add a test. **Do this before resuming your train build.**
**Your call:**

### C3 · LMDB `map_size` over-reserves ~2× on Windows
**Severity: P1 (operational — this is plausibly why your disk filled).**
**Where:** [lmdb_writer.py:53-72](../src/pedpredict/data/lmdb_writer.py#L53-L72).

The heuristic gives ≈7.4 GB map_size per 5000-sample chunk *(derived: 5000·2·512²·3·0.25·5·1.5)*;
actual JPEG payload is ≈2–3 GB. The code's own comment notes Windows **pre-allocates the file at
map_size**. Across ~20 train chunks that's ~90 GB of reserved-but-empty disk on a 1.8 TB drive that
already filled once.

**Patch (obvious):** set `data.lmdb_map_size_bytes` explicitly (measure one finished chunk, add 30%),
or post-compact each chunk (`env.copy(compact=True)` + swap), or lower the fudge factors on Windows.
Cheap, immediate disk relief.
**Your call:**

### C4 · The "NOISE" augmentation is a de-facto identity, and the plan emits exact duplicates by design
**Severity: P1 (silent ineffectiveness, not corruption).**
**Where:** [augment.py:116-118, 163-183](../src/pedpredict/data/augment.py#L116-L183), `motion_noise_std: 0.02` in `configs/augment.yaml`.

`motion_noise` adds N(0, **0.02 px**) to raw pixel-unit channels (cx ~ hundreds, dx ~ units). After
the encoder's per-sequence normalization, that's ~0.004–0.02σ — imperceptible *(derived)*. So one of
the four augmentations does nothing, and on top of that the plan intentionally emits an *identity
copy per cycle*: with `crosses_multiplier=6`, each crosses-positive exists ≈4× as byte-identical
samples + ≈2-3 transformed copies (and records positive for both crosses and looks get expanded in
both lists — 9× total). The augmented LMDB is substantially a duplication machine that the sampler
(M1) then multiplies again.

**Patch direction:** fold into the M1 redesign rather than patching in place — if augmentation
survives the lever ablation, fix the noise std (express it in *normalized* units or scale per
channel), drop identity copies (the sampler already oversamples), and dedupe the double-expansion.
**Your call:**

### C5 · `min_track_size=10` admits tracks the windower then silently discards
**Severity: P3 (config lies).**
**Where:** `configs/data.yaml`, [pie_sequences.py:139](../src/pedpredict/data/pie_sequences.py#L139).
PIE filters tracks ≥10 frames; `window_track` needs ≥20. Tracks of length 10–19 flow through and
produce zero windows. Harmless, but the config field implies a control it doesn't have. Set it to
`seq_len` or document it.
**Your call:**

Topple: All C holes are quite obvious straightforward quick fixes rather than decision making. should be safely processed as necessary.

**Resolution: all five C holes = fix, as specified — with one sequencing override that changes the
original attack order.** C2/C3 were scoped as "before resuming the train build." But the resolved M3/M4
relabel, M6 `track_id`, A4 motion fixes, and M9 ego-speed **all change the dataset**, which means:
**do NOT resume the v1 train build at all.** Finishing the remaining ~75k v1 windows on the HDD would be
days of IO thrown away and rebuilt within weeks. The existing ~20k v1 train chunks are obsolete; the
ONE v2 rebuild regenerates sequences and LMDBs for **all three splits** (val/test are v1 too — they get
rebuilt under the new labels/meta as well). C2's resume guard and C3's map_size fix still land *first*
because the v2 build itself needs them; C1 lands before any training run (val skips become hard errors).
---

## Section Q — Quality / performance (no validity risk)

- **Q1 · The warm machinery doesn't warm.** [lmdb_warm.py:24-43](../src/pedpredict/data/lmdb_warm.py#L24-L43)
  reads exactly *one* `_meta` key — the whole spawn/queue/RAM-gate apparatus pre-loads a few KB of a
  multi-GB chunk. On the HDD this means the prefetcher delivers ~none of its intended benefit. Either
  warm for real (sequential full-file read, cheap on HDD) or delete the machinery when Phase B item 7
  lands standard sharding. -> warm it up then gng.
- **Q2 · Windows worker-respawn per chunk.** `persistent_workers=False` + spawn + 4 workers + ~30+
  chunks/epoch = re-importing torch ~120×/epoch (minutes of pure overhead). Same fate as Q1. -> keep them alive then gng
- **Q3 · Two different `overall_acc` semantics** — pooled-micro in `compute()` vs. mean-of-task-accs in
  `optimal_threshold_metrics` ([metrics.py:189-220](../src/pedpredict/training/metrics.py#L189-L220)).
  Same column name in different artifacts. Rename one. -> what rename capture this better?
- **Q4 · Per-batch `float(total)`** in `train_chunk` forces a GPU sync every step; accumulate on-device
  and sync per chunk. Minor.
- **Q5 · `smooth_track` computes smoothed centers the model never reads** ([inference.py:258-272](../src/pedpredict/eval/inference.py#L258-L272)) — preserved dead computation; delete or actually use. -> if confirmed dead code then can delete safely. this is old, old, OLD artifact.
- **Q6 · `LMDBChunkDataset.__init__` prints per open** — with per-epoch re-opening × chunks this is log
  spam that buries real warnings (like the C1 fix). Route through `logging`/verbosity flag.
- **Q7 · No CI coverage floor** (Phase B item 10, unchanged).

**Resolutions (Q):** Q1 fix — warm for real (sequential full-file read of the chunk; cheap and effective
on an HDD). Q2 fix — `persistent_workers=True`. **Q3 answer:** the rename that captures it: `compute()`'s
pooled-over-all-task-decisions accuracy is **micro** averaging and keeps `overall_acc` (it's in the
`train_log.csv` schema); the sweep's mean-of-per-task-accs is **macro** averaging and becomes
`macro_acc` — and since M2 renames all swept columns to `oracle_*`, it ships as **`oracle_macro_acc`**,
leaving zero ambiguity. Q4 fix (accumulate on-device, sync per chunk). Q5 fix — delete (dead-code status
was verified by trace in this audit). Q6 fix — route through `logging` behind a verbosity flag, so the
new C1 warnings aren't buried. Q7 defer to consolidation (absorbed backlog item 10).

## Checked and cleared

For the record, these were examined and found sound: config loader/override/validation chain
(strict unknown-key rejection, type coercion, cross-field invariants — genuinely good); LMDB
write→read key contract + corrupt-chunk loud failure; collate `motion_dim` guard; AMP step order
(unscale→clip→step); checkpoint atomic-write + versioned full-state resume; `freeze_backbone`
partition; metric implementations themselves (sklearn usage, AUC degeneracy handling, single shared
accumulator); ONNX export/parity incl. the MHA fastpath workaround (well executed); run-dir/index
conventions; balance solver (golden-tested, sign-bug correctly quarantined); incremental build *plan*
logic (contiguity assumption verified against PIE's iteration order); test suite breadth (345 tests,
every module touched).

---

## Mapping to the research scope

| Hole | Feeds which workstream |
|---|---|
| M1, C4 | **Imbalance study (RQ3)** — lever ablation is the experiment; effective-distribution measurement is its instrument |
| M2, M10 | **Calibration/uncertainty** — val-tuned thresholds → temperature scaling → (control path) conformal prediction |
| M3, M4, M5, M6 | **Protocol hygiene** — decide once, before WP0 baselines; M5 optionally buys literature comparability |
| M7, M8 | **Experimental validity** — seed + selection metric *before* any baseline run |
| M9, A6 | **Ablation thesis (RQ1/RQ2/RQ4)** — kinematics-only, single-task, ego-speed rungs |
| A1, A2 | **Backbone study (RQ1)** |
| A3, A5 | **Fusion study (RQ2)** |
| A4 | **Input representation (RQ4)** — motion v2 |
| C1–C3 | **Unblock WP0** — C2/C3 before resuming the train build; C1 before trusting any val metric |

### Final attack order (resolved 2026-06-11 — supersedes the suggested order above)

> **Execution progress (2026-06-15):** step 1 (code fixes) merged to `main` (`4c2120b`); step 2 *code*
> — the v2 data-contract changes — merged to `main` (`66c6000`). **Remaining before WP0 baselines:**
> run the actual regeneration on real PIE data and re-pin the STALE Dataset Statistics
> (per [setup.md](../setup.md)), then the step-3 registry/rename wave.

The one structural change from the original suggestion: the v1 train build is **abandoned, not
resumed** — every dataset-touching decision landed `fix`, so there is exactly one v2 rebuild of all
three splits, and no v1 IO is spent first.

1. **Code fixes, now (no dataset dependency):**
   C2 (resume guard — the v2 build needs it), C3 (map_size), M7 (`train.seed`), M8
   (`train.selection_metric`, default `macro_f1`), M2 (val-tuned thresholds + `oracle_*` rename, incl.
   Q3's `oracle_macro_acc`), M1 flag (`train.use_class_weights`) + the effective-distribution
   instrument, C1 (loud skips; val skip = hard error), Q1/Q2/Q4/Q5/Q6.
2. **The ONE v2 rebuild (sequences + LMDBs for train/val/test; delete the ~20k v1 train chunks):**
   M3 relabel (actions/looks = state-at-end; measure the class-rate shift via `count_labels`),
   M4 censor filter (measure censored count first), M6 `track_id` in `SequenceRecord` + meta,
   A4 data-side motion fixes (frame-0 deltas = 0, flip-reflects-cx; norm choice stays a runtime flag),
   M9 ego-speed channel (store wide, slice via `motion_dim`), M5 benchmark-protocol eval set
   (test-split-only, same pass). Update the stats table + `test_stats` fixture + goldens in the same
   change (doc-sync checklist).
3. **WP0 — the hub:** registry wave first (kinematics-only model; `motion_only` → `ped_local` rename;
   single-task config with zeroed loss weights *and* sampler powers), then train the ~6-run ladder
   under the frozen protocol with the M1 distribution instrument logging into every run dir.
4. **WP1/WP2 — spokes vs. the hub (one axis per run; ~10–12 runs; 1-seed screen, 3-seed finals):**
   M1 lever ablation (4) · A2 backbone candidate study → swap (1–2) · A3 fusion grid incl. the
   flagged `fusion_residual` (3) · A4 norm old-vs-new (1) · M9 ego on/off (1). If two wins compose,
   test exactly one combined v2 candidate.
5. **M10 calibration workstream (post-WP0, parallel to spokes):** reliability diagrams from the
   predictions NPZ → temperature scaling on val → canonical operating-point policy (stored in run dir;
   consumed by eval and `infer_video`) → conformal only if the control framing proceeds.

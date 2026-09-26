# Re-run plan: every paper run on the pixel-free recipe

**Written 2026-09-26. Status: ARMED.** The campaign launches automatically. A cron trigger on the research
PC waits for the pixel-free ladder (`queue_v3.sh`, tmux `queue3`) to finish, runs the go/no-go gate below
once, and on GO starts `queue_v4.sh` in tmux `queue4`. Nobody has to be watching. On NO-GO nothing
launches, and the verdict says which check failed (§7).

> **2026-09-27 update.** (1) Every campaign arm now selects `best.pth` (and early-stops) on **val
> `crosses_auc`**, not F1 at the fixed 0.5 cut. Keeping F1 (old D4) was a mistake: its revisit condition
> fired on `pf_fix_s42` (F1 picked epoch 7, the AUC peak was epoch 17) and was not acted on. As a result no
> ladder run is reused (they were F1-selected), and the campaign trains all **20** runs itself.
> (2) **Gate preview on the three `pf_fix` seeds: NO-GO.** `pf_fix_s43`/`s44` peak at **epoch 1** (val AUC
> 0.870 / 0.854, trained at a tenth of the learning rate during warmup), then fall to 0.76 / 0.73 while train
> loss drops 0.54 → 0.06: memorization, which no selection metric fixes. The seed spread IS fixed, though:
> detection sd 3.1 pp at 205/hr (v1: 18.5), mean 20.2% at 41/hr (v1 binary 11.9%, GBM 38.9%), test AUC 0.785.
> The hub recipe will therefore not launch; the memorization ladder below decides instead (§11).

**The idea.** The paper's claim is about the *training objective* (binary vs hazard), not the architecture.
If the pixel-free model (`pose_kinematics`: the 58-number pose + motion vector per frame, no images) trains
stably, it becomes the canonical model, and every run the paper cites is redone on it: the motivation
matrix, the binary baseline, and all onset arms at three seeds.

---

## 1. Where things stood when this was written (2026-09-26 ~05:20 UTC)

**The training data is the fixed, reshuffled store.** `pf_ctrl_s42` reads `preprocessed_train_shuffled`
only (its `resolved_config.yaml`). Per-chunk crossing rate went from 0.2%–100% (old two-dir store) to
13.6%–15.4% across all 27 chunks (reshuffle log). The stored rate is 14.6%. The weighted sampler is still
on, so the model *sees* ~29.4% crossing windows (`train_distribution.json`). That rate is now the same in
every chunk, but it is still ~10× the 2.8% val/test rate.

**The control arm shows the old failure pattern** (no fix applied, as intended):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| train loss | 0.628 | 0.552 | 0.498 | 0.453 | 0.406 | 0.366 | 0.334 | 0.302 | 0.282 | 0.256 |
| val crosses AUC | 0.820 | 0.831 | 0.820 | **0.835** | 0.825 | 0.810 | 0.783 | 0.820 | 0.791 | 0.803 |

Train loss falls every epoch while validation peaks at epoch 4 and drifts down: memorization. The gate
below has to show that this has stopped.

**Pixel-free runs were disk-bound.** Epoch time grew 345 s → 732 s with the CPU ~70% idle and the disk
reading ~800 MB/s. The chunk warm-up worker read every JPEG in the 83 GB store each epoch, for a reader
that decodes none of them. That is fixed (P2); it only affects the page cache, so results are unchanged.

---

## 2. The go/no-go gate: "is it genuinely learning?"

`scripts/rerun_gate.py` (logic in `src/pedpredict/eval/rerun_gate.py`, pinned by `tests/test_rerun_gate.py`).
It reads the ladder's **binary** seeds only: `pf_fix_s42/43/44`. **It never looks at hazard vs binary.** It
decides whether the recipe is sound, and letting the headline comparison influence that would bias the
re-run.

| # | criterion | pass if |
|---|---|---|
| L1 | no late decline, **every seed** | val crosses AUC over the last 5 epochs stays within 0.02 of that run's peak |
| L2 | learning is not an early spike, **every seed** | peak val AUC at epoch ≥ 8 (and ≥ 10 epochs logged) |
| L3 | no collapse epochs, **every seed** | no epoch with val crosses recall > 0.8 or val loss > 1.0 (the paper's collapse definition) |
| L4 | seeds agree | detection-rate sample sd ≤ 5 pp at 205 alarms/hr, `per_window` (v1 binary: 18.5 pp; GBM probe: 1.2 pp); val AUC at each seed's selected epoch spans ≤ 0.02 |
| L5 | good enough to replace v1 | mean streaming test AUC ≥ 0.78 (v1 binary 0.746–0.777) **and** mean detection at 41 alarms/hr > 15.61% (v3 R2 measured 15.6098%) |

The thresholds were fixed on 2026-09-26, before any fixed-recipe run finished. Checked on real data: the v3
image-model R2 run fails L1, L2 and L5, as it should. The attribution arms (`pf_std`, `pf_bs32`) and the
GBM-probe ceiling (38.9 ± 1.0% at 41/hr) are reported in `v3_report`, not gated.

**Outcomes**

- **GO**: all checks pass → the campaign launches.
- **NO-GO**, stable but still noisy (L1–L3 pass, L4 fails) → the instability is not only an input-scaling or
  BatchNorm problem. Next suspects, cheap to test pixel-free: the sampler's 29% training prior, and each of
  the ~2.5k real crossing events repeated ~6.5× by offline augmentation plus the sampler.
- **NO-GO**, stable and tight but weaker (L1–L4 pass, L5 fails) → still usable, since the claim is relative,
  but it is your call. Override with `touch queue_v4/GATE_GO`.
- **NO-GO**, still declining (L1 fails) → the fix is not the cause. Try lower LR, stronger weight decay,
  selecting on val AUC, or the sampler off.

---

## 3. The canonical recipe

`pf_fix` from `queue_v3.sh` with one change: selection on val `crosses_auc` (2026-09-27). The arm flags live in
`/workspace/setup_logs/recipes_v4.sh`:

```
PF      eval.model_type=pose_kinematics  pose.enabled=true  model.motion_norm=none
        data.motion_dim=58  model.motion_dim=58  data.visual_input=none
COMMON  train.active_tasks=[crosses]  train.selection_metric=crosses_auc  augment.runtime=true
        train.lr_schedule=warmup_cosine
BS32    train.batch_size=32  train.accum_steps=1
STREAM  paths.lmdb_train=[preprocessed_train_shuffled]  pose.input_stats=<pose58_train_shuffled.json>
        data.protocol=streaming
```

Unchanged from the paper's recipe: lr 1e-4, weight decay 1e-5, 30 epochs, early-stop patience 15,
sampler `crosses^0.5`, class weights off. **Effective batch stays 32** (4×8 before, 32×1 now), so the number
of optimizer steps is identical; only the BatchNorm batch changes.

| arm | on top of the canonical recipe | may differ from the reference by |
|---|---|---|
| R2 streaming (binary baseline = Model B streaming leg) | none | seed |
| R2 anchored (Model B anchored leg; the shared anchored row) | `data.protocol=anchored`, anchored input stats | seed, protocol, `pose.input_*` |
| R1 auxiliary | `onset_head=true`, `onset_hazard_weight=0.1` | seed, `model.onset_*`, `train.onset_*` |
| R3 pure | `onset_head`, `onset_report_crosses`, hazard 1.0, crosses weight 0 | + `train.loss_weight` |
| R4 hedge | R3 + `onset_readout_weight=0.5` | as R3 |
| R3C censored | R3 on `preprocessed_train_censored_shuffled` | as R3 + `paths.lmdb_train` |
| Model A (3-task), seed 42 only | `active_tasks=[actions,looks,crosses]`, both protocols | seed, protocol, `pose.input_*`, `train.active_tasks` |

`train.selection_metric` is an allowed difference for every arm: the reference (`pf_fix_s42`) was F1-selected.

The reference is `pf_fix_s42`'s `resolved_config.yaml`. Every arm is diffed against it **before it trains**
(`scripts/check_run_config.py`). Any difference outside its column fails that arm's `_cfg` job, so it never
trains. Worker counts and eval-time knobs are ignored (they cannot change what a run learns), except
`eval.model_type`. All eight recipes were checked against a real run on the box. A deliberately wrong
recipe (sampler off) was caught, and the censored folder under a binary arm is refused by the loader guard.

---

## 4. Pre-flight: done

| # | what | status |
|---|---|---|
| P1 | Resume restores the early-stop counter and the true best epoch (`trainer.py`, `callbacks.py`; tests in `test_callbacks.py`) | ✅ deployed before any reused ladder run started, so reuse needs no "never resumed" condition |
| P2 | Warm-up reads only the `_meta` records when `data.visual_input=none`; the dataset opens LMDB without OS readahead when it decodes no image (`lmdb_warm.py`, `chunk_loader.py`, `lmdb_dataset.py`) | ✅ deployed mid-ladder 2026-09-26 ~05:43 UTC. Backward compatible with the process that was already running (older callers keep the full walk). Every run from `pf_fix_s42` on uses it |
| P3 | Threshold sweep widened 0.10–0.90 @ 0.05 → **0.01–0.99 @ 0.01** (`eval.threshold_sweep_*` default) | ✅ every ladder eval from `pf_ctrl` on already uses the wide grid, so reused runs need no re-eval |
| P4 | Commit + tag the code the campaign runs | ✅ the box's 106 `src/`/`scripts/`/`configs/` files match the laptop's, line-endings aside |
| P5 | R3C's train dir: its training data + censored, reshuffled, metadata-only (`preprocessed_train_censored_shuffled_metaonly`) | queued: `censored_shuffle` job in `queue_v4` (a few GB) |
| P6 | Anchored input stats from the anchored training set (`pose58_train_benchmark.json`) | queued: `anchored_stats` job in `queue_v4` |
| P7 | Config-diff check (`scripts/check_run_config.py`, `src/pedpredict/config/diff.py`) | ✅ runs as each arm's `_cfg` job and as the reuse check |
| P8 | 2-epoch smoke of each new recipe | **dropped** to launch sooner. What covers it: the pre-training config check (every recipe loads and validates); a crash fails fast and is retried then marked; `pf_fixonset_s42` in the ladder is the pixel-free onset-head smoke |

---

## 5. Run inventory

**No ladder run is reused** (they were F1-selected; only `best`/`last` checkpoints exist, so their selection
cannot be redone). All runs are new (tags `c4_*`):

| arm | seeds | runs | feeds |
|---|---|---|---|
| R2 streaming (binary baseline) | 42, 43, 44 | 3 | `tab:seedspread`, `tab:matrix`, `tab:detection` (**the headline**) |
| R3 pure | 42, 43, 44 | 3 | `tab:seedspread`, `tab:detection`, `tab:onset` (**the headline**) |
| R2 anchored | 42, 43, 44 | 3 | `tab:matrix`, the gap decomposition, every onset arm's anchored-trained row |
| Model A, 3-task | 42 | 2 | `tab:matrix` rows A |
| R1 auxiliary | 42, 43, 44 | 3 | `tab:onset`, `tab:detection` |
| R4 hedge | 42, 43, 44 | 3 | same |
| R3C censored | 42, 43, 44 | 3 | same + the lead-time trade |
| **total** | | **20** | |

**Per run:** config check → train → val + test on **both** protocols → streaming-test dump → report refresh.

**Regenerated on the laptop from the dumps** (no runs): separability by time-to-onset and the within-track
smoothing gain (`report_onset_timing.py`), the G decomposition (from `eval_log.csv`), collapse-epoch counts.

**Unaffected, no redo:** `tab:composition`, `tab:horizon`, `fig:onsethist`, `fig:protocols`, `fig:cases`, all
of which come from labels. The GBM probe stays as an external reference row.

---

## 6. Run order

1. **W1, headline:** R2s and R3 as a matched pair per seed (42, 43, 44). The pre-registered criterion is
   computed into `v4_report/headline.md` as soon as both arms have 3 seeds.
2. **W2, motivation:** `anchored_stats`, R2 anchored ×3, Model A streaming + anchored.
3. **W3, every arm at one seed:** `censored_shuffle`, R1 s42, R4 s42, R3C s42.
4. **W4, remaining seeds:** R1, R4, R3C at 43 then 44.
5. **Not queued** (config-only once wanted): look-ahead L ∈ {60, 96, 150}, bin width w ∈ {4, 8}. The focal /
   class-balanced loss arm needs new loss code first.

---

## 7. Operating it

| what | where / how |
|---|---|
| Trigger | user crontab `*/10` + `@reboot` → `/workspace/setup_logs/queue_v4_trigger.sh`, log `queue_v4_trigger.log` |
| Gate verdict | `outputs/diagnostics/rerun_gate/verdict.md` + marker `queue_v4/GATE_GO` \| `GATE_NOGO` \| `GATE_MISSING` |
| Override a NO-GO | `touch /workspace/setup_logs/queue_v4/GATE_GO` (the next tick launches) |
| Re-run the gate | `rm /workspace/setup_logs/queue_v4/GATE_*` |
| Pause / resume | `touch queue_v4/HOLD` / `rm queue_v4/HOLD` |
| Progress | `cat /workspace/setup_logs/queue_v4.status`; log `queue_v4.log`; markers in `queue_v4/` |
| Results | `outputs/diagnostics/v4_report/{per_window,per_track}/curves.md`, `headline.md` |
| After DONE | remove the two `queue_v4_trigger` cron lines |

Sandbox-tested cases for the trigger: ladder still running → nothing; good curves → GO and one launch;
control-like curves → NO-GO, no launch, and the gate is not re-run; a missing seed → MISSING; a manual
override → launch. The headline code reproduces the paper's v1 3-seed table exactly (INCONCLUSIVE, 0 of 5).

**Cost:** before P2, ~3–6 h per streaming run plus evals. After P2 the epoch time is unmeasured; the first
`pf_fix` epochs give the number.

---

## 8. Reporting protocol (fixed before any campaign result)

Unchanged from `SEED_PLAN_2026-09-21.md`:

- **Primary:** detection rate at {1200, 460, 205, 95, 41} alarms/hr, mean ± sd over 3 seeds, R3 vs R2
  streaming, `per_window`; `per_track` reported alongside.
- **Secondary:** lead time at the same budgets.
- **Alongside, never instead:** window AUC and val-tuned F1.
- **Success:** R3's mean beats R2's by more than the sum of their sds at ≥ 3 of 5 budgets; anything less is
  inconclusive. `headline.md` also counts the reverse direction.
- The v1 (image-model) numbers stay in `RESULTS_MATRIX.md` as history. The paper states that the canonical
  model changed and why (chunk-order bug + unscaled inputs), rather than silently swapping numbers.

## 9. Decisions (settled 2026-09-26)

D1 Model A kept at seed 42 only · D2 R3C at 3 seeds · D3 anchored arms scaled by their own training set ·
D4 ~~`selection_metric=crosses_f1` kept~~ → **`crosses_auc`** (2026-09-27; reuse dropped) · D5 warm-up fix applied to the running
ladder.

## 10. Out of scope

Image-model runs of any kind (the recipe-v2 queue stays parked), JAAD, retrained published baselines.

## 11. Memorization ladder (added 2026-09-27, armed)

The pf_fix preview failed on memorization, not on seed spread, so a second stage now sits between the ladder
and the campaign. It runs automatically when the stage-1 gate on `pf_fix_s42/43/44` says NO-GO.

**Why these three.** Each real crossing window is drawn ~15× per epoch: ~7.5 stored copies (offline
augmentation re-emits minority records) × the weighted sampler's ~2×. Two of three seeds were best after one
warmup epoch at a tenth of the learning rate. The GBM probe, trained on the base windows only with no
sampler, does not overfit this way.

| arm | tags | the one change from the hub (campaign R2-streaming recipe, AUC-selected) |
|---|---|---|
| A | `m_base_s42/43` | base windows only, no augmentation copies: `preprocessed_train_base_shuffled_metaonly` (reshuffled; per-chunk crossing 2.4–3.3%, was 0.2–5.0% in the base dir's own order) |
| B | `m_nosamp_s42/43` | `train.use_weighted_sampler=false` |
| C | `m_lr5_s42/43` | `train.lr=1e-5` (same warmup-cosine shape; "flat" would have changed a second axis) |

Input scaling stays the hub's (`pose58_train_shuffled.json`) in every arm, so each changes exactly one thing.

**Pre-registered choice.** Each arm is gated on its two seeds with the same thresholds (§2). Among arms that
pass every check, the one with the highest mean validation AUC at the selected epoch is adopted
(validation only, never test; ties A > B > C). The ladder writes `queue_v4/RECIPE.env` (the change, folded
into every campaign arm by `recipes_v4.sh`), then `GATE_GO`, and the trigger launches the campaign. If no arm
passes → `GATE_NOGO`, nothing launches. Adopting A also rebuilds R3C's censored store from base + censored.

**Stages** (`queue_v4_trigger.sh`, cron): 0 wait for queue_v3 → 1 gate `pf_fix` once (`HUB_GO` also writes
`GATE_GO`) → 2 run `queue_mem.sh` (tmux `queuem`) → 3 on `GATE_GO` run `queue_v4.sh`. All sandbox-tested:
the ladder picks the best passing arm, all-fail stays idle, a hub pass skips the ladder, the tie rule holds,
a missing seed is ineligible, and `RECIPE.env` folds in correctly. Every arm (3 ladder + 8 campaign × 4
possible outcomes) passes the config check against the real reference.

**Disk.** Both new stores are metadata-only (`reshuffle --meta-only`; config refuses them to image models).
The campaign's censored store is metadata-only too, which removed the ~88 GB copy §4 P5 budgeted.

**Results:** `outputs/diagnostics/mem_report/` (curves) and `outputs/diagnostics/mem_gate/choice.md` (verdicts + pick).


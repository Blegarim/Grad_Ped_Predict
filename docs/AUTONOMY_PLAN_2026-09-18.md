# Unattended plan — 2026-09-18 09:00 UTC → 2026-09-21

**Written for a 3-day absence with no check-ins.** It states what runs, what I decide alone, what I refuse
to decide alone, and what happens on each failure mode. Delete or retire this file once the absence ends.

**Budget:** ~72 h of research-PC time. Laptop clock = VM clock + 9 h.

---

## Where things stand right now

| | |
|---|---|
| Recipe v2 | **measured worse than v1**: streaming test AUC 0.721 / tuned F1 0.113 vs v1's 0.784 / 0.225 |
| v2 queue | **HELD** at `v2_base_trainstreaming_s42_train`. Not released during this absence |
| v1 queue | **HELD** at `r2s_train`. Released in Phase 2 below |
| Running | `probe_imgmode_frozeneval`, 5 epochs, detached (PID 4511) |
| Untouched | four v1 baselines, R1, R2-anchored, all caches, all checkpoints |

---

## RESOLVED 2026-09-18 10:55 UTC — Branch B, and R3 is running

Probe epoch-3 train_loss **0.2449** against the ≤0.29 threshold: **Branch B**. The cache is exonerated
(identical to cached v2's 0.2448); the cause is `vit_frozen_eval` and/or `train_frame_proj`, which the probe
cannot separate — see RECIPE_V2_PLAN "PROBE VERDICT" for the separating experiment. Probe stopped at the
pre-registered epoch rather than running to 5, and `queue/HOLD` was released. **R3 `onset_pure` has been
training since 10:55 UTC.**

Restart snag worth knowing: `boot_resume.sh` refused with "queue.sh already running" because an orphaned
`tmux new -d -s queue` client (PID 908) survived the earlier session kill and still matched its guard's
`pgrep`. Killing that client plus the stale `queue/queue.pid` fixed it. **Killing a queue tmux session can
leave a client that makes the queue un-restartable, and the symptom reads as "already running."**

## Phase 0 (done) — let the probe finish

The probe is BASE with `data.visual_input=images` as the **only** changed flag. Nothing else touches the box
until it ends; a concurrent job would both slow it and confound its epoch timing.

## Phase 1 — read the probe, pick a branch

The decision number is **train_loss at epoch 3**, because that is where v1 and v2 separate cleanly and it is
not calibration-sensitive. Reference points, same seed, same epoch: **v1 = 0.3934**, **v2 = 0.2448**.

| branch | epoch-3 train_loss | reading |
|---|---|---|
| **A** | **≥ 0.34** | v1-like. The **cache** caused the overfitting. `vit_frozen_eval` is exonerated |
| **B** | **≤ 0.29** | v2-like. The **BN drift was doing real regularisation work** and removing it is what cost the performance |
| **C** | 0.29 – 0.34, or the probe crashed | ambiguous |

## Phase 2 — machine track: release the v1 queue. **Same in every branch.**

`rm queue/HOLD; bash boot_resume.sh`, which runs this order — **reordered 2026-09-18** so the claim is
tested before the control (`queue.sh` edited, `bash -n` clean, `verify_queue.sh` says ALL RECIPES VALID):

1. `r3_*` — **R3 `onset_pure` — the methodological claim, never run before**, ~23 h → eval cells + timing
2. `r4_*` — R4 `onset_hedge`, ~23 h → eval cells + timing
3. `r2s_train` — R2 streaming leg (control), **resumes from epoch 4**, ~20 h → its 4 eval cells

**Why reordered.** R3/R4 are the thesis contribution and had never been run once, while ~40 h went into
infrastructure. With R2s first the first real test of the method was 43 h away; now it is 23. Safe because
`r3_train` and `r4_train` depend on nothing, and R3/R4's anchored row comes from `r2a_train`, already
`.done`. R3 can also be read against the existing no-onset baseline `20260714_134253` — the same comparison
R1 already used — so R2-streaming improves rigour later rather than gating anything now.

**Why the branch does not change this.** R3 and R4 are the thesis contribution and have never been run.
They must be comparable to R1 and the four baselines — all of which are the **v1** recipe. Training them on
any corrected recipe would also require retraining the baseline on that recipe (+23 h) and would orphan R1.
So the machine spends the 72 h on the thesis-critical arms under the recipe everything else already uses.
The probe result changes the *recommendation*, not the *run*.

**If the budget slips, R2-streaming is now the casualty** — a control the user had already considered
dropping, rather than half the method.

## ⚠️ FOUND 2026-09-18 — the threshold sweep floor will cap R3/R4's reported F1

`configs/eval.yaml`: `threshold_sweep_lo=0.10`, `hi=0.90`, `step=0.05`. **The lowest threshold ever tried is
0.10.** R3's `crosses_readout` at epoch 7 has **p95 = 0.0977**, i.e. >95% of windows score below the floor,
and R1's readout narrowed to ~0.01 as its LR decayed — so this gets worse, not better.

**Consequence:** tuned F1 can report **0.0 while AUC reads 0.808**, which looks like method failure but is
a mis-ranged instrument. It hits R3/R4 and NOT the baselines: baselines report `crosses_frame` (a softmax
centred near 0.5, well inside the grid), while R3/R4 report `crosses_readout` = `1 - Prod(1 - h_k)`, a
product over K hazard bins whose natural scale is far smaller. The grid was built for the former. The known
"anchored tuned thresholds sit at the sweep floor 0.10" note is the same defect, milder.

**NOT changed** — eval-config changes are outside this plan's authority. Nothing is lost: the queue's eval
cells write what they write, and **eval re-runs from `best.pth` in minutes**, no retraining.

**Fix when the user is back:** lower `threshold_sweep_lo` (or use a log-spaced grid) and re-run the EVAL
ONLY for R3, R4 **and** the baselines so every arm shares one protocol. Extending the grid downward cannot
move a model whose optimum is already above 0.10 — the baselines' numbers being unchanged is itself the
check that the change is fair. **Until then, compare arms on AUC**, which the grid does not touch and which
the project's own metric-discipline note already prefers for this class of reason.

## Censored-window regen — runbook (run at the R3 -> R4 boundary)

**Why:** M4 discards **7,470 train windows** (8.5%) whose future was truncated with no crossing seen. The
binary formulation has no honest label for them; the hazard loss does -- it masks the unobserved bins. So
this is the one experiment where the method has data the baseline structurally cannot use. Built as its own
LMDB dir, so `paths.lmdb_train` is untouched until an arm opts in: **generating it changes nothing**.

**Code is written and gated (696 tests, ruff clean), NOT yet deployed:** `data.emit_censored` in
`config/schema.py` + `configs/data.yaml`; the three-way M4 branch and `censored_emitted` counter in
`data/pie_sequences.py`; `_validate_censored_dirs` in `config/loader.py`; tests in `test_data_shapes.py`
and `test_config.py`.

**Order matters — do NOT start before R3's eval cells finish**, and pause the queue first so R4 does not
begin underneath it.

1. `touch /workspace/setup_logs/queue/HOLD` once `queue/r3_eval_anchored.done` exists.
2. Deploy **atomically** (scp is not atomic; a DataLoader worker importing a half-written
   `pie_sequences.py` would kill a run):
   `scp <file> research-pc:/workspace/Grad_Ped_Predict/<path>.tmp` then on the VM
   `python -m py_compile <path>.tmp && mv <path>.tmp <path>`.
   Files: `src/pedpredict/data/pie_sequences.py`, `src/pedpredict/config/schema.py`,
   `src/pedpredict/config/loader.py`, `configs/data.yaml`.
3. Generate sequences into their own output with `--set data.emit_censored=true`, then build
   **`preprocessed_train_censored`** (~7.5k windows, roughly 1.5 chunks). Full box: R3 is done, R4 not
   started. Budget ~1 h including pose.
4. **Do not edit `paths.lmdb_train` globally.** The R3-censored arm passes the extra dir itself.
5. Add the R3-censored arm to `queue.sh` (R3's flags + the censored dir). Requires killing the `queue`
   tmux session first -- and after killing it, **check for an orphaned `tmux new -d -s queue` client**, which
   otherwise makes `boot_resume.sh` refuse with "already running" (hit 2026-09-18, PID 908).
6. `rm queue/HOLD`, restart, verify `verify_queue.sh` passes.

**The guard's contract:** any training dir whose name contains `_censored` requires
`train.loss_weight['crosses'] == 0` **and** `model.onset_head=true`, else `validate_config` raises. That is
why the dir must be named `preprocessed_train_censored` and not something else.

**Caveat for the writeup:** the added windows are all non-crossers by construction, so they shift the
sampler's class frequencies. R3-censored therefore differs from R3 in data *and* effective sampling
distribution -- inherent to adding data, but it belongs in the caveat list, not a footnote.

## Phase 3 — what each branch changes (write-up, not machine time)

- **Branch A (cache):** v2 is salvageable — restore diversity in feature space (noise/mixup on cached
  vectors) or drop caching. Recorded as future work. **I will not implement it unattended**: it is new
  unvalidated code that would shape every later result.
- **Branch B (BN drift):** the "frozen backbone" bug was *load-bearing implicit regularisation*. That is a
  genuine finding and reframes the whole v2 detour from a failed fix into a result worth reporting: fixing a
  real bug made the model worse, because the bug was supplying the regularisation. Recipe v2 is a dead end
  as designed. I will write this into RESULTS_MATRIX and THESIS_ROADMAP.
- **Branch C (ambiguous):** re-run the probe once at 8 epochs if the queue has slack; otherwise record as
  unresolved and leave the decision to you.

---

## Pre-authorised: what I will do alone

- Release **`queue/HOLD`** (v1) once the probe ends — the single machine action this plan authorises.
- Clear a `.failed` marker **only** for the known open-file-limit crash (signature `received 0 items of
  ancdata` / `Pin memory thread exited unexpectedly`) — a known infrastructure fault with a known-safe fix,
  already mitigated by `ulimit -n` in the launchers. Up to 2 times per job, then I stop and report.
- Re-read and report anything: logs, metrics, configs, `nvidia-smi`, disk.
- Update docs and memory with results as they land.
- Push a short notification on every event listed below.

## Hard limits: what I will NOT do without you

- **Not** release `queue_v2/HOLD`, or run anything on recipe v2.
- **Not** change any recipe flag, `selection_metric`, seed, or config default.
- **Not** write new features (feature-space augmentation included).
- **Not** delete or overwrite any run dir, checkpoint, cache, or LMDB.
- **Not** commit, push, or touch git history.
- **Not** clear a `.failed` marker for any cause other than the open-file-limit signature.

## Failure handling

| event | what happens | what I do |
|---|---|---|
| Job fails 3× | queue marks `.failed`, dependents `.skipped`, **continues with independent jobs** | report with the traceback; act only under the pre-authorised rule above |
| Training stalls >150 min | queue watchdog kills and resumes it | report |
| VM reboot | boot service restarts both queues; v2 re-reads its HOLD and waits; v1 resumes | report; verify v2 is still held |
| SSH unreachable >20 min | owner's VPN or the gateway laptop; the VM keeps running | report once, then again on reconnect. No action |
| Disk < 20 GB free | — | hold the queue and notify (currently 207 GB, so not a live risk) |
| Gate fails | queue sets HOLD and exits | report; **do not override** — gate overrides are yours |

## Notifications

Push on: probe verdict · v1 queue released · each training completion · each arm's test numbers · any
failure, stall, gate, or reboot · one daily status line. Nothing else.

## Expected timeline (early stopping may pull these in; R1 stopped at 28/30, the v1 baseline at 19)

| when (UTC) | |
|---|---|
| Sep 18 ~12:45 | probe ends → branch decided → v1 queue released |
| Sep 19 ~12:20 | **R3 `onset_pure` done** → evals + timing report — first real test of the method |
| Sep 20 ~14:00 | R4 `onset_hedge` done → evals + timing report |
| Sep 21 ~12:00 | R2 streaming done → evals (droppable if anything slipped) |

## Available but NOT started — needs your go-ahead

**Stage 6 rare-event metric suite** (per-frame mAP, cAP/mcAP, point-AP) — laptop-only, no GPU, the roadmap's
"NEXT", and it is the instrument every later comparison is read with. I am leaving it alone deliberately:
it is substantial new code whose design choices shape how all results are interpreted, and that is a call
you should make rather than find already made.

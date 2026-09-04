# Setup — fresh machine to trained model

Commands only, in order, on the **v2 data contract**. What the labels mean, the architecture, and the
imbalance policy live in [CLAUDE.md](CLAUDE.md) / [README.md](README.md) — not repeated here. Steps 0–1
need no dataset. Rebuilding over a v1 checkout: delete `data/sequences/*.pkl` and
`preprocessed_{train,train_aug,val,test}/` first (the runtime hard-errors on v1 chunks).

## 0. Prerequisites
- **Python 3.10–3.12**; **ffmpeg** on PATH (val/test frame extraction); **CUDA GPU** for training.
- **Disk:** clips are tens of GB, LMDBs on top. The train build is self-bounding (per-chunk extract +
  delete). Tight-disk knob (**C3**): `data.lmdb_map_size_bytes` is pre-allocated per chunk on Windows
  (4 GiB default ≈ 76 GB across ~19 chunks); build one chunk, measure, pass `--set data.lmdb_map_size_bytes=<measured+30%>`.

## 1. Code + environment
```powershell
git clone <repo>; cd Grad_Ped_Predict
python -m venv .venv; .venv\Scripts\activate
pip install -e .[dev]
# GPU wheel — .[dev] pulls CPU torch; force-replace (cu126 is a safe default; torch 2.7.1 has no cu121):
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print 2.7.1+cu126 True
ruff check .; pytest -m "not slow"                                             # gate (no dataset needed)
```
⚠️ Skipping the GPU reinstall silently trains on CPU.

## 2. PIE dataset
Download from YorkU into `data/` (`pie_root`). Splits are fixed: **train=set01/02/04, val=set05/06, test=set03**.
```
data/
  PIE_clips/            set01..06/video_####.mp4
  annotations/          set01..06/*_annt.xml
  annotations_attributes/
  annotations_vehicle/  OBD — REQUIRED for the ego-speed channel (M9); missing → sequence-gen fails
  images/               created on demand (step 4)
```
(The `PIE/` toolkit folder is code only, already cloned — it holds no videos.)

## 3. Sequence windows (annotations only)
```powershell
python scripts/make_sequences.py --split all       # sequences_{train,val,test}.pkl (+ streaming benchmark test)
python scripts/make_sequences.py --benchmark       # M5 TTE-protocol test set
python scripts/count_labels.py --from-sequences    # drift canary — run NOW, before LMDBs
```
Record the printed `censored` **and `determined_positive`** counts. v2 counts differ from the stale v1
~95k/22k/76k figures by design; re-pin the Dataset Statistics table +
`tests/fixtures/golden/pie_sequences_counts.json` + the expected dict in
`tests/test_stats.py::test_reference_fixture_matches_claude_table` from this run — **all four move together
or the gate lies**.

**Recovering the confirmed positives M4 bins (`data.emit_determined_positives`).** The censor filter asks
only "was the full future observed?", so it also discards windows where the crossing was plainly *seen*
inside the truncated remainder. Those labels were never in doubt, and they are positives — the scarcest
class at ~2.9%. Turning the flag on keeps them; windows with a truncated future and **no** observed
crossing are still dropped either way, because that label really would be fabricated.
```powershell
python scripts/make_sequences.py --split all --set data.emit_determined_positives=true
python scripts/make_sequences.py --benchmark --set data.emit_determined_positives=true
python scripts/count_labels.py --from-sequences     # reads the pkls — the flag does NOT belong here
```
The flag belongs on `make_sequences.py` **only**. It decides which windows get written to the pkl; every
later step (`count_labels --from-sequences`, `build_lmdb*`, training) just reads what is already there.
⚠️ **This changes the window population**, so it invalidates comparisons against runs built without it —
the four `pose_full` baselines included. Use it only as part of a full regen where every arm is retrained,
and treat the RESULTS_MATRIX numbers from before the regen as historical. `determined_positive` in the
printed stats is exactly how many windows it recovered; report it.

> **Doing a FULL regen and you want the pose arm?** Extract pose (§11) **before** §4 and build the LMDBs
> once with the pose bundle. Extraction streams frames from `PIE_clips` in memory, so it does not need
> staged frames and is unaffected by the incremental build deleting them — but building §4 plain and then
> rebuilding with `pose.enabled` is two full LMDB passes for one result. Order: §3 → §11 extract → §4 with
> the bundle.

## 4. Build LMDBs (pkl → chunks; ImageNet norm applied at read, not here)
```powershell
# val/test/benchmark — extract that split's frames, then build (stage one split's sets at a time):
python -c "import sys; sys.path.insert(0,'.'); from PIE.utilities.pie_data import PIE; PIE(data_path='data').extract_and_save_images(extract_frame_type='annotated')"
python scripts/build_lmdb.py --split val
python scripts/build_lmdb.py --split test
python scripts/build_lmdb.py --split test_benchmark
# train — self-bounding: extracts only referenced frames per chunk, then deletes them (resumable, C2-guarded):
python scripts/build_lmdb_incremental.py --split train
```
Tight-disk alternative to the extract one-liner: `python scripts/extract_annotated_jpg.py` (JPG, ~3–5× smaller).
Storage-limited staging (all of a split's sets must be present together):

| Round | Extract | Build | Delete |
|---|---|---|---|
| 1 | set05, 06 | `build_lmdb.py --split val` | val frames |
| 2 | set03 | `build_lmdb.py --split test` + `--split test_benchmark` | test frames |
| 3 | set01, 02, 04 | `build_lmdb_incremental.py --split train` | auto, per chunk |

## 5. Augmentation
```powershell
python scripts/augment_dataset.py                  # → preprocessed_train_aug (unioned at train time)
```
⚠️ Mandatory for the default config — the trainer unions `preprocessed_train_aug`. Opt-in ablation
alternatives (do not stack): `balance_dataset.py --split train --set balance.enabled=true` (downsample), or
`--set augment.runtime=true` at train time (on-the-fly, ratio-preserving scarcity regularizer; §10).

## 6. Train
```powershell
python scripts/train.py                            # full model (default); writes outputs/runs/{timestamp}_full/
python scripts/train.py --set train.active_tasks=[crosses] train.selection_metric=crosses_f1   # crosses-only
```
Ablation arms: `--set eval.model_type=ped_local|kinematics_only|visual_only|vanilla_concat`. **The selector
is `eval.model_type`, not `model.model_type`.** Override anything inline, e.g. `--set train.lr=5e-5`.

**Crosses-only (head-selection mode).** `--set train.active_tasks=[crosses]` is the one-flag switch: it
zeros the actions/looks heads in both imbalance levers, drops them from the metric/CSV/eval columns, and
makes `macro_f1` collapse to `crosses_f1`. **Always pair it with `train.selection_metric=crosses_f1`** —
otherwise best.pth/early-stop can freeze on a transient epoch-1 value of the (now single-task) macro. Eval
inherits `active_tasks` from the checkpoint, so its `eval_log.csv` is crosses-only automatically.

## 7. Evaluate (thresholds tuned on val, applied to test)
```powershell
python scripts/evaluate.py --split val  --checkpoint outputs/runs/<run>/checkpoints/best.pth   # → thresholds_<protocol>.json
python scripts/evaluate.py --split test --checkpoint outputs/runs/<run>/checkpoints/best.pth   # applies them
```
Report `tuned_*` only (`oracle_*` = same-split leakage, diagnosis only). Lead metric `crosses_f1`.

## 8. Optional
- ONNX: `pip install -e .[export]` → `python scripts/export_onnx.py …`
- Video: `pip install -e .[infer]` → `python scripts/infer_video.py …`
- Plots: `python scripts/visualize.py …`

## 9. Anchored protocol (cross-protocol matrix)
Steps 3–7 build **streaming** (~37:1). The **anchored** (~2.5:1) protocol trains/tests the same model on
Kotseruba fixed-TTE windows; the anchored→streaming gap is the thesis headline. One toggle,
`--set data.protocol=anchored`, repoints train/val/test at the `*_benchmark` LMDBs.
```powershell
python scripts/make_sequences.py --benchmark --split train    # + --split val  (test made in step 3)
python scripts/build_lmdb_incremental.py --split train_benchmark    # fold into step-4 staging (same frames)
python scripts/build_lmdb.py --split val_benchmark
python scripts/train.py --set data.protocol=anchored train.use_weighted_sampler=false   # no aug at 2.5:1
```
Sanity-check `make_sequences` output: anchored `N − P` (negatives) should be a healthy majority, not a sliver.
**2×2 matrix** — `data.protocol` at eval selects both the test distribution and the val split thresholds tune on:
```powershell
python scripts/evaluate.py --split val  --set data.protocol=streaming --checkpoint <run>/checkpoints/best.pth
python scripts/evaluate.py --split test --set data.protocol=streaming --checkpoint <run>/checkpoints/best.pth
# ...then the same pair with data.protocol=anchored. Headline cell: train-anchored / test-streaming.
```
Thresholds are stored per protocol (`thresholds_streaming.json` / `thresholds_anchored.json`), so the two
protocols no longer clobber each other's val-tuned cutoffs — still run each protocol's `--split val` before
its own `--split test` so the file exists.

## 10. Pretrained backbone + runtime aug (RQ1 / scarcity)
Both are train-time-only, no data rebuild; see [docs/BACKBONE_STUDY.md](docs/BACKBONE_STUDY.md). First
pretrained run downloads timm weights (needs network once).
```powershell
python scripts/train.py --set model.vit_backbone=tiny_vit_5m_224      # pretrained drop-in vs legacy default
python scripts/train.py --set model.vit_backbone=tiny_vit_21m_224     # deploy-stress / Pareto arm
python scripts/train.py --set model.vit_backbone=pvt_v2_b0            # different mechanism (SRA)
python scripts/train.py --set augment.runtime=true                   # on-the-fly aug; composes with all above + protocol
# frozen pretrained visual features (field-standard on the small anchored set — trains motion+fusion+heads only):
python scripts/train.py --set model.vit_backbone=tiny_vit_5m_224 --set model.freeze_vit_backbone=true
```

## 11. Pose arm (needs a pose-enabled rebuild — fold into the final data pass)
[docs/POSE_ENCODER.md](docs/POSE_ENCODER.md). Extraction streams frames **in memory from `PIE_clips`**
(no staged images — same storage bound as the incremental build), once per unique frame; per-video npz
is merge-updated, so splits can run separately. Then rebuild LMDBs with `pose.enabled` so metas carry
raw keypoints. `--dry-run` fabricates keypoints — full-pipeline check without clips or rtmlib.
The pose bundle (always the four together; validation rejects partial):
`--set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58`
```powershell
pip install rtmlib onnxruntime-gpu                       # extractor runtime (lab PC, once)
python scripts/extract_pose.py --split all               # → pose_cache/{set}/{video}.npz
python scripts/extract_pose.py --split train_benchmark   # anchored windows, same cache
# rebuild the step-4 LMDBs with <bundle> (window population must NOT change — pose is additive):
python scripts/build_lmdb_incremental.py --split train <bundle>       # + other splits as in step 4
# train/eval: <bundle> + a pose model type
python scripts/train.py --set eval.model_type=pose_kinematics <bundle>
python scripts/train.py --set eval.model_type=pose_full <bundle>
```
Pose-enabled chunks stay readable by non-pose runs (drop the bundle → 9-dim contract, key ignored);
reading pose from a chunk built without it fails loudly. `pose.include_arms=true` ⇒ `motion_dim=70`.

## 12. Onset-timing arm (crossing onset as timing under censoring)
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) prong 2; contracts in [CLAUDE.md](CLAUDE.md) § Onset Timing.
**Default off** — the four `pose_full` baselines were trained without it, and off means no extra output
key and no extra parameter, so their checkpoints still load `strict=True`.

**Data.** The three S1 fields (`onset_offset` / `future_observed` / `track_crosses`) are written by any
build from S1-annotated pkls. Chunks built before S1 need a one-off **metadata-only** upgrade — image
blobs are never touched, so this is minutes, not a rebuild:
```powershell
python scripts/backfill_onset_meta.py --split train --split val --split test --dry-run  # verify first
python scripts/backfill_onset_meta.py --split train --split val --split test
python scripts/backfill_onset_meta.py --split test_benchmark                            # anchored too
python scripts/augment_dataset.py --set augment.enabled=true   # aug dirs inherit via the write path
```
Stop any training/eval job first — Windows refuses a write-open while a chunk is memory-mapped
(`--dry-run` opens read-only and is safe while readers are live). **Augmented dirs are not
backfillable**: oversampling breaks the positional sample→record map, and the script aborts rather than
guessing. Backfill the base dir and re-run augmentation instead.

**Disk, not just time.** Rewriting a meta is copy-on-write, so a chunk built with a tight
`data.lmdb_map_size_bytes` (step 0's disk knob) can run out of room mid-split — the
`MDB_MAP_FULL` / "environment mapsize limit reached" traceback. The pass handles that itself: writes
commit in batches, and a chunk with no room has its `map_size` grown 64 MiB at a time. Windows
pre-allocates, so growth is disk taken the moment it is asked for — `--dry-run` names the chunks short
of room and the worst-case total first. If the volume is genuinely full the run aborts saying so, and
re-running after freeing space resumes where it stopped (the pass is idempotent).

*Rebuilding from scratch instead?* Then skip the backfill entirely — steps 3–4 carry the fields already.

**Train.** Three arms, selected by weights alone (no code branches). Start with the auxiliary arm: the
reported `crosses` number still comes from `crosses_frame`, so the baselines stay exactly comparable and
any movement is attributable to the trunk learning timing.
```powershell
# 0) smoke: does the head collapse to h~0? (the predicted failure mode — check before spending hours)
python scripts/train.py --set model.onset_head=true --set train.num_epochs=2 --tag onset_smoke

# 1) AUXILIARY — hazard as a side task; crosses still reported from crosses_frame
python scripts/train.py --set model.onset_head=true --set train.onset_hazard_weight=0.03 --tag onset_aux

# 2) PURE REFORMULATION and 3) HEDGE — full one-liners in "Copy-paste training recipes" below
```
⚠️ **Scale.** The hazard term *sums* over each window's observed bins (likelihood-correct), so at
`onset_lookahead=60` it starts ~40× a per-task CE and falls as hazards saturate low. Hence ~0.03 when it
rides alongside the old objective, 1.0 when it *is* the objective. The per-epoch `onset_hazard` value is
the raw unweighted number — read it in the first minute, like `train_distribution.json`.

**If the head goes dead** (readout saturates near 0, `crosses_f1` collapses): the per-bin positive rate is
~`2.9%/K`. Raise `model.onset_bin_width` to 4 — it quadruples the positives each bin sees, costing timing
resolution. `onset_lookahead` and `onset_horizon` must both stay divisible by it.

**Eval** needs no new flags: `evaluate.py` inherits the whole `model` section from the checkpoint's
`resolved_config.yaml`, so head width and metric routing follow the checkpoint automatically.

## Copy-paste training recipes
Every line below is a **single line** (no continuations — they break differently in PowerShell and bash)
and every one has been checked to parse and pass `validate_config`. Change `--tag` per run; it becomes the
run-id suffix and the `index.csv` tag column. Defaults are spelled out where they define the experiment,
so a recipe keeps doing what it says even if someone edits `configs/`.

The pose bundle appears in full in each pose recipe — validation rejects it partially applied, so it is
written out rather than abbreviated.

**The workhorse — pose_full, streaming, onset head as an auxiliary task, runtime augmentation.**
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set augment.runtime=true --set train.lr_schedule=warmup_cosine --set model.onset_head=true --set train.onset_hazard_weight=0.03 --tag pose_onset_aux
```
The reported `crosses` number still comes from `crosses_frame`, so this stays directly comparable with the
four `pose_full` baselines. Same line with `--set data.protocol=anchored --tag pose_onset_aux_anch` for the
anchored leg.

**Its baseline twin — identical, onset head off.** Run this if you need the comparison re-made under
today's code rather than trusting a months-old run dir.
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set augment.runtime=true --set train.lr_schedule=warmup_cosine --tag pose_baseline
```

**Whole cross-protocol matrix in one command** — trains both protocols and runs val+test x anchored+streaming
per leg (10 steps). The runner owns `data.protocol`, so never pass it here. `--dry-run` prints the plan.
```powershell
python scripts/run_arm.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set augment.runtime=true --set model.onset_head=true --set train.onset_hazard_weight=0.03 --tag pose_onset_aux --save-predictions
```

### Onset arm
**Smoke test first — 2 epochs, does the head collapse to `h~0`?** Cheap insurance before a long run.
```powershell
python scripts/train.py --set model.onset_head=true --set train.num_epochs=2 --tag onset_smoke
```
**Pure reformulation** — the original crossing head switched off, reported number from the hazard readout.
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set model.onset_head=true --set model.onset_report_crosses=true --set train.onset_hazard_weight=1.0 --set "train.loss_weight={actions: 0.8, looks: 0.8, crosses: 0.0}" --tag onset_pure
```
**Hedge** — as above plus a direct gradient on the number actually reported.
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set model.onset_head=true --set model.onset_report_crosses=true --set train.onset_hazard_weight=1.0 --set train.onset_readout_weight=0.5 --set "train.loss_weight={actions: 0.8, looks: 0.8, crosses: 0.0}" --tag onset_hedge
```
**Collapse rescue** — 4-frame bins quadruple the positives each bin sees, costing timing resolution. Reach
for this only if the smoke run shows a dead head.
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set model.onset_head=true --set model.onset_bin_width=4 --set train.onset_hazard_weight=0.03 --tag onset_w4
```

### Ablations
**Crosses-only** — always pair the two flags; `selection_metric` alone is the epoch-1 dead-head trap.
```powershell
python scripts/train.py --set eval.model_type=pose_full --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --set "train.active_tasks=[crosses]" --set train.selection_metric=crosses_f1 --tag crosses_only
```
**A3 — fusion residual off** (motion reaches the heads only as an attention mask; golden-pinned arm).
```powershell
python scripts/train.py --set eval.model_type=full --set model.fusion_residual=false --tag a3_no_residual
```
**A4 — legacy per-sequence motion norm.**
```powershell
python scripts/train.py --set eval.model_type=full --set model.motion_norm=per_sequence --tag a4_per_seq
```
**RQ1 — pretrained TinyViT, frozen.** The field-standard recipe on the small anchored set.
```powershell
python scripts/train.py --set eval.model_type=full --set model.vit_backbone=tiny_vit_5m_224 --set model.freeze_vit_backbone=true --set augment.runtime=true --tag rq1_tinyvit_frozen
```
**RQ3 — imbalance levers.** One lever per run; confirm the effect in `train_distribution.json`, not by eye.
```powershell
python scripts/train.py --set eval.model_type=full --set train.use_weighted_sampler=false --tag rq3_no_sampler
python scripts/train.py --set eval.model_type=full --set train.use_class_weights=true --tag rq3_class_weights
```
**Model-type sweep** — the hub-and-spoke ablations. `kinematics_only` is the pixel-free floor.
```powershell
python scripts/train.py --set eval.model_type=kinematics_only --tag abl_kinematics
python scripts/train.py --set eval.model_type=visual_only --tag abl_visual
python scripts/train.py --set eval.model_type=vanilla_concat --tag abl_concat
python scripts/train.py --set eval.model_type=ped_local --tag abl_pedlocal
python scripts/train.py --set eval.model_type=pose_kinematics --set pose.enabled=true --set model.motion_norm=none --set data.motion_dim=58 --set model.motion_dim=58 --tag abl_pose_kin
```

### Evaluation (always this order — thresholds tune on val, apply to test)
```powershell
python scripts/evaluate.py --checkpoint outputs/runs/<run_id>/checkpoints/best.pth --split val
python scripts/evaluate.py --checkpoint outputs/runs/<run_id>/checkpoints/best.pth --split test --save-predictions --benchmark
```
`evaluate.py` inherits the whole `model` + `pose` section from the checkpoint's `resolved_config.yaml`, so
no architecture flags here — not the pose bundle, not the onset settings. Only runtime choices
(`data.protocol`, `eval.batch_size`) belong on this line.

## Run-defining knobs (pre-flight)
A wrong value here does **not** crash — it silently produces an unusable run. Every run dumps
`resolved_config.yaml` at start; skim it (and `train_distribution.json` for effective imbalance rates) in
the first minute rather than at hour three. Defaults in **bold**.

| Field | Values | Switches (silent if wrong) |
|---|---|---|
| `eval.model_type` | **full** \| ped_local \| kinematics_only \| visual_only \| vanilla_concat \| pose_kinematics \| pose_full | which model trains/evals — the selector, **not `model.model_type`** (that raises); `pose_*` need the §11 bundle |
| `data.protocol` | **streaming** (~37:1) \| anchored (~2.5:1) | repoints train+val+test LMDBs; at eval also sets the test distribution AND the val split thresholds tune on |
| `model.vit_backbone` | **legacy** \| `<timm>` (e.g. `tiny_vit_5m_224`) | from-scratch ViT vs pretrained drop-in (the RQ1 arm) |
| `model.vit_pretrained` | **true** \| false | `false` = random-init backbone (wiring tests only) — a real run is garbage |
| `model.freeze_vit_backbone` | **false** \| true | freezes the ViT (`vit.*`) for the whole run; trains motion+fusion+heads only — the field-standard PIE recipe on the small anchored set (~4.9k windows), stops the backbone memorizing. Not `ScheduleCfg.freeze_backbone` (which freezes all-but-heads) |
| `train.selection_metric` | **macro_f1** \| crosses_f1 \| val_loss | which epoch becomes `best.pth` + early stop (macro_f1 can sacrifice crosses) |
| `train.use_weighted_sampler` / `use_class_weights` | **true** / **false** | effective training distribution (imbalance levers) — confirm in `train_distribution.json` |
| `augment.runtime` | **false** \| true | on-the-fly train-time aug (scarcity regularizer); offline `augment.enabled` is a *build* flag |
| `model.motion_norm` | **image** \| per_sequence \| none | motion-feature semantics (A4 arm); `none` is pose-only (validated ⇔ `pose.enabled`) |
| `model.onset_head` | **false** \| true | builds the onset hazard head (§12). `false` = the binary baseline, byte-identical to the four `pose_full` runs |
| `model.onset_lookahead` / `onset_bin_width` | **60** / **1** | how far ahead the head predicts, and at what resolution. `lookahead` MUST exceed `onset_horizon` (validated) — at equality a crossing just past the horizon is still a flat negative, which is the failure the arm exists to remove |
| `model.onset_report_crosses` | **false** \| true | which head `crosses` is SCORED on: `crosses_frame` (= the baselines) vs the hazard readout. **Metrics only** — never the loss routing |
| `train.onset_hazard_weight` / `onset_readout_weight` | **1.0** / **0.0** | selects the arm (§12). The hazard term sums over bins, so it starts ~40× a CE — use ~0.03 in the auxiliary arm, 1.0 where it is the objective |
| `train.num_epochs` / `lr` / `lr_schedule` | **30** / **1e-4** / **warmup_cosine** | training budget + optimization (wrong `lr` = diverge / no-learn) |
| `train.warmup_epochs` / `warmup_start_factor` | **1** / **0.1** | `warmup_cosine` linear-warmup length **in epochs** (not steps — the scheduler steps once per epoch; `0` disables warmup) + its start LR (`warmup_start_factor * lr`, = 1e-5 at default `lr`) |

## Config overrides
`--config-dir DIR` and repeatable `--set section.field=value` (also `--section.field value`), validated
against [config/schema.py](src/pedpredict/config/schema.py) — the full field list + defaults live there.
- scalars `--set train.lr=5e-5` · bools `--set augment.runtime=true` · null `--set data.lmdb_map_size_bytes=null`
- lists `--set "model.window_size=[8,4,2,null]"` · dicts replace wholesale, pass every key:
  `--set "train.sampler_powers={crosses: 0.5, actions: 0.3, looks: 0.3}"`
- `section.field` only (2 levels); for `schedule.phases` edit `configs/schedule.yaml`.

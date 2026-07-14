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
Record the printed `censored` count. v2 counts differ from the stale v1 ~95k/22k/76k figures by design; re-pin
the Dataset Statistics table + `tests/fixtures/golden/pie_sequences_counts.json` from this run.

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
| `train.num_epochs` / `lr` / `lr_schedule` | **30** / **1e-4** / **warmup_cosine** | training budget + optimization (wrong `lr` = diverge / no-learn) |
| `train.warmup_epochs` / `warmup_start_factor` | **1** / **0.1** | `warmup_cosine` linear-warmup length **in epochs** (not steps — the scheduler steps once per epoch; `0` disables warmup) + its start LR (`warmup_start_factor * lr`, = 1e-5 at default `lr`) |

## Config overrides
`--config-dir DIR` and repeatable `--set section.field=value` (also `--section.field value`), validated
against [config/schema.py](src/pedpredict/config/schema.py) — the full field list + defaults live there.
- scalars `--set train.lr=5e-5` · bools `--set augment.runtime=true` · null `--set data.lmdb_map_size_bytes=null`
- lists `--set "model.window_size=[8,4,2,null]"` · dicts replace wholesale, pass every key:
  `--set "train.sampler_powers={crosses: 0.5, actions: 0.3, looks: 0.3}"`
- `section.field` only (2 levels); for `schedule.phases` edit `configs/schedule.yaml`.

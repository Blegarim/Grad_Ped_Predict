# Setup — fresh machine to trained model

The whole pipeline, start to finish, on the current **v2 data contract**. What `actions`/`looks`/
`crosses` mean and how the data is built is in [CLAUDE.md](CLAUDE.md) (Data Pipeline); this file is the
runbook. Steps 0–1 need no dataset, so you can install and pass the gate before downloading anything.

> **Rebuilding over an old v1 checkout?** The v1 sequence pkls and LMDBs are obsolete (the runtime
> hard-errors on v1 chunks). Delete `data/sequences/*.pkl` and `preprocessed_{train,train_aug,val,test}/`
> before step 3, then follow the steps as written — they regenerate everything under the v2 contract.

## 0. Prerequisites
- **Python 3.10–3.12** (3.13+ won't build the pinned torch/numpy — README).
- **git**; **ffmpeg** on PATH (PIE's frame extractor shells out to it — needed for val/test only).
- **CUDA GPU + driver** for training in reasonable time (CPU is fine for the test gate, not for training).
- **Disk:** PIE clips are tens of GB; extracting a *full split's* frames is hundreds of GB; LMDBs sit on
  top. The **train** build avoids full-split extraction (it extracts per-chunk and deletes — step 4), so
  budget for clips + the growing LMDB, not clips + all frames. Low-storage knob (**C3**):
  `data.lmdb_map_size_bytes` is **pre-allocated per chunk** on Windows — the 4 GiB default reserves
  ~76 GB across the ~19 train chunks even though real payload is ~2–3 GB/chunk. On a tight disk, build
  one chunk, measure it, and pass `--set data.lmdb_map_size_bytes=<measured+30%>`.

## 1. Code + environment
```powershell
git clone <repo>           # brings the vendored PIE/ toolkit with it
cd Grad_Ped_Predict
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
# GPU wheels — the .[dev] install above pulls the CPU-only torch from PyPI; force-replace it.
# torch 2.7.1 ships cu118/cu126/cu128 (NOT cu121); pick the tag your driver supports (cu126 is a safe default).
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```
Verify the GPU build took (must print `2.7.1+cu126 True`, not `+cpu False`):
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Gate (no dataset needed — confirms the install is sound):
```powershell
ruff check .
pytest -m "not slow"
```

## 2. Acquire the PIE dataset
"PIE" is two things — don't conflate them:
- **PIE toolkit (code)** = the `PIE/` folder the clone already brought in. Only job: be importable
  (`from PIE.utilities.pie_data import PIE`). No videos go here.
- **PIE dataset (videos + annotations)** = a separate multi-GB download from York University, placed under
  `data/` (`paths.yaml` → `pie_root: data`).

Download and lay out under `data/`:
```
Grad_Ped_Predict/
  PIE/                       # toolkit (code) — already cloned; holds no videos
  data/                      # ← pie_root
    PIE_clips/               # set01..set06/video_####.mp4   (from YorkU)
    annotations/             # set01..set06/*_annt.xml
    annotations_attributes/
    annotations_vehicle/     # OBD — REQUIRED for the v2 ego-speed channel (M9); don't skip it
    images/                  # created on demand (val/test extraction, step 4)
    sequences/               # created by step 3
```
Split mapping is fixed by PIE: **train = set01/02/04, val = set05/06, test = set03**.

## 3. Generate sequence windows (annotations only — no frames yet)
```powershell
python scripts/make_sequences.py --split all      # sequences_{train,val,test}.pkl + *_stats.json
python scripts/make_sequences.py --benchmark      # M5 TTE-protocol eval set (test split only)
```
Windowing params (`seq_len=20`, `stride=3`, `future_offset=30`, `tol=2`) come from
[configs/data.yaml](configs/data.yaml). v2 labels: `actions`/`looks` = state at the **last observed
frame**; `crosses` = any crossing in the future window; **right-censored windows are dropped and
counted**. Record the printed **`censored`** count — it's the thesis sentence "N windows excluded as
right-censored."

Then run the drift canary **now**, before spending hours on LMDBs — `--from-sequences` counts the pkls
you just generated (the base LMDB is a 1:1 image of them, so the numbers are identical to a post-build
scan; without the flag the script needs the LMDBs to exist):
```powershell
python scripts/count_labels.py --from-sequences
```
⚠️ The legacy ~95,684 train / 22,665 val / 76,048 test figures are **v1 and STALE** — the v2 relabel
(state-at-end + censor-drop) deflates N and every positive rate (`looks` hardest), so v2 counts *will*
differ legitimately. The gate is currently relaxed to structural checks. **Re-pin from this run** (one
doc-sync change): update the Dataset Statistics table in [CLAUDE.md](CLAUDE.md), re-pin
`tests/fixtures/golden/pie_sequences_counts.json`, and re-check `train.sampler_powers` for `looks` if its
rate fell far. After re-pinning, a nonzero exit again means real drift — stop and investigate.

## 4. Build LMDBs (pkl → preprocessed chunks)
ImageNet normalization is applied at read time, not here. Two paths:

**val / test / benchmark — extract that split's frames, then build** (small enough to stage whole):
```powershell
python -c "import sys; sys.path.insert(0,'.'); from PIE.utilities.pie_data import PIE; PIE(data_path='data').extract_and_save_images(extract_frame_type='annotated')"
python scripts/build_lmdb.py --split val
python scripts/build_lmdb.py --split test
python scripts/build_lmdb.py --split test_benchmark
```
The Python extractor processes whatever set folders exist under `data/PIE_clips/`, so stage one split's
sets at a time and delete its frames before the next. (Use `'all'` instead of `'annotated'` only if
sequence-gen later reports missing frames.) It writes lossless **PNG**; on a tight disk, swap the
one-liner for the JPG variant — same annotated selection, ~3–5× smaller transient cache, found
transparently by the runtime's same-stem `.jpg` fallback:
```powershell
python scripts/extract_annotated_jpg.py
```

**train — the self-bounding builder (no full-split extraction):**
```powershell
python scripts/build_lmdb_incremental.py --split train
```
It consumes `sequences_train.pkl` and, chunk by chunk, extracts **only** the frames those records
reference straight from `data/PIE_clips/` (cv2, byte-identical to PIE's extractor), builds the chunk, then

deletes the spent frames. Peak disk = the videos straddling one chunk + the growing LMDB — never the whole
split, no pre-extracted `images/` or ffmpeg needed. It's resumable: the **C2** guard counts the committed
records in the highest chunk and, if short, refuses to continue and names the partial `chunk_NNNNNN.lmdb`
to delete — a crashed build can no longer silently skip a half-written chunk. Set the **C3** map_size knob
(step 0) on a tight disk. Override with `--start-idx N` / `--keep-frames`.

Storage-limited staging order (all of a split's sets must be present together to build it):

| Round | Extract sets | Build | Then delete |
|---|---|---|---|
| 1 | set05, set06 | `build_lmdb.py --split val` | val frames |
| 2 | set03 | `build_lmdb.py --split test` + `--split test_benchmark` | test frames |
| 3 | set01, 02, 04 | `build_lmdb_incremental.py --split train` | (auto, per chunk) |

## 5. Augmentation (default-ON imbalance lever)
The default imbalance policy is augmentation + online sampler + loss weights; the trainer unions
`preprocessed_train` with `preprocessed_train_aug`, so this dir must exist:
```powershell
python scripts/augment_dataset.py      # augment.enabled defaults true; oversamples crosses/looks → preprocessed_train_aug
```
Augment reads its source crops **straight from the built `preprocessed_train` LMDB** (step 4), not the
PIE frames — so it runs cleanly after the incremental build that deleted them. Run it after step 4;
augment errors if `preprocessed_train` is missing. (To augment a balanced base instead, point it at that
dir: `--in-dir preprocessed_train_balanced`.)
Opt-in alternative for ablation only (the downsample path — do **not** stack with augmentation):
```powershell
python scripts/balance_dataset.py --split train --set balance.enabled=true
```

## 6. Train
```powershell
python scripts/train.py --set eval.model_type=full
```
`full` is the default, so the bare `python scripts/train.py` trains the full model; the ablation arms are
`--set eval.model_type=motion_only|visual_only|vanilla_concat`. **The model selector lives in the `eval`
section, not `model`** — `--set model.model_type=...` raises `Unknown field 'model_type'`.
Writes `outputs/runs/{timestamp}_full/` with `resolved_config.yaml` (incl. the seed), `train_log.csv`,
`train_distribution.json`, `checkpoints/{best,last}.pth`, `plots/`. Override anything inline, e.g.
`--set train.lr=5e-5` — see the [override reference](#appendix--config-override-reference---set) below.

## 7. Evaluate (two passes — thresholds tuned on val, applied to test)
```powershell
python scripts/evaluate.py --split val  --checkpoint outputs/runs/<run>/checkpoints/best.pth   # tunes + stores thresholds.json
python scripts/evaluate.py --split test --checkpoint outputs/runs/<run>/checkpoints/best.pth   # loads + applies them
```
Report the `tuned_*` columns only; `oracle_*` are same-split (leakage) diagnostics. Lead metric is
`crosses_f1` (accuracy is misleading at ~37:1). `index.csv` tracks runs for cross-run comparison.

## 8. Optional downstream
- ONNX export + parity: `pip install -e .[export]` → `python scripts/export_onnx.py …`
- Video inference (YOLO): `pip install -e .[infer]` → `python scripts/infer_video.py …`
- Plots / qualitative: `python scripts/visualize.py …`

---

### Critical path
install (`.[dev]` + GPU torch) → gate → drop PIE into `data/` (incl. `annotations_vehicle/`) →
`make_sequences.py --split all` + `--benchmark` → `count_labels.py --from-sequences` (re-pin) → build LMDBs (val/test
standard, train incremental, + `test_benchmark`) → `augment_dataset.py` → `train.py` → `evaluate.py`
(val then test).

### Gotchas
- The **GPU torch reinstall** (step 1) is easy to miss — without it you silently train on CPU.
- **`count_labels.py --from-sequences` is your canary** — it reads the pkls, so run it right after step 3,
  not after burning hours on LMDBs (the bare `count_labels.py` form scans the LMDBs and only works post-build).
- **Augment is mandatory for the default config** even though it's described as a "lever" (the trainer
  unions `preprocessed_train_aug`).
- **`annotations_vehicle/` is required** for the ego-speed channel — a missing OBD file fails sequence-gen.

---

## Appendix — config override reference (`--set`)

Every CLI script (`train.py`, `evaluate.py`, …) exposes the **same two argparse flags** — there are no
per-field flags:

| Flag | Default | Meaning |
|---|---|---|
| `--config-dir DIR` | `configs` | Directory holding the `*.yaml` config files. |
| `--set section.field=value` | — | Override one config value. **Repeatable.** Also accepts `--section.field value`. |

So **everything below is reached through `--set`**, validated against the dataclass schema
([`config/schema.py`](src/pedpredict/config/schema.py)). An unknown section or field raises
`ConfigError` (e.g. the common `model.model_type` mistake — the selector is `eval.model_type`).

**Override syntax:**
- **Scalars:** `--set train.lr=5e-5`, `--set train.batch_size=2`.
- **Bools:** `true/false/yes/no/on/off/1/0` — `--set train.use_weighted_sampler=false`.
- **Optional (nullable) fields:** pass yaml `null` — `--set train.chunk_warm_mem_timeout=null`,
  `--set data.lmdb_map_size_bytes=null`.
- **Lists / tuples:** yaml-parsed — `--set "model.window_size=[8, 4, 2, null]"`.
- **Dicts:** yaml-parsed and **replace the whole dict** (no deep-merge) — you must pass every key:
  `--set "train.sampler_powers={crosses: 0.5, actions: 0.3, looks: 0.7}"`.
- **Depth limit:** overrides are exactly `section.field` (two parts). Nested dataclasses (i.e.
  `schedule.phases[*]`) are **not** addressable via `--set` — edit `configs/schedule.yaml` instead.

> **Imbalance levers (the RQ3 ablation axis).** Only `train.use_weighted_sampler`,
> `train.use_class_weights`, and `train.sampler_powers` are *live at train time*. `balance.enabled` and
> `augment.enabled` are **offline build-time** flags (`balance_dataset.py` / `augment_dataset.py`); passing
> them to `train.py` does nothing — augmentation is "on" at train time iff `preprocessed_train_aug/` exists.
> `train_distribution.json` (written to every run dir) reports the *effective* per-task draw rate so you
> never toggle blind.

### `paths.*` — artifact locations
| Field | Default |
|---|---|
| `pie_root` | `data` |
| `sequences_dir` | `data/sequences` |
| `lmdb_train` | `[preprocessed_train, preprocessed_train_aug]` |
| `lmdb_train_balanced` | `[preprocessed_train_balanced]` |
| `lmdb_val` | `preprocessed_val` |
| `lmdb_test` | `preprocessed_test` |
| `lmdb_test_benchmark` | `preprocessed_test_benchmark` |
| `log_dir` / `ckpt_dir` / `run_ckpt_dir` | `training_log` / `best_model_outputs` / `model_outputs` (legacy) |
| `runs_dir` | `outputs/runs` |

### `data.*` — data-layer constants
| Field | Default | Note |
|---|---|---|
| `max_seq_len` | `20` | |
| `motion_dim` | `8` | consumed motion width; 8 = no ego, 9 = with ego (must equal `model.motion_dim`) |
| `source_width` / `source_height` | `1920` / `1080` | PIE frame px; used by flip aug + motion-norm cross-check |
| `img_height` / `img_width` | `128` / `128` | tight-crop write+read size |
| `read_context_height` / `read_context_width` | `224` / `224` | context model input (must be square; tiles ViT windows) |
| `context_scale` | `3.0` | context crop = scale × tight bbox |
| `jpeg_quality` | `90` | |
| `chunk_size` | `5000` | samples per LMDB chunk |
| `lmdb_map_size_bytes` | `4294967296` (4 GiB) | per-chunk pre-alloc on Windows; `null` → heuristic (**C3**) |
| `lmdb_map_size_floor_gib` / `lmdb_map_size_safety` | `4.0` / `1.5` | heuristic floor + multiplier (used when bytes is null) |
| `preprocess_num_workers` / `preprocess_prefetch_factor` | `8` / `2` | offline writer DataLoader |
| `seq_len` / `stride` / `future_offset` / `tol` | `20` / `3` / `30` / `2` | sliding-window gen params |
| `benchmark_obs_len` | `16` | M5 TTE-protocol obs length |
| `benchmark_tte_min` / `benchmark_tte_max` | `30` / `60` | TTE sampling window |
| `benchmark_overlap` | `0.6` | benchmark window overlap |
| `min_track_size` / `fstride` | `10` / `1` | PIE source opts |
| `data_split_type` / `seq_type` | `default` / `all` | |
| `squarify_ratio` / `height_min` / `height_max` | `0.0` / `0.0` / `null` | PIE bbox opts (`null` → +inf) |
| `norm_mean` / `norm_std` | ImageNet | applied at read time |

### `model.*` — architecture (no `model_type` here — see `eval.model_type`)
| Field | Default | Note |
|---|---|---|
| `d_model` | `128` | unified across ALL modules — never change one alone |
| `in_channels` | `3` | |
| `motion_dim` | `8` | must equal `data.motion_dim` |
| `motion_norm` | `image` | `image` (fixed frame-dim scale) \| `per_sequence` (legacy z-norm, A4 arm) |
| `motion_norm_image_size` | `[1920, 1080]` | must equal `(data.source_width, source_height)` |
| `ego_speed_scale` | `50.0` | km/h scale for ego channel under `image` norm |
| `stage_dims` | `[36, 36, 288, 36]` | ViT (lengths of the 5 stage lists must match) |
| `layer_nums` | `[2, 4, 5, 7]` | |
| `head_nums` | `[2, 2, 16, 2]` | each `stage_dims[i]` must divide by `head_nums[i]` |
| `window_size` | `[8, 4, 2, null]` | `null` = global window |
| `mlp_ratio` | `[4, 4, 4, 4]` | |
| `drop_path` / `attn_dropout` / `proj_dropout` / `dropout` | `0.15` | ViT dropouts |
| `motion_hidden_dim` | `168` | must divide by `motion_num_heads` |
| `motion_num_layers` / `motion_num_heads` / `motion_dropout` | `2` / `8` / `0.3` | |
| `head_dropout` | `0.1` | classifier + cross-attn dropout |
| `num_classes` | `{actions: 2, looks: 2, crosses: 2}` | dict — keys fixed to the 3 tasks |
| `cross_attn_num_heads` | `4` | must divide `d_model` |
| `use_frame_crosses` | `true` | |
| `frame_pool` | `logsumexp` | `logsumexp` \| `max` \| `mean` |
| `emit_crosses_pooled` | `true` | live-but-unsupervised aux head; `false` drops it |

### `train.*` — training loop
| Field | Default | Note |
|---|---|---|
| `lr` / `weight_decay` | `1e-4` / `1e-5` | Adam |
| `batch_size` / `num_epochs` / `num_workers` | `4` / `30` / `4` | |
| `use_amp` | `true` | request; runtime-gated by CUDA |
| `seed` | `42` | global RNG seed (M7) |
| `selection_metric` | `macro_f1` | `val_loss` \| `macro_f1` \| `crosses_f1` (best.pth + early stop) |
| `loss_weight` | `{actions: 0.8, looks: 0.8, crosses: 1.2}` | per-task loss scalar (dict) |
| `use_class_weights` | `true` | **imbalance lever 3** — inverse-freq CE weights |
| `use_weighted_sampler` | `true` | **imbalance lever 2** — `WeightedRandomSampler` |
| `sampler_powers` | `{crosses: 1.5, actions: 0.3, looks: 0.7}` | sampler intensity (dict) |
| `sampler_min_weight` | `1e-6` | per-sample weight floor |
| `grad_clip_max_norm` | `1.0` | |
| `early_stop_patience` / `early_stop_min_delta` | `20` / `0.001` | patience wide enough for the cosine to finish |
| `lr_schedule` | `warmup_cosine` | `warmup_cosine` (linear warmup → cosine to `lr_min`, deterministic) \| `plateau` (`ReduceLROnPlateau` on val_loss) |
| `warmup_epochs` / `warmup_start_factor` | `1` / `0.1` | warmup_cosine: first epoch runs at `warmup_start_factor·lr`, peak from epoch 1 (`0` = no warmup) |
| `lr_min` | `1e-6` | cosine `eta_min`; also the `ReduceLROnPlateau` floor under `lr_schedule=plateau` |
| `sched_factor` / `sched_patience` / `sched_threshold` | `0.5` / `2` / `1e-4` | `ReduceLROnPlateau` knobs — **`lr_schedule=plateau` only** |
| `chunk_preload_depth` | `3` | warm-ahead window |
| `chunk_warm_ram_threshold` / `chunk_warm_mem_interval` | `96.0` / `1.0` | RAM gate before each warm spawn |
| `chunk_warm_mem_timeout` | `null` | opt-in cap on the infinite RAM wait |
| `chunk_queue_timeout` | `300.0` | skip-on-timeout for a stuck warm worker |
| `dataloader_prefetch_factor` | `2` | (when `num_workers > 0`) |

### `eval.*` — evaluation / benchmark
| Field | Default | Note |
|---|---|---|
| `batch_size` / `num_workers` | `16` / `4` | |
| `model_type` | `full` | **THE model selector** — `full` \| `motion_only` \| `visual_only` \| `vanilla_concat` |
| `bench_batch_size` / `bench_warmup` / `latency_trials` | `1` / `10` / `50` | efficiency benchmark knobs |
| `threshold_sweep_lo` / `threshold_sweep_hi` / `threshold_sweep_step` | `0.10` / `0.90` / `0.05` | val F1-threshold sweep |

### `infer.*` — video inference (needs `.[infer]`)
| Field | Default |
|---|---|
| `detector_weights` / `detector_class_idx` / `detector_conf` | `yolo11n.pt` / `0` / `0.3` |
| `window_stride` / `batch_size` / `default_fps` | `1` / `32` / `30.0` |
| `draw_color_chips` | `true` |

### `balance.*` — offline balance (lever 1, OPT-IN, build-time only)
| Field | Default | Note |
|---|---|---|
| `enabled` | `false` | the majority-downsample *alternative* to augmentation |
| `cross_pos_ratio` | `0.30` | target crosses=1 fraction |
| `target_action_rate` / `target_look_rate` | `0.5` / `0.5` | |
| `x11_select` | `lower` | `lower` \| `upper` |
| `subsample_cross1` / `allow_approx` | `true` / `true` | |
| `on_infeasible` | `empty` | `raise` \| `empty` |
| `legacy_x00_sign_bug` | `false` | parity-only |
| `seed` | `0` | |

### `augment.*` — offline minority augmentation (default lever, build-time only)
| Field | Default | Note |
|---|---|---|
| `enabled` | `true` | builds `preprocessed_train_aug` — **read by `augment_dataset.py`, not `train.py`** |
| `n_augs_min` / `n_augs_max` | `2` / `4` | per-call transform count |
| `p_flip` / `p_color` / `p_noise` / `p_erase` | `0.5` / `0.4` / `0.3` / `0.2` | per-transform probs |
| `color_brightness` / `color_contrast` / `color_saturation` / `color_hue` | `0.2` / `0.2` / `0.3` / `0.1` | ColorJitter |
| `motion_noise_std` / `erase_n_frames` | `0.02` / `2` | |
| `crosses_multiplier` / `looks_multiplier` | `6` / `3` | oversampling |
| `seed` | `42` | |

### `schedule.*` — multi-phase training
| Field | Default | Note |
|---|---|---|
| `enabled` | `false` | `true` → 3-phase schedule (`run_phase_schedule`) |
| `phases` | 3-phase default | tuple of `PhaseCfg` — **not `--set`-able per-field; edit `configs/schedule.yaml`** |

### `export.*` — ONNX export (needs `.[export]`)
| Field | Default |
|---|---|
| `opset` / `output_dir` | `17` / `outputs/onnx` |
| `include_temporal_weights` | `false` |
| `parity_atol` / `parity_rtol` | `1e-4` / `1e-4` |
| `parity_batch_size` / `parity_seq_len` | `2` / `4` |

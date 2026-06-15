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
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121   # GPU wheel; default resolves to CPU
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

Then run the drift canary **now**, before spending hours on LMDBs:
```powershell
python scripts/count_labels.py
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
sequence-gen later reports missing frames.)

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
Opt-in alternative for ablation only (the downsample path — do **not** stack with augmentation):
```powershell
python scripts/balance_dataset.py --split train --set balance.enabled=true
```

## 6. Train
```powershell
python scripts/train.py --set model.model_type=full
```
Writes `outputs/runs/{timestamp}_full/` with `resolved_config.yaml` (incl. the seed), `train_log.csv`,
`train_distribution.json`, `checkpoints/{best,last}.pth`, `plots/`. Override anything inline, e.g.
`--set train.lr=5e-5`.

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
`make_sequences.py --split all` + `--benchmark` → `count_labels.py` (re-pin) → build LMDBs (val/test
standard, train incremental, + `test_benchmark`) → `augment_dataset.py` → `train.py` → `evaluate.py`
(val then test).

### Gotchas
- The **GPU torch reinstall** (step 1) is easy to miss — without it you silently train on CPU.
- **`count_labels.py` is your canary** — run it right after step 3, not after burning hours on LMDBs.
- **Augment is mandatory for the default config** even though it's described as a "lever" (the trainer
  unions `preprocessed_train_aug`).
- **`annotations_vehicle/` is required** for the ego-speed channel — a missing OBD file fails sequence-gen.

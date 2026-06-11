0. Machine prerequisites
 Python 3.10–3.12 (3.13+ won't build the pinned torch/numpy — README:92). Your .venv is already 3.10 per memory, but this is a fresh PC, so install it.
 git, and a CUDA-capable GPU + driver if you want to train in a reasonable time (CPU works for tests, not for real training).
 ffmpeg on PATH — PIE's frame extractor shells out to it.
 Disk: budget large. PIE clips are tens of GB; extracted annotated frames add a lot, extracting all frames is hundreds of GB. Then LMDBs on top. Plan for ~0.5 TB headroom if extracting all.

1. Get the code + environment
 Clone the repo (brings the vendored PIE/ toolkit with it).
 Create venv and install:

python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
 GPU torch (the pinned torch==2.7.1 resolves to a CPU wheel by default — README:105):

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu121
 Sanity gate before touching data:

ruff check .
pytest -m "not slow"
These don't need the dataset and confirm the install is sound.

2. Acquire the PIE dataset
 Heads-up — "PIE" is two separate things, don't conflate them:
   - PIE toolkit (code) = the `PIE/` folder the repo clone already brought in. Its only job is to be importable (`from PIE.utilities.pie_data import PIE`, make_sequences.py:29). Don't put dataset videos in here.
   - PIE dataset (videos + annotations) = a separate multi-GB download from York University. It goes under `data/` (paths.yaml `pie_root: data`; the pipeline constructs `PIE(data_path='data')`).

 Request/download PIE from the official source (York University PIE dataset — videos PIE_clips/ for set01–set06, plus annotations/, annotations_attributes/, annotations_vehicle/).
 Place them under `data/` so the layout matches the tree below. The annotations bundled in the `PIE/` clone must also live at `data/annotations/` — copy them across (or just download the full annotation set into `data/`). The PIE class reads/writes these subfolders under data_path (pie_data.py:59-64):

```
Grad_Ped_Predict/
  PIE/                       # toolkit (code) — already cloned; importable, holds no videos
  data/                      # ← pie_root: the PIE class's data_path
    PIE_clips/               # set01..set06/video_####.mp4   (download from YorkU)
    annotations/             # set01..set06/*_annt.xml
    annotations_attributes/
    annotations_vehicle/
    images/                  # created by step 3 (extract_and_save_images); empty for now
    sequences/               # created by step 4 (make_sequences.py)
```

 Split mapping is fixed by PIE: train = set01/02/04, val = set05/06, test = set03 (pie_data.py:90-94).

3. Extract frames from clips (PIE's own tool)
 Run PIE's extract_and_save_images — it reads data/PIE_clips/ and writes data/images/setXX/video_YYYY/00000.png:

python -c "import sys; sys.path.insert(0,'.'); from PIE.utilities.pie_data import PIE; PIE(data_path='data').extract_and_save_images(extract_frame_type='annotated')"
Use 'annotated' first (smaller — it's the frames behavior sequences reference). If sequence generation later complains about missing frames, re-run with 'all'. The Python extractor processes whatever set folders exist in data/PIE_clips/ (pie_data.py:229), so it's already per-set if you only stage the sets you're working on.

Incremental extraction (storage-limited PCs)
 Extracting ALL frames for the whole dataset is ~3 TB; even 'annotated' is hundreds of GB. If the disk can't hold it all at once, process one SPLIT at a time — extract → build its LMDB → delete its frames → next. Granularity is the split, NOT arbitrary sets: build_lmdb crops every frame a split's pkl references, and PIE's split→set mapping is fixed (pie_data.py:90-94):

   - val   = set05 + set06
   - test  = set03
   - train = set01 + set02 + set04   ← all three must be present together to build the train LMDB

 So "set01+02 all the way to LMDB then delete" does NOT work — the train LMDB also needs set04. Generate all three sequence pkls up front (step 4 — annotations only, tiny, frame-free), then loop:

 | Round | Extract sets   | Build                                         | Then delete  |
 |-------|----------------|-----------------------------------------------|--------------|
 | 1     | set05, set06   | build_lmdb --split val                        | val frames   |
 | 2     | set03          | build_lmdb --split test                        | test frames  |
 | 3     | set01,02,04    | build_lmdb --split train  → augment_dataset    | train frames |

 To extract specific sets with PIE's own ffmpeg script (all frames), use the parametrized helper scripts/split_sets.sh (a per-set variant of PIE/annotations/split_clips_to_frames.sh). Run it from inside data/ (relative paths) in Git Bash with ffmpeg on PATH:

   cd /d/Grad_Ped_Predict/data
   bash ../scripts/split_sets.sh set05 set06

 Prefer the Python 'annotated' extractor above when you only need annotated frames (~10x smaller); use split_sets.sh only if you need all frames.

Resumable per-video build (train, or any disk that can't hold a whole split)
 The per-split table above still needs ALL of a split's frames staged at once — fine for val/test, but
 train (set01+02+04) peaks at hundreds of GB and is the usual cause of a mid-build disk-full crash. For
 train specifically (or any split too big to stage), skip steps 3+5 and use the self-bounding builder:

   python scripts/build_lmdb_incremental.py --split train

 It generates nothing new — it consumes the existing sequences_train.pkl (run step 4 first) and, chunk by
 chunk, extracts ONLY the frames those records reference (cv2, byte-identical to PIE's extractor), builds
 the chunk, then deletes the spent video frames. Peak disk = the videos straddling one chunk + the growing
 LMDB, never the whole split. No ffmpeg or pre-extracted images/ needed — just data/PIE_clips/.

 It is resumable: it auto-detects the completed chunk_NNNNNN.lmdb dirs and continues from the next index, so
 a crashed build picks up where it stopped. ⚠️ First delete the partial final chunk (the one being written
 when the disk filled — e.g. chunk_020000.lmdb if you got ~20k in); the builder refuses to resume onto an
 existing chunk dir. Override resume/cleanup with --start-idx N / --keep-frames.

4. Generate sequence windows (PIE → pkl)
 ```powershell python scripts/make_sequences.py --split all

Produces `data/sequences/sequences_{train,val,test}.pkl`. Windowing params (`seq_len=20`, `stride=3`, `future_offset=30`, `tol=2`) come from [configs/data.yaml](configs/data.yaml). This is the step that imports PIE and unwraps tracks ([make_sequences.py](scripts/make_sequences.py)).
 Verify counts match the documented table (drift gate):

python scripts/count_labels.py
Expect ~95,684 train / 22,665 val / 76,048 test. Nonzero exit = your sequences drifted from the documented stats — stop and investigate before training.

5. Build LMDBs (pkl → preprocessed chunks)
 ```powershell python scripts/build_lmdb.py --split all

Writes `preprocessed_train/`, `preprocessed_val/`, `preprocessed_test/` (JPEG crops + motion/labels; ImageNet norm applied at read time, not here).

6. Augmentation (default-ON imbalance lever)
The default imbalance policy is augmentation + online sampler + loss weights (CLAUDE.md). The trainer expects preprocessed_train_aug to exist because lmdb_train = [preprocessed_train, preprocessed_train_aug] are unioned at train time.

 ```powershell python scripts/augment_dataset.py --set augment.enabled=true

Reads `sequences_train.pkl`, oversamples minority (crosses/looks) records, writes `preprocessed_train_aug/` ([augment_dataset.py](scripts/augment_dataset.py)).
 Optional, opt-in alternative — offline balance (do not stack with augmentation; it`s the downsample alternative for ablation). Skip unless ablating:

python scripts/balance_dataset.py --split train --set balance.enabled=true

7. Train
 ```powershell python scripts/train.py --set model.model_type=full

Writes a run dir `outputs/runs/{timestamp}_full/` with `resolved_config.yaml`, `train_log.csv`, `checkpoints/{best,last}.pth`, `plots/`. Override anything inline, e.g. `--set train.lr=5e-5`.

8. Evaluate on test
 ```powershell python scripts/evaluate.py --set model.model_type=full

Writes `eval_log.csv` into the run dir. Lead metric is `crosses_f1` (accuracy is misleading at ~37:1 imbalance). `rebuild_index` / the index regen keeps `outputs/runs/index.csv` for cross-run comparison.

9. Optional downstream
 ONNX export + parity: pip install -e .[export] then python scripts/export_onnx.py …
 Video inference (YOLO): pip install -e .[infer] then python scripts/infer_video.py …
 Plots/qualitative: python scripts/visualize.py …
The critical-path short version
install (.[dev] + GPU torch) → 2. drop PIE into data/ → 3. extract_and_save_images → 4. make_sequences.py --split all → count_labels.py (gate) → 5. build_lmdb.py --split all → 6. augment_dataset.py --set augment.enabled=true → 7. train.py → 8. evaluate.py.

A few gotchas worth flagging up front: the GPU torch reinstall (step 1) is easy to miss and you`ll silently train on CPU; augment is mandatory for the default config even though it`s described as a "lever" (the trainer unions that dir); and count_labels.py is your canary — run it right after sequence generation, not after you`ve burned hours on LMDBs.


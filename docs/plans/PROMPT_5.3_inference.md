# Sub-plan — Prompt 5.3: Video inference (`eval/inference.py` + `scripts/infer_video.py`)

> Phase-A port of `OLD/Undergrad_thesis_project/main.py` (+ `extract_frames.py`,
> `scripts/pedestrian_detection.py`). **Deliverable: a detailed sub-plan** (skeletons + signatures),
> not final production code.
> Dependency status: **green** — 1.1 (`SequenceRecord`), 1.2 (`crop_tight`/`crop_context`/`compute_motion`/
> `build_read_transforms`), 1.5 (`collate`), 2.4 (`registry`), 5.1 (`load_eval_weights`), 0.2/0.3
> (config/paths). The `[infer]` extra (`ultralytics`, `lap`) and core `opencv-python` already exist in
> `pyproject.toml`.

---

## 0. Scope & guiding principle

`OLD/main.py` is a 326-line top-level script that conflates **six** concerns: (1) detection+tracking
(YOLO via `pedestrian_detection.extract_tracks_from_video`/`smooth_track`), (2) track→sequence assembly
(sliding window), (3) preprocessing (re-using `PIESequenceDataset` with **in-memory crops** and a bespoke
`inference_collate_fn`), (4) model build+load, (5) forward + per-task argmax, (6) frame-result aggregation
+ cv2 overlay + `VideoWriter`. `extract_frames.py` is an unrelated frame-dumper; `pedestrian_detection.py`
also carries a dead 4-dim `extract_sequences_from_track` path.

The rebuild keeps **only the orchestration** in `eval/inference.py` and routes the *math* to modules that
already own it — the same discipline as 5.1. The pipeline becomes five named, separable stages:

```
detect/track ──▶ assemble sequences ──▶ preprocess (Phase-1 reuse) ──▶ model forward (2.4/5.1) ──▶ overlay
  (YOLO,           (smooth + sliding        (crop_tight/crop_context/      (build_model +              (cv2
   [infer] extra)   window → records)        compute_motion + read xf)      forward_model)             VideoWriter)
```

| OLD `main.py` piece | Rebuild home | Why |
|---|---|---|
| `extract_tracks_from_video`, `smooth_track` | **port** → `inference.detect_tracks` / `smooth_track` (lazy `ultralytics` import) | B5; YOLO stays an optional `[infer]` dep |
| track→window loop (`main.py` 92–117) | **new** `assemble_windows` (pure, testable) | B5; isolate the windowing from PIE/YOLO |
| `PIESequenceDataset(..., preload, in-memory)` + `inference_collate_fn` | **reuse** `crop_tight`/`crop_context`/`compute_motion`/`build_read_transforms` (1.2) | one preprocessing path, no second collate (B5/B7) |
| `get_model` + `model(...)` + per-key argmax | **reuse** `build_model`/`forward_model` (2.4) + `load_eval_weights` (5.1) | B10; no stringly dispatch, strict load (B2) |
| frame-result bbox-matching aggregation (162–228) | **replace** — carry `(track_id, frame_idx)` on each window so no fragile bbox re-match | removes an O(n²) `==`-on-bboxes lookup smell |
| cv2 overlay + `VideoWriter` (230–325) | **port** `render_overlays` (cosmetic, kept) | parity of the visual output |
| `extract_frames.py` | **fold** into a `FrameSource` (dir-of-frames input) | B11 one-off → library helper |
| `pie._map_scalar_to_text` label text | **inline** a tiny `{0,1}` label map (no PIE toolkit at inference) | drop the PIE runtime dep for inference |

`inference.py` stays a thin orchestrator: *open source → detect → assemble → preprocess+forward → render*.

---

## 1. Target files & public API

### 1a. `src/pedpredict/eval/inference.py`

```python
"""Video / frame-sequence inference (Prompt 5.3) — ports OLD main.py.

Reuses the Phase-1 preprocessing math (crop_tight/crop_context/compute_motion/build_read_transforms, 1.2)
instead of re-running PIESequenceDataset on in-memory crops (B5/B7); the typed registry
build_model/forward_model (2.4, B10) + load_eval_weights (5.1, B2) for build/load/forward. Detection/
tracking is YOLO via the optional [infer] extra (lazy import). Five separable stages: detect -> assemble
-> preprocess -> forward -> overlay.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from pedpredict.config.schema import DataCfg, InferenceCfg, RootCfg
from pedpredict.data.transforms import compute_motion, crop_context, crop_tight, build_read_transforms
from pedpredict.models.registry import build_model, forward_model
from pedpredict.eval.evaluate import load_eval_weights
from pedpredict.utils.amp import autocast_ctx, resolve_amp
from pedpredict.utils.device import get_device

__all__ = [
    "Detection", "Track", "TrackWindow", "FramePrediction", "InferenceResult",
    "FrameSource", "VideoFrameSource", "DirFrameSource", "open_frame_source",
    "detect_tracks", "smooth_track", "assemble_windows",
    "preprocess_window", "predict_windows", "aggregate_by_frame",
    "render_overlays", "run_video_inference",
]

# ---- data types (typed replacements for OLD's dict-of-dicts) --------------------------------
@dataclass(slots=True, frozen=True)
class Detection:
    frame_idx: int
    bbox: tuple[int, int, int, int]      # (x1, y1, x2, y2), clamped to frame bounds
    image: np.ndarray                    # RGB HxWx3 crop (BGR->RGB done at detection; see B-flag)

Track = list[Detection]                  # one tracked pedestrian, frame-ordered

@dataclass(slots=True, frozen=True)
class TrackWindow:
    track_id: int
    detections: tuple[Detection, ...]    # exactly seq_len frames
    @property
    def frame_idxs(self) -> tuple[int, ...]: ...
    @property
    def bboxes(self) -> tuple[tuple[int, int, int, int], ...]: ...

@dataclass(slots=True, frozen=True)
class FramePrediction:
    track_id: int
    bbox: tuple[int, int, int, int]
    actions: int                         # argmax {0,1}
    looks: int
    crosses: int                         # from crosses_frame (B4)

@dataclass(slots=True, frozen=True)
class InferenceResult:
    frame_results: dict[int, list[FramePrediction]]   # frame_idx -> detections
    n_frames: int
    n_windows: int
    out_video: Path | None

# ---- stage 0: frame source (video file OR dir of frames; folds extract_frames.py) -----------
class FrameSource(Sequence[np.ndarray]):
    """Abstract RGB-frame provider; __len__/__getitem__ + fps/size metadata."""
    fps: float; width: int; height: int
    def __iter__(self) -> Iterator[np.ndarray]: ...

class VideoFrameSource(FrameSource): ...   # cv2.VideoCapture, BGR->RGB
class DirFrameSource(FrameSource): ...      # sorted *.jpg/*.png dir (qualitative_visualize/), default fps

def open_frame_source(path: str | Path, *, fps: float | None = None) -> FrameSource:
    """Dispatch on path: a file -> VideoFrameSource, a directory -> DirFrameSource."""
    ...

# ---- stage 1: detection + tracking (lazy ultralytics; [infer] extra) ------------------------
def detect_tracks(source: FrameSource, cfg: InferenceCfg) -> dict[int, Track]:
    """YOLO detect+track over `source`; returns {track_id: Track} (RGB crops, clamped bbox).

    Lazy-imports ultralytics so importing this module without the [infer] extra works; raises a clear
    'pip install pedpredict[infer]' error only when called. Ports pedestrian_detection.extract_tracks.
    """
    ...

def smooth_track(track: Track, window: int) -> Track:
    """Centered moving-average of bbox centers (OLD smooth_track); bbox/image/frame_idx carried through."""
    ...

# ---- stage 2: track -> fixed-length windows (pure) ------------------------------------------
def assemble_windows(tracks: dict[int, Track], cfg: DataCfg, icfg: InferenceCfg) -> list[TrackWindow]:
    """Sliding window (len=cfg.seq_len, step=icfg.window_stride) over each smoothed track >= seq_len.

    Carries (track_id, frame_idx) on every window so aggregation needs no bbox re-matching (OLD smell).
    """
    ...

# ---- stage 3: preprocess one window via Phase-1 math (no PIESequenceDataset) ----------------
def preprocess_window(
    win: TrackWindow, cfg: DataCfg,
    transform_tight, transform_context,
) -> tuple[Tensor, Tensor, Tensor]:
    """One window -> (images_tight[T,3,h,w], images_context[T,3,H,W], motions[T,8]).

    Wraps each RGB crop as PIL, runs crop_tight/crop_context against the full frame is NOT possible here
    (we only kept the ped crop), so: the crop IS the tight image; context is rebuilt from the bbox on the
    full frame at detection time (see §2.3). Motion is compute_motion(int-boxes) — the SAME 8-dim function
    as the data pipeline, so the [..., :8] slice never reappears (B7).
    """
    ...

# ---- stage 4: batched forward -------------------------------------------------------------
def predict_windows(
    model: nn.Module, windows: list[TrackWindow], cfg: RootCfg, device: torch.device, *, use_amp: bool,
) -> list[dict[str, int]]:
    """Batch windows -> forward_model -> argmax per task. crosses from `crosses_frame` only (B4)."""
    ...

# ---- stage 5: aggregate + overlay ---------------------------------------------------------
def aggregate_by_frame(
    windows: list[TrackWindow], preds: list[dict[str, int]],
) -> dict[int, list[FramePrediction]]:
    """Scatter each window's prediction to its frames (no bbox-equality matching; uses carried indices)."""
    ...

def render_overlays(source: FrameSource, frame_results, icfg: InferenceCfg, out_path: Path) -> Path:
    """cv2 overlay (bbox + ID + action/look/cross labels + color chips) -> mp4 (OLD 230-325)."""
    ...

# ---- top-level orchestrator ----------------------------------------------------------------
def run_video_inference(
    cfg: RootCfg, *, video: str | Path, checkpoint: str | Path,
    out_video: str | Path | None = None, device: torch.device | None = None, strict: bool = True,
) -> InferenceResult:
    """open -> detect_tracks -> assemble_windows -> build_model+load_eval_weights -> predict -> aggregate -> render."""
    ...
```

### 1b. `scripts/infer_video.py` (thin CLI, mirrors `scripts/evaluate.py`)

```python
"""Video inference entry point (Prompt 5.3).

    python scripts/infer_video.py --video test_clip.mp4 --checkpoint outputs/runs/<id>/checkpoints/best.pth
    python scripts/infer_video.py --video qualitative_visualize/ --checkpoint <path> --out out.mp4   # frame dir
"""
from __future__ import annotations
import multiprocessing as mp, sys
from pedpredict.config import build_argparser, load_config
from pedpredict.eval.inference import run_video_inference
from pedpredict.utils.device import get_device

def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--video", required=True)          # file OR directory of frames
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)
    res = run_video_inference(cfg, video=args.video, checkpoint=args.checkpoint,
                              out_video=args.out, device=get_device())
    print(f"Inference complete: {res.n_windows} windows over {res.n_frames} frames -> {res.out_video}")
    return 0

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
```

### 1c. New config: `InferenceCfg` (added to `schema.py` + `RootCfg`, defaults from OLD `main.py`)

```python
@dataclass(frozen=True, slots=True)
class InferenceCfg:
    """Video-inference knobs (Prompt 5.3) — replaces main.py top-of-file literals (B1)."""
    detector_weights: str = "yolo11n.pt"   # OLD main.py 'yolo11n.pt'
    detector_class_idx: int = 0            # pedestrian class (OLD class_idx=0)
    detector_conf: float = 0.3             # OLD conf=0.3
    smooth_window: int = 3                 # OLD smooth_track(window=3)
    window_stride: int = 1                 # OLD used stride 1 (every window); data pipeline uses 3
    batch_size: int = 32                   # OLD inference batch_size=32
    default_fps: float = 30.0              # DirFrameSource fps when frames have no container fps
    draw_color_chips: bool = True
```
> `seq_len`, `context_scale`, `img_height/width`, `read_context_height/width`, `motion_dim`, `norm_*` are
> **reused from `DataCfg`** — inference must use the SAME geometry the model was trained on (see §2.3 B-flag
> on `context_scale`). Add an `infer:` block to `configs/eval.yaml` (or a new `configs/infer.yaml`).

---

## 2. Step-by-step port procedure (referencing OLD `main.py`)

1. **Frame source (folds `extract_frames.py`).** `open_frame_source` returns a `VideoFrameSource`
   (`cv2.VideoCapture`, reading `fps/width/height`) for a file, or a `DirFrameSource` (sorted image dir,
   `default_fps`) for a directory. Both yield **RGB** `np.ndarray` (cv2 reads BGR → convert once here).
   This lets the manual smoke run on `qualitative_visualize/` frames (§5) and removes the need for a
   separate extract step.
2. **Detect + track** (OLD `pedestrian_detection.py` 17–67). Port `extract_tracks_from_video` body into
   `detect_tracks(source, cfg)`: `YOLO(cfg.detector_weights)`, iterate frames, `model.track(persist=True,
   classes=[cfg.detector_class_idx], conf=cfg.detector_conf)`, clamp boxes, skip degenerate boxes, store
   `Detection(frame_idx, bbox, rgb_crop)`. **Lazy-import** `ultralytics` inside the function with a clear
   `[infer]`-extra install hint on `ImportError`. Port `smooth_track` verbatim (centered MA, `window` from
   `InferenceCfg`).
3. **Assemble windows** (OLD `main.py` 92–117). For each track with `len >= seq_len`, slide a `seq_len`
   window with `step=window_stride`; build `TrackWindow(track_id, detections)`. Drop the placeholder label
   dict (`actions/looks/crosses = 0`) — labels are predicted, not stored.
4. **Preprocess (Phase-1 reuse, replaces `PIESequenceDataset`/`inference_collate_fn`).** The detection
   already holds the **tight ped crop** (= `crop_tight` output) and, separately, must hold the **context
   crop**. Decision (§2.3): compute BOTH crops from the *full frame + bbox* at detection time via
   `crop_tight`/`crop_context(box, cfg.context_scale)`, storing two arrays per `Detection`, OR keep only the
   full frame ref and crop in `preprocess_window`. **Chosen:** store full-frame-derived tight+context PIL
   crops in `Detection` (keeps `preprocess_window` pure tensor work). Apply `build_read_transforms(cfg.data)`
   → `images_tight[T,3,128,128]`, `images_context[T,3,224,224]`. Motion: `compute_motion([d.bbox …])` —
   the identical 8-dim function the writer uses (B7: no `[...,:8]` slice).
5. **Build + load + forward** (OLD 53–74, 164–191). `build_model(cfg, cfg.eval.model_type)` →
   `load_eval_weights(model, checkpoint, device=…, strict=…)` (5.1; strict load, no `relative_position_bias`
   fix-up — B2). Batch windows (`InferenceCfg.batch_size`), `with torch.inference_mode(), autocast_ctx(amp):
   outputs = forward_model(model, t, c, m)`. Per task: `argmax(dim=1)`; `crosses` from `outputs["crosses_frame"]`
   only (B4) — delete OLD's key-name `if/elif` branching over `outputs`.
6. **Aggregate** (OLD 162–228). Replace the fragile "match window by `bboxes == bboxes`" loop with a direct
   scatter: each `TrackWindow` already carries `frame_idxs` + `bboxes`, so `aggregate_by_frame` appends a
   `FramePrediction` to every `frame_idx` the window covers. (When a frame is covered by multiple windows of
   the same track, last-window-wins, matching OLD's overwrite-by-iteration behavior — confirm in §6.)
7. **Overlay** (OLD 230–325). Port `render_overlays`: reopen the source, draw green bbox, `ID {track_id}`,
   the three labels (action/look/cross) with the OLD color scheme, and the on/off color chips. Replace
   `pie._map_scalar_to_text` with an inline `{0:"not", 1:"yes"}`-style map (no PIE toolkit at inference).
   Write mp4 via `cv2.VideoWriter` at `source.fps`.
8. **Orchestrate.** `run_video_inference` wires 1→7 and returns `InferenceResult`.

### 2.3 The `context_scale` behavior change (FLAG in MIGRATION.md)

OLD `main.py` built the dataset with **`context_scale=2.0`** (line 131) while the *training/data* pipeline
uses **`3.0`** (`DataCfg.context_scale`, CLAUDE.md). The model was trained on 3.0 context crops, so OLD
inference fed an **out-of-distribution** context. The rebuild uses `cfg.data.context_scale` (3.0) — a
deliberate **correctness fix**, NOT pure parity. Because full video inference has no golden oracle
(non-deterministic YOLO + needs a trained checkpoint + a video), this divergence cannot be caught by a
parity test; it is recorded explicitly and justified (matches the trained distribution).

---

## 3. Reuse map to Phase-1 / earlier prompts

| Need | Reused symbol | Prompt |
|---|---|---|
| tight crop geometry | `data.transforms.crop_tight` | 1.2 |
| context crop geometry | `data.transforms.crop_context(box, scale)` | 1.2 |
| 8-dim motion feature | `data.transforms.compute_motion` (identical fn → B7 slice never returns) | 1.2 |
| read-time resize+ImageNet-normalize | `data.transforms.build_read_transforms(cfg.data)` | 1.2/1.5 |
| model build / forward | `models.registry.build_model` / `forward_model` | 2.4 |
| strict weight load | `eval.evaluate.load_eval_weights` | 5.1 |
| AMP context / device | `utils.amp.autocast_ctx`,`resolve_amp`; `utils.device.get_device` | 0.x |
| config + paths | `config.load_config`, `paths.resolve_paths` | 0.2/0.3 |

**Not reused (deliberately):** `PIESequenceDataset` (the OLD in-memory dataset path) and
`inference_collate_fn` — superseded by the function-level Phase-1 math + a trivial `torch.stack`.

---

## 4. Band-aids removed (and how)

| Band-aid | How 5.3 removes it |
|---|---|
| **B5** (fragmented data/inference scripts) | `main.py` + `extract_frames.py` + `pedestrian_detection.py` collapse into one staged `eval/inference.py` + a thin `scripts/infer_video.py`; the dead 4-dim `extract_sequences_from_track` is dropped. |
| **B7** (`motions[...,:8]` slice, magic seq len) | Motion via the canonical `compute_motion` (8-dim); window length = `cfg.data.seq_len`; no slice, no `MAX_SEQ_LEN` literal. |
| **B10** (stringly dispatch) | `get_model(str)` + key-name `if/elif` over outputs → `build_model`/`forward_model` (intrinsic `ModelType`) + `crosses_frame`-only read. |
| **B2** (lazy ViT param fix-up at load) | `load_eval_weights` strict-loads rebuilt ckpts; no `relative_position_bias` reconstruction. |
| **B4** (dead crosses head) | crosses read from `crosses_frame`; `crosses_pooled`/`temporal_weights` untouched at inference. |
| **B11** (root one-offs / artifact sprawl) | `extract_frames.py` folded into `DirFrameSource`; PIE-text mapping inlined (no PIE runtime dep); output path is caller-specified, not a hardcoded `output_with_predictions.mp4`. |

**Behavior changes flagged (not neutral):** (a) `context_scale` 2.0→3.0 (§2.3, correctness fix);
(b) **BGR→RGB** conversion at frame read — OLD applied `ToTensor`+`Normalize` to cv2 **BGR** crops (a latent
channel-swap bug vs the RGB-trained model); the rebuild converts to RGB once in the `FrameSource`. Both are
documented in MIGRATION.md as intentional fixes with no golden oracle to contradict.

---

## 5. Golden fixtures & test list (behavior preservation)

Full video inference is **not golden-able** end-to-end (YOLO is external + non-deterministic across
versions; needs a trained checkpoint + a real video). The preservation strategy is therefore **seam-level**:
the preprocessing math is already golden (`test_transforms.py`), the forward is already golden
(`ensemble.pt`), so 5.3 only needs to prove the **new glue** (windowing, in-memory preprocessing equals the
file-path path, aggregation) and that detection is correctly isolated/mocked.

New file `tests/test_inference.py`; capture script `tests/_capture/capture_inference_golden.py` →
`tests/fixtures/golden/inference_cases.pt` (synthetic deterministic track; no YOLO, no video).

| # | Test | Asserts |
|---|---|---|
| 1 | `test_smooth_track_matches_legacy` | `smooth_track` == OLD `smooth_track` on a synthetic track (exact). |
| 2 | `test_assemble_windows_matches_legacy` | window count + per-window frame_idxs/bboxes == OLD `main.py` 92–117 sliding loop (`stride=1`). |
| 3 | `test_preprocess_equals_process_record` | `preprocess_window` tensors == `transforms.process_record` for the SAME crops/boxes written to disk (proves the in-memory path matches the golden file-path path within float tol). |
| 4 | `test_motion_is_canonical_compute_motion` | `preprocess_window` motion == `compute_motion(boxes)` (B7 — same fn, no slice). |
| 5 | `test_predict_windows_reads_crosses_frame` | crosses prediction comes from `crosses_frame`; perturbing `crosses_pooled`/`temporal_weights` does not change preds (B4). |
| 6 | `test_aggregate_by_frame_scatter` | each frame gets one `FramePrediction` per covering window; indices match carried `frame_idxs` (no bbox-equality matching). |
| 7 | `test_detect_tracks_lazy_import` | importing `inference` without `ultralytics` succeeds; `detect_tracks` raises a clear `[infer]`-extra message (monkeypatch the import). |
| 8 | `test_run_inference_with_stub_detector` | `run_video_inference` with a **stub** detector (fixed tracks) + random-init model over `DirFrameSource` produces an `InferenceResult` with non-empty `frame_results`, no crash. |
| 9 | **SMOKE** `test_render_overlays_writes_video` | `render_overlays` on a few `tmp_path` frames writes a readable mp4 (`cv2.VideoCapture` reopens, frame count > 0). |
| 10 | `test_context_scale_uses_data_cfg` | `preprocess_window` uses `cfg.data.context_scale` (3.0), not a hardcoded 2.0 (guards the §2.3 fix). |

**Golden capture** (tests 1–4): build a deterministic synthetic track (fixed bboxes + tiny random RGB
crops, `seed=0`); run OLD `smooth_track` + the OLD windowing loop; for the preprocess parity, write the
crops to a temp dir and run `process_record` as the oracle. Tests 5–10 use stubs/`tmp_path` (no fixture).

---

## 6. Manual smoke-test procedure (on existing `qualitative_visualize/` frames)

The prompt requires a manual smoke on the already-extracted frames. `qualitative_visualize/` is gitignored
but exists locally as a directory of frame images — perfect for `DirFrameSource` (no video, no YOLO video
decode needed for the render path).

1. **Frame-source sanity (no model, no YOLO):**
   `python -c "from pedpredict.eval.inference import open_frame_source as o; s=o('qualitative_visualize'); print(len(s), s.fps, s.width, s.height, next(iter(s)).shape)"`
   → expect a frame count, a default fps (30), frame size, and an RGB `(H,W,3)` array.
2. **End-to-end with a stub detector (no `[infer]` extra needed):** in a REPL, monkeypatch `detect_tracks`
   to return two synthetic tracks whose bboxes walk across the frames, build a random-init model
   (`build_model(RootCfg())`), save its `state_dict` to a temp `.pth`, then call
   `run_video_inference(cfg, video="qualitative_visualize", checkpoint=<tmp.pth>, out_video="smoke_out.mp4")`.
   → expect `smoke_out.mp4` written, `n_frames == len(source)`, `n_windows > 0`, overlays drawn.
3. **Full pipeline (requires `pip install -e .[infer]` + a YOLO weight + a real checkpoint):**
   `python scripts/infer_video.py --video test_clip.mp4 --checkpoint outputs/runs/<id>/checkpoints/best.pth --out out.mp4`
   → expect a console summary and a playable annotated mp4. (Numbers depend on the checkpoint; this is a
   does-it-run smoke, not a parity check.)

---

## 7. Risks & open questions (confirm before coding)

1. **Detection in-scope vs external.** **Proposed:** `pedestrian_detection.py` is **ported** (its
   `extract_tracks`/`smooth_track` become package functions) but `ultralytics`/`lap` stay an **optional
   `[infer]` extra** (already in `pyproject.toml`) with a **lazy import** — so the package imports and
   tests run without YOLO, and `detect_tracks` errors clearly only when actually invoked. Confirm we don't
   instead want a pluggable detector interface (any external detector returning `dict[int, Track]`).
2. **`context_scale` fix (§2.3).** Confirm using `cfg.data.context_scale` (3.0, matches training) over
   OLD's 2.0 is desired — it changes outputs vs OLD but matches the trained distribution.
3. **BGR→RGB fix.** Confirm converting cv2 frames to RGB (OLD fed BGR to an RGB-trained model). Also changes
   outputs vs OLD; almost certainly correct, but it is a flagged behavior change.
4. **Window stride at inference.** OLD used stride 1 (a prediction per possible window → dense, slow). The
   data pipeline uses stride 3. **Proposed default** `InferenceCfg.window_stride=1` (parity with OLD's dense
   coverage); confirm or set to 3 for speed.
5. **Multi-window frame conflict.** When several windows of one track cover a frame, OLD effectively
   last-write-wins via dict append+overwrite ordering. **Proposed:** keep last-window-wins (most recent
   observation context). Confirm vs averaging logits across overlapping windows.
6. **No golden oracle for E2E.** Confirm the seam-level test strategy (§5) is acceptable — i.e. parity is
   proven at the reused-math seams, and the two flagged fixes (context_scale, BGR) are accepted as
   documented divergences rather than parity failures.
7. **Where context crop comes from.** OLD's in-memory `PIESequenceDataset` cropped context from the **ped
   crop**, not the full frame, when given pre-cropped images — meaning OLD's "context" was a scaled-up
   *crop of the crop* (degenerate). **Proposed:** crop context from the **full frame** at detection time
   (correct, matches the data pipeline). Confirm — this is another latent OLD divergence the rebuild fixes.

---

## 8. Coupling notes (keep the contract singular)

- Honors the **output-contract** siblings (2.3/2.4/2.5/3.2/5.1): crosses read from `crosses_frame`;
  `crosses_pooled`/`temporal_weights` untouched.
- Reuses **5.1** `load_eval_weights` verbatim (no second loader); reuses **2.4** registry (no stringly path).
- Reuses **1.2** preprocessing math verbatim (the `compute_motion` 8-dim + the `context_scale` channel
  definition stay singular across writer / flip / motion / inference — the motion-channel coupling set).
- The **6.x** viz seam: `temporal_weights` overlay is the *viz* prompt's job (6.2), not inference; 5.3
  emits only the action/look/cross overlay (kept parity with OLD `main.py`).
```

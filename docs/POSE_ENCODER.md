# Pose-Keypoint Motion Arm — design + implementation plan

**Prepared:** July 2026 · **Status:** steps 0–4 implemented on `pose-encoder-arm` (💻 code complete,
gate green); steps 5–8 (🖥️ extraction pass, experiments, verdict) pending. **Companions:**
[THESIS_ROADMAP.md](THESIS_ROADMAP.md) (the data pass this piggybacks on — read Stage 3),
[BACKBONE_STUDY.md](BACKBONE_STUDY.md) (RQ1 — must be locked before the `pose_full` comparison),
[project-context-streaming-crossing-onset.md](project-context-streaming-crossing-onset.md) (the thesis spine).

> **What this is.** A concrete, code-grounded plan to add a **pose-keypoint motion arm**: replace the
> tight-crop image in the motion branch with 2D pose-keypoint features, keeping the ViT context stream and
> the 9-dim bbox motion. It records the locked decisions (extractor, joint set, feature math,
> normalization), the plumbing into *this* repo's contracts, the experiment, and the branch→merge workflow.
> It is a **supporting arm, not the thesis spine** (§2).

---

## 1. The idea

The full model's motion branch (`MotionEncoder`, [motion_encoder.py](../src/pedpredict/models/motion_encoder.py))
fuses two per-frame signals: a CNN over the **tight pedestrian crop** and a Conv1d over the **9-dim bbox
motion**. The tight crop is a weak, noisy ensemble member — a low-res, background-polluted patch of a
pedestrian who is often 40–100 px tall on dashcam. The hypothesis: **2D pose is a cleaner geometric
substitute** — it hands the model articulation (gait, limb configuration) and body/head **facing angle**
directly, instead of asking a small CNN to re-derive them from pixels.

Division of labor after the swap:

| Stream | Carries | Status |
|---|---|---|
| ViT context crop | scene / environment | unchanged (RQ1 backbone) |
| 9-dim bbox motion | coarse trajectory, position, scale, approach, ego | unchanged (stored) |
| **pose keypoints** | **articulation + body/head facing angle** | **new (this doc)** |

The tight crop is **dropped** from the pose arm — testing "the crop is noise" is the point, not a thing to
keep alongside pose.

## 2. Scope, novelty, and ordering

**This is a supporting arm.** Pose+bbox for PIE crossing is well-studied and strong (~0.9 F1 reported) —
but **only under the anchored protocol**, so those numbers are not comparable to our streaming F1 (~0.08).
The novel angle, and the only reason this ties to the thesis, is to **evaluate the pose arm under the
streaming protocol and inside the anchored→streaming gap decomposition** (see
[THESIS_ROADMAP.md](THESIS_ROADMAP.md)). Encoder sophistication is *not* the contribution — see
the encoder audit conclusion in §5 (flatten into the existing temporal stack, not an ST-GCN).

**Ordering constraints:**
- The pose **extraction pass** piggybacks on **streaming-onset Phase A** (the one final data pass). Do not
  schedule a separate lab-PC data pass for it.
- `pose_kinematics` (pixel-free) has **no ViT dependency** → can proceed in parallel immediately.
- `pose_full` uses the ViT context stream → its **final** comparison waits on the **RQ1 backbone lock**
  ([BACKBONE_STUDY.md](BACKBONE_STUDY.md)), so the visual stream isn't a moving confound.

## 3. The keypoint specification (locked)

### 3.1 Extractor

Whole-body **top-down** pose with the **PIE GT bounding boxes** as the person prior (no detector stage —
we already have boxes), run on the **full frame** (never the stored crop, which is too small/normalized).

- **Primary: DWPose / RTMW whole-body (COCO-WholeBody 133).** ONNX-friendly, robust confidences, strong on
  low-res bodies, easy to run bbox-conditioned. Gives body-17 + **feet-6** (the signal we want).
- **Alternative: AlphaPose-Halpe (136).** Only if the **head-top / neck / mid-hip** points earn their keep;
  Halpe adds those three over WholeBody-133. We derive head yaw from the face-5 either way (§3.3), so
  head-top is a nice-to-have, not load-bearing → default to DWPose.

Store **per-joint confidence**; **temporally smooth / interpolate** missing joints before storage.

### 3.2 Joint set — what we keep, and how we feed each

The extractor emits 133; we do **not** feed all of them. On dashcam peds the **68 face + 42 hand** points
are near-hallucinated noise ("no way the human eye discerns the face, much less the model") — they are
**dropped, not down-weighted** (removing input *dimensionality* is what helps in the low-data regime;
down-weighting can't). The kept set:

| Joints | n | fed as | why |
|---|---|---|---|
| nose | 1 | raw coord | head position anchor |
| eyes ×2, ears ×2 | 4 | **collapsed → head-facing angle; coords dropped** | 4 jittery px-close points; their only signal is head yaw |
| shoulders ×2 | 2 | raw coord | torso facing, upper body |
| hips ×2 | 2 | raw coord | torso facing |
| knees ×2 | 2 | raw coord | gait |
| ankles ×2 | 2 | raw coord | gait |
| feet ×6 (big toe, small toe, heel ×2) | 6 | raw coord | foot heading + weight shift (3/foot = a foot-direction vector) |
| elbows ×2, wrists ×2 | 4 | raw coord — **optional, first to ablate** | weak arm-swing cue; noisy |

**Kept coordinate joints: 15 (core)** — nose, 2 shoulders, 2 hips, 2 knees, 2 ankles, 6 feet — or **19**
with arms. The 4 eye/ear points become **one head-facing angle**, not 8 coordinates.

### 3.3 Per-frame pose feature vector (itemized)

**Core (recommended), 49 dims:**

| block | dims | cum |
|---|---|---|
| 15 joint coords (x, y), bbox-normalized (§3.4) | 30 | 30 |
| 15 per-joint confidences | 15 | 45 |
| head-facing angle (sin, cos) — from ear→nose geometry | 2 | 47 |
| body-facing angle (sin, cos) — from shoulder line / hip line | 2 | **49** |

**With arms** (elbows+wrists): +8 coords +4 conf → **61**.

**Deliberately excluded:** per-joint *velocity*. The Conv1d-over-time + GRU in the encoder already extract
dynamics — feeding explicit deltas would double the block for signal the temporal stack recovers (same
dedup logic as §5's position argument). **Foot-heading angle** is also excluded: it is redundant with the
raw feet coordinates we already feed.

Two judgment calls baked in (flag if reversing): (a) head collapsed to nose + yaw angle — the net can no
longer learn head pose from raw points, only the angle we hand it; (b) body-facing angle is mildly
redundant with the shoulder/hip coords but kept as a cheap, low-noise prior (head-facing is **not**
redundant — its raw joints were dropped).

### 3.4 Normalization (grounded in stored data)

The intrinsic/extrinsic split is what guarantees **no double-encoding** with the 9-dim motion: normalization
surgically removes exactly the position + scale that the motion vector carries.

**Key realization — the normalizing box is already stored.** The 9-dim motion holds `cx, cy` (idx 0,1) and
`w, h` (idx 4,5) per frame ([transforms.compute_motion](../src/pedpredict/data/transforms.py)). So the pose
feature build needs **no extra stored geometry** — it reads the bbox from the motion channels:

- **Origin (translation):** subtract the bbox center `(cx, cy)` = `motion[:, 0:2]` from every joint. Removes
  absolute position → position lives **only** in the motion vector. (Bbox-center from the GT box is more
  robust than pelvis-centering on distant peds, where hips are low-confidence; note pelvis-centering as an
  alternative if box tightness varies.)
- **Scale:** divide by bbox height `h` = `motion[:, 5]`. Removes scale → scale lives **only** in the motion
  vector (`h`). Coords land ≈ `[-1, 1]`.
- **Angles** are inherently translation/scale-invariant; encode `(sin, cos)` to avoid the 0/2π wrap.
  - body-facing: perpendicular of the shoulder vector `(R_sh − L_sh)`, cross-checked against the hip vector.
  - head-facing: from `ear_R − ear_L` and the nose offset (yaw proxy).
- **Confidence gating:** feed raw per-joint score `[0,1]`; do **not** pre-zero low-conf joints — let the net
  gate. Each derived angle gets its own confidence = product of its contributing joints' scores (an angle
  from two dead joints reads as untrusted).

## 4. The merged input — one vector into the motion branch

Instead of pose and bbox as two overlapping inputs, they merge into **one geometric descriptor** (the
"fusion dance"): intrinsic pose ⊕ extrinsic motion, with zero overlap by construction (§3.4).

```
pose_arm_input[T, 58] = [ motion_9 (image-normalized) | pose_49 (bbox-normalized, §3.3) ]
                          ^ idx 0:9, unchanged first-8 semantics    ^ idx 9:58
```

**motion_dim arithmetic:** core = **9 + 49 = 58** (keep ego — it is a studied streaming signal, RQ4). No-ego
= 8 + 49 = 57; with arms = 9 + 61 = 70. Pin the **core at `motion_dim = 58`**.

This makes the pose arm reuse the existing pixel-free machinery: it **is**
`KinematicsEncoder`/`KinematicsOnlyModel` ([ablations.py](../src/pedpredict/models/ablations.py)) with a
wider input. The B7 contract already says the Conv1d "uses the channel *count*, never the per-channel
semantics" — widening `motion_dim` is the sanctioned extension path.

## 5. Encoder choice (audit conclusion — flatten, not ST-GCN)

The pose features flatten into the **existing GRU + MultiheadAttention temporal stack**, not a bespoke
ST-GCN. Rationale (full audit in session notes): (a) the masked/partial, per-joint-confidence skeleton
**violates ST-GCN's fixed-complete-graph premise**; (b) ST-GCN's temporal side (fixed-kernel 1D conv) is
**weaker** than the GRU+MHA we already have and would fight the frame-level `crosses_frame` contract; (c) a
supporting arm shouldn't spend its budget on encoder novelty. Spend sophistication on the **features and
normalization** (§3), not the graph.

**Concrete model surface:**
- **`PoseMotionEncoder`** — `KinematicsEncoder` with one change: a **`motion_norm="none"`** (pass-through)
  mode, because all normalization now happens at read time (§7), and its `motion_scale` buffer must be
  guarded (the current `scale_full[:motion_dim]` raises for `motion_dim=58 > 9`). The golden `image` /
  `per_sequence` modes are untouched.
- **`pose_kinematics`** — standalone, pixel-free: `KinematicsOnlyModel` fed the 58-dim input,
  `motion_norm="none"`. Signature `("motions",)` — reuses the existing `KINEMATICS_ONLY` forward path.
- **`pose_full`** — the pose features as the **cross-attention query** over the ViT context: an
  `EnsembleModel` variant whose motion branch is `PoseMotionEncoder` (no tight crop) instead of
  `MotionEncoder`. Signature `("images_context", "motions")` — a new registry entry (additive, like
  `kinematics_only` was).
- **Control:** the tight-crop **`full`** model, unchanged — the frozen baseline that "the crop is noise" is
  tested against.

## 6. The experiment

The ablation ladder falls out for free (each rung isolates one contribution), run under **both protocols**
to tie into the anchored→streaming gap:

| model_type | input | isolates |
|---|---|---|
| `kinematics_only` | 9-dim motion | position/kinematics floor (exists) |
| **`pose_kinematics`** | + pose articulation (pixel-free) | **what articulation adds over kinematics** |
| `full` (control) | tight crop + 9-dim + ViT | the current model |
| **`pose_full`** | pose + 9-dim + ViT | **pose vs crop as the motion query** |

**Primary questions:**
1. `pose_kinematics − kinematics_only` > 0? → articulation carries crossing signal beyond trajectory.
2. `pose_full ≥ full`? → **the tight crop is noise** (the headline of this arm). If pose can't beat the
   crop, that is the answer — no encoder change would rescue it.
3. Does the pose arm **narrow the anchored→streaming gap** more than the crop arm? → the thesis tie-in.

**Metrics** (per [training/metrics.py](../src/pedpredict/training/metrics.py)): Accuracy, F1, AUC,
Precision, Recall per task + macro-F1; efficiency (params/FLOPs/latency/FPS/VRAM). **Thresholds tuned on
val, applied on test** (`tuned_*` only; M2). **Seeds:** screen at 1 seed, confirm finalists at 3, report
mean±std. Report **streaming and anchored** columns (`data.protocol` switch).

## 7. Data-pipeline integration

Route decision: **pose rides the `motions` tensor** (Route A), **not** a new 4th collate/`forward_model`
tensor (Route B). Route A keeps the golden-pinned shared contracts — the 4-tuple `collate_sequences` and
`forward_model` signatures used by *every* model — **frozen**, and makes the pose arm additive/reversible. Route
B is documented-but-rejected for v1 (it churns parity surfaces for cosmetic separation). Store **raw
keypoints**, build features **at read time**, so the normalization/angle math (§3) stays iterable without
re-running the expensive extraction.

Flow (💻 personal PC = code · 🖥️ lab PC = data/GPU):

1. **🖥️ Extraction pass — `scripts/extract_pose.py` (new).** Iterate unique PIE `(set, video, frame)`; run
   the bbox-conditioned whole-body model on the **full frame** with GT boxes for **every** ped; temporally
   smooth/interpolate; write a cache `pose_cache/{set}/{video}.npz` keyed by `(frame, pid) → [23, 3]`
   (x, y, conf, absolute frame px). Decoupled from windowing so overlapping windows don't re-extract the
   same frame ~`seq_len/stride`× . **Piggybacks streaming-onset Phase A.**
2. **💻 `transforms.process_record`** — attach `pose[T, 23, 3]` (absolute px) to `ProcessedSample` from the
   cache by `(track_id, frame)`, gated by `pose.enabled`. New field on the dataclass
   ([transforms.py](../src/pedpredict/data/transforms.py)), mirroring `track_id`/`tte`.
3. **💻 `lmdb_writer.pack_meta`** — store `pose` in the meta pickle, gated (mirrors the `tte` line). LMDB
   schema note: an **additive** meta key; existing consumers ignore it.
4. **💻 `lmdb_dataset.__getitem__`** — if `pose.enabled`, read raw `pose`, call
   **`build_pose_features(pose, motions)`** → `[T, 49]`, **image-normalize the 9-dim motion block** (same
   scale `KinematicsEncoder` applies today), concat → `motions_out[T, 58]`, return as `"motions"`. Mirrors
   the existing read-time ImageNet-norm + motion-slice already living in `__getitem__`.
5. **💻 `data/pose.py` (new)** — single source of truth for pose math: the joint-index constants, the kept
   set, `build_pose_features`, the normalization + angle formulas. Analogous to
   `transforms.compute_motion` owning the motion contract.
6. Collate / `forward_model` / registry dispatch: **unchanged for `pose_kinematics`**; `pose_full` adds one
   enum member + one `MODEL_INPUT_SIGNATURE` + one `forward_model` branch.

`build_pose_features` sketch (pins the contract):

```python
# data/pose.py — read-time, cheap, iterable (no re-extraction to change normalization)
KEPT_JOINTS: tuple[int, ...] = (...)   # 15 core indices into the 23-joint layout
def build_pose_features(pose: Tensor, motions: Tensor) -> Tensor:  # [T,23,3],[T,9] -> [T,49]
    cx, cy, h = motions[:, 0], motions[:, 1], motions[:, 5].clamp_min(1.0)
    xy   = pose[:, KEPT_JOINTS, :2]                       # [T,15,2] absolute px
    conf = pose[:, KEPT_JOINTS,  2]                       # [T,15]
    xy_n = (xy - torch.stack([cx, cy], -1)[:, None]) / h[:, None, None]   # bbox-normalized
    body = facing_body(pose); head = facing_head(pose)   # (sin,cos) each, conf-gated
    return torch.cat([xy_n.flatten(1), conf, body, head], dim=-1)         # [T,49]
```

## 8. Config

New top-level **`PoseCfg`** (mirrors `AugmentCfg`/`BalanceCfg`; add to `RootCfg` + `config/loader.py` +
`configs/pose.yaml`):

| field | default | role |
|---|---|---|
| `enabled` | `False` | master gate — read-time build + writer storage |
| `extractor` | `"dwpose"` | `"dwpose"` \| `"alphapose_halpe"` |
| `include_arms` | `False` | 15 vs 19 kept joints (58 vs 70 `motion_dim`) |
| `conf_channel` | `True` | feed per-joint confidence |
| `smooth_window` | `5` | temporal smoothing / interpolation window (extraction) |
| `min_conf` | `0.3` | below → treat joint as missing (interpolate) |
| `cache_dir` | `"pose_cache"` | extraction output root |

`validate_config` cross-check: when `pose.enabled`, assert `model.motion_dim == data.motion_dim ==
9 + pose_feature_dim(pose)` (with a `pose_feature_dim()` helper), and force `model.motion_norm == "none"`
for `pose_*` model types. **Doc-sync** (CLAUDE.md checklist): config schema → `configs/*.yaml` + schema
docstrings; new `src/` module + `scripts/extract_pose.py` → README layout + command list; new model types +
`motion_dim` → Architecture table (CLAUDE.md + README) + registry; the v2 labeling/data-contract note gains
a pose line.

## 9. Tests (offline, no data — gates on 💻)

- `tests/test_pose.py` — `build_pose_features` on **synthetic** keypoints: shape `[T,49]`; translation-
  invariance (shift all joints → identical features); scale-invariance (scale about bbox center → identical);
  a known skeleton → expected facing angles; low-conf joint → flagged, not NaN; missing-joint interpolation.
- `tests/test_pose_model.py` — `PoseMotionEncoder` `[B,T,58]→[B,T,128]`; `motion_norm="none"` buffer guard;
  `pose_kinematics` / `pose_full` build via registry + emit the 4 supervised keys (`actions`, `looks`,
  `crosses_pooled`, `crosses_frame`); `pose_full` also emits `temporal_weights`.
- **Golden safety:** the change is **purely additive** — `full`/`ped_local`/`kinematics_only`/`visual_only`/
  `vanilla_concat` and the stored 9-dim `motions` contract are untouched, so the existing goldens must stay
  green with **zero** re-pinning. The new modules carry no golden (no legacy equivalent, like
  `KinematicsOnlyModel`).
- **Gate:** `ruff check .` and `pytest -m "not slow"` green.

## 10. Implementation plan — branch → experiment → merge

Work on a branch off `main`; everything below lands there. Legend: 💻 personal PC · 🖥️ lab PC.

```
git checkout -b pose-encoder-arm
```

**Step 0 — 💻 scaffold + config.** `PoseCfg` + `configs/pose.yaml` + `RootCfg`/loader wiring +
`validate_config` width check. Gate green (no behavior yet). Commit.

**Step 1 — 💻 pose math (`data/pose.py`) + tests.** Joint constants, `build_pose_features`, normalization +
angle formulas, `pose_feature_dim()`. `tests/test_pose.py` (synthetic). This is the load-bearing correctness
step — invariances must pass. Commit.

**Step 2 — 💻 model surface.** `PoseMotionEncoder` (add `motion_norm="none"` + buffer guard),
`pose_kinematics` + `pose_full` in `ablations.py`/`ensemble.py`, register in
[registry.py](../src/pedpredict/models/registry.py) (enum, `MODEL_INPUT_SIGNATURE`, builder,
`forward_model` branch for `pose_full`). `tests/test_pose_model.py`. Confirm existing goldens still green.
Commit.

**Step 3 — 💻 data plumbing.** `pose` field on `ProcessedSample`; store in `pack_meta`; read + build + concat
in `LMDBChunkDataset.__getitem__`; all gated by `pose.enabled` so a no-pose run is byte-identical. Extend
`read_raw_sample` for parity. Commit.

**Step 4 — 💻 extraction script.** `scripts/extract_pose.py` (bbox-conditioned whole-body → cache), plus a
`--dry-run` that fabricates random keypoints so the **whole pipeline is exercisable end-to-end on the
personal PC without frames or the extractor installed**. Commit. **Gate the branch:** `ruff` +
`pytest -m "not slow"` green; README/CLAUDE doc-sync done.

**Step 5 — 🖥️ extraction pass (piggyback streaming-onset Phase A).** Run `extract_pose.py` over PIE →
`pose_cache/`; regenerate the standard LMDBs **with pose** in the same Phase-A pass (streaming + anchored);
re-pin stats / `count_labels` gate if window population shifts (it must **not** — pose is additive; if it
does, that's a bug). **Notify: this step needs the data + GPU.**

**Step 6 — 🖥️ experiments (§6).** Screen at 1 seed under **streaming**: `kinematics_only`,
`pose_kinematics`, `full`, `pose_full`. If `pose_kinematics > kinematics_only` and/or `pose_full ≥ full`,
confirm finalists at 3 seeds and add the **anchored** column. Log to `outputs/runs/index.csv`; the
comparison table is the deliverable that closes this doc.

**Step 7 — compare + write up.** Fill §6's table (mean±std, both protocols). Answer the three questions.
Update this doc's Status → resolved, with the numbers and the "is the crop noise?" verdict.

**Step 8 — merge.** Merge to `main` only when: gate green, doc-sync complete, the experiment table is filled
and the verdict recorded. If `pose_full` underperforms `full`, that is a **publishable negative** (pose ≱
crop under streaming) — still merge the arm + the finding; do not bury it.

## 11. Implementation notes (deviations from the plan above)

Recorded July 2026, at the step 0–4 implementation:

- **No `PoseMotionEncoder` class.** §5's "KinematicsEncoder with one change" was taken literally:
  `KinematicsEncoder` itself gained the `motion_norm="none"` pass-through (the `motion_scale` buffer is
  now built only under `"image"`, which is also where the >9-channel guard lives). `pose_kinematics`
  therefore builds the existing `KinematicsOnlyModel`; only `pose_full` got a new class
  (`PoseFullModel`, `models/ablations.py`).
- **Feature order** pinned to the §3.3 table: `coords(2n) | conf(n) | head(2) | body(2)` — the §7 sketch's
  body/head order was not used. Angle confidence rides *inside* the angle: each `(sin, cos)` pair is
  scaled by the product of its contributing joints' confidences (a dead angle reads ≈ (0, 0)).
- **Read path**: `PoseMotionTransform` (picklable class, `data/pose.py`) not a closure — it crosses the
  Windows-spawn DataLoader boundary. `pose_motion_transform(root)` is the gate; `validate_config`
  enforces `pose.enabled ⇔ model.motion_norm="none"` and the strict `motion_dim = 9 + feature_dim`
  width (the no-ego 8+dim variant is documented but not wired).
- **Augmentation kept compatible** (not in the plan): `SequenceAugmenter.horizontal_flip` now mirrors
  the raw pose too (`flip_pose`: reflect x + swap L/R joints), so the offline aug lever and
  `augment.runtime` remain usable in pose-enabled builds instead of silently corrupting flipped copies.
- **Cache writes merge-update** per-video npz, so streaming and anchored splits can be extracted in
  separate runs; `extract_pose.py --split X` accepts every pkl `build_lmdb` does.
- **Frames stream from `PIE_clips`, in memory.** §7's "iterate unique PIE frames" originally assumed
  staged image files, which the storage-limited lab PC never has (it builds LMDBs incrementally from
  clips). `extract_pose.py` now decodes each video with a sequential cv2 scan (same decode as
  `incremental.extract_video_frames`) and feeds the BGR frame straight to the extractor — nothing is
  written to or read from an images dir. Side effect: rtmlib receives cv2-native BGR, its expected
  channel order.

## 12. Risks / open forks

1. **Extractor quality on small peds is the real risk**, not the encoder. If DWPose confidences collapse
   below `min_conf` for most feet on distant peds, the feet block is dead weight — mitigate with the
   confidence channel (the model gates) and report the joint-availability distribution as a diagnostic.
2. **`motions` semantic overload** (Route A): the tensor means "9-dim" for baseline models and "58-dim" for
   pose models. Contained to the encoder; flagged here so future readers aren't surprised. Revisit Route B
   only if a second pose consumer appears.
3. **`pose_full` confounded by the ViT** until RQ1 is locked — hence the §2 ordering. `pose_kinematics` is
   the clean, ViT-free read and can carry the arm's headline alone if `pose_full` must wait.
4. **Pelvis vs bbox-center normalization** (§3.4) — bbox-center chosen for robustness; if GT boxes are loose
   enough that intra-skeleton scale drifts, switch origin to pelvis (a one-line change in `data/pose.py`,
   re-run Step 1 invariance tests).

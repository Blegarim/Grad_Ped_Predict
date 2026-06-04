# MIGRATION.md

Running log of the Phase-A ground-up rebuild (behavior-preserving restructure). One row per ported
module. See [REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md) for the prompts and [CLAUDE.md](CLAUDE.md) for the
architecture, band-aid table (B1–B13), and imbalance policy. The porting workflow itself is the
`behavior-preserving-port` skill.

**OLD repo (read-only reference):** `OLD/Undergrad_thesis_project` (vendored into this repo; golden reference samples in `OLD/golden/`)

## How to use this file

For each module you port:
1. Capture a golden fixture from the OLD repo **before** writing new code.
2. Port into the target file(s); resolve the band-aids the prompt lists.
3. Add a parity test; record the result here.
4. Fill in the row: golden fixture path, band-aids resolved, behavior changes flagged, parity result.

**Parity legend:** ✅ exact (within tol) · ⚠️ intentional behavior change (justified in Notes) ·
❌ failing/blocked · — not started.

## Progress

| Prompt | Module(s) | Source (OLD) | Golden fixture | Band-aids resolved | Parity | Notes |
|---|---|---|---|---|---|---|
| 0.1 | repo scaffold, `pyproject.toml`, `.gitignore` | `requirements.txt`, repo root | n/a | B11, B12 | ✅ | gate green on py3.10: ruff + 3 smoke tests + editable src-layout import |
| 0.2 | `config/schema.py`, `config/loader.py`, `configs/*.yaml` | `config.py`, hardcoded args | `tests/fixtures/golden/legacy_config.json` | B1, B6, B7 | ✅ | `vit_kwargs()`/`motion_kwargs()` reproduce OLD `config.py` dicts exactly; 23 config tests green on py3.10. See Config decisions below. |
| 0.3 | `utils/{seed,device,amp,memory,logging}.py`, `paths.py` | `train.py` perf/AMP/mem idioms | n/a | B8, part B1/B9 | ✅ | infra (no numeric fixture). B8 = `to_float_logits` value-parity tested; perf flags/AMP gate/mem-poll relocated 1:1 from `train.py:244-255`,`347`,`train_utils.py:74-77`. 14 utils tests green on py3.10. Added `runs_dir` to `PathsCfg`; `outputs/` gitignored. See Utils decisions below. |
| 1.1 | `data/pie_sequences.py`, `scripts/make_sequences.py` | `scripts/generate_sequences.py` | `tests/fixtures/golden/pie_sequences_counts.json` | B5 | ✅ | EXACT (int labels, deterministic PIE 'all' path → tol=0). Legacy-oracle parity (synthetic) + count gate vs the legacy pkls green (train 95,684 / val 22,665 / test 76,048 reproduced). `has_onset` dropped; abs image paths re-rooted to this repo. See Data decisions below. |
| 1.2 | `data/transforms.py`, `data/lmdb_writer.py`, `scripts/build_lmdb.py` | `scripts/preprocess_data_lmdb.py`, `PIE_sequence_Dataset_1.py` | `tests/fixtures/golden/lmdb_process_record.pt` | B5, upstream B7 | ✅ / ⚠️ | geometry+motion EXACT vs OLD `_process_sequence` (motions/labels tol=0, crops atol=1e-6). ⚠️ flagged: `bboxes` dropped from meta; TurboJPEG→PIL; write-time img_augment + `_dataaug` dropped; `context_scale` unified to 3.0. 15 tests green. See Data decisions (1.2) below. |
| 1.3 | `data/balance.py`, `scripts/balance_dataset.py`, `config/{schema,loader}.py`, `configs/balance.yaml` | `scripts/balance_sequences.py`, `split_balance_sequences_all.py` | `tests/fixtures/golden/balance_cases.json` | B3 (offline), B5 | ✅ / ⚠️ | Both legacy balancers reproduced EXACTLY (selected indices + all 3 solvers, tol=0) via `BALANCE_EQUAL` / `BALANCE_RATIO_30_70` presets. ⚠️ flagged: default solver fixes the `solve_exact` sign bug (legacy reachable only via `legacy_x00_sign_bug=True`); `summarize` now clamps crosses; random `split_indices` dropped (PIE provides splits, 1.1). Balance is OPT-IN (`enabled=false`). 21 balance tests green. See Data decisions (1.3) below. |
| 1.4 | `data/augment.py`, `scripts/augment_dataset.py`, `config/{schema,loader}.py`, `configs/augment.yaml` | `scripts/augment_sequences.py` | `tests/fixtures/golden/augment_cases.pt` | B5 | ✅ / ⚠️ | All 4 transform kernels reproduce OLD `SequenceAugmenter` EXACTLY (flip/erase tol=0; color/noise atol=1e-6 under matched seeding). Flip-negated channel = **idx 2 (dx)**, guarded by a cross-module test vs `compute_motion`. ⚠️ flagged: OLD augmenter was DEAD (assumed tensor pkls; real pkl is path-based) → re-homed onto `ProcessedSample` at write time; negatives excluded from aug LMDB; per-transform copies (not composed); seedable per-item RNG (distribution-, not draw-, parity for the *plan*). 12 augment tests green. See Data decisions (1.4). |
| 1.5 | `data/lmdb_dataset.py`, `data/collate.py` | `scripts/lmdb_dataset.py`, `train_utils.py` | `tests/fixtures/golden/lmdb_dataset_cases.pt` | B7 | ✅ | Read parity vs OLD dataset+`collate_fn` (images atol=1e-6, motions/labels tol=0). Worker-safe env (pid-keyed) + picklable dataset/collate tested under `num_workers=2`. B7: `MAX_SEQ_LEN→DataCfg.max_seq_len`; `motions[...,:8]` slice **deleted** + guarded. Read transforms config-driven; read context=224. 4 tests green. See Data decisions (1.5). |
| 1.6 | `data/sampler.py`, `config/{schema,loader}.py`, `configs/train.yaml` | `train.py:34-123` | `tests/fixtures/golden/sampler_cases.json` | B3 (online, dedup scans), part B1 | ✅ / ⚠️ | Both legacy weight fns reproduced EXACTLY (global CE class weights + per-chunk sampler weights, atol=1e-6) via ONE `scan_chunk_labels` + `LabelScanCache` (replaces the two scan loops + inline `weight_cache`). The two inverse-freq *formulas* kept verbatim (only the scan is shared). ⚠️ flagged: single canonical crosses clamp (`clamp_cross`) replaces the two legacy clamps — coincide on in-contract `{0,1}` data, diverge only on never-occurring `2`. Added `TrainCfg.sampler_min_weight`. 16 sampler tests green. See Data decisions (1.6). |
| 1.7 | `data/stats.py`, `scripts/count_labels.py` | `label_count.py` | reuses `tests/fixtures/golden/pie_sequences_counts.json` (1.1) | B5, B3 (reuse) | ✅ / ⚠️ | No new scanner — aggregates the 1.6 `LabelScanCache` over base split LMDBs; drift = EXACT integer equality vs the 1.1 fixture (the table's numeric source). ⚠️ flagged: per-chunk rows → per-split aggregate; `crosses[-1]` column dropped (clamped at 1.1); aug dir excluded from the canonical table (opt-in `--include-aug` bypasses the gate). `tabulate` dropped (unused → hand-rolled table). 7 tests green. See Data decisions (1.7). |
| 2.1 | `models/vit.py`, `models/geometry.py` | `models/Vision_Transformer.py` | `tests/fixtures/golden/vit.pt` | B2, B13, B6 | ✅ / ⚠️ | Output parity vs OLD ViT EXACT under shared state_dict (atol=1e-6, rtol=1e-5, eval mode); strict=True load with **no dummy forward** proven. ⚠️ flagged: eager resolution-bound rel-pos tables (no forward-time rebuild) — a 224-trained ckpt won't strict-load into another resolution by design (OLD lazy path silently reinit'd it). 6 model + 2 config tests green. See Model decisions (2.1). |
| 2.2 | `models/motion_encoder.py` | `models/Motion_Encoder.py` | `tests/fixtures/golden/motion_encoder.pt` | B6, B7 (confirm) | ✅ / ⚠️ | Output parity vs OLD `MotionEncoder` EXACT under shared state_dict (atol=1e-6, rtol=1e-5, eval mode), 51 keys / 685,010 params; strict=True load with **no dummy forward** (no lazy params — not a B2 case). ⚠️ flagged: T>capacity now raises a clear error vs an opaque broadcast crash (numerically neutral, `T=20≤200`); `max_positions` constructor-only (kept out of `motion_kwargs`, like `img_size` in 2.1). Resolution-agnostic (adaptive pool). 5 model tests green. See Model decisions (2.2). |
| 2.3 | `models/cross_attention.py`, `models/heads.py` | `models/Cross_Attention_Module.py` | `tests/fixtures/golden/cross_attention.pt` | B4 | ✅ / ⚠️ | Output parity vs OLD `CrossAttentionModule` EXACT under shared state_dict (atol=1e-6, rtol=1e-5, eval), 22 keys / 124,937 params; strict=True load (no lazy params), incl. the legacy-dead `classifier.crosses`. ⚠️ flagged: **`crosses_pooled` made LIVE** (default on) but **unsupervised** — legacy ALLOCATED the head yet never emitted it (dead param); rebuild emits it as a ready-to-swap auxiliary (golden ref recomputed from legacy weights), never routed to loss. `key_padding_mask` removed (always `None` at every call site; fixed seq_len=20, no padding — 1.5). Heads factored into `heads.py` as builders (state_dict keys byte-identical) shared with ablations (2.5). `num_heads=4` (get_model), not class default 8. 8 cross/heads tests green. See Model decisions (2.3). |
| 2.4 | `models/ensemble.py`, `models/registry.py` | `models/Unified_Module.py`, `scripts/model_utils.py` | | B10 | — | typed factory + forward adapter |
| 2.5 | `models/ablations.py` | `models/AblationModels.py` | | B11 | — | per-ablation output keys |
| 3.1 | `losses/multitask.py` | `train.py:144-153,341-345` | | B3 (loss), part B1 | — | imbalance policy (w/ 1.3, 1.6) |
| 3.2 | `training/metrics.py` | `train.py` val, `test.py` eval | | B1 | — | shared by train+test |
| 4.1 | `training/trainer.py` | `train.py:125-175,236-632` | | B1, B2 (consumer), B8 | — | no dummy-forward |
| 4.2 | `training/chunk_loader.py` | `train.py:368-504`, `train_utils.py:80-98` | | B9 | — | crash-safe ChunkPrefetcher |
| 4.3 | `training/callbacks.py` | `train.py` ckpt/early-stop/sched | | B2 (load), B11, B1 | — | full-state resume, strict=True |
| 4.4 | two-phase schedule on Trainer | `train_two_phase.py` | | B1 | — | phases as config, not god-script |
| 4.5 | `utils/logging.py` CSV conventions | `train.py`/`test.py` CSV writers | n/a | B11, B1 | — | run-dir + index.csv |
| 5.1 | `eval/evaluate.py`, `scripts/evaluate.py` | `test.py` | | B1, B10 | — | |
| 5.2 | `eval/benchmark.py` | `Vision_Transformer.py` fvcore use | n/a | — | — | params/FLOPs/latency/FPS/VRAM |
| 5.3 | `eval/inference.py` | `main.py`, `extract_frames.py`, `pedestrian_detection.py` | | — | — | reuse Phase-1 preprocessing |
| 6.1 | `viz/plots.py`, `scripts/visualize.py` | `scripts/plot_results.py` | n/a | — | — | consume new CSV schema |
| 6.2 | `viz/qualitative.py` | `visualize_comparison.py`, `visualize_gt.py` | n/a | B11 | — | temporal_weights overlays |
| 7.1 | `export/onnx.py`, `scripts/export_onnx.py` | `onnx/onnx_export.py` | | B2 | — | onnxruntime parity check |
| 8.1 | `tests/`, CI gate | OLD `test_*.py` ad-hoc scripts | n/a | B12 | — | golden fixtures + ruff/pytest |
| 8.2 | `CLAUDE.md`, `README.md`, docstrings | OLD `CLAUDE.md`, `README.md`, `GUIDELINE.md` | n/a | — | — | keep stat table in sync |
| 9.1 | cutover & legacy retirement | whole OLD repo | n/a | B11 | — | parity gate per model_type |

## Decisions Log

Record cross-cutting decisions here as they're made (so coupled prompts stay consistent):

- **Imbalance policy** (1.3 / 1.6 / 3.1): _DECIDED (1.3). Default = **online sampler (1.6) + inverse-freq
  loss class weights (3.1)**, both already ON in `TrainCfg`, layered on offline **augmentation** (1.4) —
  this is what OLD `train.py` actually ran (`['preprocessed_train','preprocessed_train_aug']` +
  `compute_class_weights_from_lmdb` + `build_sampler_weights`). Offline **balance** (1.3) is the OPT-IN
  majority-downsample **alternative** to augment, `BalanceCfg.enabled=false` by default; when enabled,
  relax the online levers so the three never silently triple-stack (B3). The balance scripts did NOT feed
  the legacy final pipeline. 1.6 owns the single metadata scan feeding both sampler + loss; 1.3 scans the
  sequence pkls (a separate offline artifact), not the LMDB._
- **`crosses_pooled` fate** (B4, Prompt 2.3): _DECIDED (2.3) — **LIVE-but-unsupervised auxiliary head,
  default ON** (`ModelCfg.emit_crosses_pooled=True`). Rationale: the OLD `forward` never emitted
  `crosses_pooled` (the `classifier["crosses"]` MLP was allocated but skipped — a dead *parameter*, not
  dead compute), yet the docs advertised a 5-key contract. We reconcile by keeping the head and computing
  it every forward, emitted as `crosses_pooled`, kept ready to swap in for `crosses_frame` later — but
  **never routed to the loss/metrics** (3.1/3.2 supervise `crosses_frame` only). This is an intentional,
  flagged ADDITION over legacy: the 4 genuine legacy keys keep EXACT golden parity; `crosses_pooled` has
  its own golden reference recomputed from the legacy weights. The `classifier["crosses"]` param is
  retained 1:1 so OLD checkpoints still `strict=True`-load. A `emit_crosses_pooled=False` switch exists
  (gating must not perturb the legacy keys — tested). Coupled siblings: keep `crosses → crosses_frame`
  routing singular in loss (3.1), metrics (3.2), eval (5.1)._
- **`key_padding_mask` removed** (B4-adjacent, Prompt 2.3): _the legacy `forward` accepted it but every
  call site (`EnsembleModel`, `model_forward`) passed only 2 positional args (permanently `None`), and the
  data layer emits fixed-length `seq_len=20` windows with no padding (1.5 fixed-length policy). Dropped the
  unused param — behavior-neutral. Note for 2.4: `EnsembleModel.forward` calls `cross_attention(motion, image)`._
- **8-dim motion channel definition** (1.2 / 1.4 / 2.2): _LOCKED in `transforms.compute_motion`.
  Order `(cx, cy, dx, dy, w, h, dw, dh)` from the int-truncated bbox. **Flip-negated channel = index 2
  (dx)** for 1.4. ⚠️ Two preserved legacy quirks: frame-0 dx/dy hold the first *delta* but dw/dh (idx
  6/7) hold the *raw* w0/h0 (not a delta); and the absolute `cx` (idx 0) is not reflected under flip —
  1.4 negates only dx, so reconsider reflecting cx there (Phase-B candidates). See Data decisions (1.2)._
- **`context_scale` = 3.0 uniform** (1.2 / 5.2): _user-mandated single value (matches what OLD
  `preprocess_data_lmdb.__main__` actually ran; the earlier "2.0" claim was drift). `DataCfg.context_scale`
  now 3.0; kept config-flexible for ablation. `EvalCfg.bench_context_scale=3.0` is now redundant with it —
  unify/drop in 5.2._
- **Sequence-length policy** (1.5): _DECIDED — **fixed-length, truncate, no pad**. Windows are exactly
  `seq_len=20` frames by construction (1.1 drops short tracks); the collate `[:max_seq_len]` cap is a
  defensive truncation that is a no-op while `seq_len <= max_seq_len`. No padding path (stacking needs a
  common T); variable-length is a Phase-B concern._
- **Custom chunk prefetch vs torch DataLoader** (4.2): _default to preserving custom prefetch this phase._

### Data decisions (Prompt 1.1)

Locked so the downstream data prompts (1.2 writer, 1.5 collate) stay consistent:

- **Parity class = EXACT (tol=0), not float tolerance.** Labels are integers and PIE's `'all'`
  path is deterministic (sorted iteration, no RNG), so the ported windowing reproduces the legacy
  output bit-for-bit. Captured golden = `tests/fixtures/golden/pie_sequences_counts.json` (exact
  per-split N + actions/looks/crosses totals, read from the legacy `sequences_*.pkl`; matches the
  CLAUDE.md table). The behavior-preserving test asserts equality against a **verbatim transcription
  of the OLD windowing loop** (`_legacy_window_track` in `tests/test_data_shapes.py`).
- **`has_onset` → DROPPED** (B5). Dead in the legacy code; onset-based labeling would change the
  label table and is a Phase-B concern. The future-window labeling rule is isolated in one helper
  (`_label_future_window`) so it can be swapped without touching the windowing loop — the "keep it
  adaptable" requirement, met by structure, not by retaining dead code.
- **Image paths: absolute but re-rooted to THIS repo.** PIE is to be cloned in-repo and its toolkit
  builds paths from `paths.pie_root` (resolved via `find_project_root`), so records carry
  `<this-repo>/data/images/...`. ⚠️ This is the one non-label difference vs the OLD pkls (whose
  paths point at the OLD repo). Golden parity is asserted on **labels + window structure** (counts,
  lengths, slicing), NOT on the absolute path string. 1.2 (lmdb_writer) must read images via the
  same `pie_root`. Relativizing paths is deferred (Phase B).
- **Record key contract (frozen, consumed by 1.2):** `{images: list[str], bboxes: list[list[float]],
  actions: int, looks: int, crosses: int}` — keep these keys stable.
- **`.gitignore` fix (latent 0.1 bug, B11).** The unanchored `data/` rule (ported from the OLD
  `.gitignore`) was *also* matching `src/pedpredict/data/`, silently making the entire data package
  un-committable (`git ls-files src/pedpredict/data/` was empty). Anchored the dataset patterns to
  the repo root (`/PIE/`, `/data/`) so they ignore only the root dataset dir + generated pkls, not
  the package. Verified: package trackable, `data/sequences/*.pkl` + `PIE/` still ignored.
- **Config additions (additive, defaults = OLD `data_opts` literals):** `DataCfg` gains
  `min_track_size, fstride, data_split_type, seq_type, squarify_ratio, height_min, height_max`
  (`None`→`inf`); `PathsCfg` gains `pie_root`, `sequences_dir`. `validate_config` now also asserts
  `data.seq_len / stride / min_track_size > 0`. `pie_data_opts(DataCfg())` reproduces the exact OLD
  PIE-call dict (asserted).
- **Verification status:** the count gate currently diffs the legacy `sequences_*.pkl` (the
  deterministic legacy output) — green. The full **regenerate-from-PIE → diff** check
  (`scripts/make_sequences.py --split val`) is ready but **deferred until the PIE toolkit + dataset
  are cloned into the repo** (`test_legacy_pkl_counts_match_fixture` is `@pytest.mark.slow`, skips if
  pkls/PIE absent).

### Data decisions (Prompt 1.2)

Locked so the downstream data prompts (1.4 augment, 1.5 collate/dataset) stay consistent:

- **Concern split (B5).** OLD `PIE_sequence_Dataset_1.py` + `preprocess_data_lmdb.py` → `data/transforms.py`
  (crop geometry + motion + resize/normalize, the math) and `data/lmdb_writer.py` (serialization/chunking
  only) + thin `scripts/build_lmdb.py`. Dead `scripts/preprocess_data.py` stays dropped.
- **Parity class.** Geometry/motion EXACT vs OLD `_process_sequence`: motions + labels `tol=0`, resized
  crops `atol=1e-6` (same Pillow 11.2.1 / torchvision 0.22.1 captured in `.venv`). Golden =
  `tests/fixtures/golden/lmdb_process_record.pt`, produced by `tests/_capture/capture_lmdb_golden.py`
  (run against the OLD repo, TurboJPEG force-disabled). Plus a verbatim motion oracle in `test_transforms.py`.
- **8-dim motion (upstream B7).** `compute_motion` emits exactly `motion_dim` (=8) channels
  `(cx, cy, dx, dy, w, h, dw, dh)` from the **int-truncated** bbox → the collate `motions[..., :8]` slice
  (1.5) is now a provable no-op to delete. Channel table + the frame-0 dw/dh quirk documented in the
  `compute_motion` docstring. Flip-negated channel for 1.4 = **index 2 (dx)**.
- **⚠️ Intentional behavior changes (flagged, not silent):**
  1. **`bboxes` dropped from `_meta`.** OLD stored "everything not `images*`", silently including `bboxes`;
     the frozen contract is `{motions, actions, looks, crosses}`. Image JPEG bytes + motions + labels are
     unchanged. 1.5 must not read `meta['bboxes']` (it doesn't need to — motions encode bbox geometry).
  2. **TurboJPEG dropped → PIL decode only.** Removes the hardcoded `C:\libjpeg-turbo64` DLL path. Golden
     captured on the PIL path so parity holds.
  3. **Write-time `img_augment` / `data_aug` / `_dataaug` LMDB dropped** (user-confirmed dead artifact;
     real augmentation is offline sequence-level in 1.4). Writer is deterministic resize-only.
  4. **`context_scale` = 3.0 uniform** (was the OLD `__main__` value; the schema's 2.0 was drift).
- **LMDB key contract (frozen, consumed by 1.5):** keys reset **per chunk** — `f"{j}_{t}_tight"`,
  `f"{j}_{t}_context"` (JPEG, un-normalized `[0,1]*255`), `f"{j}_meta"` (pickle). No global length key
  (1.5 derives N by counting `_meta`). ImageNet normalize is **read-time** (`imagenet_normalize` defined in
  transforms but applied in 1.5), never at write time.
- **`map_size` heuristic.** OLD lines 52-54 reproduced in `compute_map_size` with named/documented factors
  + `lmdb_map_size_bytes` override (and `lmdb_map_size_floor_gib`/`_safety` config fields). ⚠️ LMDB
  **pre-allocates the file on Windows**, so the 4 GiB floor reserves 4 GiB/chunk there — tests pass an
  explicit small `lmdb_map_size_bytes`.
- **Config additions (additive, defaults = OLD literals):** `DataCfg` gains `lmdb_map_size_bytes`,
  `lmdb_map_size_floor_gib`, `lmdb_map_size_safety`, `preprocess_num_workers`, `preprocess_prefetch_factor`;
  `validate_config` now also asserts `context_scale > 0`, `jpeg_quality ∈ [1,100]`, `img_height/width > 0`.
- **Worker parallelism preserved.** `CropSequenceDataset` + `DataLoader(bs=1, shuffle=False, num_workers)` keep
  the OLD parallelism (behavior-neutral, deterministic order); the OLD `unbatch` hack → module-level
  `_passthrough_collate` (picklable for Windows `spawn`).

### Data decisions (Prompt 1.3)

Locked so the coupled imbalance prompts (1.6 sampler, 3.1 loss) and the data DAG stay consistent:

- **One module, two legacy presets (B5).** OLD `balance_sequences.py` + `split_balance_sequences_all.py`
  → `data/balance.py` (pure, deterministic, stdlib `random.Random(seed)`) + thin `scripts/balance_dataset.py`.
  The two scripts used *genuinely different* solvers; both are reproduced EXACTLY by `BALANCE_EQUAL`
  (50/50, `x11=upper`, keep-all-cross1, raise) and `BALANCE_RATIO_30_70` (30/70, `x11=lower`, subsample,
  approx, empty). Parity class = EXACT (integer indices, tol=0); golden = `tests/fixtures/golden/balance_cases.json`
  (captured by `tests/_capture/capture_balance_golden.py`, run against the OLD scripts — pure stdlib, no venv).
- **⚠️ Intentional behavior changes (flagged, not silent):**
  1. **`solve_exact` sign bug fixed.** OLD `split_balance.solve_exact` used `n0 - a - l` for the `x00`
     bound instead of `a + l - n0` (sign-flipped), which can silently miscount in the 30/70 regime
     (`pick` clamps a negative/over-large `x00`). The default `solve_cross0_counts` ships the **corrected**
     constraint; the buggy interval is reachable only via `legacy_x00_sign_bug=True` (used by the `RATIO`
     golden test). `test_legacy_sign_bug_rejects_a_solvable_case` documents the divergence numerically.
     Phase-B: drop the flag.
  2. **`summarize` clamps crosses.** OLD `balance_sequences.summarize` summed raw `{-1,0,1}` (reported
     ~0.46 on the equal preset); the new `summarize` clamps (reports 0.5). Selected-index parity is
     unaffected (grouping always clamped); only the *reported rate* changes.
  3. **Random `split_indices` dropped from the DAG.** The canonical train/val/test come from PIE's default
     split (1.1, which reproduced N=95,684/22,665/76,048). The `split_balance` random-split half is
     superseded; not ported (no `scripts/split_dataset.py`).
- **Opt-in, outside the structure loop (B3).** `BalanceCfg.enabled=false`. Balance is a transform on the
  *sequence pkl* artifact: `list[SequenceRecord] → subset[SequenceRecord]`, identical keys/format, so a
  balanced pkl flows through the LMDB writer (1.2) → dataset (1.5) → training like any other. It is never a
  branch in the model/sampler/loss/collate path.
- **Config (additive, new top-level section).** Added `BalanceCfg` + `RootCfg.balance` + `configs/balance.yaml`,
  registered in `loader._SECTIONS`. A **top-level** section (not `data.balance`) because `apply_overrides`
  caps overrides at 2 levels (`section.field`); `--set balance.cross_pos_ratio=0.25` works, `data.balance.*`
  would not. `validate_config` gains balance invariants (`0<ratio<1`, rates ∈ `[0,1]`, enum fields).
- **Determinism locked to stdlib `random`** (not numpy) — required to reproduce the legacy index sets; the
  group-build → per-group `pick` → `cross1+cross0` → `shuffle` call order is the parity contract.

### Data decisions (Prompt 1.4)

Locked so the coupled imbalance prompts (1.3 balance / 1.6 sampler / 3.1 loss) stay consistent:

- **OLD `augment_sequences.py` was dead/broken code (B5).** Its `SequenceAugmenter` indexed
  `seq['images_tight']/'images_context']/'motions']` — pre-cropped *tensor* sequences — but the real
  pipeline pkl is path-based (`{images, bboxes, actions, looks, crosses}`, confirmed by loading
  `sequences_train.pkl`) and the writer crops from paths, so it could never have run. The
  `sequences_train_augmented.pkl` / `preprocessed_train_aug` artifacts are gone, so "reproduce what
  trained the weights" is unrecoverable for this module. **Parity target = the transform MATH**, re-homed
  onto `ProcessedSample` and applied at write time (`AugmentedCropSequenceDataset`).
- **Parity class.** The four kernels reproduce OLD EXACTLY: flip + random-erase are tol=0; color-jitter and
  motion-noise are atol=1e-6 under **matched seeding** (`apply()` seeds `torch.manual_seed`/`random.Random`
  the same way the capture did, so the same RNG stream → identical draws). Golden =
  `tests/fixtures/golden/augment_cases.pt` (`tests/_capture/capture_augment_golden.py`, run vs the OLD repo
  on a synthetic tensor dict). The *oversampling plan* is deterministic (seeded) but is **distribution-, not
  draw-, parity** vs OLD's single global-RNG stream — an intentional consequence of per-item seedability.
- **⚠️ Intentional behavior changes (flagged, not silent):**
  1. **Re-homed to write time.** Augmentation transforms `ProcessedSample` (post-crop) inside the writer
     dataset, not a phantom tensor pkl. Required: the OLD tensor-pkl path was non-functional.
  2. **Negatives excluded (Q1, decided).** OLD `augment_minority_sequences` re-emitted all `crosses=0`
     records into its output; unioning that with `preprocessed_train` would double-count negatives. The aug
     LMDB now holds **only** minority records (crosses=1 ×6, looks=1 ×3) + their copies; the union
     `['preprocessed_train','preprocessed_train_aug']` = full base set + boosted minorities.
  3. **Single transform per copy.** OLD `__call__` appended a fresh copy per *selected* transform (it never
     composed); preserved — one `AugItem` carries one transform (or `None` = identity).
- **Flip↔motion-channel coupling made explicit (the schematic's silent-corruption risk).** Module constant
  `_FLIP_NEGATE_IDX = 2` (= `dx`), cross-checked against `compute_motion`'s layout by
  `test_flip_index_matches_motion_channel_def`. ⚠️ Preserved Phase-A quirks: `cx` (idx 0) is NOT reflected
  under flip; motion noise hits absolute channels too (Phase-B candidates, per the 1.2 motion decision).
- **Config (additive, new top-level section).** `AugmentCfg` (+ `RootCfg.augment` + `configs/augment.yaml`,
  registered in `loader._SECTIONS`). Top-level (not `data.augment`) because overrides cap at `section.field`
  (same rationale as `BalanceCfg`). `enabled=True` by default — augmentation is the default imbalance lever
  (policy 1.3). `validate_config` gains augment invariants (probs ∈ [0,1], `1≤n_augs_min≤n_augs_max≤4`,
  multipliers ≥ 1, σ ≥ 0, erase_n_frames ≥ 0).
- **Writer seam (behavior-neutral).** `lmdb_writer` gained `write_dataset_to_lmdb` + `write_dataset_chunks_from`
  (a generalized chunker over any `Dataset[ProcessedSample]` via `Subset`); `write_chunk`/`write_dataset_chunks`
  now delegate to them with identical signatures — all 1.2 roundtrip tests still green.

### Data decisions (Prompt 1.5)

Locked so the chunk loader (4.2) and training (4.1) stay consistent:

- **Concern split.** OLD `scripts/lmdb_dataset.py` → `data/lmdb_dataset.py` (`LMDBChunkDataset`); OLD
  `train_utils.collate_fn` → `data/collate.py`. The rest of `train_utils.py` ports elsewhere per the
  disposition table (`EarlyStopping`→4.3, `mp_async_load`/`wait_for_memory`→4.2/0.3, `gather_chunks`→4.2).
- **Parity class.** Read parity is EXACT for motions/labels (`tol=0`) and `atol=1e-6` for the decoded+
  resized+normalized crops — OLD and new read **byte-identical** JPEG from the *same* deterministic 1.2
  writer, then apply identical torchvision transforms. Golden = `tests/fixtures/golden/lmdb_dataset_cases.pt`
  (`tests/_capture/capture_lmdb_dataset_golden.py`, run vs OLD `LMDBChunkDataset` + `collate_fn`).
- **Worker-safety preserved verbatim.** Per-process env via `_get_env` (pid-keyed, reopens on pid change),
  `__getstate__` drops `_env`/`_pid` so the dataset pickles to workers, `__del__` closes. Lexicographic
  `_meta` cursor order for `seq_ids` kept. The corrupt-chunk frame-count-mismatch raise is kept (fails
  loudly, never silently short). Tested under `DataLoader(num_workers=2)`.
- **B7 closure (completes 0.2's partial).** `MAX_SEQ_LEN` → `DataCfg.max_seq_len`; the `motions[..., :8]`
  slice is **deleted** (writer emits exactly `motion_dim`, 1.2-locked) and replaced by a cheap guard in
  `collate_sequences` that raises on a stale wider-motion LMDB (`test_collate_guard_rejects_wide_motion`).
- **Read transforms are config-driven (lifted from `train.py:355-366`).** New `transforms.build_read_transforms(cfg)`
  → `(tight, context)` = `Resize → ToTensor → ImageNet Normalize`; `LMDBChunkDataset.from_config` consumes it.
  Normalize is **read-time only** (writer stores un-normalized — 1.2 contract).
- **Config addition (additive): read-time context size.** OLD `train.py` hardcoded the runtime context
  resize to **224** (`Resize((224,224))`), distinct from the *write-time* context size
  (`img_* * context_scale = 384`). Added `DataCfg.read_context_height/width = 224` (+ `configs/data.yaml`,
  `validate_config` positivity check); tight read reuses `img_height/img_width = 128`. ⚠️ The stored 384
  crops are re-decoded and shrunk to 224 at read time — a legacy inefficiency preserved this phase.
- **`build_collate(cfg)` returns a `functools.partial`** (picklable under Windows `spawn`), not a closure —
  so 4.2's prefetcher / DataLoader workers can carry it without a pickling failure.
- **Couples to 4.2 (chunk loader):** it opens one `LMDBChunkDataset.from_config(chunk, cfg)` per chunk and
  uses `build_collate(cfg.data)` for the DataLoader; `__getstate__` + the partial-collate keep it crash- and
  pickle-safe under prefetch.

### Data decisions (Prompt 1.6)

The online imbalance lever + the **single LMDB scanner**. Locked jointly with 1.3 (offline balance) and
3.1 (loss weights) so the three levers read as one policy (B3), not three accidents:

- **The dedup is the SCAN, not the math.** OLD `compute_class_weights_from_lmdb` (train.py:34-72) and
  `build_sampler_weights`/`_inverse_class_weights` (74-123) duplicated the `_meta` cursor pass but use
  *legitimately different* inverse-freq formulas (loss `t/(2·max(c,1))` over fixed 2 classes vs. sampler
  `t/(len(counts)·c)` over observed classes) at *different scopes* (loss=GLOBAL over all train chunks;
  sampler=PER-CHUNK). New design: one `scan_chunk_labels` → `ChunkLabelScan`, cached per chunk by
  `LabelScanCache` (replaces the inline `weight_cache` at train.py:426). `class_weights_ce` and
  `sample_weights` are byte-for-byte ports of the two formulas; `aggregate_counts` sums cached per-chunk
  counts for the global loss path (no second scan). **Prompt 3.1 imports `class_weights_ce` from here** —
  it must not add a third scanner.
- **Scope asymmetry preserved (not "fixed").** Sampler weights stay PER-CHUNK and class weights GLOBAL,
  exactly as legacy ran — fixing the sampler to global frequencies would be a Phase-B behavior change.
- **Parity class.** EXACT (`atol=1e-6`) for both levers vs the legacy oracle (transcribed verbatim into
  `tests/_capture/capture_sampler_golden.py`; OLD `train.py` is not importable). Golden =
  `tests/fixtures/golden/sampler_cases.json` (two synthetic label-only chunks, one with single-class
  crosses to exercise `n_classes=1`).
- **⚠️ Flagged behavior change — single canonical clamp.** The scan clamps `crosses` once via
  `data.balance.clamp_cross` (`==1 → 1 else 0`) instead of the two legacy clamps (loss `max(0,min(1,·))`,
  sampler `<0 → 0`). All three agree on in-contract data (writer guarantees `crosses ∈ {0,1}`, clamped at
  1.1); they diverge only on the never-occurring value `2` (where the legacy sampler's `n_classes=len(counts)`
  would spuriously inflate to 3 classes). Zero effect on real data → parity stays exact.
- **Legacy quirk kept on purpose.** Sampler `_inverse_class_weights` uses `n_classes = len(counts)` (observed
  classes), so a single-class chunk uses `n_classes=1`; and the `if power > 0` guards let a zeroed task drop
  out entirely. Both preserved + tested.
- **Config addition (additive).** `TrainCfg.sampler_min_weight = 1e-6` (was the `build_sampler_weights`
  default literal) + `configs/train.yaml` + a `validate_config` guard (`> 0`, powers keyed to the 3 tasks,
  powers `>= 0`). `sampler_powers` already existed.
- **Resource safety.** Every `lmdb.open` is paired with a `try/finally … close()`; no env handle escapes the
  scan. The module reads only the paths it is handed (train chunks) — no val/test leakage into the levers.

### Data decisions (Prompt 1.7)

The data-layer verification tool. Locked so the drift gate has one numeric source of truth:

- **No third scanner (B3).** `data/stats.py` aggregates the 1.6 `LabelScanCache.aggregate_counts` over a
  split's chunk envs; it adds no counting logic of its own. OLD `label_counts` (the per-chunk Counter helper)
  is **deleted**, not ported. `test_aggregation_matches_scanner` pins stats == summing `scan_chunk_labels`.
- **Canonical scope = sequence-level (base LMDB).** A split's stats come from its base dir(s) — `train` from
  `lmdb_train[0]` only (`split_lmdb_dirs`). `preprocessed_train_aug` intentionally redistributes classes (the
  imbalance lever, 1.3/1.4), so it is excluded; `--include-aug` opts in and **bypasses the drift gate**. The
  base-train LMDB is a deterministic 1:1 image of `sequences_train.pkl`, so these counts reproduce the table.
- **Drift = EXACT integer equality, reference = the 1.1 fixture.** `check_drift` diffs `N` + the 3 positive
  counts against `tests/fixtures/golden/pie_sequences_counts.json` (already the table's numeric source — no
  third transcription). `test_reference_fixture_matches_claude_table` guards the fixture↔CLAUDE.md prose link.
  No tolerance/threshold config needed → **no new `StatsCfg` section** (config-first is satisfied: nothing to
  configure; scope/gating are CLI flags).
- **⚠️ Intentional changes vs OLD `label_count.py` (flagged, not silent):** per-chunk rows → per-split
  aggregate (the documented form); the `crosses[irrelevant]` (-1) column dropped (crosses clamped to {0,1} at
  1.1, none survive into the LMDB).
- **`tabulate` watch RESOLVED → dropped.** It was core-but-unused; the printed/markdown table is hand-rolled
  in `format_table` (one trivial table doesn't justify a dep — keeps the stack minimal). Removed from
  `pyproject.toml` core deps.
- **CI-friendliness.** The CLI `skip_missing=True` → exit 0 with a notice when no LMDB is built (CI without
  data stays green); the real regenerate-and-diff gate is `@pytest.mark.slow` (deferred until LMDBs exist,
  mirrors 1.1). `iter_chunk_lmdbs` (sorted `chunk_*.lmdb` glob) homes here for now — relocation candidate for
  4.2 (chunk_loader), which needs the same enumeration.

### Model decisions (Prompt 2.1)

The first model port. Locked so the downstream model prompts (2.2–2.5, 4.1 trainer, 7.1 export) stay consistent:

- **Parity class.** Output EXACT vs OLD `ViT_Hierarchical` under a **shared `state_dict`** loaded
  `strict=True`, eval mode (Dropout/DropPath → identity ⇒ pure deterministic fp32 math), `atol=1e-6`,
  `rtol=1e-5`. Golden = `tests/fixtures/golden/vit.pt` (`tests/_capture/capture_vit_golden.py`, run vs OLD;
  the OLD state_dict is captured **after** a dummy forward so it contains the global table). Module/attr
  names preserved verbatim (`stem`, `stages[i]["down_sample"|"block"]`, `attn.relative_position_bias_table`,
  `relative_position_index`, …) so the 265-key dict maps 1:1.
- **B2 RESOLVED — eager params, resolution-bound (the headline).** Only the *global* stage
  (`window_size=None`) deferred its `relative_position_bias_table` to first forward (stages 0–2 were already
  eager); that defer forced the `train.py:311-317` dummy forward and broke `strict=True`. The global window is
  now resolved at `__init__` from a scalar `img_size` (the last stage's feature-map side: 224 → 7×7 → table
  `[169,2]`), so **every param exists after construction**. `test_strict_load_without_dummy_forward` proves the
  OLD dict loads with zero missing/unexpected keys and no forward. The stem/downsample geometry lives in a
  torch-free `models/geometry.py` (`feature_map_size`) shared by the ViT and `config.validate_config`.
- **⚠️ Intentional behavior change — no forward-time resolution adaptivity.** The OLD lazy path *rebuilt* the
  global table for whatever feature size showed up at forward (and `load_state_dict(strict=False)` then
  silently dropped a mismatched table — a latent weight-loss bug on resolution change). The new model is
  **built for one resolution** (`ViT_Hierarchical.from_config(model_cfg, img_size=cfg.data.read_context_height)`)
  and fixed within a run — matches the actual workflow (train+eval at a designated resolution; resolution
  varied across runs, never within one model). A different resolution is an **explicit** `rebuild_position_bias(img_size)`
  call (benchmark/export), never silent in `forward`. Consequence: a 224-trained checkpoint legitimately will
  **not** `strict`-load into a model built for another resolution (the rel-pos table is a resolution-specific
  weight) — now an explicit error instead of a silent reinit. Decided with the user.
- **B13 RESOLVED (behavior-neutral).** `WindowTransformerBlock.forward` MLP residual rewritten with an
  unambiguous `shortcut_flat = x_perm.reshape(B, H*W, C)`; identical math (both the MLP-internal dropout and
  the post-MLP `self.dropout` preserved). Parity holds exactly.
- **B6 RESOLVED.** `__main__` is now a `ModelCfg`-driven smoke test (`from_config(ModelCfg(), img_size=224)` →
  asserts `[B,T,d_model]`); the drifting legacy kwargs (`d_model=224`, `stage_dims=[48,96,168,96]`) are gone.
- **`img_size` is NOT a `ModelCfg` field (config hygiene).** It flows from the already-flexible
  `DataCfg.read_context_height/width` at the call site, so the ViT can't drift from the crops it's fed and
  `ModelCfg.vit_kwargs()` stays byte-identical to OLD `vit_args_config()` (0.2 golden untouched). New
  `validate_config` checks (additive): context crop must be **square** and must **tile every stage window**
  (`feature_map_size(ctx, i) % window_i == 0`), turning a runtime crash into a config-time error. Two new
  `test_config.py` cases cover both.
- **Windows generalized to `(Wh, Ww)`** in `window_partition`/`window_reverse`/`WindowAttention` (numerically
  identical to the legacy square partition when `Wh==Ww`, which is the only case today) — keeps non-square
  windows expressible without touching parity. `timm` watch: confirmed **`DropPath` is the only `timm`
  symbol** used; kept this phase (identity in eval ⇒ parity-neutral), vendoring is a Phase-B simplification.

### Model decisions (Prompt 2.2)

The second model port. Locked so the downstream prompts (2.4 ensemble, 2.5 ablations, 4.1 trainer) stay consistent:

- **Parity class.** Output EXACT vs OLD `MotionEncoder` under a **shared `state_dict`** loaded `strict=True`,
  eval mode (Dropout / GRU-dropout / MultiheadAttention-dropout → identity; BatchNorm uses captured running
  stats), `atol=1e-6`, `rtol=1e-5`. Golden = `tests/fixtures/golden/motion_encoder.pt`
  (`tests/_capture/capture_motion_golden.py`, run vs OLD; B=2, T=3, tight 128×128). Module/attr names
  preserved verbatim (`img_encoder`, `motion_encoder`, `fusion`, `gru`, `temporal_attn`, `pos_encoding`,
  `norm`, `dropout`, `proj`) so the 51-key dict maps 1:1. Math + op order copied verbatim, incl. the
  in-forward per-sequence motion normalization (`(x-mean)/(std+1e-6)`, **unbiased `std`**).
- **NOT a B2 case (the contrast with 2.1).** MotionEncoder has **no lazy parameters** — every weight exists
  at `__init__`, so the state_dict is captured right after construction (no dummy forward) and
  `test_strict_load_motion_no_lazy_params` proves a zero-missing/unexpected strict load with no forward.
  `img_encoder` ends in `AdaptiveAvgPool2d(1)` ⇒ the module is **resolution-agnostic** (no `img_size`
  constructor arg, no `rebuild_*` machinery, no resolution-strict-load caveat — unlike the ViT).
- **⚠️ Intentional behavior change — T≤capacity guard (numerically neutral).** OLD `pos_encoding[:, :T]`
  silently yields `[1,200,hidden]` for `T>200` then crashes on broadcast with an opaque `RuntimeError`. The
  port raises a clear `ValueError` ("exceeds positional-encoding capacity") at the top of `forward`. For every
  valid `T` the numbers are identical (runtime `T = seq_len = 20 ≤ DataCfg.max_seq_len = 20 ≪ 200`) — both
  paths already errored for `T>200`, so this only improves the message, not the math.
- **`max_positions` is a constructor arg, NOT a `ModelCfg` field (config hygiene).** Default `200` keeps
  `pos_encoding`'s shape — and thus the OLD state_dict — unchanged, and keeps `ModelCfg.motion_kwargs()`
  byte-identical to OLD `motion_enc_args_config()` (the 0.2 config golden untouched). Exact parallel to the
  2.1 decision to keep `img_size` out of `vit_kwargs`.
- **B6 RESOLVED.** `__main__` is a `ModelCfg`-driven shape test (`from_config(ModelCfg())` → asserts
  `[B,T,d_model]`); the drifting legacy kwargs (`hidden_dim=224`) are gone. With training dims `hidden_dim=168
  ≠ d_model=128`, `proj` is a **real `Linear`** (residual projection), present in the golden dict.
- **B7 confirmed (motion-dim contract).** The Conv1d input width is the only coupling to the 8-channel motion
  definition (1.2 writer / 1.4 flip-negation) and it uses the channel *count*, never the per-channel
  semantics — so it cannot corrupt the flip contract. `test_motion_conv_in_channels_matches_datacfg` pins
  `motion_encoder[0].in_channels == DataCfg.motion_dim == ModelCfg.motion_dim`.
- **`forward(motion, tight)` arg order** (= OLD `(motion_data, images_data)`) — the 2.4 ensemble / 2.5
  ablation call sites must pass `(motion, tight-crops)` in this order when wiring.

### Model decisions (Prompt 2.3)

The fusion + heads port. Locked so the downstream prompts (2.4 ensemble, 2.5 ablations, 3.1 loss, 3.2
metrics, 5.1 eval) stay consistent:

- **Parity class.** The 4 genuinely-legacy outputs (`actions`, `looks`, `crosses_frame`,
  `temporal_weights`) are EXACT vs OLD `CrossAttentionModule` under a **shared `state_dict`** loaded
  `strict=True`, eval mode (Dropout + MultiheadAttention dropout → identity), `atol=1e-6`, `rtol=1e-5`.
  Golden = `tests/fixtures/golden/cross_attention.pt` (`tests/_capture/capture_cross_attention_golden.py`;
  B=2, T=3, D=128). 22-key dict / 124,937 params maps 1:1 — attrs preserved verbatim (`cross_attn`,
  `pool_mlp`, `classifier`, `crosses_frame_head`). No lazy params (not a B2 case).
- **B4 RESOLVED — `crosses_pooled` is LIVE-but-unsupervised (default ON).** The OLD `forward` ALLOCATED
  `classifier["crosses"]` but **skipped it** (`if key != "crosses"`) → `crosses_pooled` was never emitted;
  the head was a dead *parameter* (checkpointed, optimizer-tracked, never run), while the docs claimed a
  5-key contract. ⚠️ Intentional ADDITION: the head now runs every forward and is emitted as
  `crosses_pooled`, kept ready to swap in for `crosses_frame`, but **never routed to loss/metrics**
  (3.1/3.2/5.1 supervise `crosses_frame` only). Its golden reference is recomputed from the legacy weights
  in the capture script (legacy `out` had 4 keys). `emit_crosses_pooled=False` disables it; gating must not
  perturb the 4 legacy keys (`test_cross_attention_emit_flag_default_on`). The `classifier["crosses"]`
  param is retained 1:1 so OLD checkpoints `strict`-load — the residual dead param under the off-switch is
  a Phase-B / 9.1 drop candidate.
- **⚠️ `key_padding_mask` removed.** Legacy `forward` accepted it; every call site passed it as `None`
  (fixed `seq_len=20`, no padding — 1.5). Behavior-neutral removal. **2.4 note:** `EnsembleModel.forward`
  must call `self.cross_attention(motion_feats, image_feats)` (two positional args).
- **Heads factored into `heads.py` as BUILDERS, not a wrapping module.** `build_pool_mlp` /
  `build_task_classifiers` / `build_crosses_frame_head` return bare `nn.Sequential`/`ModuleDict`/`Linear`
  assigned to the OLD attr names → state_dict keys byte-identical (a wrapping `nn.Module` would prefix
  `heads.*` and break strict-load). Stateless `temporal_attention_pool` + `frame_pool_reduce` live there
  too. **2.5 must reuse these** to kill the pool-MLP/classifier/frame-pool duplication across the 3
  ablations (same attr names → their OLD checkpoints also strict-load).
- **`num_heads=4` (config hygiene).** `get_model` wired the full model's cross-attn with `num_heads=4`
  (NOT the legacy class default 8) and did NOT forward dropout (so `cross_attn`+classifier use
  `head_dropout=0.1`). New `ModelCfg.cross_attn_num_heads=4`, `frame_pool="logsumexp"`,
  `use_frame_crosses=True`; `cross_kwargs()` mirrors the `get_model` call. `emit_crosses_pooled` is a
  constructor-only field kept OUT of `cross_kwargs()` (same precedent as `img_size`/`max_positions`).
  `validate_config` now checks `d_model % cross_attn_num_heads == 0` and `frame_pool ∈ {logsumexp,max,mean}`.

### Config decisions (Prompt 0.2)

Locked so the later prompts that consume config stay consistent:

- **Defaults source = `config.py` + `train.py`/`test.py` literals**, NOT the `__main__` smoke-test kwargs.
  B6 drift recorded: `Vision_Transformer.__main__` used `d_model=224`, `stage_dims=[48,96,168,96]`,
  `head_nums=[2,4,7,4]`, `dropout=0.1`; `Motion_Encoder.__main__` used `hidden_dim=224`. These never fed
  training (train.py imports `vit_args_config()`/`motion_enc_args_config()`); `ModelCfg` mirrors the latter.
  The drifting `__main__` blocks get rebuilt to consume `ModelCfg` in Prompts 2.1/2.2 — drift closes there.
- **Q1 — dict overrides REPLACE** the whole dict (no deep-merge), e.g. `train.loss_weight={crosses:2.0}`
  drops `actions`/`looks`. Documented in `loader.py`; asserted by `test_override_dict_replaces_not_merges`.
- **Q2 — `TrainCfg.use_amp: bool` is the *request*;** runtime ANDs it with CUDA availability in
  `utils/amp.py` (Prompt 0.3). Schema stores intent, not the resolved value.
- **Q3 — `context_scale` unified to 3.0 (RESOLVED in 1.2).** The original note kept `DataCfg.context_scale=2.0`
  ("fixed by how the LMDB crops were written") vs `EvalCfg.bench_context_scale=3.0`. That 2.0 was **wrong**:
  OLD `preprocess_data_lmdb.__main__` actually wrote at `context_scale=3.0`. Per user mandate, `context_scale`
  is now a single uniform **3.0** (`DataCfg.context_scale=3.0`), kept config-flexible for ablation.
  `bench_context_scale` is now redundant (also 3.0) → unify/drop in 5.2.
- **Q4 — `DataCfg.chunk_size = 5000`** is canonical (OLD `preprocess_data_lmdb.main()` default was 4500).
  Affects only future LMDB re-writes, never model parity.
- **Q5 — frozen dataclasses with `slots=True`;** list-like fields are `tuple`s (immutable, hashable),
  `dict` defaults via `field(default_factory=...)`. Adapters convert tuples back to lists for the OLD
  model constructors (the parity surface). Verified on py3.10.
- **B7 closure (partial):** `MAX_SEQ_LEN → DataCfg.max_seq_len`; the `motions[..., :8]` slice → `motion_dim`
  on both `DataCfg` and `ModelCfg`, cross-checked equal in `validate_config`. The collate slice is deleted
  in Prompt 1.5 once the writer (1.2) is confirmed to emit exactly `motion_dim` channels.
- **Override channel:** CLIs use a repeatable `--set section.field=value` (via `build_argparser`), not
  `argparse.REMAINDER`, so overrides can't swallow real subcommand flags.

### Utils decisions (Prompt 0.3)

Locked so the training/eval prompts that consume these helpers stay consistent:

- **Q-A — run-dir home:** added `PathsCfg.runs_dir = "outputs/runs"` (+ `paths.yaml`). Per-run layout is
  `outputs/runs/{run_id}/{checkpoints,plots}/` (matches the `experiment-tracking` skill). The three legacy
  flat fields (`log_dir`/`ckpt_dir`/`run_ckpt_dir`) are **kept** for reading OLD artifacts; revisit dropping
  them in 4.5/9.1. `outputs/` added to `.gitignore` (R2, B11).
- **R1 — seed vs perf flags:** call order is `set_seed()` → `enable_perf_flags()`. `set_seed(deterministic=True)`
  sets `cudnn.deterministic=True`/`benchmark=False` + `use_deterministic_algorithms(True, warn_only=True)`;
  `enable_perf_flags` then **skips** `cudnn.benchmark` when `deterministic` is set (mutually exclusive).
- **R3 — `to_float_logits` is dict-wide:** upcasts *every* floating tensor in the output dict to fp32
  (superset of the OLD per-key `.float()`); int/bool tensors and non-tensors pass through; input not mutated;
  no-op outside autocast → behavior-neutral. Harmless on the unused `crosses_pooled` head (B4).
- **R4 — logging boundary:** 0.3 ships only the generic `CsvLogger` + run-dir scaffold + `make_run_id`. The
  concrete `train_log.csv`/`eval_log.csv`/`index.csv` column schemas belong to `training/metrics.py` (3.2) and
  the logging conventions of Prompt 4.5 — they pass `fieldnames` in; nothing here hardcodes columns.
- **Net-new (additive, not parity breaks):** `seed.set_seed` (OLD had no global seed) and
  `wait_for_memory(timeout=...)` (default `None` = legacy infinite wait, OLD `train_utils.py:74-77`).
- **Q2 closure:** `resolve_amp(requested, device)` realises the schema decision — `TrainCfg.use_amp` is the
  request, ANDed with `device.type == 'cuda'` at runtime (OLD `use_amp = device.type == 'cuda'`).

## Parity Gate (Phase A → cutover)

Before retiring the legacy repo (Prompt 9.1): the new repo must reproduce the OLD test metrics per
`model_type` within tolerance, using ported weights. Targets tracked in the `experiment-tracking` skill's
`references/baseline-results.md`.

## OLD Top-Level File Disposition (Prompt 0.1 audit trail)

Disposition contract for every OLD top-level file: **PORT** (logic migrated), **FOLD-INTO-TESTS**
(behavior captured as a test), **DROP** (superseded/dead/artifact), **DECIDE** (open). No file is moved
in 0.1 — this table is the contract the later prompts execute against.

| OLD path | Disposition | New location / reason |
|---|---|---|
| `config.py` | PORT | `config/schema.py` + `loader.py` (0.2). B6. |
| `train.py` (635-line god-script) | PORT (split) | `training/{trainer,chunk_loader,callbacks,metrics}.py` + `losses/multitask.py` + `data/sampler.py` (P3/P4). B1. |
| `train_two_phase.py` | PORT | phase-schedule on `training/trainer.py` (4.4). B1. |
| `test.py` | PORT | `eval/evaluate.py` + `scripts/evaluate.py` (5.1). B1. |
| `main.py` | PORT | `eval/inference.py` (5.3). |
| `label_count.py` | PORT | `data/stats.py` + `scripts/count_labels.py` (1.7). |
| `class_imbalance_strategies.py` | DROP / fold | unified imbalance policy in `data/sampler.py` + `losses/multitask.py` (B3). Salvage formulas, then drop. |
| `imbalance_config.py` | DROP / fold | merged into `TrainCfg` imbalance fields (B3). |
| `models/Vision_Transformer.py` | PORT | `models/vit.py` (2.1). B2, B13, B6. |
| `models/Motion_Encoder.py` | PORT | `models/motion_encoder.py` (2.2). |
| `models/Cross_Attention_Module.py` | PORT | `models/cross_attention.py` + `heads.py` (2.3). B4. |
| `models/Unified_Module.py` | PORT | `models/ensemble.py` (2.4). |
| `models/AblationModels.py` | PORT | `models/ablations.py` (2.5). |
| `models/__init__.py` | DROP | replaced by `models/registry.py` typed factory (B10). |
| `scripts/generate_sequences.py` | PORT | `data/pie_sequences.py` (1.1). B5. |
| `scripts/preprocess_data_lmdb.py` | PORT | `data/lmdb_writer.py` (1.2). B5. |
| `scripts/PIE_sequence_Dataset_1.py` | PORT | `data/lmdb_writer.py` + `transforms.py` (1.2). B5. |
| `scripts/balance_sequences.py` | PORT | `data/balance.py` (1.3). B3/B5. |
| `scripts/split_balance_sequences_all.py` | PORT | `data/balance.py` (1.3). B5. |
| `scripts/augment_sequences.py` | PORT | `data/augment.py` (1.4). B5. |
| `scripts/lmdb_dataset.py` | PORT | `data/lmdb_dataset.py` (1.5). |
| `scripts/train_utils.py` | PORT (split) | collate→`data/collate.py`, EarlyStopping→`training/callbacks.py`, mp prefetch→`training/chunk_loader.py`, memory poll→`utils/memory.py` (B7/B9). |
| `scripts/model_utils.py` | DROP / replace | `models/registry.py` typed dispatch (B10). |
| `scripts/plot_results.py` | PORT | `viz/plots.py` + `scripts/visualize.py` (6.1). |
| `scripts/pedestrian_detection.py` | PORT (conditional) | `eval/inference.py` helper or external `[infer]` extra (decide in 5.3). |
| `scripts/preprocess_data.py` | DROP | dead non-LMDB variant (B5). |
| `onnx/onnx_export.py` | PORT | `export/onnx.py` + `scripts/export_onnx.py` (7.1). |
| `ablation_usage_example.py` | DROP | usage doc only; superseded by README/docstrings. B11. |
| `final_ablation_verification.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py` (2.5/8.1). B11/B12. |
| `test_ablation_models.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py`. B12. |
| `test_ablation_structure_clean.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py`. B12. |
| `test_imbalance_setup.py` | FOLD-INTO-TESTS | `tests/test_losses.py` / test_sampler (1.6/3.1). B12. |
| `visualize_comparison.py` | PORT | `viz/qualitative.py` (6.2). B11. |
| `visualize_gt.py` | PORT | `viz/qualitative.py` (6.2). B11. |
| `extract_frames.py` | PORT / fold | helper for `eval/inference.py` (5.3). B11. |
| `run_env.bat` | DROP | replaced by `pip install -e .[dev]` + README. B11. |
| `requirements.txt` | DROP (absorb) | deps moved into `pyproject.toml`. |
| `CLAUDE.md` / `README.md` / `GUIDELINE.md` | PORT (rewrite) | regenerated in 8.2; new CLAUDE.md/README already exist. |
| `training_log/*.csv`, `*.xlsx` | DROP (gitignore) | run artifacts; archive originals out-of-repo (B11). |
| `plots/*.png`, `qualitative_visualize/*.jpg` | DROP (gitignore) | generated artifacts (B11). |
| `model_outputs/`, `best_model_outputs/`, `venv/` | DROP (gitignore) | weights/env never tracked (B11). |
| `.claude/`, `.understand-anything/`, `.vscode/` | DECIDE → resolved | keep `.claude/skills` tracked; gitignore `.claude/settings.local.json` + `.understand-anything/` caches (open question 5). |

### Dropped from OLD `requirements.txt`

- **Unused (0 imports):** `yacs`, `torchview`.
- **Platform shim:** `pywin32` (Windows-only; left to the resolver / installed ad hoc).
- **Pure transitive** (~25 pins) left to the resolver per the "deliberately minimal" mandate:
  `certifi`, `charset-normalizer`, `colorama`, `contourpy`, `cycler`, `filelock`, `fonttools`, `fsspec`,
  `huggingface-hub`, `idna`, `iopath`, `Jinja2`, `joblib`, `kiwisolver`, `MarkupSafe`, `mpmath`,
  `networkx`, `packaging`, `portalocker`, `py-cpuinfo`, `pyparsing`, `requests`, `safetensors`, `sympy`,
  `termcolor`, `threadpoolctl`, `typing_extensions`, `ultralytics-thop`, `urllib3`.

### Deliberate additions (not behavior changes)

- `onnx` / `onnxruntime` (`[export]` extra) — schematic mandates an onnxruntime parity check (P7);
  absent from OLD requirements.
- `ruff`, `pytest`, `pytest-cov` (`[dev]` extra) — lint + test gate (B12).

### `tabulate` / `timm` watch

- `tabulate==0.9.0` — RESOLVED (1.7): unused across `src/` → **dropped** from core deps; the 1.7 stat table is
  hand-rolled (`stats.format_table`).
- `timm==1.0.20` — RESOLVED (2.1): `models/vit.py` uses **only `timm.layers.DropPath`**. Kept this phase
  (behavior-preserving; identity in eval ⇒ parity-neutral). Vendoring the ~10-line DropPath to drop the dep
  is a Phase-B simplification candidate.

### Prompt 0.1 verification checklist

Verified on the repo `.venv` (Python 3.10.0):

- [x] Editable src-layout install succeeds (`pip install -e .` → `pedpredict 0.0.0`), with the full pinned
      core stack installed (`torch==2.7.1+cpu`, `numpy`, …) — the earlier `--no-deps` shortcut is no longer
      needed now that `.venv` carries the real dependencies.
- [x] `ruff check .` exits 0 ("All checks passed!").
- [x] `pytest -m "not slow"` exits 0 (3 passed).
- [x] `import pedpredict` from an arbitrary cwd resolves to `src/pedpredict/__init__.py`, version
      `0.0.0` (proves src-layout install, not accidental cwd import).

> **Environment note:** the repo's `.venv` is now **Python 3.10.0** with `pedpredict` editable-installed —
> the canonical env for all tests/lint; run `.venv/Scripts/python.exe` directly (no `PYTHONPATH`, no side
> venvs). It satisfies `requires-python = ">=3.10,<3.13"`; the pinned `torch==2.7.1` / `numpy==2.2.6` have
> no wheels for 3.13+. (Bare `py` / `python` on this machine still resolve to 3.14 — invoke the venv
> interpreter explicitly.)


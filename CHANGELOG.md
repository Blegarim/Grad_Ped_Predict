# Changelog

## Unreleased — v2 data contract (hole audit, attack-order step 2 code) (2026-06-12)

All dataset-touching fixes from [docs/HOLE_AUDIT.md](docs/HOLE_AUDIT.md), bundled for the ONE v2
rebuild (execution: [docs/V2_REBUILD_RUNBOOK.md](docs/V2_REBUILD_RUNBOOK.md)). **Deliberate behavior
changes** — v1 sequence pkls and LMDBs are obsolete; the runtime dataset hard-errors on v1 chunks.

- **M3** — `actions`/`looks` relabeled as state-at-end-of-observation; `crosses` stays future-any.
- **M4** — right-censored windows dropped (not labeled 0) and counted per split
  (`WindowStats` → `sequences_<split>_stats.json`).
- **M6** — `track_id` (PIE pedestrian id) carried through `SequenceRecord` → LMDB meta → dataset items.
- **M9 + A4 (motion v2)** — stored motion vector is 9-dim (`MOTION_STORE_DIM`): frame-0 deltas are true
  zeros (legacy raw-size dw/dh quirk removed), ego-speed (PIE OBD) is channel 8. Store wide, slice
  narrow: consumers read `data.motion_dim` (8 default / 9 = ego). Flip augmentation now also reflects
  `cx` about `data.source_width`. Motion normalization is a runtime flag `model.motion_norm`
  (`image` fixed frame-dim scale = new default; `per_sequence` = legacy z-norm, pinned by parity tests).
- **M5** — benchmark-protocol eval set: `make_sequences.py --benchmark` (TTE-sampled, event-labeled,
  `tte` in meta) + `build_lmdb.py --split test_benchmark` → `paths.lmdb_test_benchmark`.
- Dataset Statistics table marked STALE pending the v2 regen re-pin (counts fixture relaxed to
  structural checks until then).

## v1.0.0 — Clean baseline (2026-06-09)

First standalone release. The codebase began as a behavior-preserving rebuild of an undergraduate
thesis project; this release completes that effort (P0–P9) and retires the rebuild scaffolding.

### Highlights
- Multimodal pedestrian behavior prediction on PIE (`actions` / `looks` / `crosses`) with a typed
  model registry (`full`, `motion_only`, `visual_only`, `vanilla_concat`).
- Config-driven throughout: yaml → typed dataclasses → `--set section.field=value` overrides. No
  hardcoded paths or hyperparameters; CSV-only tracking.
- Full pipeline: sequence generation → LMDB build → balance/augment → train (single-phase or
  scheduled) → evaluate → benchmark → visualize → ONNX export.
- Test + lint gate (`ruff` + `pytest -m "not slow"`) with golden characterization fixtures pinning
  per-module numerics.

### Cutover (P9)
- Retired the vendored legacy repo (`OLD/`) and the rebuild ledgers
  (`MIGRATION.md`, `REBUILD_SCHEMATIC.md`, per-prompt sub-plans). All are preserved in the
  **`legacy-archive`** git tag; the ledgers also live under `docs/archive/`.
- Flipped `CLAUDE.md` / `README.md` to standalone docs (removed the Rebuild Context section and the
  band-aid inventory).
- Stripped prompt-number provenance asides from module docstrings; repointed surviving references to
  `docs/archive/`.
- Golden fixtures reframed from "legacy parity" to "characterization"; their regenerators now require
  the `legacy-archive` tag (see `tests/_capture/README.md`).
- Added `docs/PHASE_B_BACKLOG.md` (architecture-redesign backlog) and
  `docs/archive/legacy_baselines.md`.

### Known limitations
- End-to-end per-`model_type` test-set metrics are not bundled (no trained weights / PIE data in the
  repo). Behavior preservation is established at the module level by the golden fixtures; regenerate
  end-to-end metrics with `scripts/evaluate.py` when data + weights are available.

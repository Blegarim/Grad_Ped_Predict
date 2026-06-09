# Changelog

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

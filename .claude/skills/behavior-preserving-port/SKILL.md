---
name: behavior-preserving-port
description: The repeated workflow for porting a module from the OLD repo into the new pedpredict layout during the Phase-A rebuild. Use whenever porting, migrating, or rebuilding any module per REBUILD_SCHEMATIC.md — data, model, loss, training, eval, viz, or export code. Trigger on "port X", "rebuild X", "migrate X", "implement prompt N.M", "capture golden output", "verify parity", "behavior-preserving", or any task that moves legacy code into src/pedpredict/. Governs HOW to port safely; combine with debugging-playbook for failures and experiment-tracking for run conventions.
---

# Behavior-Preserving Port

The Phase-A rebuild ports ~25 modules from the OLD repo into the new `src/pedpredict/` layout.
Every port follows the **same loop**. This skill is that loop. Do not skip the golden-fixture step —
it is the only thing that makes "behavior-preserving" verifiable rather than aspirational.

**OLD repo (read-only reference):** `OLD/Undergrad_thesis_project` (vendored into this repo; golden reference samples in `OLD/golden/`)
**Contract:** the new module must produce numerically equivalent outputs (within float tolerance) to the
legacy module for the same inputs and weights — UNLESS a listed band-aid (see CLAUDE.md table) changes
behavior, in which case the change must be called out and justified.

## The Port Loop (do these in order)

```
1. LOCATE   → find the source in the OLD repo (paths are in the schematic prompt)
2. CAPTURE  → snapshot a golden fixture from the OLD code BEFORE writing anything new
3. PORT     → implement into the target file(s); config-first, no hardcoded paths
4. VERIFY   → parity test new output vs golden fixture within tolerance
5. RESOLVE  → remove the band-aids the prompt lists; flag any intentional behavior change
6. RECORD   → update MIGRATION.md row + add the test to tests/
```

### 1. Locate
Read the source files named in the schematic prompt from the OLD repo. Read them fully (per
code-review-protocol) before porting. Note any coupled decisions the prompt flags.

### 2. Capture the golden fixture (BEFORE porting)
Run the OLD code with a fixed seed and fixed inputs; save inputs + outputs so the new module can be
diffed against them. Store fixtures under `tests/fixtures/golden/<module>.pt` (or `.npz`).

```python
# scripts run against the OLD repo
import torch
torch.manual_seed(0)
inputs = {...}                      # deterministic dummy or a tiny real sample
out = legacy_module(**inputs)       # or legacy_model(...) with fixed weights
torch.save({"inputs": inputs, "outputs": out, "seed": 0,
            "src": "<old/path.py>", "tol": 1e-5}, "tests/fixtures/golden/<module>.pt")
```

- For **models**: also save the **weights** used (`state_dict`) so the new model can load identical
  weights — parity is meaningless with different initializations.
- For **data**: capture exact tensor shapes/dtypes and a few sample values, plus label counts.
- Record the tolerance you'll assert at, in the fixture metadata.

### 3. Port
Implement into the target file(s) from the schematic layout. Always:
- **Config-first**: every constant/hyperparameter/path comes from the dataclass schema + yaml — never
  hardcode. Magic constants (`MAX_SEQ_LEN`, `motions[...,:8]`, `context_scale`, etc.) become config fields.
- **Typed**: type hints on public signatures; model selection via the typed `registry`, not strings.
- Keep the math identical to legacy except for the band-aids the prompt explicitly resolves.

### 4. Verify parity
Add a golden-output test that loads the fixture and asserts equivalence:

```python
def test_golden_<module>():
    fx = torch.load("tests/fixtures/golden/<module>.pt")
    new_out = new_module(**fx["inputs"])          # models: load fx weights first
    for k in fx["outputs"]:
        torch.testing.assert_close(new_out[k], fx["outputs"][k],
                                   rtol=fx["tol"], atol=fx["tol"])
```

If a key intentionally differs (a resolved band-aid), assert the *new* expected behavior and document why.

### 5. Resolve the listed band-aids
Each prompt names the band-aids it fixes (B1–B13, see CLAUDE.md). For each:
- If the fix is **behavior-neutral** (e.g. B13 residual cleanup, B6 config dedup, B8 dtype centralizing) →
  parity must still hold exactly.
- If the fix **changes behavior** (e.g. B2 eager params, B4 gating `crosses_pooled`, B7 removing a silent
  truncation) → call it out explicitly in MIGRATION.md and justify; the golden test asserts the new behavior.
- **Never silently keep dead compute** and never silently change numbers.

### 6. Record
Add a MIGRATION.md row (see its template) and commit the test alongside the module.

## Coupled Decisions — Keep Singular

When porting one of a coupled set, honor the decisions already made in its siblings:
- **Imbalance policy** (Prompts 1.3 offline balance / 1.6 sampler / 3.1 loss weights) — one documented
  policy, one metadata scan. See CLAUDE.md "Imbalance Policy".
- **Output contract** (Prompts 2.3 / 2.4 / 2.5 / 3.1 / 3.2 / 5.1) — output dict keys
  `actions, looks, crosses_pooled, crosses_frame, temporal_weights`; only `crosses_frame` is supervised.
- **Motion channels** (Prompts 1.2 writer / 1.4 flip-negation / 2.2 motion_enc) — the 8-dim definition and
  the flip-negated index must agree across all three or augmented data corrupts silently.

## Common Pitfalls

- Porting before capturing the fixture — then there's nothing to verify against. Capture FIRST.
- Comparing model outputs with mismatched weights — load the OLD `state_dict` into the new module.
- Asserting too tight a tolerance after a legitimate AMP/dtype reorder — pick tolerance deliberately, record it.
- Treating a behavior change as neutral — if counts/outputs move, it's a flagged change, not a refactor.
- Re-deriving this loop from scratch each session — follow the steps, update MIGRATION.md as you go.

## See Also

- `REBUILD_SCHEMATIC.md` — the per-module prompts (source paths, band-aids, deliverables).
- `CLAUDE.md` — architecture, output contract, dataset stats, band-aid table, imbalance policy.
- `experiment-tracking` — run-dir / CSV / checkpoint conventions for the new repo.
- `debugging-playbook` — when a port fails or outputs drift unexpectedly.

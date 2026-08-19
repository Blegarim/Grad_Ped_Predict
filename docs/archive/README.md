# Archive

Retired documents. **Nothing here is a plan, a spec, or a source of truth** — each file states something
that was true when it was written and has since been superseded, completed, or abandoned. Every file
carries a ⛔ RETIRED banner explaining what replaced it.

Kept rather than deleted because the *reasoning* is often still worth reading even when the *conclusion*
has moved on: why a decision went the way it did, what alternatives were rejected, what a number used to
be. Deleting that turns a settled question back into an open one.

**The live docs are:** [`../../CLAUDE.md`](../../CLAUDE.md) (contracts + orientation) ·
[`../../README.md`](../../README.md) (repo overview) · [`../../setup.md`](../../setup.md) (runbook) ·
[`../THESIS_ROADMAP.md`](../THESIS_ROADMAP.md) (tracker) · [`../METHODOLOGY.md`](../METHODOLOGY.md)
(the method) · [`../../outputs/runs/RESULTS_MATRIX.md`](../../outputs/runs/RESULTS_MATRIX.md) (the numbers).

## Retired in the 2026-08-19 doc consolidation

| File | Was | Why retired | Where its live content went |
|---|---|---|---|
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | The thesis plan (June 2026) — RQ1–RQ6, WP0–WP4, 12-month schedule | Spine superseded by the July pivot and again by the August reframe; demoted twice by prepended banner, body never rewritten, so several claims are now factually wrong | RQ list → [`../THESIS_ROADMAP.md`](../THESIS_ROADMAP.md) § Supporting studies |
| [`HOLE_AUDIT.md`](HOLE_AUDIT.md) | Two-pass engineering audit of v1.0 — every validity/architecture/correctness hole, each with a resolution | Fully executed; its Final attack order is complete | Decisions → [`../../CLAUDE.md`](../../CLAUDE.md) (data contract, imbalance policy, validity rules). Still the long-form *why behind the why* |
| [`PROPOSAL.md`](PROPOSAL.md) | Plain-language research proposal (June 2026) | Describes the pre-pivot thesis; the only doc that never got a pivot banner | Superseded by the brief + roadmap + methodology. Kept as raw material for a thesis intro |
| [`PHASE_B_BACKLOG.md`](PHASE_B_BACKLOG.md) | Architecture-redesign backlog (Phase A cutover) | Superseded by the hole audit in June 2026; kept "for reference only" ever since | Two surviving items → [`../THESIS_ROADMAP.md`](../THESIS_ROADMAP.md) § Remaining spoke work |
| [`CHANGELOG.md`](CHANGELOG.md) | Repo changelog | Abandoned at 2026-06-12; two months of work unlogged | Not replaced — `git log` + the roadmap carry the history at both altitudes |

## Retired earlier (P9 cutover, 2026-06-09)

The behavior-preserving-rebuild scaffolding. Also preserved in the **`legacy-archive`** git tag, which is
what the golden-fixture regenerators require.

| File | Was |
|---|---|
| [`MIGRATION.md`](MIGRATION.md) | The port ledger — every module moved from the undergrad prototype, with its band-aid inventory |
| [`REBUILD_SCHEMATIC.md`](REBUILD_SCHEMATIC.md) | The phase/prompt plan that drove the rebuild |
| [`plans/`](plans/) | Per-prompt sub-plans (evaluation, inference) |
| [`legacy_baselines.md`](legacy_baselines.md) | Placeholder for legacy end-to-end metrics (never filled — no weights/data at cutover) |

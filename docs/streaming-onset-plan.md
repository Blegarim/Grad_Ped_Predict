# Streaming Crossing-Onset — Reformulated Plan of Attack

**Prepared:** July 2026 · **Companion to:** [`project-context-streaming-crossing-onset.md`](project-context-streaming-crossing-onset.md) (the self-contained research brief — read that first for the *why*, the ruled-out dead ends, and the reviewer objections).

> **What this document is.** The brief's §5 "plan of attack" is written model-agnostically ("take 2–3
> published models"). This document re-grounds it against *this repository* and the two-machine
> execution reality: what is already built, what is a seed, what is net-new, and the order that
> exploits the code we have. It is a **full spine-pivot**: the anchored→streaming decomposition becomes
> the thesis, and the [RESEARCH_PLAN.md](RESEARCH_PLAN.md) RQs are re-slotted as support (not discarded).
> RESEARCH_PLAN.md, [HOLE_AUDIT.md](HOLE_AUDIT.md), and CLAUDE.md still describe the older
> architecture-ablation framing and are pending reconciliation (see §8).

---

## 1. The core reframe: we already built one half of this thesis

The existing pipeline **is** the streaming detector the brief proposes. In
[`pie_sequences.window_track`](../src/pedpredict/data/pie_sequences.py) (standard mode):

- dense sliding window at `stride` over *all* tracks,
- filter #2 drops windows where crossing is already underway during observation,
- M4 drops right-censored windows,
- `crosses = any(crosses[end : end + future_offset + tol])`.

Because obs-time crossings are dropped, every retained positive **is** an onset-in-horizon label —
exactly §2.1's streaming formulation, and the ~37:1 ratio is §2.2's true base rate. This recontextualizes
the entire HOLE_AUDIT imbalance saga: the M1 triple-stack collapse (run `20260616_153511_full`: AUC
~0.57, `crosses` F1 0.081, effective-crosses ~89%) is **§2.3's "raw 37:1 collapses a from-scratch,
data-hungry ViT," not a bug to tune away.** The friction was the finding.

The anchored side already has a seed:
[`pie_sequences.window_track_benchmark`](../src/pedpredict/data/pie_sequences.py) (M5) is a
Kotseruba-faithful anchored **eval** set — fixed-TTE windows, labeled by the crossing event, ~2.5:1.
That is one column of the 2×2 matrix already coded.

## 2. Asset map (grounded in code)

| Need for the streaming thesis | Status | Where / gap |
|---|---|---|
| Streaming sampler (dense, onset-in-horizon) | **In hand** | `window_track`, standard mode |
| Anchored *eval* set (Kotseruba-faithful) | **Seed** | `window_track_benchmark`; CLI `make_sequences.py --benchmark --split {train,val,test}` already supports any split |
| Anchored *training* set (train/val under anchored protocol) | **Wired (code done)** | `data.protocol={streaming,anchored}` switch (via `paths.protocol_lmdb_dirs`) repoints train+val+test at the `lmdb_{train,val}_benchmark` dirs together across train.py / ChunkPrefetcher / evaluate.py; `make_sequences.py --benchmark --split {train,val}` + `build_lmdb[_incremental].py --split {train,val}_benchmark` produce the pkls/LMDBs. **Lab-PC data+train pass remains.** Non-crosser negatives are **already handled**: PIE gives every ped a `crossing_point` anchor + `activities=0` (`_get_all`, pie_data.py:1280-1308), which `windows_from_pie_benchmark` reads → sanity-check via the `(P event-positive)` count `make_sequences` prints |
| Per-window onset info in *standard* records | **DONE (S1, this change)** | `onset_offset` / `future_observed` / `track_crosses` — see §3 |
| Horizon-H as a runtime knob (no re-gen per H) | **DONE (S1)** | derivable from `onset_offset` + `future_observed` |
| Negative typing (genuine / hard-temporal / already-crossed / junk) | **Partial (S1)** | crossing-based types from S1; **junk** (occlusion/far/low-info) needs bbox-size / occlusion signals — net-new |
| ODAS/OAD metrics (per-frame mAP, cAP/mcAP, point-level AP w/ offset tolerance) | **Net-new** | only F1/AUC/P/R exist today; onset-frame ground truth for p-AP is now recoverable from S1 |
| Prior correction / recalibration at true prior | **Seed** | M2 val-tuned thresholds; needs an explicit true-prior correction layer |
| Curated streaming training (pretrained backbone, focal, hard-neg mining, junk filter) | **Seed** | RQ1 backbone study ([BACKBONE_STUDY.md](BACKBONE_STUDY.md)) + RQ3 levers already scoped; recast as the G_hardneg recipe |
| Published baselines (PCPA, SF-GRU, transformer) | **Net-new, biggest risk** | see §7 — de-scopable stretch |

## 3. The S1 change (landed in this branch)

`SequenceRecord` (standard windows only) now carries three **pure per-window annotations** that do
**not** change the `crosses` label or the emitted-window population — they only annotate:

- **`onset_offset`** — frames from end-of-observation to the first future crossing frame; `-1` if the
  pedestrian never crosses again within the observed track. Horizon-H streaming label is
  `0 <= onset_offset < H`.
- **`future_observed`** — frames of future actually observed after the window (`n - end`). A window is
  usable for horizon H only if `future_observed >= H` or a crossing was already seen — this is what
  makes the **H-sweep rigorous against right-censoring** (objection #2's rebuttal needs a clean sweep).
- **`track_crosses`** — 1 if the track ever crosses (before or after the window). Separates a genuine
  non-crosser (`0`) from a will-cross / already-crossed negative.

**Why this matters and why it's cheap.** The H-sweep *and* the negative-composition analysis both need
per-window onset info that the old binary label didn't localize — so a standard-sequence re-gen was
forced regardless. S1 folds everything the streaming thesis needs into that one pass. And because these
are pure functions of the window, they **ride the sequence pkls** (fast regen — no JPEG re-encode) and
join to *test* predictions by index (the test set is unaugmented, so 1:1 order holds). **No LMDB schema
rebuild is required for the eval-side H-sweep / negative-typing.** (Optional: thread through
`transforms.ProcessedSample` + `lmdb_writer.pack_meta`, mirroring `track_id`/`tte`, only if a
LMDB-only consumer such as track-aggregation needs them — that would need a full crop rebuild, so defer
unless required.)

Tests: `tests/test_data_shapes.py::test_onset_*` cover genuine-non-crosser, in-horizon positive,
hard-temporal-negative (crosses==0 but onset beyond canonical window), and already-crossed negative.
`ruff` + `pytest -m "not slow"` green.

## 4. Reformulated phased plan (reordered for this repo + hardware)

Legend: 💻 personal PC (code, no data) · 🖥️ lab PC (needs PIE data / GPU).

**Phase 0 — Kill-check** 💻 *(done, this session).* No published work applies OAD/ODAS streaming
metrics to PIE/JAAD *crossing intention*. One nearest-neighbor to differentiate in write-up:
[Temporal-contextual Event Learning (arXiv 2504.06292)](https://arxiv.org/abs/2504.06292), plus the
brief's existing neighbor *Coupling Intent and Action* (arXiv 2105.04133). Runway clear.

**Phase A — One final data pass** 🖥️. The forced last sequence-gen: regenerate standard pkls with S1
fields (this change); generate anchored train/val via `--benchmark --split {train,val}` and build their
LMDBs (`build_lmdb_incremental.py --split train_benchmark`, `build_lmdb.py --split val_benchmark`);
**verify non-crosser negatives** via the `(P event-positive)` count `make_sequences` prints so
anchored-train isn't thinned by early-anchor drops; re-pin stats + `count_labels` gate + `test_stats`
fixture. Fold in any still-open WP0 rebuild items. **The train/eval wiring (the `data.protocol` switch)
is already merged — Phase A is now purely the lab-PC data build + the Phase B train.** Runbook: [setup.md §9](../setup.md).

**Phase B — Own-model de-risk (brief Phase 1, regrounded)** 🖥️. Train **our** model **anchored** —
mild 2.5:1, pretrained-backbone-friendly, so it should actually converge (and finally give a working
model). Evaluate it on anchored **and** streaming test sets. Does precision collapse at the true prior,
and does recalibration recover it? **First read on G_prior in days.** The anchored model is
simultaneously the easy-to-train baseline *and* the comparison arm — we do not have to first fix the
collapsing streaming-trained model.

**Phase C — The streaming leg (pulls brief Phase 5's recipe early)** 🖥️. Get a *streaming-trained*
model to actually work: pretrained backbone (RQ1), focal / class-balanced loss, hard-negative emphasis,
junk filtering. Gates G_hardneg; absorbs RQ1 + RQ3. **Graceful degradation:** if this stays hard, we
still have G_prior from Phase B — the decomposition reports "G is mostly prior" (the brief's
demote-to-cautionary-paper branch), it does not collapse.

**Phase D — Metrics + negative composition (brief Phase 2)** 💻 code → 🖥️ apply. Implement the
ODAS/OAD metric suite (per-frame mAP, cAP/mcAP, point-level AP with offset tolerance) — unit-testable on
synthetic streams locally. Build the negative-composition report from S1 (+ a junk signal). Both are
unpublished sub-contributions.

**Phase E — Matrix + decomposition (headline)** 🖥️. {train, test} × {anchored, streaming} on **our
model first**; report G = G_prior + G_hardneg. Published baselines are **stretch/v2** (§7).

**Phase F — Supporting + constructive close** 🖥️. H-sweep (near-free via S1) → objection-#2 rebuttal;
ego-speed-under-streaming (RQ4 leakage probe reconnected); ship the streaming ODAS-style protocol +
curated recipe as the community artifact.

## 5. Old RQs re-slotted (nothing wasted)

| Old RQ (RESEARCH_PLAN) | Fate under the streaming spine |
|---|---|
| RQ3 imbalance levers | **Elevated → the point.** 37:1 is the true prior; the lever study becomes the curated recipe (Phase C) that narrows G_hardneg. |
| RQ6 calibration / thresholds (M2, M10) | **Elevated → half the headline.** Prior correction / recalibration *is* G_prior. |
| RQ1 backbone swap (TinyViT etc.) | **Absorbed into Phase C** — a pretrained backbone is what makes curated 37:1 training converge. |
| RQ4 ego-speed | **Named supporting finding** (Phase F: does streaming lean *harder* on the shortcut?). |
| RQ2 fusion / RQ4 motion-norm / RQ5 efficiency | **Demoted to within-model support** — our model is "one model in the matrix." |

## 6. Staged data-contract work & open forks (for the lab PC)

1. **Standard pkl regen with S1** — code ready; runs on lab PC in Phase A.
2. **Anchored train/val** — **wired.** `data.protocol` + `lmdb_{train,val}_benchmark` paths +
   `paths.protocol_lmdb_dirs` route train.py / ChunkPrefetcher / evaluate.py at the benchmark dirs;
   `build_lmdb[_incremental].py --split {train,val}_benchmark` build them. **Non-crosser negatives:**
   PIE assigns every pedestrian a `crossing_point` (a valid in-track frame — the decision/closest-approach
   moment for non-crossers) and an `activities` flag (`_get_all`, PIE/utilities/pie_data.py:1280-1308);
   `windows_from_pie_benchmark` builds `crosses=0` windows for them. Remaining lab-PC step is the
   **count sanity-check** that early-anchor drops (`test_benchmark_early_event_yields_no_windows`)
   don't thin the negatives too far — read the `(P event-positive)` line `make_sequences --benchmark` prints.
3. **LMDB threading of S1 (optional)** — only if a LMDB-only consumer needs onset fields; costs a full
   crop rebuild, so default to the pkl-sidecar join instead.
4. **Junk signal for negative typing** — pick a definition (bbox height below a threshold / occlusion
   flag / low future_observed) and store or compute it; needed for the §2.3 three-way composition.

## 7. Execution risks specific to this setup (not a re-stress of the idea)

1. **Published-baseline retraining is the schedule killer.** PCPA/SF-GRU expect their own preprocessing
   (VGG/pose features); integrating each into our PIE pipeline on the lab PC is multi-week, uncertain
   work. **The decomposition on our own model alone is a defensible thesis + workshop paper;** the
   multi-model matrix is the reach for a top venue, explicitly de-scopable (mirrors the RESEARCH_PLAN
   risk row).
2. **G_hardneg depends on a streaming model we have not yet made converge.** The brief treats the
   curated recipe as a closing flourish; here it is load-bearing for half the headline. Phase B/C
   ordering (G_prior first, independent of the streaming model) is the mitigation.
3. **ODAS point-level AP needs onset-frame ground truth** — now recoverable from S1; another reason
   the S1 pass is a prerequisite, not a nice-to-have.

## 8. Doc reconciliation (follow-up, not blocking)

Full pivot confirmed → these need updating once the plan is locked: RESEARCH_PLAN.md (spine = streaming
decomposition; RQs re-slotted per §5), CLAUDE.md (problem framing + the v2 labeling contract gains an
S1 line), HOLE_AUDIT.md (add a note that the imbalance chain is now the streaming base-rate finding).
Left for a dedicated reconciliation pass to avoid churning three anchor docs mid-design. (Partial: the
`data.protocol` switch is now documented in [setup.md §9](../setup.md) and CLAUDE.md's Data Pipeline /
Evaluation sections.)

**Deferred code follow-up (Phase E hygiene) — RESOLVED (July 2026).** `thresholds.json` is now keyed by
`data.protocol` (`thresholds_{streaming,anchored}.json`) in `eval/evaluate.py`
`save_thresholds`/`load_thresholds`, so evaluating one checkpoint under both protocols writes two files
instead of the second val pass overwriting the first's thresholds. The legacy `thresholds.json` is still
read as a fallback so pre-change runs keep loading. (Runs eval'd before this change kept only the
last-calibrated protocol's file on disk, but their `tuned_*` CSV columns were captured inline at eval
time and remain valid.)

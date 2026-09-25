# Paper draft

Draft of the streaming-vs-anchored protocol paper. IEEE conference format
(`IEEEtran`), targeting IV / ITSC.

Current state: 9 pages with the inline TODO boxes rendered, 8 with them
disabled. A typical IV/ITSC limit is 6–8, so roughly one page of trimming is
owed once the outstanding experiments land and the boxes come out.

## Build

```bash
cd paper/figures && ../../.venv/Scripts/python.exe make_figures.py
cd .. && max_print_line=1000 latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
./check_numbers.sh
```

Both steps are needed after any change to `NUMBERS`. `latexmk -C` cleans.
`max_print_line=1000` stops TeX wrapping the result-status warnings that
`check_numbers.sh` reads; without it the longer tags are truncated.

**Gate before considering a build good:** zero errors, zero unresolved
references *in the final pass* (`grep -c "undefined on input" main.log` — not
`build.log`, whose early passes always show some), and zero overfull boxes.
`./check_numbers.sh` reports all four at once.

To hide the TODO boxes for a clean read, swap the `todonotes` options for
`[disable]`.

## Unfilled results

Results are still landing, so every number in `main.tex` carries one of three
states, defined in the "result-status machinery" block of the preamble:

| Macro | Meaning | Renders as | Log trace |
|---|---|---|---|
| `\seedconf{v}` | measured, confirmed across seeds | the value | silent |
| `\oneseed{v}{tag}` | measured on one seed; the seed mean supersedes it | blue value + † | `PROVISIONAL` |
| `\pending{tag}` | not measured yet | red `[tag]` | `PENDING` |

`./check_numbers.sh` counts each from `main.log` and lists what is outstanding;
`--strict` makes it exit nonzero. Setting `\draftnumbersfalse` in the preamble
hides the marks for a clean read and turns any surviving `\pending` into a hard
compile error, so the paper cannot be submitted with a hole in it.

Values live in the `NUMBERS` block at the top of `main.tex`, each with its run
id and source file in a trailing comment. Nothing in the body hardcodes a number.
Earlier sections (§III–§VI) still carry their numbers inline from before this
convention; migrating them is outstanding.

## Where the numbers come from

Every measured value in the figures lives in the `NUMBERS` dict at the top of
[`figures/make_figures.py`](figures/make_figures.py); nothing is hardcoded
elsewhere in the figure code. Table values live in `main.tex` directly.

| Quantity | Source |
|---|---|
| Window population per split | `CLAUDE.md` § Dataset Statistics (re-pinned 2026-08-19) |
| Negative composition, time-to-onset buckets, horizon sweep | `scripts/report_negative_composition.py`, re-run 2026-09-09 → [`composition_report.json`](composition_report.json) |
| Cross-protocol matrix, G-decomposition | `outputs/runs/RESULTS_MATRIX.md` (Models A and B) |
| Look-ahead sizing | `docs/METHODOLOGY.md` § How far ahead the head looks |

When a new eval pass lands, update `RESULTS_MATRIX.md` first (it is the ledger),
then mirror into `main.tex`.

### Confusable-band definition (resolved 2026-09-09)

The band is **windows whose onset falls in the 64 frames (≈2.1 s) immediately
after the horizon**. At `H=32` that is 4,186 train windows, 1.65× the positive
set. The earlier 3,955 figure in `METHODOLOGY.md` used a 60-frame band; the
~4,200 figure summed histogram buckets running to 96 frames. The sweep in
`main.tex` Table III was regenerated at `--band-frames 64` so the table, the
figure, and the prose now use one definition. Verified against cumulative counts
at 32/64/118 frames: 2,220 / 4,186 / 7,074, which reproduce the published
buckets exactly (2,220 · 1,966 · 2,888 · 15,560).

## Outstanding items

Run `./check_numbers.sh` for the authoritative list; it reads the build log. One
`\todo` box remains (planned ablations).

**Closed 2026-09-24/25 — the seed plan completed, every job succeeded:**

1. ~~Populate the onset table~~ — all four configurations trained, evaluated and
   dumped; three seeds each for the binary baseline and the pure hazard arm.
2. ~~Multi-seed the primary comparison~~ — done, and it **is** the paper's
   result: the pre-registered criterion is met at **0 of 5 budgets**, so the
   objective comparison is reported as inconclusive. Both objectives vary more
   across their own seeds than they differ from each other. A single seed
   reports 3.1×.
3. ~~Predictability-limit measurement~~ — computed from the dumps. Separability
   decays 0.78 → 0.52 AUC from the first second out past ten seconds,
   identically for both objectives. Now in §VIII-A.
4. ~~Verify the Yao et al. sampling characterization~~ — confirmed against
   arXiv:2105.04133 §4.2. Their settings are named "original data" (whole track)
   and "event-to-crossing" (anchored), and they prefer the latter because early
   windows show little action change. §II-A now uses their terms.
5. ~~`last.pth` checkpoint-robustness check~~ — **run, then discarded.** All
   three hazard `last.pth` read-outs are collapsed (spread 0.19 / 0.10 / 0.03
   against 0.41–0.79 at `best.pth`; r3_s44_last sits under the project's own
   0.05 dead-head gate), so those checkpoints break the read-out's
   comparability with a binary head. Their apparently lower seed variance is
   consistent with uniform head collapse. Dumps kept under
   `outputs/diagnostics/*_last/` for the record; **no result rests on them.**

**Still open:**

6. **Matched-size streaming control** (~4.9k windows). One training run. The only
   `\pending` left in the build, and the confound a reviewer attacks first in §V.
7. **OAD metric suite** — calibrated AP, point-level AP with temporal tolerance.
   Implementation, not a run. Optional now the detection curve is primary and
   reported with seed spread.
8. Optionally add two or three more recent PIE methods to §II-A.

**Not in the paper, by decision:** the recipe-v2 detour, the feature-cache
rebuild, the image-mode probe, BatchNorm-drift work. Repo history, at most a
thesis appendix. Collapse and seed-dependence appear only as a single
threats-to-validity paragraph, never as a results subsection.

**Closed 2026-09-09:** confusable-band inconsistency (above); all §II citations
verified against publisher/arXiv records; the OAD transformer line entered
(OadTR, LSTR, TeSTra, MAT); the censoring novelty check run — see below.

### Novelty check outcome

A targeted search found that survival analysis **has** been applied to
pedestrian behavior: Kalatian & Farooq, *DeepWait* (ITSC 2019), estimate
waiting time at unsignalized crosswalks with a Cox model and a neural log-risk
function. It uses virtual-reality participant data, not dashcam video, and
predicts waiting duration rather than detecting onset in a stream. It is now
cited and differentiated in §II-C. Discrete-time neural survival estimation is
cited via DeepHit. No prior work was found applying censored time-to-event
modeling to crossing-onset detection under the PIE protocol, but the search was
targeted rather than systematic, and §II-C is phrased to hold either way.

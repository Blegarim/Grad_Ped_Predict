# Paper draft

Draft of the streaming-vs-anchored protocol paper. IEEE conference format
(`IEEEtran`), targeting IV / ITSC.

Current state: 9 pages with the inline TODO boxes rendered, 8 with them
disabled. A typical IV/ITSC limit is 6–8, so roughly one page of trimming is
owed once the outstanding experiments land and the boxes come out.

## Build

```bash
cd paper/figures && ../../.venv/Scripts/python.exe make_figures.py
cd .. && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Both steps are needed after any change to `NUMBERS`. `latexmk -C` cleans.

**Gate before considering a build good:** zero errors, zero unresolved
references *in the final pass* (`grep -c "undefined on input" main.log` — not
`build.log`, whose early passes always show some), and zero overfull boxes.

To hide the TODO boxes for a clean read, swap the `todonotes` options for
`[disable]`.

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

The draft carries its own to-do list as `\todo` boxes, which render inline in
the PDF. The pre-submission checklist is at the end of the Conclusion.

**Blocked on the lab PC (compute):**

1. **Populate Table VI** — the onset-timing configurations. Implemented and
   unit-tested, never trained. Needs: metadata backfill, a short run to confirm
   the read-out distribution is not degenerate, then the auxiliary and pure
   configurations.
2. **Matched-size streaming control** (~4.9k windows). One run. Converts the
   headline claim from "consistent with" to "demonstrated"; this is the
   confound a reviewer attacks first.
3. **Multi-seed** the cross-protocol matrix (3 seeds, mean ± std).
4. **Predictability-limit measurement** — stratify separability by time to
   onset on any trained model. Cheap once anything trains, and no prior work in
   the area appears to report it.

**Doable on this machine:**

5. **Verify the Yao et al. sampling characterization** against the paper body.
   Title/authors/venue are confirmed (IJCAI 2021); the specific claim about
   their two sampling settings comes from project notes, not the paper.
6. **Adopt the OAD metric suite** — calibrated AP and point-level AP with
   temporal tolerance. Implementation work, not a run. Should land before the
   method results, or the arms cannot be compared at 34:1.
7. Optionally add two or three more recent PIE methods to §II-A.

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

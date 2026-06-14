# Research Proposal — Will the Pedestrian Cross?

**Nguyen Bao Viet · Master's research · June 2026**
*Companion plan: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) · Engineering audit: [`HOLE_AUDIT.md`](HOLE_AUDIT.md)*

---

## The one-paragraph version

A self-driving or driver-assist car has to answer one question, over and over, in a fraction of a
second: *is that person about to step into the road?* This project builds and studies a model that
watches a short clip of dashcam video and predicts, for each pedestrian, three things — whether they
are **walking**, whether they are **looking at traffic**, and, most importantly, whether they will
**cross** in the next second. I already have a working prototype. The research is about making it
*trustworthy*: getting the rare, dangerous "they will cross" case right, and making the model's
confidence mean what it says. The intended outcome is a thesis and a conference/workshop paper.

---

## Why this is hard (the relatable version)

1. **The dangerous case is rare.** In normal driving, almost nobody is about to cross right now.
   In our data, only about **1 in 38** moments is a real "will-cross." A model can score 97% accuracy
   by always saying "won't cross" — and be useless exactly when it matters. So accuracy is the wrong
   scoreboard; we measure how well it catches the rare crossings without crying wolf.

2. **Fixing rarity can quietly break honesty.** The standard trick is to show the model far more
   crossing examples during training than really occur. Do that too aggressively and the model starts
   thinking crossings are *common*, so it over-predicts and its "90% sure" no longer means 90%. Our
   prototype did exactly this — and then, worse, it had been *tuning its decision cutoff on the test
   set*, which hides the problem instead of revealing it. **The headline numbers were both too rosy
   and self-concealing.** Fixing this honestly is the spine of the thesis.

3. **A decision is only useful if you can trust the confidence.** A downstream system (braking,
   planning) doesn't want a yes/no — it wants a *number it can rely on*: "80% likely to cross." That
   only works if, among all the "80%" calls, about 80% really do cross. Nobody had ever checked
   whether ours does. (This is also the bridge to a possible control-systems follow-up.)

4. **You have to compare fairly, on a budget.** The model has many moving parts (the vision module,
   the motion module, the way they're combined, the imbalance settings). Testing every combination
   would be dozens of training runs we can't afford on one GPU. And many "samples" in the data are
   near-duplicate video frames of the same person, so naive scores look more certain than they are.

---

## What I'm actually asking (research questions)

| # | Question | Plain-language stake |
|---|---|---|
| **RQ1** | Does a modern, pretrained vision backbone beat our hand-built one? | The current one was trained from scratch on a small dataset and has an odd internal design; pretrained models usually win. |
| **RQ2** | Are the two information streams (appearance + motion) combined the right way? | Surprisingly, in the current design the motion stream only *steers attention* — its content never directly reaches the decision. Is that wise? |
| **RQ3** | What's the right way to handle the rare-crossing imbalance? | The crux. Decide it by measurement, not by stacking three half-understood tricks. |
| **RQ4** | Do better motion features help — including the car's own speed? | Current features throw away *where* the person is. The car's speed is a strong cue (drivers brake for crossers) — and a fascinating "is this cheating?" probe. |
| **RQ5** | Is the full multimodal model worth its cost? | Measure accuracy vs. size/speed/memory; maybe a simpler model is enough. |
| **RQ6** | Are the model's probabilities honest, and what's the single decision cutoff? | Make the confidence trustworthy and pick one principled threshold used everywhere. |

---

## How I'll answer them (the approach)

**Foundation first — make every comparison fair.** Before any experiment: seed everything so runs
are reproducible, pick the best model by the metric we actually care about (not a noisy proxy), tune
the decision cutoff on validation data (*never* the test set), and add an instrument that *reports the
true training mix* so we stop guessing. **This part is already built and committed.**

**One clean data rebuild.** Several fixes all touch the dataset (correcting labels, dropping moments
whose future we never actually observe, repairing the motion features, adding the car's speed, and
tagging each person so we can score per-*pedestrian* not per-frame). Rather than rebuild repeatedly,
they're batched into **one** rebuild.

**Hub-and-spoke experiments, not brute force.** Establish one solid baseline (the "hub"). Then every
experiment changes **exactly one thing** and is compared straight back to the hub — never against each
other. Screen ideas with one run; confirm the few finalists with three runs and report the spread.
Total: about **20–25 training runs**, which fits the hardware. If two improvements look like they
stack, I test that *one* combination — I don't search the whole grid.

**Then make it honest.** Plot whether the confidences are calibrated, correct them if not
(temperature scaling), and lock in one decision threshold used by both evaluation and live video
inference.

---

## Scope (what's in, what's out)

**In:** the PIE dashcam dataset; the three behaviors (walking / looking / crossing, with crossing as
the priority); the imbalance and calibration study; the backbone and fusion redesigns; corrected
motion features + ego-speed; a full ablation + efficiency comparison; an extra "literature-comparable"
evaluation so results can be placed against published work.

**Out (deliberately):** predicting *where* the pedestrian will walk (we do yes/no, not trajectories);
extra datasets beyond PIE (a JAAD sanity-check only if time allows); real-time in-car deployment (we
use ONNX export + speed benchmarks as a stand-in); and detection/tracking research (an off-the-shelf
detector handles that part).

---

## Why it's feasible

- The codebase is already clean, tested, and reproducible — a year of engineering is done.
- The validity fixes that make results trustworthy are **already implemented**.
- The experiment budget is sized to the actual hardware (one A4500 GPU).
- Even a *negative* result is publishable here: "which of these popular imbalance tricks actually
  help?" is a genuine, citable contribution because almost everyone stacks them blindly.

---

## What comes out of it (deliverables)

1. **A thesis** with an ablation-backed redesign of the model.
2. **A paper** (workshop/conference) — the strongest candidate angle is the honest imbalance +
   calibration story, since the field routinely over-claims here.
3. **An evidence-based imbalance recipe** and a **trustworthy-confidence (calibration) recipe** for
   rare-event prediction.
4. **A reusable, reproducible codebase** for students who come after.

---

## Rough timeline (12 months)

| Phase | Months | What |
|---|---|---|
| Baseline | 1–3 | Validity fixes (done) → one data rebuild → reference results |
| Data & imbalance | 3–6 | Imbalance study, motion features, ego-speed, fair-comparison metrics |
| Architecture | 5–8 | Pretrained backbone swap; fusion redesign |
| Calibration & wrap-up | 7–10 | Honest confidences; final multi-seed runs; efficiency + failure analysis |
| Writing | 10–12 | Thesis + paper draft |

---

## The headline, in one line

*Take a promising-but-overclaiming pedestrian-crossing model and turn it into an honest, calibrated,
properly-ablated one — and report, for the first time on this pipeline, which of the field's standard
imbalance tricks actually earn their place.*

# Pre-specification: what information decides crossing onset? (scene audit)

**Written 2026-10-06, before any scene feature has been computed.** Known beforehand, from the intention check of the
same day:
- 75% of test never-crossers have PIE human intention ≥ 0.5.
- The human intention rating separates crossers from never-crossers at AUC 0.739 per pedestrian; the tree model
  reaches 0.755.
- 80–90% of the tree's false alarms fall on high-intention never-crossers.

No scene, traffic or layout feature has been looked at.

## Question

Every change to the *model* has left streaming detection where it was. That includes the objective, the learner,
images, re-weighting and data volume. The hypothesis is that crossing in front of the car is decided mostly by
right-of-way: the light, the traffic and the driver. Our inputs never contain those. PIE annotates them in every
frame. Giving the same tree model that information, one group at a time, measures how much of the problem it
explains. If nothing moves, it measures a ceiling instead.

## Data and gate

The windows are the existing exports (train, val, test). Each window is matched to its video frame through the
sequence files: the export row equals the sequence record's row, which is **checked**, not assumed (`track_id`,
`crosses` and the three onset fields must match on every row). The last observed frame is t.

## Feature groups (computed at t; every deployable group uses only frames ≤ t)

- **R, right-of-way (deployable: a camera sees traffic lights).**
  - For the largest *regular* traffic light visible at t: its state (none / red / yellow / green); frames since its
    state last changed (from that light's own track up to t); whether it changed within the last 20 frames.
  - The number of regular lights visible at t, and whether any of them is red or green.
  - The same state, time since change and visibility for the largest *pedestrian*-type light.
- **T, traffic (deployable).**
  - The number of other vehicles at t.
  - The horizontal image distance from the pedestrian to the nearest vehicle, divided by image width.
  - That vehicle's box height, divided by image height, as a nearness proxy.
  - The change in that distance since t − 19.
  - The number of vehicles within 0.15 image widths.
- **S, layout (deployable with a map) and crosswalk (camera).**
  - The pedestrian's intersection type, number of lanes, signalization and traffic direction (PIE attributes).
  - Whether a crosswalk is visible at t.
  - The distance from the pedestrian's box bottom-centre to the nearest crosswalk box, divided by image width
    (0 when inside it).
  - Whether the pedestrian's centre lies within a crosswalk's horizontal extent.
- **H, oracle: human intention** (PIE `intention_prob`). Not deployable. An upper bound on "who".
- **E, oracle: the ego car's next second.** OBD speed at t + 15 and t + 30 minus the speed at t. Not deployable, and
  confounded: the driver slows *because* of the pedestrian. It bounds how much of the outcome is the interaction,
  and it is never presented as a method.

Missing values stay missing (NaN; the learner handles them natively).

## Models

All arms use the same learner, settings, training windows and label as tree-study arm **B**: binary, stored
`crosses`, train split, balanced weighting inherited from B, and seeds 42–46, each cut at its validation-best
iteration. Every arm's input is B's 232 summary features plus the named groups:

`base` (= B; it must reproduce B's test scores exactly), `R`, `T`, `S`, `RTS`, `H`, `E`, `RTS_H`, `RTS_E`, `RTS_HE`.

## Questions and how each answer will be read

The test is the paired pedestrian bootstrap of the tree study: 2,000 resamples, seeds resampled. A verdict needs 3 of
5 budgets, primary `per_window`, with `per_track` reported beside it. The 1200/hr `per_track` budget is degenerate
(every arm detects 100%) and is **excluded** from `per_track` verdicts.

- **Q1 (primary): RTS − base.** Does deployable scene information raise detection?
  - **Large** if the rule says RTS is better *and* the gap at 205/hr is at least 10 points.
  - **Small** if RTS is better but the gap is under 10.
  - **None** if equivalent within 5 points.
  - Otherwise **inconclusive**.
- **Q2: which group carries it.** R − base, T − base and S − base.
- **Q3: what the oracles add.** RTS_H − RTS (does human-level intent add beyond the scene?) and RTS_E − RTS (the
  interaction share). RTS_HE is reported as the bound: "even knowing all of this, detection at 205/hr is X".
- **Q4: by crossing type.** base and RTS on signalized pedestrians and on unsignalized ones separately
  (descriptive, with paired intervals).
- **Q5: how far ahead.** Separability by time to onset for base, RTS and RTS_HE (descriptive). Does right-of-way
  extend the horizon at which crossing is predictable beyond the current ~1–3 s?

Everything above is reported whatever it shows. Anything added after the numbers is labelled post hoc.

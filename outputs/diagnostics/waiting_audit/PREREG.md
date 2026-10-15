# Pre-specification: does waiting history move the timing ("when") factor?

**Written 2026-10-06, after the scene audit and before any history feature is computed.** This is an exploratory
follow-up, motivated by the scene audit's results. Its questions are fixed here so that its answers count as
measurements.

## Why

The scene audit (`../scene_audit/`) found:
- Scene layout lifts "who" (incidence) from 0.755 to 0.853.
- Nothing observable lifts "when" (latency: imminent vs later windows among crossers) beyond 0.805 → 0.845.
- The remaining false alarms sit on intending pedestrians waiting at crosswalks.

The model has only ever seen the last 20 frames (0.67 s). Studies of waiting times at crossings report that the chance
of stepping out rises with time already waited (impatience). The one piece of timing information no model has had is
therefore the pedestrian's own history: how long they have been standing, and how they have been looking.

## Features (per window, at its last observed frame t, from the pedestrian's own annotated frames ≤ t)

They come from PIE's per-frame behaviour annotations: `action` (standing / walking) and `look` (looking /
not-looking). These are observable in principle; here they are taken from the annotations, like the scene features.
The per-frame `cross` attribute is never used.

- **W, waiting history.**
  - Frames since the pedestrian was first annotated.
  - Standing at t (yes/no).
  - Length of the current standing bout.
  - Length of the current walking bout.
  - Total standing frames so far.
  - Number of walking-to-standing transitions so far.
- **L, looking history.**
  - Frames since the pedestrian last looked at traffic (NaN if never).
  - Looking frames in the last 90 frames (3 s).
  - Number of separate looking bouts so far.

## Arms

All arms share tree-study arm B's learner, settings, label, training windows, inherited balanced weighting and seeds
42–46:

- `base` (must reproduce B exactly);
- `W` (+ waiting);
- `WL` (+ waiting + looking);
- `RTS` (+ the deployable scene groups of the scene audit; must reproduce its RTS exactly);
- `RTS_WL` (scene + waiting + looking).

## Questions and readings

The primary outcome is the **latency factor**: the "when" AUC (among crossers' windows with the crossing still ahead,
onset within 32 frames vs later). It is averaged over seeds, with a 95% interval from resampling crossing pedestrians
(1,000 resamples).

- **H1 (primary): W − base on "when".** **Moves latency** if the gain is at least 0.02 *and* its interval excludes
  zero. **Small** if the interval excludes zero with a gain under 0.02. **No** otherwise.
- **H2: WL − W** (does looking add to waiting?) and **RTS_WL − RTS** (does behaviour history add beyond the scene?),
  on "when", with the same readings.
- **H3: detection** (context). The paired pedestrian bootstrap of the earlier studies on the detection curve, `per_window` and
  `per_track` (with 1200/hr excluded from `per_track`), for W − base and RTS_WL − RTS. "Who" and the time-to-onset
  separability are reported for every arm.

Everything is reported whatever it shows.

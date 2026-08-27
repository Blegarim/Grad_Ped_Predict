# Methodology — building a method for streaming crossing-onset

**Started:** 2026-08-18 · **Status:** active working reference.

## What this document is

The thesis needs a **method**, not just a measurement. This document holds the direction: what we are going
to build, why we think it will work, what other people have already tried, and what we are choosing not to
do. It is the day-to-day reference for the method work.

How it sits next to the other documents:

| Document | What it is for |
|---|---|
| [`project-context-streaming-crossing-onset.md`](project-context-streaming-crossing-onset.md) | The research argument — why streaming evaluation matters at all |
| [`THESIS_ROADMAP.md`](THESIS_ROADMAP.md) | The overall tracker — every stage, what's done, what's left |
| **`METHODOLOGY.md`** (this file) | **The method itself — onset timing under censoring, and how it was chosen** |
| [`../outputs/runs/RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md) | The numbers, the baselines, and the caveats attached to them |

The supporting side-studies (backbone choice, fusion, imbalance levers, calibration) used to live in a
separate research plan; that document was retired 2026-08-19 and the surviving RQ list now sits in
[`THESIS_ROADMAP.md`](THESIS_ROADMAP.md) § Supporting studies.

Write in plain language here. This document has to still make sense in six months, to a reader who has been
away from it, and to an examiner who has never seen the code.

---

## Why we are building a method

Short version; the roadmap's "Where the thesis stands" header carries the same framing at tracker altitude.

We measured something real: take a model trained the standard way — on a dataset built around crossing
events, where roughly one in three examples is a crossing — and test it on a realistic continuous stream,
where roughly one in thirty-seven is. Its ability to rank pedestrians by risk falls to near-chance (area
under the ROC curve drops from 0.88 to 0.53). Re-tuning the decision threshold, which is the obvious cheap
fix, recovers essentially nothing.

That is a genuine finding but it is a **negative** one: it says the usual benchmark flatters models. On its
own it is a warning, not a contribution. The decision (August 2026) is that it becomes the **motivation**,
and the thesis becomes: *here is a way to train for this that the standard setup cannot produce.*

**Consequence to keep in mind throughout:** the four existing training runs are now **baselines**. Every
method we build gets compared against them. That makes their configurations load-bearing — a difference
between two runs used to be a footnote, and is now something that can quietly make a comparison
meaningless. One such difference has already been found and is recorded in the results file.

---

## What the evidence rules out

Worth stating before the options, because it eliminates a lot of plausible-sounding work.

**The problem is ranking, not the threshold.** When the model is tested on the stream it cannot order
pedestrians by risk at all — 0.53 is barely better than shuffling them. Anything whose only effect is to
move the decision boundary therefore cannot help: threshold tuning, prior correction, and to a large extent
class weighting and focal loss. This is not a guess; it is what a near-chance ranking score means, and it is
why re-tuning the threshold recovered nothing.

**Good ranking is not enough either.** The model trained *on* the stream reaches 0.78 — clearly better than
chance, but at a 1-in-37 base rate that still gives poor precision. The best precision observed anywhere in
these runs is about 33%: two false alarms for every real one. So "improve the ranking somewhat" is not a
finish line; the target is a large improvement in separating one specific kind of confusing example.

**Training at a realistic base rate is itself unstable.** The streaming run's log shows the validation loss
jumping to 4–8 on several epochs, with recall snapping to 1.0 — meaning the model briefly collapsed into
calling everything a crossing. Its best epoch looks like a lucky dip rather than a settled plateau. So part
of the work is making training *behave*, separate from making the model *better*.

**The two protocols train genuinely different models.** A model trained on the stream also does badly on the
event-anchored test set (0.51 and 0.68 across the two model pairs). So this is not "one setting is simply
harder." They are different problems, and a model good at both is a real achievement rather than a given.

---

## What the negatives are actually made of

*Measured 2026-08-20. Rebuild with `python scripts/report_negative_composition.py --annotations
PIE/annotations/annotations.zip` — laptop only, no database, no GPU, no image frames.*

This was open decision #2, and it sized everything else. "Not crossing" splits three ways:

| Split | N | positive | never crosses | **will cross, later** | already crossed |
|---|---|---|---|---|---|
| train | 88,214 | 2.9% (2,530) | 64.3% | **25.7% (22,634)** | 7.2% |
| val | 20,490 | 2.8% (569) | 77.5% | **15.2% (3,108)** | 4.6% |
| test | 69,875 | 3.1% (2,140) | 53.4% | **39.2% (27,411)** | 4.3% |

Read the bold column as: one training window in four, and two test windows in five, show a person who
*does* cross — just later than the question asks about. They outnumber the real positives 8.9 to 1 in
train and 12.8 to 1 in test. The hard case is a large slice, so the direction holds.

Two things fell out that nobody was looking for.

**Train and test are not built the same way.** 25.7% hard-temporal in train against 39.2% in test, and
64.3% never-crossers against 53.4%. The test split is harder in a specific structural way, not merely
bigger. Any train-to-test difference has to be read with that in mind, which is why it is also flagged
in [`RESULTS_MATRIX.md`](../outputs/runs/RESULTS_MATRIX.md).

**Most of the hard mass is not actually hard.** Bucketed by how far off the crossing is (train):

| Time to onset | Windows | Share of hard-temporal |
|---|---|---|
| 1.1–2.1 s | 2,220 | 9.8% |
| 2.1–3.2 s | 1,966 | 8.7% |
| 3.2–5.0 s | 2,888 | 12.8% |
| **more than 5 s** | **15,560** | **68.7%** |

Two-thirds sit more than five seconds out — a person walking down a street who happens to cross
eventually, which is not a confusing example by any useful definition. The genuinely confusable band
is one to three seconds out: about **4,200 windows in train, roughly 1.7× the positive set** (test:
3,700, 1.7×). That is the population matched-pair training would draw from, so one or two natural
partners exist per positive. Enough to work with, not so many that they would swamp training.

---

## Why not simply predict further ahead

The obvious reply to a 34:1 imbalance is to ask a longer question. Re-labelling the same windows at
every horizon shows exactly what that buys (train; same script):

| Horizon | usable | positives | imbalance | confusable band | band : pos | windows lost |
|---|---|---|---|---|---|---|
| 1.1 s *(current)* | 88,214 | 2,530 | 33.9:1 | 3,955 | 1.56:1 | 0% |
| 2.0 s | 82,020 | 4,481 | 17.3:1 | 3,590 | 0.80:1 | 7% |
| 3.0 s | 75,767 | 6,367 | 10.9:1 | 3,237 | 0.51:1 | 14% |
| 5.0 s | 65,326 | 9,604 | 5.8:1 | 2,360 | 0.25:1 | 26% |
| 10.0 s | 49,325 | 14,599 | 2.4:1 | 1,408 | 0.10:1 | 44% |

It buys more than expected: 34:1 becomes 6:1 at five seconds, and the confusable band drops from 1.6×
the positives to a quarter. That is a real effect and the table is worth reporting on its own. It is
still the wrong move here, for three reasons.

**The windows it deletes are the ones that matter.** A longer horizon needs more observed future, so
26% of windows fall away at five seconds and 44% at ten — and they fall away *by track length*. Short
tracks go first, meaning pedestrians who appear suddenly or are quickly occluded. Those are precisely
the cases the project's short-window rationale exists to cover. That is buying balance by deleting the
hard part of the dataset.

**The band does not get easier, it gets relabelled.** The ratio improves because windows that were
hard negatives at one second *become positives* at five. The model still cannot separate a 4.9-second
window from a 5.1-second one; it simply now has to answer "yes" on the first, on evidence that is not
in the frame. The errors move from false alarms to misses. The confusion is conserved.

**The horizon picks which phenomenon you are detecting.** At about one second the observable signal is
the *body committing* — foot lift, weight shift, the turn. At five seconds no such cue exists yet;
what is observable is *where the person is heading* — position relative to the kerb, heading, distance
to a crossing point. Those are two different problems needing two different feature sets. Prong 1 aims
squarely at the first and would be close to useless for the second. A longer horizon does not extend
the method, it replaces it.

Worth noting where ten seconds lands: 2.4:1, which is essentially the event-anchored benchmark's
2.5:1. Different mechanism, same arithmetic.

**Decision: the canonical horizon stays at ~1 second** (32 frames), which is also inside the
benchmark's own 1–2 s time-to-event band, so comparability is preserved. The sweep is reported as the
justification for that choice rather than as a change to it.

---

## The ceiling, and what it does to the claim

Take a pedestrian walking down the sidewalk exactly like everyone else, who then turns abruptly and
steps out. Three seconds earlier, **the information is not in the video.** The decision has not been
made, or has not reached the body. No feature, no architecture and no loss function recovers it.

So an unknown fraction of the confusable band is not hard but *impossible*, and "handle the hard
negatives" overclaims. Three consequences, accepted up front rather than discovered by an examiner:

1. **The target is not accuracy on that band.** It is being confident where evidence exists and
   uncertain where it does not — ranking and calibration rather than classification. That also matches
   the measured failure: what collapsed on the stream was the *ordering*, and ordering is a weaker and
   reachable goal.
2. **Where the ceiling sits is itself a result.** Train once, then measure separability as a function
   of how far off the crossing is. The distance at which performance decays to chance is the empirical
   limit of prediction from body cues. No paper in this area appears to report it, and it converts the
   objection above from a hole into a finding.
3. **The binary label is our own doing.** A window at 0.9 s and a window at 1.1 s are near-identical
   inputs carrying opposite labels. That is not a hard learning problem, it is a badly posed one, and
   it is what prong 2 now exists to remove.

---

## Prong 1 — Turn pose keypoints into features about movement

**Where the code is:** [`src/pedpredict/data/pose.py`](../src/pedpredict/data/pose.py) (feature math),
[`scripts/extract_pose.py`](../scripts/extract_pose.py) (the extraction pass, already run),
[`POSE_ENCODER.md`](POSE_ENCODER.md) (the arm's design doc).

### What exists now

The extraction pass found 23 points on each pedestrian's body in each frame: nose, eyes, ears, shoulders,
elbows, wrists, hips, knees, ankles, and **six points on the feet** (big toes, small toes, heels). Each point
comes with a position in the image and a confidence score saying how sure the estimator was.

Those raw points are converted into 49 numbers per frame:

| Block | Count | What it is |
|---|---|---|
| Positions | 30 | 15 of the points, re-centred on the person's bounding box and divided by their height, so that where they are in the frame and how far away they are don't matter |
| Confidence | 15 | how much to trust each of those points |
| Head direction | 2 | which way the face is pointing, expressed as a pair of numbers so there is no wrap-around problem at 360° |
| Body direction | 2 | which way the torso is pointing, the same way |

Those 49 travel alongside 9 numbers describing the bounding box: its centre, its size, how both changed since
the previous frame, and the car's own speed. 58 numbers per frame in total.

### The gap

**Every one of those 49 pose numbers describes a single frozen frame.** Nothing in them describes change over
time. The model is handed a flipbook and only ever allowed to look at one page.

The network does contain a memory layer that could in principle work out motion for itself. But it is being
asked to rediscover, from limited and badly imbalanced data, something that can be computed exactly and for
almost no cost.

This matters more here than it would in most problems, because **the thing being detected is a movement, not
a posture.** A person standing at the kerb about to step out and a person standing at the kerb who will stay
put can look nearly identical in any single frame. What separates them is what their legs do next. The
current features are, almost by construction, unable to express that.

### What to add

Roughly in order of expected payoff:

1. **Movement of each point** — how far every joint shifted since the previous frame. Exactly the arithmetic
   already done for the bounding box, applied per joint.
2. **Foot contact and lift.** Vertical movement of each foot separates the planted foot from the swinging
   one. The instant a planted foot leaves the ground is the physical beginning of a step — the closest thing
   to a direct measurement of crossing onset available anywhere in this data.
3. **Walking rhythm.** How quickly the legs swing back and forth. Standing still is flat; walking is regular
   oscillation; **standing and then starting to walk is a flat stretch followed by a sudden onset** — which
   is the event we are trying to catch, expressed as a signal rather than inferred from appearance.
4. **Stance width and weight shift.** People widen their stance or shift their weight before stepping off a
   kerb, before the foot actually moves.
5. **Rate of turning.** The body-direction angle is stored but not how fast it is changing. Turning to face
   the road is a strong cue and is currently thrown away.
6. **Head direction relative to the car**, rather than relative to the image. "Looking at the oncoming
   vehicle" carries more meaning than "looking left."

### Why this is a good place to start

- It is **arithmetic on data already extracted**. The keypoints are on the lab machine's disk. No
  re-extraction, no GPU.
- It can be **written and tested on the laptop** — the existing tests in
  [`tests/test_pose.py`](../tests/test_pose.py) run on fabricated keypoints, so a synthetic walker with a
  known step rhythm is a perfectly good test subject.
- The **feet are a real advantage**. The keypoint format most published work in this area uses (the standard
  17-point COCO layout, via OpenPose or similar) stops at the ankle and has no foot points at all. Methods
  built on it structurally cannot compute foot contact. Ours can. *(The 17-point layout's contents are from
  my own knowledge, not from a source I checked here — worth a sanity check before it goes in writing.)*

### The risk, and the check that resolves it

Dashcam pedestrians are often small, distant, and partly hidden by cars. **Feet are the body part most
likely to be estimated badly.** If foot confidence is mostly low, the strongest version of this prong
weakens considerably.

So check before building: read the cached keypoint files and report the confidence distribution for the six
foot points against the ankles and hips. This is a short script and a genuine decision point. It needs the
lab machine (the cache is not on the laptop), so the script gets written first and runs at the next visit.

**Do not block on the answer.** Build the features with a configurable set of joints, so that
"feet are unreliable, fall back to knees and ankles" is a configuration change rather than a rewrite. Both
outcomes then land the same code.

### Two correctness requirements

- **Mirroring.** Training data is augmented by flipping images left-to-right. The existing code already has
  to negate horizontal box movement and swap left/right joints to stay correct under that flip. **New
  movement features need the same treatment.** Get it wrong and mirrored training data is silently corrupt —
  the kind of bug that costs a full training run and never announces itself. This belongs in a test, not in
  a reviewer's memory.
- **Low-confidence points.** A foot that vanishes for one frame and reappears looks like it teleported, which
  produces a nonsense velocity. Movement features must be gated by confidence. There is already a smoothing
  helper (`smooth_pose`) in the file for this purpose.

### What published work does here

Pose-based crossing prediction is a crowded area: multi-branch networks over 2D skeleton sequences,
3D-joint spatio-temporal representations, and models combining keypoints with trajectories and group
context. Gait does appear, but **coarsely** — knees and ankles used to infer a binary walking-versus-standing
state. Detailed gait analysis (stride length, cadence, heel-strike and toe-off phases) lives in the
biomechanics and wearable-sensor literature, not in dashcam crossing prediction.

So the crowded part is "use pose." The much less crowded part is **use the derivatives of pose, at foot
level, as an onset cue.** That is where this prong aims.

---

## Prong 2 — Predict *when*, and treat "not yet" as censored rather than negative

> **This is the spine of the method** (decided 2026-08-20). It began as one of three parallel
> directions; the ceiling argument above promoted it, because it is the only one of the three that
> removes the confusing case instead of fighting it.

**Where the code is:** [`src/pedpredict/data/pie_sequences.py`](../src/pedpredict/data/pie_sequences.py)
(the fields are computed here), [`src/pedpredict/data/lmdb_writer.py`](../src/pedpredict/data/lmdb_writer.py)
(where they get dropped).

### First, what a "window" is

The data is not whole videos. It is chopped into short clips, each one a fixed number of consecutive frames
following a single pedestrian. For each clip, three labels are currently stored: was the person walking, were
they looking toward traffic, and will they cross soon — the last one a plain yes or no.

### The three extra numbers

They are already computed. Then they are thrown away.

| Field name | In plain terms |
|---|---|
| `onset_offset` | **How many frames until this person starts crossing**, counted from the last frame the model is allowed to see. If they never cross, it is −1. |
| `future_observed` | **How much future footage actually exists** after this clip ends. |
| `track_crosses` | **Does this person ever cross at all** — before this clip, after it, or never. |

Why the second one matters: if a video ends five frames after a clip, you cannot honestly claim the person
did not cross in the following two seconds. You simply don't know. Right now those clips are labelled "no"
anyway, which is a small but real source of wrong labels.

### Where they used to get stranded — fixed 2026-08-26

They were computed in `pie_sequences.py` and saved into the intermediate sequence files, then dropped when
those files were packed into the database training actually reads. They now survive the whole path, and the
model has a head and a loss that use them. What is left is a one-off pass on the lab machine to write the
fields into the databases that were built before the fix, plus the training runs themselves.

The engineering contracts — the four cases, the two hard rules about horizon and look-ahead, and the weight
table that selects between the three versions of the objective — are stated once in **CLAUDE.md § Onset
Timing** and not repeated here. This document keeps the *reasoning*; that one keeps the *rules*.

### What these three numbers actually are

They are a **time-to-event dataset**, which is a well-established shape of statistical problem with its
own century of theory. The correspondence is exact, not loose:

| Our field | What it is in time-to-event terms |
|---|---|
| `onset_offset` | the time until the event, when the event was observed |
| `future_observed` | how long we watched — the *censoring* time, for the ones we never saw cross |
| `track_crosses` | which of the two situations a "no crossing seen" window is in |

The vocabulary matters because it names the thing we keep getting wrong. A window where the person has
not crossed *within the footage we have* is not a negative. It is an observation that was cut short —
**censored**, in the technical sense. Right now the pipeline handles that in the only two ways a binary
label allows: label it 0 and be wrong (the pre-M4 behaviour), or throw it away (M4, current). Time-to-event
methods have a third option, which is to use it for what it does say: *no crossing before this point*.

### What that changes

**The confusing case stops existing.** With no fixed cut-off there is no boundary, so there is no pair
of near-identical windows sitting either side of it. A window two seconds from onset is not a mislabelled
negative; it is an example whose answer is "two seconds". The hardest examples become the most
informative ones instead of label noise.

**Some of the discarded windows come back — and one group matters much more than the rest.** M4 drops
every window whose future was not fully observed: 7,470 in train at the canonical horizon. Looking at what
those actually are, they split three ways:

* **Confirmed positives.** The pedestrian was *seen* stepping off inside the truncated remainder. The label
  was never in doubt; the filter binned them anyway, because it asks about time remaining without asking
  whether anything happened in it. These are the most valuable windows in the dataset — imminent onsets, in
  a task with 2.9% positives. Recovered by `data.emit_determined_positives`, and worth having **whether or
  not the onset head is ever switched on**, since they are confirmed positives for the ordinary binary task
  too.
* **Genuinely censored.** Watched briefly, nothing seen. Real but weak information, and every one is a
  partial *negative* — so it adds nothing to the positive class and slightly worsens the imbalance. Left
  dropped: the gain is a few percent of extra supervision, against a fabricated `crosses` bit that would
  only be safe in one of the three training arms.
* **Already crossed.** Recovered by a regen, then dropped again by the hazard loss as not at risk. Net zero.

The earlier framing here — "a straight recovery of data" — oversold the second group and missed the first.
Note also that the censoring machinery is **already exercised without any regen**: at `lookahead=60` every
window with `future_observed` between 32 and 60 is a censored observation, and generation only guarantees
32. The demonstration does not depend on recovering the short-future windows.

**Any horizon can be read afterwards.** The model estimates *when*; "will they cross within H" is then a
question asked of the output, at whatever H the reader wants, rather than a decision frozen into the
training labels.

**Comparability is preserved, and this is a hard requirement.** The four existing runs are baselines at
the binary 32-frame question. A model that predicts timing must still emit a directly comparable number —
the probability of onset within 32 frames — or every existing comparison is lost. This maps cleanly, but
it must be built in from the start, not bolted on.

### What still holds from the binary view

**Counting the negatives — done.** See [What the negatives are actually made of](#what-the-negatives-are-actually-made-of):
it is a large slice, so the direction is confirmed. Answered on the laptop from the annotation XMLs; the
database backfill was not needed for it.

**Matched comparisons — still wanted.** Pair a window where the person crosses with a window of **the
same person, same scene, a couple of seconds earlier**. Clothing, lighting, street and camera are held
constant, so the only thing separating the pair is the onset cue itself. Measured above: roughly 1.7
natural partners per positive. See prong 3, direction (b).

### What the field does with timing today

In pedestrian crossing papers, time-to-event is used to **define the task** — "predict two seconds ahead" —
and then a yes/no classifier is trained at that horizon. There is a smaller line of work predicting
time-to-cross directly, using statistical rather than deep methods, and multi-task models that add auxiliary
outputs. Using onset *time* as a training signal for the crossing decision does not appear to be the standard
move.

**Honesty flag:** that is based on a handful of searches, not a systematic review. Before this is claimed as
novel in writing, it needs a proper literature check. The direction is worth pursuing either way — if
someone has done it, that is a baseline to compare against rather than a reason to stop.

**The same flag applies twice as hard to the censoring framing**, which is the newer half of this prong.
Time-to-event modelling with right-censoring is standard practice in other fields and is certainly not new
in itself; the open question is whether anyone has applied it to pedestrian crossing prediction. That check
is now **open decision #3** and it gates how the contribution is described, not whether it is built.

### What the fix involves

Plumbing first — everything else depends on it, and it is the same three steps whichever way the
supervision is finally shaped:

1. ~~Add the three fields to the packing function.~~ **Done.**
2. ~~Add them to the read path.~~ **Done** — they arrive with each training example, carried in the same
   bundle as the ordinary labels, so nothing in the trainer had to change to receive them.
3. ~~A patch script for the existing databases.~~ **Written; still to run on the lab machine.** It checks
   each stored window against the record it is supposed to be before writing anything, so pointing it at
   the wrong file stops rather than corrupts.

Then the method itself:

4. ~~A **timing output** and a loss that handles censored observations.~~ **Done.** Off by default. The
   loss turned out to be an ordinary per-bin cross-entropy with a mask — the masking is the whole
   contribution, and there is a test that checks with autograd that unobserved bins receive exactly zero
   gradient rather than a small one.
5. ~~A **conversion back to the baseline question**.~~ **Done**, and pinned by a test that reproduces the
   generator's own yes/no label from the timing output over a real track. That test is what keeps the four
   existing runs valid comparisons.
6. **Censored windows restored** — still open, and it needs a **regeneration**, not the patch script. The
   generator drops those windows before writing, so they are not in the files to be patched. Worth
   measuring on its own: it is a data-quantity change, separable from the objective change.

**One thing found while building it that is worth knowing.** The timing loss adds up a score for every
future step a window actually saw. A window watched for sixty steps therefore contributes about sixty times
as much as one watched for one step. That is correct — it genuinely knows sixty times more — but it means
the timing term starts out roughly forty times larger than the ordinary label terms, and would drown them
if it were simply switched on at equal weight. Its weight has to be turned down when it is riding alongside
the old objective, and left at full strength when it *is* the objective.

**The image data does not need to be touched.** Labels are stored under separate keys from the image blobs,
so this is a fast pass over the small text-like part of each entry, not a rebuild. Steps 1–3 are laptop work;
only running the patch needs the lab machine.

---

## Prong 3 — Import a rare-event mechanism that already exists

### The literature we should be reading

The relevant field is **online action detection**, and within it, **detection of action start**. The setup:
video arrives one frame at a time, the system must announce that something has just begun, and it cannot look
into the future. That is structurally our problem, described in a different research community's vocabulary.

The paper that matters most states our failure mode directly. From Shou et al. (2018):

> *"Due to the shared contents (background scene and object), the feature of the start window may be closer
> to the preceding background window than the actual action window after the start."*

In plain terms: **the moment just before something begins looks more like the boring footage than like the
event itself** — same scene, same objects, same person. That is exactly our confusing case, written down in
2018.

**Reading list:**

- Shou et al. 2018, *Online Detection of Action Start in Untrimmed, Streaming Videos* — the problem statement
  and three proposed fixes.
- Gao et al. 2019, *StartNet* — the direct follow-up, which splits the problem into two stages.
- The newer transformer-based work in this area. *(Names I recall are TRN, OadTR, LSTR, TeSTra and MAT — from
  memory, not verified here. Check before citing.)*

**Status flag:** the quote above and the three fixes below come from search summaries of the 2018 paper, not
from reading it end to end. Reading both papers properly is a task in its own right, and the details will
shape which mechanism gets built.

The three fixes that paper proposes, since they map onto our options: **generating** hard examples to sharpen
the boundary; **explicitly modelling** how features behave across the start point; and **adaptive sampling**
to cope with scarcity.

### Four directions

**(a) Make the model say *when*, not just *whether*.**

Right now the model scores every frame of the clip and then collapses those scores into one answer using a
soft version of "take the strongest frame." One confident frame decides the whole clip.

That is precisely the wrong design against someone who is *about to* cross, because the last frames of their
clip genuinely do resemble crossing frames. The online-detection approach keeps the per-frame answer instead
of collapsing it. Our model already computes the per-frame scores internally, so structurally this is a small
change with large consequences.

There is also a **free experiment available today**: the collapsing rule is a configuration option
(`model.frame_pool`, currently `logsumexp`, with `max` and `mean` available). Trying the alternatives costs
one training run and no new code. Worth queueing simply because it is free.

Frame-level supervision needs prong 2 first — a per-frame target requires knowing when the crossing starts.

**(b) Train directly against the confusion.**

Take a clip where the person crosses and pair it with a clip of the same person, same scene, a couple of
seconds earlier. Require the model to score the first higher than the second.

Because clothing, lighting, street, and camera are all held constant across the pair, **the only thing the
model can learn from that comparison is the actual onset cue.** It cannot fall back on recognising the person
or the location. This is the "hard example" family; the 2018 paper generates such examples artificially,
whereas our dataset already contains them — we just need prong 2 to find them.

**(c) Change what the training objective rewards.**

The loss function currently treats every mistake as equally bad
([`losses/multitask.py`](../src/pedpredict/losses/multitask.py) is plain cross-entropy with optional
inverse-frequency weights and a per-task scale — no focal loss, no class-balanced loss, nothing
rare-event-specific). At one crossing in thirty-seven clips that is a poor match for what we care about,
which is being right when we raise an alarm. Objectives that optimise ranking or average precision directly
exist and are a better fit.

**Being straight about expectations:** focal loss and class-balanced loss belong here, but the evidence
section above argues they will not be the answer, because they adjust where the decision line sits and our
problem is that the ordering itself has collapsed. They go in as the **comparison baseline** — a methods
thesis has to measure the obvious alternative rather than dismiss it — not as the plan.

**(d) Cheap model first, expensive model second.**

A cheap model using only bounding-box motion looks at everything, tuned to almost never miss a crossing while
accepting many false alarms. The expensive pose-and-vision model only examines what survives.

This is standard practice for rare events, it is a direct answer to the "two models checking each other"
suggestion, and it produces a genuine speed argument for a system meant to run live — which the existing
efficiency measurement harness can quantify. The pixel-free `kinematics_only` model already in the registry
is a natural candidate for the cheap stage.

### Measure the way that field measures

That community stopped using F1 for exactly our reason: at extreme imbalance it swings too much to compare
methods. They use calibrated average precision, and a point-level score that gives credit for detecting the
start close to the right time. **If we adopt their methods we should adopt their measurements**, or our
numbers will not be comparable to theirs — and our own before-and-after comparisons will be too noisy to
read.

This is the argument for building the metrics *before* the method, and it is why the roadmap puts Stage 6
ahead of Stage 7.

---

## How the three fit together

They are no longer three equal prongs. After the August 2026 reframe:

- **Prong 2 is the method.** Predicting onset *time* under censoring removes the confusing case rather
  than fighting it, which is the only one of the three responses the ceiling argument leaves standing.
- **Prong 1 supplies the feature** that makes onset physically expressible at a one-second horizon.
  Without movement features there is nothing in the input for a timing model to read.
- **Prong 3 is now a source of mechanisms and, more importantly, of measurements.** That community
  solved the metric problem for exactly this shape of data; its objectives remain available where they
  fit the timing formulation, and its comparison baselines still have to be run.

One sentence: **treat streaming crossing prediction as onset timing under censoring rather than
classification at a fixed horizon, with foot-level gait dynamics as the cue and the online-action-detection
metric suite as the instrument.** The decomposition finding is the motivation for why that is needed.

Each part also has standalone value, which matters if the combination disappoints:

- better pose features are a useful result on their own, and cheap to ablate;
- the negative-composition count is already a reportable observation about what the standard benchmark
  leaves out — measured, and it stands whatever happens to the method;
- the ceiling measurement is a finding regardless of which side of it the method lands on;
- the metric suite is a contribution to how this task gets evaluated.

That spread is deliberate. It means a null result on the headline method still leaves reportable work.

---

## Open decisions

Recorded so they get made deliberately rather than by default.

1. **Are the foot keypoints good enough?** Resolved by the confidence check. Determines whether prong 1's
   strongest version is available or whether it falls back to knees and ankles. **Still open**, and now
   the largest un-run laptop-blocked item — the cache is on the lab machine.
2. ~~**How much of the imbalance is the hard case?**~~ **Answered 2026-08-20.** A large slice: 25.7% of
   train windows and 39.2% of test windows are people who cross later, 8.9× and 12.8× the positives.
   Of that mass, the genuinely confusable one-to-three-second band is ~1.7× the positive set. The
   direction is confirmed and matched pairs have enough material.
3. **Has this been done for crossing prediction?** Two separate checks, both needed before any novelty
   claim reaches writing: onset *time* as a training signal, and time-to-event modelling with censoring.
   Changes how the contribution is described, not whether it is built.
4. **Which supporting mechanism to add on top** — (a) frame-level scoring, (b) matched pairs,
   (c) a ranking objective, or (d) a two-stage cascade. Now a second-order choice rather than the main
   fork, since the timing formulation is the method. Left open until the two papers are read properly.
5. **Do we fix the streaming training instability, or report it?** It may be a finding in itself. Probably
   both — attempt a fix, report what the attempt reveals.
6. **How far ahead is crossing predictable at all?** The ceiling measurement. Needs one trained model
   plus a stratified evaluation, so it is cheap once anything is training; the answer shapes what the
   thesis is allowed to claim.

---

## Deliberately not doing

- **The matched-size control experiment.** It was essential when the gap was the headline. As motivation, the
  size confound is an acknowledged caveat in a paragraph. The knob is cheap to add if it becomes wanted
  again.
- **Retraining other papers' models for comparison.** Long-standing schedule risk, already out of scope. The
  method is measured against our own four runs.
- **Scene segmentation** (kerb and crosswalk geometry as explicit inputs). Plausibly useful, expensive, and
  away from the onset argument.
- **Chasing better numbers on the event-anchored benchmark.** That is the setting the thesis argues is
  misleading. Improving on it would be beside the point.

---

## Status

| Part | Where it stands | Next concrete step | Machine |
|---|---|---|---|
| **2 — onset timing (the method)** | **built 2026-08-26**, default off, full test gate green | run the patch script, then a short run to check the head does not go dead, then the auxiliary arm | 🖥️ |
| 1 — pose movement features | not started | confidence-check script, then the feature math with a configurable joint set | 💻 write · 🖥️ check |
| 3 — mechanisms + baselines | reading not done | read both papers properly; queue the free pooling experiment | 💻 |
| Metrics (prerequisite) | not started | calibrated average precision + point-level score, tested on made-up streams | 💻 |
| Negative composition | ✅ done 2026-08-20 | — (`scripts/report_negative_composition.py`) | 💻 |
| Horizon sweep | ✅ done 2026-08-20 | reported as the justification for keeping ~1 s | 💻 |
| Ceiling measurement | not started | stratify separability by time-to-onset on any trained model | 🖥️ |
| Baseline hygiene | mismatch found | re-run the anchored crosses-only half with the sampler on | 🖥️ |

💻 = laptop, no data needed · 🖥️ = needs the lab machine's data or GPU

---

## Sources

Consulted for this document (search summaries unless noted):

- [Shou et al. 2018, *Online Detection of Action Start in Untrimmed, Streaming Videos*](https://arxiv.org/abs/1802.06822)
- [Gao et al. 2019, *StartNet*](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_StartNet_Online_Detection_of_Action_Start_in_Untrimmed_Videos_ICCV_2019_paper.pdf)
- [Pedestrian intention prediction from 2D skeletal pose sequences](https://www.mdpi.com/1999-4893/13/12/331)
- [Multi-scale pedestrian intent prediction using 3D joint information](https://www.sciencedirect.com/science/article/pii/S0957417423005791)
- [TrajFusionNet](https://arxiv.org/pdf/2508.19866)
- [VRUNet — multi-task intent prediction](https://arxiv.org/pdf/2007.05397)
- [Crossing intention forecasting from naturalistic trajectories](https://pmc.ncbi.nlm.nih.gov/articles/PMC10006956/)
- [Pedestrian intention prediction — topic overview](https://www.emergentmind.com/topics/pedestrian-intention-prediction)

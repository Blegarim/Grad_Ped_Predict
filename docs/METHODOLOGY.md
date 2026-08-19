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
| **`METHODOLOGY.md`** (this file) | **The method itself — the three directions and how they were chosen** |
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

## Prong 2 — Supervise *when*, not just *whether*

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

### Where they get stranded

They are computed in `pie_sequences.py` and saved into the intermediate sequence files. When those files are
packed into the database that training and evaluation actually read, the packing function keeps the three
labels and the box motion and drops these three fields. **So they exist in the code and are invisible during
training.** Nothing in the trainer, the sampler, or the loss can see them.

### Four things they unlock

**1. Counting what the negatives are actually made of.** Today "not crossing" lumps together three
completely different situations:

- someone who never crosses at all;
- someone who crosses four seconds from now — visually almost identical to a positive;
- someone who has already finished crossing.

We currently cannot say how much of the 1-in-37 imbalance is each kind. **This count should come early,
because it sizes everything else.** If the about-to-cross case is a small slice of the negatives, the
hard-case story shrinks and attention should move elsewhere. If it is a large slice, the direction is
confirmed. Either way it is a counting exercise, not a training run.

**2. Teaching the model the timing.** Add a second output that estimates how many frames until the person
steps out, trained alongside the yes/no answer. One bit of supervision becomes a real number. Even if the
timing estimate is never used at deployment, having to produce it forces the model to represent something
about *when*, which is the information the current target discards.

**3. Building matched comparisons.** Pair a clip where the person crosses with a clip of **the same person,
in the same scene, a couple of seconds earlier**, before they crossed. Finding those pairs needs
`track_crosses` (who eventually crosses) and `onset_offset` (when). See prong 3, direction (b), for what to
do with them.

**4. Changing the question after the fact.** "Will they cross in the next two seconds" versus one second
versus four — with these numbers the question can be re-asked at any horizon using clips already built, no
regeneration needed. `future_observed` is what makes that honest, by excluding clips where the answer isn't
knowable.

### What the field does with timing today

In pedestrian crossing papers, time-to-event is used to **define the task** — "predict two seconds ahead" —
and then a yes/no classifier is trained at that horizon. There is a smaller line of work predicting
time-to-cross directly, using statistical rather than deep methods, and multi-task models that add auxiliary
outputs. Using onset *time* as a training signal for the crossing decision does not appear to be the standard
move.

**Honesty flag:** that is based on a handful of searches, not a systematic review. Before this is claimed as
novel in writing, it needs a proper literature check. The direction is worth pursuing either way — if
someone has done it, that is a baseline to compare against rather than a reason to stop.

### What the fix involves

1. Add the three fields to the packing function, so future builds carry them.
2. Add them to the read path so they arrive with each training example.
3. Write a patch script that reopens the existing databases and fills the fields into entries already there.

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

- Prong 3 supplies a **mechanism** that a different community has already shown works on this shape of
  problem.
- Prong 2 supplies the **label** that mechanism needs.
- Prong 1 supplies the **feature** that makes the distinction physically expressible.

One sentence: **port onset-detection supervision from the online-action-detection literature into pedestrian
crossing prediction, using foot-level gait dynamics as the discriminative cue.** The decomposition finding is
the motivation for why that is needed.

Each prong also has standalone value, which matters if the combination disappoints:

- better pose features are a useful result on their own, and cheap to ablate;
- the negative-composition count is a publishable observation about what the standard benchmark leaves out;
- the metric suite is a contribution to how this task gets evaluated.

That spread is deliberate. It means a null result on the headline method still leaves reportable work.

---

## Open decisions

Recorded so they get made deliberately rather than by default.

1. **Are the foot keypoints good enough?** Resolved by the confidence check. Determines whether prong 1's
   strongest version is available or whether it falls back to knees and ankles.
2. **How much of the imbalance is the hard case?** Resolved by the negative-composition count. Determines how
   much prongs 2 and 3 have to work with. **This is the most decision-relevant number not yet measured.**
3. **Has onset-time supervision been done for crossing prediction?** Needs a proper literature check. Changes
   whether it is framed as novel or as improving on existing work.
4. **Which rare-event mechanism to build first** — (a) frame-level, (b) matched pairs, (c) objective, or
   (d) two-stage. Deliberately left open until decisions 1–3 land and both papers have been read properly.
5. **Do we fix the streaming training instability, or report it?** It may be a finding in itself. Probably
   both — attempt a fix, report what the attempt reveals.

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

| Prong | Where it stands | Next concrete step | Machine |
|---|---|---|---|
| 1 — pose movement features | not started | confidence-check script, then the feature math with a configurable joint set | 💻 write · 🖥️ check |
| 2 — onset supervision | fields computed, stranded | add to the packing function and read path; write the patch script | 💻 write · 🖥️ patch |
| 3 — rare-event mechanism | reading not done | read both papers properly; queue the free pooling experiment | 💻 |
| Metrics (prerequisite) | not started | calibrated average precision + point-level score, tested on made-up streams | 💻 |
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

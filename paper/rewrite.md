## 1 Introduction

A pedestrian-crossing detector on a moving vehicle must make a prediction at every moment for every pedestrian in view. Most of those moments do not precede a crossing. The standard evaluation protocol asks a narrower question. Introduced with the Pedestrian Intention Estimation (PIE) and Joint Attention in Autonomous Driving (JAAD) datasets [1, 2], and later standardised in an evaluation suite [3], it samples a few windows from each pedestrian track. These windows end one to two seconds before a reference moment and are labelled according to whether the pedestrian crosses (Figure 1a).

Interpreting a score from this protocol requires separating two questions. The first is *who*: will the pedestrian cross in front of the vehicle at all? The second is *when*: if the pedestrian will cross, will the crossing begin within the next second? A streaming detector has to answer both. It should remain silent, for example, when a pedestrian will cross only five seconds later (Figure 1b). We show that the streaming label is the product of these two components. The anchored protocol holds the second component fixed and therefore scores only the first. A single anchored score cannot reveal which component makes the streaming task difficult.

We trained a neural network and boosted trees on the same pose and kinematic inputs. We then added scene, behaviour and oracle inputs to the trees, one group at a time; the oracles provide information unavailable to a deployed system. Each study was specified in writing and time-stamped before its first result. The study makes five contributions:

1. **Decomposition.** We factor the streaming crossing label into who and when, corresponding to incidence and latency in a mixture cure model from survival analysis. The anchored label estimates who alone (Section 3).
2. **An explanation of the protocol gap.** In both model families, anchored training ranks who well but reverses the ordering of when. The area under the receiver operating characteristic curve (AUC) falls to chance on streaming data. Neither threshold re-tuning nor re-weighted losses remove this gap, and the smaller size of the anchored training set does not explain it (Section 5).
3. **Decomposed evaluation.** We score who and when separately and count false alarms both per window and per pedestrian. These two ways of counting lead to different conclusions (Section 6).
4. **An information audit.** Scene information raises who above a human intention rating, although that comparison is post hoc. When improves by at most 0.043, less than half the gain in who (Section 6).
5. **A timing objective.** On the same training windows, an objective that models time to crossing does no better than binary training in either model family. This agrees with the decomposition if the inputs contain no timing information beyond what binary training already uses (Section 7).

Streaming evaluation is established in online action detection [4, 5], and traffic research has distinguished immediate crossing from waiting time (Section 2). However, to our knowledge and based on a targeted search, no previous study of crossing prediction from vehicle video separates who from when or shows that the anchored score measures only who.

## 2 Related Work

### 2.1 Crossing Prediction on PIE and JAAD

Most studies using PIE [1] and JAAD [2] follow the benchmark of Kotseruba et al. [3]. They assign labels per track and use an observation period ending one to two seconds before the crossing. Several studies have questioned what the resulting scores measure. Bounding boxes alone can yield high accuracy on PIE [6], predictors transfer poorly between datasets [7], and models depend heavily on ego-vehicle speed, which may reflect the driver's behaviour rather than the pedestrian's [8].

Rasouli and Kotseruba [9] point out that intention and action are often used interchangeably. They also report much higher action-prediction scores when the ego vehicle is stationary, a setting in which most samples are crossers. Kotseruba et al. [10] provide a human reference for assessing a pedestrian's intention to cross. Yao et al. [11] report results for both whole tracks and an “event of crossing” setting restricted to the one to two seconds before crossing. Most of these studies retain the window population of the anchored protocol, and none separates the label into who and when. Our evaluation changes both the sampled windows and the label.

### 2.2 When Pedestrians Start to Cross

Early work on path prediction found that a pedestrian approaching the kerb could be reliably classified as stopping or crossing only a fraction of a second before the event [12]. Later work incorporated context, including awareness of the vehicle and distance to the kerb, to predict transitions between walking and standing [13]. More recent studies predict whether a pedestrian will start or stop walking within two seconds [14], collect frame-by-frame intent labels from many annotators [15], and report crossing-prediction performance against time to the event [16]. These studies model timing or vary it in evaluation. We show that the standard score is insensitive to it.

### 2.3 Online Action Detection

Online action detection assigns a label to each frame in a stream using only present and past evidence [4]. In the action-start variant, the system must signal the beginning of an action as early as possible [5]. We adopt this streaming setting and its approach to counting false alarms.

### 2.4 Time-to-Event Modelling

Survival analysis models time to an event when some observations end before the event occurs [17]. Mixture cure models separate whether an event occurs from when it occurs [18, 19], giving the structure used in Section 3.4. In traffic research, survival and duration models have been used to describe pedestrians' waiting times at the kerb [20, 21]. Yang et al. [22] combine a logit model for pedestrians who cross immediately with a duration model for those who wait, a separation close to ours. Kalatian and Farooq [23] fitted a neural survival model to waiting times in virtual reality. We use the cure structure to interpret an existing vision benchmark and use a hazard objective for time to crossing as a further check (Section 7).

## 8 Discussion

### 8.1 What This Means for Evaluation

An anchored score measures who alone. It can be high even when a model reverses the ordering of crossing times (Table 3), so it tells us little about how the model will behave on streaming data. Three changes to evaluation could be made with the existing annotations. Who and when should be reported separately. Evaluation should use streaming windows and a detection curve at fixed false-alarm budgets. False alarms should also be counted per pedestrian as well as per window, because a few waiting pedestrians can dominate the per-window count.

### 8.2 What When Would Need

When remains the open problem. On PIE, none of the inputs we added improved it by more than 0.043. In the confusable band, it remained between 0.61 and 0.69 in a post hoc analysis. The most persistent false alarms, examined post hoc as well, were who errors involving intending pedestrians at signalised crosswalks who might cross after the footage ends. Their crossing time may depend on information missing from the data, such as the signal controlling the pedestrian, gaps in oncoming traffic or interaction with the driver. Frame-level intent labels [15], longer footage after the observation window and waiting-time models [20, 21, 22] are natural next steps.

### 8.3 Limitations

**Pre-registration.** All studies were specified and time-stamped before their first result, with two amendments. Before any results were obtained, we loosened a check that the exported inputs reproduced the earlier network outputs, because those stored outputs had been computed in half precision. After the results were obtained, we found that the tree hazard readout had been stored in single precision. This rounded confident scores to exactly one, leaving three hazard arms with no threshold that could meet the tightest alarm budgets. Recomputing the readout in double precision changed two verdicts—from “binary better” to inconclusive—for the tree hazard comparisons against binary training, with and without ego speed. Both versions are retained.

The pre-registered verdicts not discussed above were all inconclusive per window: lights, traffic and layout individually against the pedestrian-only model; the scene with both oracles against that model, although this was better per pedestrian; the intent oracle beyond the scene; the detection contrasts in the waiting audit; hazard against binary training at each smaller training fraction; a split of the scene result by signalisation; and an earlier test combining an anchored model's score with a streaming model's score.

**Accounting.** The scene audit was inconclusive on its primary, per-window measure. Its clear result was on the pre-registered secondary measure, counted per pedestrian. Our explanation for the difference between these measures in Section 6.4 is post hoc.

**Inputs.** Scene and behaviour features came from PIE annotations, so they assume perfect perception of information a camera could see. Noisier, real-world perception would weaken the who result and strengthen the negative result for when. The light feature represents the largest light in view, which need not control the pedestrian. The finding that lights add little therefore applies to this feature rather than to right-of-way. Neither oracle is deployable, and the ego-future oracle is confounded. Its arms also reached the iteration cap while validation performance was still improving, so its effect may be understated.

**Power and scope.** The test split contains 205 crossing pedestrians. Two seeds of one tree arm can differ by 6.8 points per window, and the tree comparisons resolved differences of about 5 points. We used the neural-network recipe despite its failure on a pre-registered training check: the networks overfit after the first few epochs, and we report the checkpoint with the best validation AUC. All results come from PIE, which was recorded in one city in daylight. Our anchored protocol is a reimplementation, and its scores cannot be compared directly with published results. We chose the detection metric after obtaining the first window-level results. Earlier findings motivated the scene and waiting audits, but their questions were fixed before their own results were obtained. Excluding the 21 never-crossers who cross a different road changes detection per window by at most 2.8 points in a post hoc check.

## 9 Conclusion

The streaming crossing label is the product of two factors: the probability that a pedestrian will cross at all, or who, and the probability that the crossing will begin within the horizon, or when. The event-anchored protocol measures only who. A model can therefore score well on anchored data while reversing the ordering of when, as we observed on PIE with both a neural network and boosted trees.

Scene information that a camera or map could in principle provide raised who above a human intention rating in a post hoc comparison. When, by contrast, improved by at most 0.043 and remained low just beyond the horizon. With the same training windows, a hazard objective did no better than binary training, consistent with the decomposition. Further progress in streaming crossing prediction is likely to depend on information about when pedestrians cross. Evaluations should report who and when separately.

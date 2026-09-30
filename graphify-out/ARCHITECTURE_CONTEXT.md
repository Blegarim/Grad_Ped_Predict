# Implementation context

Snapshot: local working tree on 2026-09-29, HEAD `fd507a2`. This is an analysis artifact, not a change to the research protocol.

## Data and task

The project uses PIE pedestrian tracks. The streaming generator takes 20 observed frames at stride 3 and asks whether crossing starts in the next 32 frames (`future_offset=30`, `tol=2`). At 30 frames/s these are approximately 0.67 s of observations and 1.07 s of forecast horizon; “one second of observation” in broad prose is approximate. Actions and looks describe the last observed state; crossing is the future outcome. The event-anchored dataset answers a different sampling question. Protocol selection is centralized in `paths.protocol_lmdb_dirs`.

Sources: `configs/data.yaml:24`, `src/pedpredict/data/pie_sequences.py:201`, `src/pedpredict/paths.py`.

The offline path is annotations → sequence windows → LMDB metadata and image crops → optional balancing/augmentation → runtime dataset → collation. Raw motion storage has nine channels; the pose branch constructs a 58-dimensional input at read time: nine motion channels plus 49 pose features. The core pose representation uses 15 joints, their normalized coordinates and confidences, and head/body facing cues. Raw stored pose is `[T,23,3]`. This separates reusable pose extraction from feature design.

Sources: `src/pedpredict/data/pose.py:1`, `src/pedpredict/data/lmdb_dataset.py:257`, `src/pedpredict/data/transforms.py`, `src/pedpredict/data/lmdb_writer.py`.

Pixel-free training is literal: `visual_input=none` bypasses crop decoding and returns empty image slots. Pose features still originate from an upstream pose extractor, so this is a pixel-free prediction model, not an end-to-end system that never consumes imagery. Cached visual features are a separate route for frozen context backbones.

Sources: `src/pedpredict/data/lmdb_dataset.py:143`, `src/pedpredict/data/lmdb_dataset.py:282`, `src/pedpredict/data/feature_cache.py`.

## Model families and the current experiment

`models.registry` is the single model factory and forward adapter. Seven model types are supported: full, pedestrian-local, kinematics-only, visual-only, concatenation fusion, pose-kinematics, and pose-full. The registry's top-level prose claiming ablations are placeholder stubs is stale; its builders point to implemented classes.

Sources: `src/pedpredict/models/registry.py:1`, `src/pedpredict/models/registry.py:90`.

The full visual architecture combines a context backbone with a motion/tight-crop encoder. Pose-full instead uses the pose-kinematics encoder as queries over context features. Cross-attention uses motion as query and visual features as keys/values. The configurable residual adds motion content to the attention output; otherwise the motion stream influences the attention weights without directly contributing its content to the output. Feature interfaces use `d_model=128`.

Sources: `src/pedpredict/models/ensemble.py`, `src/pedpredict/models/ablations.py:455`, `src/pedpredict/models/cross_attention.py:111`, `configs/model.yaml:3`, `configs/model.yaml:48`.

The recent v4 campaign runs **pose_kinematics**, using `KinematicsOnlyModel`, rather than the expensive visual fusion model. Its encoder is Conv1d/BatchNorm/ReLU → feature projection and normalization → GRU → learned positional encoding and temporal self-attention → projection to 128 features. Shared head code allows binary and hazard objectives to be compared on the same encoder family.

Sources: `src/pedpredict/models/registry.py:98`, `src/pedpredict/models/ablations.py:157`, `src/pedpredict/models/ablations.py:286`, and per-run `resolved_config.yaml` files described in `EXPERIMENT_CONTEXT.md`.

The configured visual default remains TinyViT 5M with freezing. Historical v1 freezing disabled parameter gradients but did not freeze BatchNorm running statistics, and also froze a random projection. The explicit v2 options separate eval-mode backbone freezing from training the frame projection. Thus “frozen” is insufficient to establish that two visual experiments used equivalent recipes.

Sources: `configs/model.yaml:27`, `src/pedpredict/models/timm_backbone.py`, `src/pedpredict/training/schedule.py:93`, `tests/test_frozen_backbone.py`.

## Binary and hazard heads

The ordinary crossing score is the per-observed-frame classifier reduced by logsumexp. `crosses_pooled` is emitted optionally but is not the ordinary crossing loss/metric input. A temporal attention pool separately produces the feature vector for the hazard head. Consequently the **observed-time** axis in the binary head and the **future-time** axis in the hazard head have different meanings.

Sources: `src/pedpredict/models/heads.py:124`, `src/pedpredict/models/heads.py:137`, `src/pedpredict/models/heads.py:148`, `src/pedpredict/losses/multitask.py`.

The default hazard geometry is 96 future frames in bins of 4, yielding 24 hazards. The first eight bins produce the comparable 32-frame probability: `1 - product(1-h_k)`. The implementation computes stable log-survival and log-event probabilities in float32. The head is absent, including parameters, when disabled.

Sources: `configs/model.yaml:56`, `src/pedpredict/models/heads.py:111`, `src/pedpredict/models/heads.py:198`.

Three metadata fields—onset offset, observed future length, and whether the track ever crosses—reach the trainer through the dataset and `labels` dictionary. **This plumbing is implemented.** The early statement in CLAUDE.md that these fields are dropped is no longer true.

Sources: `src/pedpredict/data/lmdb_dataset.py:321`, `src/pedpredict/data/collate.py:48`, `src/pedpredict/data/onset_target.py:152`.

Hazard targets mark zero before the observed event, one in its bin, and mask later bins. Without an event, only fully observed bins are supervised. Already-crossed windows are excluded. The loss is binary cross-entropy summed over observed bins per window and averaged over valid windows; masked bins contribute zero gradient. Pure hazard, auxiliary hazard, and readout-regularized hazard use the same code with different weights. Metric routing is separate from loss routing.

Sources: `src/pedpredict/data/onset_target.py:152`, `src/pedpredict/losses/onset.py:77`, `src/pedpredict/losses/onset.py:111`; tests `test_onset_target.py`, `test_onset_loss.py`, `test_onset_plumbing.py` encode the intended contracts. Tests were inspected, not rerun as part of this mapping task.

## Evaluation and interpretation

Training selection and the primary research outcome are different. Current campaign checkpoint selection uses validation crossing AUC. The research comparison uses detection rate and lead time at five matched false-alarm budgets: 1200, 460, 205, 95, and 41 per hour. A detected crossing track has at least one alarm while crossing is still ahead. Lead time is the largest qualifying onset offset, which avoids relying on unavailable absolute ordering across occlusion segments.

Sources: `src/pedpredict/training/trainer.py:402`, `src/pedpredict/eval/detection_curve.py:63`, `src/pedpredict/eval/detection_curve.py:174`.

False alarms count either every alarming negative window or each negative track that ever alarms. Both normalize by negative-window exposure, derived from 30 fps and stride 3. These are pedestrian-window exposure rates, not measured driving-hour rates from a deployed alarm system. Detection thresholds are chosen from negative scores in the **same supplied dump**. Test curves therefore describe a matched operating-point comparison on the test distribution; they do not establish a fixed validation-calibrated threshold's deployed false-alarm rate. Validation-tuned window F1 is a separate reporting path.

Sources: `src/pedpredict/eval/detection_curve.py:72`, `src/pedpredict/eval/detection_curve.py:139`, `src/pedpredict/eval/detection_curve.py:204`, `src/pedpredict/eval/calibration.py`.

The preregistered decision rule compares mean detection-rate gaps against the sum of sample standard deviations and requires wins at at least three budgets. It is a declared screening rule, **not** a confidence interval, formal significance test, or equivalence test. An inconclusive result must not be translated into proof that objectives are equal.

Source: `src/pedpredict/eval/campaign_report.py:101`.

The combination script pairs seeds and checks window alignment, multiplies anchored and streaming scores for its primary rule, and fits the stacking secondary on validation only. Its causal smoothing uses decreasing remaining-future length as a time-order proxy; ordering across occlusion segments is explicitly approximate.

Sources: `scripts/report_combination.py:44`, `src/pedpredict/eval/campaign_report.py:73`, `src/pedpredict/eval/campaign_report.py:121`, `src/pedpredict/eval/campaign_report.py:133`.

## Engineering boundaries

The package has typed dataclass/YAML configuration, config validation and run-difference checking, per-run snapshots, CSV logs, golden parity tests, thin CLI wrappers, and separate train/eval/report paths. Configuration defaults describe the general codebase; historical and current experiment recipes must be read from the individual run snapshots. No training or model checkpoint loading was needed for this context build.

The mapped tree includes diagnostic arrays and logs that older orientation prose says are absent. That does not establish that the lab's raw data or checkpoint set is present. Conclusions here are grounded in visible code, documents, logs, local history, and checked derived artifacts. External paper citations have not been independently verified against their publishers.

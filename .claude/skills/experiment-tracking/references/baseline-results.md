# Baseline Results

**Update this file whenever a new best run is established.**
During Phase A the baseline is the **OLD repo's test metrics** — the parity target the rebuilt repo must
reproduce (within tolerance) per `model_type` before legacy is retired (Prompt 9.1).
Format: run name / source, date, key change, metrics.

---

## Phase-A Parity Target (OLD repo)

Fill these in by running the OLD repo's `test.py` per model_type. The rebuilt repo must match within
tolerance before cutover.

| model_type | Task | Accuracy | F1 | AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| full | crosses | — | — | — | — | — |
| full | looks   | — | — | — | — | — |
| full | actions | — | — | — | — | — |
| motion_only | crosses | — | — | — | — | — |
| visual_only | crosses | — | — | — | — | — |
| vanilla_concat | crosses | — | — | — | — | — |

**Compute (per model_type):** params — | FLOPs — | Latency —ms | FPS —

---

## Current Best Run (rebuilt repo)

**Run:** `<fill in run_id>`
**Date:** `<fill in>`
**Change vs previous:** `<what was different>`

### eval_log.csv summary (model_type = full)

| Task | Accuracy | F1 | AUC | Precision | Recall |
|------|----------|----|-----|-----------|--------|
| crosses | — | — | — | — | — |
| looks   | — | — | — | — | — |
| actions | — | — | — | — | — |

---

## Run History

| run_id | model_type | f1_crosses | auc_crosses | Notes |
|--------|------------|------------|-------------|-------|
| *(add rows as runs complete)* | | | | |

---

## How to Update

After `scripts/evaluate.py` completes on a new best:
1. Copy `eval_log.csv` values into the table above
2. Move previous best to Run History
3. Note what changed in "Change vs previous"
4. Update `outputs/runs/index.csv` accordingly

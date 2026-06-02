---
name: experiment-tracking
description: Conventions for logging, naming, checkpointing, and interpreting training runs in the rebuilt pedpredict repo (minimal yaml+CSV stack, no W&B). Use whenever starting a training run, saving/loading checkpoints, comparing runs, reading CSV logs, deciding if a result is an improvement, or writing eval/logging code. Trigger on "run", "checkpoint", "results", "metrics", "is this better", "compare runs", "save model", "resume training", or any question about whether a model improved.
---

# Experiment Tracking

Conventions for the **rebuilt** repo. The tracking stack is deliberately minimal: resolved-yaml config
snapshot + CSV logs, one run-dir per run, a cross-run index. No Hydra, no W&B. The implementation lives in
`src/pedpredict/utils/logging.py` (Prompt 4.5) and shares CSV schemas with `training/metrics.py` (Prompt 3.2).

> Canonical task names are `actions`, `looks`, `crosses` (the model output keys). Only `crosses_frame` is
> the supervised crosses logit. Use these names everywhere — do not reintroduce `crossing/looking/action`.

## Run Naming Convention

```
run_id = {timestamp}_{model_type}_{tag}
e.g.: 20260602_143015_full_lrsched
      20260602_181030_motion_only_baseline
```

- `timestamp`: YYYYMMDD_HHMMSS
- `model_type`: one of `full`, `motion_only`, `visual_only`, `vanilla_concat` (from the typed registry)
- `tag`: what changed vs previous run (1-3 words, underscored), from `TrainCfg.tag`
- Filesystem-safe — no spaces, no slashes

## Directory Structure Per Run

```
outputs/runs/
└── {run_id}/
    ├── config_resolved.yaml   # full resolved config (yaml→dataclass→CLI) snapshot at run start
    ├── train_log.csv          # per-epoch train + val metrics
    ├── eval_log.csv           # output of scripts/evaluate.py on this run
    ├── checkpoints/
    │   ├── best.pth           # best val metric (see "What Better Means")
    │   └── last.pth           # full-state end-of-epoch checkpoint
    └── plots/                 # figures regenerated from the CSVs
outputs/runs/index.csv         # one row per run, for cross-run comparison
```

Always snapshot the **resolved** config at run start (after yaml load + CLI overrides) — reconstructing
hyperparams later is unreliable. Paths come from `PathsCfg`; never commit weights/CSVs (see `.gitignore`, B11).

## CSV Log Schemas

`train_log.csv` columns:
```
epoch, train_loss, val_loss,
loss_actions, loss_looks, loss_crosses,                  # per-task loss breakdown
acc_actions, f1_actions, auc_actions,
acc_looks,   f1_looks,   auc_looks,
acc_crosses, f1_crosses, auc_crosses,
macro_f1, lr, epoch_time_s
```

`eval_log.csv` columns (from `scripts/evaluate.py`):
```
model_type, task, accuracy, f1, auc, precision, recall
```

`benchmark.csv` columns (from `eval/benchmark.py`, separate from accuracy metrics):
```
model_type, params, flops, latency_ms, fps, peak_vram_mb
```

`outputs/runs/index.csv` columns (cross-run comparison):
```
run_id, model_type, tag, best_epoch, f1_crosses, auc_crosses, f1_actions, f1_looks, val_loss
```

Keep these schemas in sync with `training/metrics.py`. Define columns once; both train-validation and
test use the same `MetricAccumulator` (no divergence).

## Checkpoint Save/Load Pattern

The B2 fix (eager ViT params, Prompt 2.1) means checkpoints load with `strict=True`. Save **full** training
state for true resume — not just the model weights.

**Save:**
```python
torch.save({
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "scaler_state": scaler.state_dict(),
    "best_metric": best_metric,
    "model_type": cfg.model.model_type,
    "config": cfg_dict,
}, path)
```

**Load for resume:**
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state"], strict=True)   # strict=True now that B2 is fixed
optimizer.load_state_dict(ckpt["optimizer_state"])
scheduler.load_state_dict(ckpt["scheduler_state"])
scaler.load_state_dict(ckpt["scaler_state"])
start_epoch = ckpt["epoch"] + 1
```

**Load for eval only:**
```python
ckpt = torch.load(path, map_location=device)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()                                              # don't load optimizer/scheduler/scaler
```

`strict=False` is now an explicit opt-in for debugging only — if it's needed for a normal load, that's a
regression of B2, not a workaround.

## What "Better" Means

**Primary metric: `f1_crosses`** (most task-critical; ~37:1 imbalance makes accuracy misleading).

**Decision rule:**
```
New run is better if:
  f1_crosses improves by > 0.5pp   ← meaningful
  AND val_loss does not increase by > 5%
```

**Metric priority:**
1. `f1_crosses` — primary
2. `auc_crosses` — secondary confirmation
3. `f1_looks`, `f1_actions` — should not regress > 1pp vs best run
4. `val_loss` — sanity check, not primary criterion

Do not optimize for accuracy alone. AUC is confirmation, not the target.

## Baseline Numbers

See `references/baseline-results.md`. During Phase A the meaningful baseline is **the OLD repo's test
metrics per model_type** — the parity target. Update the file when a clean-baseline run is established.
Never compare to numbers recalled from memory — read the file.

## Comparing Runs

1. Read `eval_log.csv` for each run (and `outputs/runs/index.csv` for a quick scan)
2. Diff `config_resolved.yaml` to identify what changed
3. Apply the metric priority above
4. Flag if any non-target task regressed > 1pp

## Common Pitfalls

- `best.pth` tracks the val metric, not the test F1 — always run `scripts/evaluate.py` before declaring a run "best".
- Resuming from `last.pth` restores scheduler + scaler state — double-check LR is as expected after resume.
- `benchmark.csv` numbers are hardware-specific — don't compare across machines.
- Phase A: a run only "counts" once its modules pass their golden-parity tests (see behavior-preserving-port).

## See Also

- `references/baseline-results.md` — current best / parity-target metrics per task.
- `behavior-preserving-port` — the porting workflow that gates Phase-A runs.

---
name: debugging-playbook
description: Decision-tree debugging guide for the rebuilt pedpredict PyTorch project. Use whenever diagnosing errors, crashes, silent failures, NaN losses, OOM, shape mismatches, slow training, golden-parity drift, or unexpected model outputs. Trigger on any error traceback, "why is X happening", "model not converging", "CUDA error", "loss is NaN", "shape mismatch", "parity failed", or "it's slow" in the context of this project.
---

# Debugging Playbook

Targets the rebuilt `src/pedpredict/` layout: config is yaml→dataclass→CLI (no `config.py`), model
selection is via the typed `registry`, hyperparameters live in `configs/*.yaml`, and the training loop is
`training/trainer.py` (not a god-script). Task/output keys are `actions, looks, crosses` /
`crosses_pooled, crosses_frame, temporal_weights`.

## Rule 0: Identify the Stage First

```
Data loading → Forward pass → Loss → Backward → Optimizer → Metrics
```
Isolate which stage fails before fixing anything. Use the checklist in `references/debug-checklist.md`.

---

## Golden-Parity Drift (Phase A)

A ported module's output no longer matches its golden fixture (see behavior-preserving-port).

1. **Is the difference expected?** If this module resolves a band-aid that changes behavior (B2/B4/B7),
   the golden test should assert the NEW behavior — update the test, document in MIGRATION.md.
2. **Weights identical?** Models: confirm the new module loaded the OLD `state_dict` from the fixture.
   Parity with different init is meaningless.
3. **Seed + inputs identical?** Re-run with the fixture's saved `seed` and exact inputs.
4. **Tolerance realistic?** An AMP/dtype reorder can exceed `1e-6`. Compare in fp32, pick tolerance
   deliberately, record it in the fixture.
5. **Find the first divergent op** — print intermediate tensors at each block boundary, old vs new.

---

## OOM (CUDA Out of Memory)

**Order of fixes — try in sequence, stop when resolved:**

1. Lower `train.batch_size` (yaml or `--train.batch_size`; halve it)
2. Ensure `torch.no_grad()` wraps all eval/inference code
3. Check for tensor accumulation in loops — `.detach()` before appending to lists
4. Use `utils.memory.free_cuda()` between chunks; check the `ChunkPrefetcher` RAM threshold
5. Lower `data.max_seq_len` or crop size (last resort — affects accuracy/parity)
6. Enable gradient checkpointing in the ViT

**Diagnosis:**
```python
print(torch.cuda.memory_summary())  # before crash
```

---

## NaN / Inf Loss

**Causes in priority order:**

1. **LR too high** — lower `train.lr`, most common cause
2. **AMP overflow** — check `scaler.get_scale()` isn't repeatedly collapsing to 1.0
3. **Zero division in loss** — check class weights and that label range matches each task's num_classes
4. **Bad input** — NaN in batch:
   ```python
   assert not torch.isnan(images_tight).any(), "NaN in tight crops"
   assert not torch.isnan(motion).any(), "NaN in motion"
   ```
5. **LogSumExp instability** in the crosses_frame pooling — inspect `frame_pool` path in `models/heads.py`

**Quick check:**
```python
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"NaN grad: {name}")
```

---

## Shape Mismatch

**Always print shapes at the boundary:**
```python
print(f"tight: {images_tight.shape}")    # expect (B,T,3,H,W)
print(f"context: {images_context.shape}")
print(f"motion: {motion.shape}")          # expect (B,T,8)
```

**Common mismatches:**

| Error | Cause | Fix |
|-------|-------|-----|
| `mat1 dim 1 != mat2 dim 0` | Embedding dim mismatch | Check `model.d_model` (=128) consistent across components |
| `Expected 4D tensor` | Missing batch or time dim | Check `data/collate.py` |
| `size mismatch at dim 1` | seq-len inconsistency | Check `data.max_seq_len`; confirm collate truncate/pad policy |
| `Expected input batch_size == target batch_size` | Loss got wrong labels | Verify label keys match the output contract (`MultiTaskLoss`) |
| motion last dim ≠ 8 | writer/collate disagreement (B7) | Confirm LMDB writer emits exactly 8 motion dims |

---

## CUDA / Device Errors

**`RuntimeError: Expected all tensors to be on same device`**
- Print `.device` for each input before forward
- Ensure `model.to(device)` after the model is assembled by the registry
- Ensure labels moved: `labels = {k: v.to(device) for k, v in labels.items()}`

**`CUDA error: device-side assert triggered`**
- Re-run with `CUDA_LAUNCH_BLOCKING=1` to get the real traceback
- Usually a label index out of range — check `crosses` is clamped to `{0,1}` (raw PIE labels are `{-1,0,1}`)

---

## Checkpoint Load Fails (`strict=True`)

The B2 fix means all params exist at `__init__`, so `strict=True` should succeed.

- **Missing/unexpected keys** → a param is still being created lazily (B2 regressed). Fix the model
  `__init__`, don't paper over with `strict=False`.
- **Don't** reach for the old dummy-forward materialization hack — it's gone on purpose.
- For resume, confirm the checkpoint carries optimizer/scheduler/scaler state, not just `model_state`.

---

## Slow Training

**Benchmark first:**
```python
import time
t = time.time(); next(iter(dataloader)); print(f"Batch load: {time.time()-t:.2f}s")
```

**If load time > forward time:** bottleneck is data
- Increase `train.num_workers`; confirm `pin_memory=True`
- Check `ChunkPrefetcher` preload depth and RAM threshold
- Confirm LMDB is on SSD, not a network drive

**If forward time is slow:**
- Confirm `utils.device.enable_perf_flags()` ran (cudnn.benchmark, TF32, high matmul precision)
- Confirm AMP is active (`autocast` + `GradScaler`)
- Profile a single batch

---

## Model Not Converging

1. Overfit one batch (tiny chunk, many steps) — loss should approach ~0; if not, bug in loss/labels
2. Check the LR schedule / warmup in `train.yaml`
3. Verify all three heads (`actions`, `looks`, `crosses`) contribute non-zero loss (per-task breakdown in train_log.csv)
4. Check per-task `loss_weight` — one task dominating?
5. Inspect predictions — collapsed to one class? (expected-ish on raw `crosses` accuracy; check F1/recall)

---

## Component Import Errors

Quick isolation against the new package:
```bash
python -c "from pedpredict.models.vit import ViT_Hierarchical; print('ViT OK')"
python -c "from pedpredict.models.motion_encoder import MotionEncoder; print('ME OK')"
python -c "from pedpredict.models.cross_attention import CrossAttentionModule; print('CA OK')"
python -c "from pedpredict.models.ensemble import EnsembleModel; print('Ensemble OK')"
python -c "from pedpredict.models.registry import build_model; print('registry OK')"
```

---

## See Also

- `references/debug-checklist.md` — step-by-step checklist before changing code.
- `behavior-preserving-port` — golden-fixture workflow; start here for parity-drift bugs.

"""Capture the Prompt 4.1 trainer-step golden (provenance, not a test).

OLD ``train.py`` cannot be imported (module-load side effects + ``mp`` + LMDB), and the full loop is
stochastic (sampler / shuffle / workers). What 4.1 must preserve is the **orchestration**, not new math —
so the oracle TRANSCRIBES VERBATIM the deterministic inner loops:

  * train step — ``OLD/Undergrad_thesis_project/train.py:140-164`` (zero_grad -> forward -> per-head CE
    accumulation -> backward -> clip_grad_norm_(1.0) -> step), AMP OFF (the ``scaler is None`` branch).
  * validate  — ``train.py:204-228`` (forward -> per-head CE summed over samples -> argmax accuracy) and
    the ``val_loss = loss_sum / n_samples`` reduction at ``:572``.

It runs them on the NEW model (``registry.build_model``) under a fixed init ``state_dict`` + fixed
synthetic batches, so the test can replay the same init and assert the new ``Trainer`` reproduces:
the per-batch loss, the post-step weights, the val loss, and the per-task correct counts.

Determinism note: the train step runs ``model.train()`` (Dropout/DropPath active). Both this capture and
the test seed ``torch.manual_seed(STEP_SEED)`` immediately before the loop and consume RNG identically
(only the forward draws), so the dropout masks match -> bitwise-stable weights. Validation runs
``model.eval()`` (dropout = identity), so it needs no seeding. Re-run after touching the source::

    python tests/_capture/capture_trainer_golden.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from pedpredict.config.schema import RootCfg
from pedpredict.models.registry import build_model, forward_model

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "trainer_step.pt"

TASKS = ("actions", "looks", "crosses")
BUILD_SEED = 4101
TRAIN_DATA_SEED = 4102
VAL_DATA_SEED = 4103
STEP_SEED = 4104
B, T = 2, 2
# Inverse-frequency-style class weights (crosses minority strongly upweighted) — mirrors 3.1's capture.
CLASS_WEIGHTS = {"actions": [0.9, 1.1], "looks": [0.6, 3.0], "crosses": [0.51, 19.0]}


def _make_batch(seed: int) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    images_tight = torch.randn(B, T, 3, 128, 128, generator=gen)
    images_context = torch.randn(B, T, 3, 224, 224, generator=gen)
    motions = torch.randn(B, T, 8, generator=gen)
    labels = {task: torch.randint(0, 2, (B,), generator=gen) for task in TASKS}
    return images_tight, images_context, motions, labels


# ----------------------------------------------------------------- OLD oracle (verbatim transcription)


def _legacy_train_step(model, batches, criterion, loss_weight, optimizer):
    """Transcribed from train.py:140-164 (scaler=None / AMP-off branch). Returns per-batch totals."""
    model.train()
    torch.manual_seed(STEP_SEED)
    per_batch_total = []
    for images_tight, images_context, motions, labels in batches:
        labels = {k: v.long() for k, v in labels.items()}
        labels["crosses"] = torch.clamp(labels["crosses"], 0, 1)   # remap_cross_labels
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_model(model, images_tight, images_context, motions)
        total = torch.tensor(0.0)
        for name in TASKS:
            logits = outputs["crosses_frame"] if name == "crosses" else outputs[name]
            head_loss = criterion[name](logits.float(), labels[name])
            total = total + loss_weight[name] * head_loss
        total.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        per_batch_total.append(total.detach().clone())
    return per_batch_total


def _legacy_validate(model, batches, criterion, loss_weight):
    """Transcribed from train.py:204-228,572. Returns (val_loss, correct, n_samples, per_task_acc)."""
    model.eval()
    loss_sum = 0.0
    n_samples = 0
    correct = {name: 0 for name in TASKS}
    with torch.inference_mode():
        for images_tight, images_context, motions, labels in batches:
            batch_size = images_tight.size(0)
            labels = {k: v.long() for k, v in labels.items()}
            labels["crosses"] = torch.clamp(labels["crosses"], 0, 1)
            outputs = forward_model(model, images_tight, images_context, motions)
            for name in TASKS:
                logits = outputs["crosses_frame"] if name == "crosses" else outputs[name]
                loss_i = criterion[name](logits.float(), labels[name])
                loss_sum += loss_weight[name] * loss_i.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                correct[name] += int((preds == labels[name]).sum())
            n_samples += batch_size
    val_loss = loss_sum / n_samples
    per_task_acc = {name: correct[name] / n_samples for name in TASKS}
    return val_loss, correct, n_samples, per_task_acc


def main() -> None:
    torch.manual_seed(BUILD_SEED)
    cfg = RootCfg()
    model = build_model(cfg)                                  # full model, cpu, img_size=224 (B2: eager)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    train_batches = [_make_batch(TRAIN_DATA_SEED)]
    val_batches = [_make_batch(VAL_DATA_SEED)]
    class_weights = {t: torch.tensor(CLASS_WEIGHTS[t], dtype=torch.float32) for t in TASKS}
    loss_weight = dict(cfg.train.loss_weight)
    criterion = {t: nn.CrossEntropyLoss(weight=class_weights[t]) for t in TASKS}

    # --- train-step oracle (starts from init_state) ---
    model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    per_batch_total = _legacy_train_step(model, train_batches, criterion, loss_weight, optimizer)
    post_step_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # --- validate oracle (on the init weights, independent of the train step) ---
    model.load_state_dict(init_state)
    val_loss, correct, n_samples, per_task_acc = _legacy_validate(
        model, val_batches, criterion, loss_weight
    )

    fixture = {
        "init_state": init_state,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "class_weights": class_weights,
        "loss_weight": loss_weight,
        "lr": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "grad_clip_max_norm": cfg.train.grad_clip_max_norm,
        "step_seed": STEP_SEED,
        "tasks": list(TASKS),
        "tol": 1e-6,
        "expected": {
            "per_batch_total": per_batch_total,
            "post_step_state": post_step_state,
            "val_loss": val_loss,
            "val_correct": correct,
            "val_n_samples": n_samples,
            "val_per_task_acc": per_task_acc,
        },
        "meta": {
            "src": "OLD/Undergrad_thesis_project/train.py:140-164,204-228,572",
            "build_seed": BUILD_SEED,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)
    print(f"wrote {OUT}")
    print(f"  per_batch_total={[round(t.item(), 6) for t in per_batch_total]}")
    print(f"  val_loss={val_loss:.6f} correct={correct} n={n_samples}")


if __name__ == "__main__":
    main()

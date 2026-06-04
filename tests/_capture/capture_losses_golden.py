"""Capture golden fixtures for Prompt 3.1 by running the OLD loss code (provenance, not a test).

OLD ``train.py`` cannot be imported (it pulls torch/PIE/model imports at module load), but the loss
accumulation it performs is self-contained tensor math. The oracle is TRANSCRIBED VERBATIM below from
``OLD/Undergrad_thesis_project/train.py:144-153`` (the per-head CE accumulation in ``train_one_chunk``)
together with the criterion build at ``train.py:341-345``. If that source changes, re-transcribe and
rerun::

    python tests/_capture/capture_losses_golden.py

Parity notes:
  * The loss is pure tensor math, so synthetic ``outputs`` / ``labels`` / ``class_weights`` give a
    fully deterministic (tol=0 up to float round-off) fixture — no LMDB needed.
  * ``outputs`` deliberately carries the FULL output contract: ``actions`` / ``looks`` / ``crosses_frame``
    (supervised) PLUS ``crosses_pooled`` / ``temporal_weights`` (emitted but UNSUPERVISED, B4). The oracle
    reads ONLY ``crosses_frame`` for the crosses task — the fixture thus pins that ``crosses_pooled`` never
    enters the loss.
  * ``loss_weight`` and ``class_weights`` mirror the training-effective values: loss_weight =
    ``{actions:0.8, looks:0.8, crosses:1.2}`` (TrainCfg.loss_weight); class weights are an inverse-freq
    style imbalance (crosses minority strongly upweighted).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "losses_cases.pt"

GEN_SEED = 3001
TASKS = ("actions", "looks", "crosses")
LOSS_WEIGHT = {"actions": 0.8, "looks": 0.8, "crosses": 1.2}   # TrainCfg.loss_weight
# Inverse-frequency-style class weights (class-1 minority upweighted, crosses most severe).
CLASS_WEIGHTS = {
    "actions": [0.9, 1.1],
    "looks": [0.6, 3.0],
    "crosses": [0.51, 19.0],
}


# ----------------------------------------------------------------- OLD oracle (verbatim transcription)
# Transcribed from OLD/Undergrad_thesis_project/train.py:144-153 + :341-345. Do not "improve".


def _legacy_total_and_heads(outputs, labels, class_weights, loss_weight, device):
    criterion = {                                          # train.py:341-345
        name: nn.CrossEntropyLoss(weight=class_weights[name]) for name in TASKS
    }
    per_task = {}
    total = torch.tensor(0.0, device=device)               # train.py:144
    for name in TASKS:                                     # train.py:145-153
        if name == "crosses":
            logits = outputs["crosses_frame"]
        else:
            logits = outputs[name]
        targets = labels[name]
        head_loss = criterion[name](logits.float(), targets)
        per_task[name] = head_loss.detach().clone()
        total = total + loss_weight.get(name, 1.0) * head_loss
    return total, per_task


# ----------------------------------------------------------------- synthetic batch


def _make_batch(batch_size: int = 12, seq_len: int = 20):
    gen = torch.Generator().manual_seed(GEN_SEED)
    outputs = {
        "actions": torch.randn(batch_size, 2, generator=gen),
        "looks": torch.randn(batch_size, 2, generator=gen),
        "crosses_frame": torch.randn(batch_size, 2, generator=gen),
        # UNSUPERVISED extras — present in the contract, must never affect the loss.
        "crosses_pooled": torch.randn(batch_size, 2, generator=gen),
        "temporal_weights": torch.softmax(torch.randn(batch_size, seq_len, generator=gen), dim=1),
    }
    labels = {
        "actions": torch.randint(0, 2, (batch_size,), generator=gen),
        "looks": torch.randint(0, 2, (batch_size,), generator=gen),
        "crosses": torch.randint(0, 2, (batch_size,), generator=gen),
    }
    return outputs, labels


def main() -> None:
    device = torch.device("cpu")
    outputs, labels = _make_batch()
    class_weights = {t: torch.tensor(CLASS_WEIGHTS[t], dtype=torch.float32) for t in TASKS}

    total, per_task = _legacy_total_and_heads(outputs, labels, class_weights, LOSS_WEIGHT, device)
    weighted = {t: LOSS_WEIGHT[t] * per_task[t] for t in TASKS}

    fixture = {
        "outputs": outputs,
        "labels": labels,
        "class_weights": {t: class_weights[t] for t in TASKS},
        "loss_weight": LOSS_WEIGHT,
        "tasks": list(TASKS),
        "tol": 1e-6,
        "expected": {
            "total": total.detach().clone(),
            "per_task": {t: per_task[t] for t in TASKS},        # raw mean-CE per task
            "weighted": {t: weighted[t].detach().clone() for t in TASKS},
        },
        "meta": {
            "src": "OLD/Undergrad_thesis_project/train.py:144-153,341-345",
            "gen_seed": GEN_SEED,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    print(f"wrote {OUT}")
    print(f"  total={total.item():.6f}")
    for t in TASKS:
        print(f"  {t}: raw_ce={per_task[t].item():.6f} weighted={weighted[t].item():.6f}")


if __name__ == "__main__":
    main()

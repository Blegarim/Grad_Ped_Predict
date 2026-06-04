"""Capture golden fixtures for Prompt 1.6 by running the OLD weight code (provenance, not a test).

OLD ``train.py`` cannot be imported (it pulls torch/PIE/model imports at module load), but the two
functions being unified are self-contained. They are TRANSCRIBED VERBATIM below from
``OLD/Undergrad_thesis_project/train.py:34-123`` (``compute_class_weights_from_lmdb`` and
``build_sampler_weights`` / ``_inverse_class_weights``) — the parity oracle. If that source changes,
re-transcribe and rerun::

    python tests/_capture/capture_sampler_golden.py

Parity notes:
  * The unified module dedups only the LMDB *scan*; the two inverse-frequency *formulas* legitimately
    differ (loss: ``t/(2·max(c,1))``; sampler: ``t/(len(counts)·c)``) and are preserved verbatim, so
    this fixture snapshots BOTH oracles independently.
  * Determinism: synthetic label rows are generated once with a fixed seed AND saved verbatim into the
    fixture; the test rebuilds tiny label-only LMDBs from them, so the test never re-generates data.
  * ``seq_ids`` are captured in LMDB cursor (lexicographic) order — the same order
    ``LMDBChunkDataset.seq_ids`` and ``scan_chunk_labels`` use; per-sample weight alignment depends on it.
  * Data deliberately spans ``{-1, 0, 1}`` crosses (exercises the clamp) and includes a chunk whose
    crosses are all 0 (exercises the absent-class branch: sampler ``0.0`` weight, loss ``max(c,1)`` floor).
"""

from __future__ import annotations

import json
import pickle
import random
import tempfile
from collections import Counter
from pathlib import Path

import lmdb
import torch

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "sampler_cases.json"

GEN_SEED = 2024
POWERS = {"crosses": 1.5, "actions": 0.3, "looks": 0.7}   # TrainCfg.sampler_powers
MIN_WEIGHT = 1e-6                                          # TrainCfg.sampler_min_weight


# ----------------------------------------------------------------- OLD oracle (verbatim transcription)
# Transcribed from OLD/Undergrad_thesis_project/train.py:34-123. Do not "improve" — parity surface.


def compute_class_weights_from_lmdb(lmdb_paths, device):  # train.py:34-72
    counts = {task: [0, 0] for task in ["actions", "looks", "crosses"]}
    for path in lmdb_paths:
        env = lmdb.open(path, readonly=True, lock=False)
        try:
            with env.begin(write=False) as txn:
                for key, value in txn.cursor():
                    key_str = key.decode()
                    if key_str.endswith("_meta"):
                        meta = pickle.loads(value)
                        for task in counts:
                            label = int(meta.get(task, 0))
                            if task == "crosses":
                                label = max(0, min(1, label))
                            if label in [0, 1]:
                                counts[task][label] += 1
        finally:
            env.close()
    weights = {}
    for task, cnt in counts.items():
        total = sum(cnt)
        if total > 0:
            cnt0, cnt1 = cnt[0], cnt[1]
            weights[task] = torch.tensor(
                [total / (2 * max(cnt0, 1)), total / (2 * max(cnt1, 1))],
                dtype=torch.float32, device=device,
            )
        else:
            weights[task] = torch.tensor([1.0, 1.0], device=device)
    return weights


def _inverse_class_weights(counts):  # train.py:74-83
    total = sum(counts.values())
    n_classes = len(counts)
    weights = {}
    for k, v in counts.items():
        if v == 0:
            weights[k] = 0.0
        else:
            weights[k] = total / (n_classes * v)
    return weights


def build_sampler_weights(lmdb_path, seq_ids, cross_pow=1.0, action_pow=0.5, look_pow=0.5,
                          min_weight=1e-6):  # train.py:85-123
    label_rows = []
    counts = {"actions": Counter(), "looks": Counter(), "crosses": Counter()}
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    try:
        with env.begin(write=False) as txn:
            for seq_id in seq_ids:
                meta = pickle.loads(txn.get(f"{seq_id}_meta".encode()))
                actions = int(meta["actions"])
                looks = int(meta["looks"])
                crosses = int(meta["crosses"])
                if crosses < 0:
                    crosses = 0
                label_rows.append((actions, looks, crosses))
                counts["actions"][actions] += 1
                counts["looks"][looks] += 1
                counts["crosses"][crosses] += 1
    finally:
        env.close()
    action_w = _inverse_class_weights(counts["actions"])
    look_w = _inverse_class_weights(counts["looks"])
    cross_w = _inverse_class_weights(counts["crosses"])
    weights = []
    for actions, looks, crosses in label_rows:
        weight = max(min_weight, cross_w.get(crosses, min_weight)) ** cross_pow
        if action_pow > 0:
            weight *= max(min_weight, action_w.get(actions, min_weight)) ** action_pow
        if look_pow > 0:
            weight *= max(min_weight, look_w.get(looks, min_weight)) ** look_pow
        weights.append(weight)
    return weights, counts


# ----------------------------------------------------------------- synthetic data + LMDB writer


def _make_chunks() -> list[list[dict]]:
    """Two deterministic label-only chunks. Chunk 0: rare crosses w/ a few -1; chunk 1: crosses all 0."""
    rng = random.Random(GEN_SEED)
    chunk0: list[dict] = []
    for _ in range(40):
        roll = rng.random()
        crosses = 1 if roll < 0.10 else (-1 if roll < 0.15 else 0)   # -1 clamps to 0
        chunk0.append({
            "actions": 1 if rng.random() < 0.45 else 0,
            "looks": 1 if rng.random() < 0.17 else 0,
            "crosses": crosses,
        })
    chunk1: list[dict] = []
    for _ in range(15):                                              # no crosses=1 → absent-class branch
        chunk1.append({
            "actions": 1 if rng.random() < 0.50 else 0,
            "looks": 1 if rng.random() < 0.20 else 0,
            "crosses": 0,
        })
    return [chunk0, chunk1]


def _write_label_lmdb(records: list[dict], path: str) -> list[str]:
    """Write label-only ``<i>_meta`` pickles; return seq_ids in LMDB cursor (lexicographic) order."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for i, rec in enumerate(records):
                txn.put(f"{i}_meta".encode(), pickle.dumps(rec))
    finally:
        env.close()
    env = lmdb.open(path, readonly=True, lock=False)
    seq_ids: list[str] = []
    try:
        with env.begin(write=False) as txn:
            for key, _ in txn.cursor():
                ks = key.decode()
                if ks.endswith("_meta"):
                    seq_ids.append(ks.split("_")[0])
    finally:
        env.close()
    return seq_ids


def main() -> None:
    chunks = _make_chunks()
    with tempfile.TemporaryDirectory() as tmp:
        paths = [str(Path(tmp) / f"chunk{i}.lmdb") for i in range(len(chunks))]
        seq_ids = [_write_label_lmdb(recs, p) for recs, p in zip(chunks, paths, strict=True)]

        # GLOBAL class weights (loss lever) — one scan over all chunks
        cls_w = compute_class_weights_from_lmdb(paths, device="cpu")
        class_weights = {task: cls_w[task].tolist() for task in ("actions", "looks", "crosses")}

        # PER-CHUNK sampler weights (online lever)
        chunk_cases = []
        for recs, p, sids in zip(chunks, paths, seq_ids, strict=True):
            w, counts = build_sampler_weights(
                p, sids,
                cross_pow=POWERS["crosses"], action_pow=POWERS["actions"], look_pow=POWERS["looks"],
                min_weight=MIN_WEIGHT,
            )
            chunk_cases.append({
                "records": recs,                  # label dicts in original write order
                "seq_ids": sids,                  # cursor order (parity-critical)
                "sample_weights": w,              # aligned to seq_ids order
                "counts": {t: dict(counts[t]) for t in ("actions", "looks", "crosses")},
            })

    fixture = {
        "powers": POWERS,
        "min_weight": MIN_WEIGHT,
        "tol": 1e-6,
        "class_weights": class_weights,           # GLOBAL, aggregated over both chunks
        "chunks": chunk_cases,
        "meta": {
            "src": "OLD/Undergrad_thesis_project/train.py:34-123",
            "gen_seed": GEN_SEED,
            "n_chunks": len(chunks),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2)

    print(f"wrote {OUT}")
    print(f"  class_weights (global): {class_weights}")
    for i, c in enumerate(chunk_cases):
        print(f"  chunk{i}: n={len(c['seq_ids'])} counts={c['counts']}")


if __name__ == "__main__":
    main()

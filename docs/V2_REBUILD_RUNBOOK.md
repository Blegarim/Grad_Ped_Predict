# v2 Rebuild Runbook (hole audit, attack-order step 2)

The dataset-touching code for the ONE v2 rebuild is landed (M3 relabel, M4 censor filter, M6
`track_id`, M9 ego-speed, A4 motion fixes, M5 benchmark eval set). This is the execution checklist
for the machine that has the PIE dataset (`data/` per [setup.md](../setup.md) — annotations are
enough for step 1; frames/clips only matter from step 3 on).

**v1 artifacts are obsolete**: the ~20k v1 train chunks AND the v1 val/test LMDBs all predate the
v2 meta contract (the runtime dataset now hard-errors on v1 chunks). Sequence pkls are also v1.

## 1. Regenerate sequences (annotations only — fast, no frames needed)

```bash
python scripts/make_sequences.py --split all
python scripts/make_sequences.py --benchmark
```

- Prints per-split windowing stats and writes `sequences_<split>_stats.json`. The **`censored`**
  count is the thesis sentence "N windows excluded as right-censored" (M4) — record it.
- Expect N and all positive rates to shift vs v1; `actions`/`looks` rates should **drop**
  (state-at-end labeling can only deflate vs `any()`-over-32-frames), `looks` hardest.

## 2. Pin the new statistics (same change — doc-sync checklist)

1. Update the **Dataset Statistics** table in `CLAUDE.md` (drop the ⚠️ STALE banner) — the v2
   counts come straight from the step-1 output.
2. Re-pin `tests/fixtures/golden/pie_sequences_counts.json` with the new per-split
   `{N, actions, looks, crosses}` (the slow test `test_pkl_counts_match_fixture` and
   `scripts/count_labels.py` both diff against it).
3. Re-check `train.sampler_powers` for `looks` if its rate fell far (M3 resolution note).

## 3. Delete v1 LMDBs, rebuild all splits

```bash
# delete: preprocessed_train/ preprocessed_train_aug/ preprocessed_val/ preprocessed_test/
python scripts/build_lmdb.py --split val
python scripts/build_lmdb.py --split test
python scripts/build_lmdb.py --split test_benchmark        # M5 eval set (small, test split only)
python scripts/build_lmdb_incremental.py --split train     # disk-bounded; C2 resume guard is active
python scripts/augment_dataset.py                          # rebuild preprocessed_train_aug from v2 pkl
python scripts/count_labels.py                             # drift gate vs the re-pinned fixture
```

Frame staging order per [setup.md](../setup.md) (val → test → train rounds on a storage-limited disk).

## What changed underneath (for orientation)

| Hole | Landed as |
|---|---|
| M3 | `actions`/`looks` = state at last observed frame; `crosses` stays future-any (`pie_sequences._label_window`) |
| M4 | windows with truncated future are dropped + counted (`WindowStats`) |
| M6 | `track_id` in `SequenceRecord` → LMDB meta → dataset items (eval aggregation comes next, eval-side) |
| M9 | ego-speed stored as motion channel 8; consumed width = `data.motion_dim` (8 default, 9 = ego run) |
| A4 | frame-0 deltas = 0; flip reflects `cx`; norm choice = `model.motion_norm` runtime flag (`image` default, `per_sequence` = legacy arm) |
| M5 | `--benchmark` sequence mode + `test_benchmark` build target → `preprocessed_test_benchmark` with `tte` in meta |

Still pending after the rebuild (not data-side): eval-side track aggregation (M6 second half),
benchmark-split evaluation wiring in `evaluate.py` (M5 eval row), then WP0 registry wave
(kinematics-only, `motion_only`→`ped_local` rename, single-task configs).

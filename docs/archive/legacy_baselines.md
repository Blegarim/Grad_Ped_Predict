# Legacy baselines (archived at cutover)

Salvaged from the OLD undergrad repo's untracked `training_log/` before it was retired at the P9
cutover. The full OLD repo (code + these logs) is preserved in the **`legacy-archive`** git tag:

```bash
git checkout legacy-archive   # OLD/Undergrad_thesis_project/training_log/*.csv
```

These are recorded for historical reference only. Phase-A behavior preservation is proven by the
module-level golden tests (`tests/fixtures/golden/`), **not** by these end-to-end numbers.

## Dataset label totals (`label_count.csv`)

Per-class counts over the generated sequences (raw `crosses` includes the pre-clamp `-1` bucket):

| Task | label=1 | label=0 | label=-1 |
|---|---|---|---|
| actions | 27,124 | 24,831 | — |
| looks | 5,034 | 46,921 | — |
| crosses | 10,075 | 40,141 | 1,739 |

(The canonical, clamped positive-class rates the project uses are the Dataset Statistics table in
[CLAUDE.md](../../CLAUDE.md), sourced from `tests/fixtures/golden/pie_sequences_counts.json`.)

## Test metrics (`test_log_*.csv`)

The OLD `test.py` wrote **per-chunk** rows (not per-model aggregates), and its model-suffix logging
was buggy — the `_motion_only` log was byte-identical to the unsuffixed one (the suffix-naming defect
noted during the ablation port). They are therefore **not reliable per-`model_type` baselines** and are
not reproduced here. The raw CSVs remain in the `legacy-archive` tag for anyone who wants them.

When trained weights + the PIE dataset are available, regenerate clean per-`model_type` test metrics
via the rebuilt evaluator (`python scripts/evaluate.py`) and record them here.

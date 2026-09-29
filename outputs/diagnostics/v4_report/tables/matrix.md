# Cross-protocol matrix (test split; AUC · raw F1 → val-tuned F1; mean ± sd over seeds)

## Model B (crossing only, n=3)

| trained on ↓ / tested on → | anchored | streaming |
|---|---|---|
| anchored | 0.889 ± 0.007 · 0.730 ± 0.017 → 0.729 ± 0.015 | 0.516 ± 0.010 · 0.063 ± 0.002 → 0.062 ± 0.002 |
| streaming | 0.696 ± 0.028 · 0.388 ± 0.071 → 0.459 ± 0.031 | 0.783 ± 0.010 · 0.237 ± 0.023 → 0.243 ± 0.008 |

## Model A (3-task, n=1)

| trained on ↓ / tested on → | anchored | streaming |
|---|---|---|
| anchored | 0.869 · 0.710 → 0.711 | 0.511 · 0.062 → 0.064 |
| streaming | 0.711 · 0.397 → 0.510 | 0.785 · 0.225 → 0.265 |

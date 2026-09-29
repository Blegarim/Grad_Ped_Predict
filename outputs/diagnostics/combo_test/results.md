# Anchored who × streaming when — results (pre-registered: PREREG.md)

Detection rate % (`per_window`, streaming test, mean ± sd over 3 paired seeds) and window AUC.

| score | AUC | @1200/hr | @460/hr | @205/hr | @95/hr | @41/hr |
|---|---|---|---|---|---|---|
| R2 alone (baseline) | 0.783 ± 0.010 | 69.6 ± 6.0 | 55.0 ± 6.2 | 42.3 ± 3.2 | 32.0 ± 4.2 | 22.9 ± 3.5 |
| anchored alone | 0.516 ± 0.010 | 35.9 ± 6.2 | 24.1 ± 2.9 | 15.1 ± 2.6 | 8.9 ± 2.9 | 4.1 ± 2.3 |
| R3 alone | 0.793 ± 0.010 | 72.0 ± 2.8 | 59.2 ± 4.9 | 44.1 ± 4.7 | 31.4 ± 5.9 | 17.4 ± 9.1 |
| PRIMARY: anchored × R2 | 0.757 ± 0.021 | 69.8 ± 5.2 | 54.3 ± 4.3 | 42.3 ± 2.0 | 31.2 ± 1.8 | 23.1 ± 4.3 |
| S1: causal intent (running mean of anchored) × R2 | 0.760 ± 0.020 | 63.9 ± 9.1 | 45.5 ± 5.7 | 33.3 ± 5.2 | 22.9 ± 3.5 | 15.9 ± 4.5 |
| S2: stacked on val (anchored, R2) | 0.781 ± 0.010 | 69.3 ± 6.3 | 54.6 ± 6.4 | 41.6 ± 4.1 | 32.2 ± 3.9 | 22.8 ± 4.5 |
| S3: anchored × R3 | 0.750 ± 0.013 | 70.4 ± 0.7 | 55.8 ± 2.4 | 45.0 ± 1.1 | 33.7 ± 3.8 | 23.9 ± 2.1 |

## Pre-registered criterion vs R2 alone (beats R2 beyond the sd sum at >= 3 of 5 budgets)

- anchored alone, `per_window`: **R2 alone (baseline) WINS**
- anchored alone, `per_track`: **INCONCLUSIVE**
- R3 alone, `per_window`: **INCONCLUSIVE**
- R3 alone, `per_track`: **INCONCLUSIVE**
- PRIMARY: anchored × R2, `per_window` **(PRIMARY)**: **INCONCLUSIVE**
- PRIMARY: anchored × R2, `per_track`: **INCONCLUSIVE**
- S1: causal intent (running mean of anchored) × R2, `per_window`: **INCONCLUSIVE**
- S1: causal intent (running mean of anchored) × R2, `per_track`: **INCONCLUSIVE**
- S2: stacked on val (anchored, R2), `per_window`: **INCONCLUSIVE**
- S2: stacked on val (anchored, R2), `per_track`: **INCONCLUSIVE**
- S3: anchored × R3, `per_window`: **INCONCLUSIVE**
- S3: anchored × R3, `per_track`: **INCONCLUSIVE**

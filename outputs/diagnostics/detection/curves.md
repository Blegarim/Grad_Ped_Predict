## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | binary (R1 frame) (n=1) | R3 pure (n=1) | R4 hedge (n=1) | R3C censored (n=1) |
|---|---|---|---|---|
| 1200 | 62.0% @ 6.09 s | 66.8% @ 6.52 s | 49.8% @ 4.76 s | 65.4% @ 8.11 s |
| 460 | 34.6% @ 3.40 s | 56.6% @ 4.40 s | 28.8% @ 3.16 s | 45.4% @ 4.55 s |
| 205 | 22.9% @ 2.97 s | 48.8% @ 3.05 s | 14.6% @ 2.73 s | 31.2% @ 3.12 s |
| 95 | 14.1% @ 1.99 s | 41.0% @ 1.84 s | 9.3% @ 2.65 s | 18.5% @ 2.36 s |
| 41 | 5.9% @ 0.92 s | 34.1% @ 0.72 s | 3.9% @ 1.25 s | 10.7% @ 2.51 s |

## binary (R1 frame) — r1 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6026 | 62.0% (127/205) | 6.09 s | 1.90 s |
| 460 | 459.2 | 0.9325 | 34.6% (71/205) | 3.40 s | 1.47 s |
| 205 | 204.5 | 0.9777 | 22.9% (47/205) | 2.97 s | 1.40 s |
| 95 | 94.5 | 0.9924 | 14.1% (29/205) | 1.99 s | 1.07 s |
| 41 | 40.5 | 0.9967 | 5.9% (12/205) | 0.92 s | 0.70 s |

## R3 pure — r3 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.4574 | 66.8% (137/205) | 6.52 s | 2.50 s |
| 460 | 459.2 | 0.6786 | 56.6% (116/205) | 4.40 s | 1.58 s |
| 205 | 204.5 | 0.8031 | 48.8% (100/205) | 3.05 s | 0.68 s |
| 95 | 94.5 | 0.8762 | 41.0% (84/205) | 1.84 s | 0.17 s |
| 41 | 40.5 | 0.9215 | 34.1% (70/205) | 0.72 s | 0.07 s |

## R4 hedge — r4 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.8586 | 49.8% (102/205) | 4.76 s | 1.47 s |
| 460 | 459.2 | 0.9577 | 28.8% (59/205) | 3.16 s | 0.90 s |
| 205 | 204.5 | 0.9817 | 14.6% (30/205) | 2.73 s | 0.73 s |
| 95 | 94.5 | 0.9894 | 9.3% (19/205) | 2.65 s | 0.57 s |
| 41 | 40.5 | 0.9963 | 3.9% (8/205) | 1.25 s | 1.10 s |

## R3C censored — r3c (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5911 | 65.4% (134/205) | 8.11 s | 2.63 s |
| 460 | 459.2 | 0.8008 | 45.4% (93/205) | 4.55 s | 2.27 s |
| 205 | 204.5 | 0.8982 | 31.2% (64/205) | 3.12 s | 1.60 s |
| 95 | 94.5 | 0.9579 | 18.5% (38/205) | 2.36 s | 0.90 s |
| 41 | 40.5 | 0.9763 | 10.7% (22/205) | 2.51 s | 0.83 s |


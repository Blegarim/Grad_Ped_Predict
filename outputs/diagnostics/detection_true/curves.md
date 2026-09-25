## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | binary (n=1) | hazard (n=1) |
|---|---|---|
| 1200 | 100.0% @ 14.34 s | 100.0% @ 14.34 s |
| 460 | 98.0% @ 14.02 s | 98.5% @ 14.00 s |
| 205 | 84.9% @ 9.58 s | 84.4% @ 10.00 s |
| 95 | 49.8% @ 4.64 s | 60.5% @ 5.52 s |
| 41 | 27.3% @ 3.76 s | 47.8% @ 3.03 s |

## binary — r2s_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0025 | 98.0% (201/205) | 14.02 s | 6.97 s |
| 205 | 204.5 | 0.1266 | 84.9% (174/205) | 9.58 s | 3.55 s |
| 95 | 94.5 | 0.6190 | 49.8% (102/205) | 4.64 s | 2.05 s |
| 41 | 40.5 | 0.8468 | 27.3% (56/205) | 3.76 s | 1.60 s |

## hazard — r3 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0084 | 98.5% (202/205) | 14.00 s | 6.43 s |
| 205 | 204.5 | 0.2362 | 84.4% (173/205) | 10.00 s | 3.40 s |
| 95 | 94.5 | 0.5556 | 60.5% (124/205) | 5.52 s | 2.10 s |
| 41 | 40.5 | 0.8152 | 47.8% (98/205) | 3.03 s | 0.67 s |


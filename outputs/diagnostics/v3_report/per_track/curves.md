## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | GBM probe (n=3) | R2s v1 block-order (n=3) | R3 v1 block-order (n=3) | R2 v3 shuffled (n=1) | R3 v3 shuffled (n=1) | PF control (n=1) | PF +std (n=1) | PF +bs32 (n=1) | PF fix (n=3) | PF fix onset (n=1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1200 | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0% @ 14.34 s |
| 460 | 99.8±0.3% @ 13.66 s | 98.2±0.3% @ 13.83 s | 98.5±0.0% @ 14.01 s | 98.0% @ 13.98 s | 99.0% @ 14.05 s | 99.5% @ 14.19 s | 99.0% @ 14.16 s | 100.0% @ 14.26 s | 98.2±1.0% @ 13.84 s | 98.0% @ 13.93 s |
| 205 | 85.7±0.6% @ 8.55 s | 86.0±3.3% @ 9.87 s | 82.9±3.0% @ 9.87 s | 85.4% @ 11.72 s | 89.8% @ 11.29 s | 88.3% @ 10.48 s | 90.7% @ 10.83 s | 89.8% @ 11.44 s | 85.4±0.5% @ 9.94 s | 88.8% @ 10.02 s |
| 95 | 69.8±1.8% @ 4.81 s | 62.1±13.5% @ 6.05 s | 56.7±5.3% @ 6.20 s | 58.5% @ 8.22 s | 70.7% @ 7.75 s | 58.0% @ 7.43 s | 69.8% @ 7.98 s | 68.8% @ 8.10 s | 65.5±2.3% @ 6.35 s | 71.2% @ 6.36 s |
| 41 | 56.9±1.0% @ 3.26 s | 39.2±14.7% @ 3.86 s | 38.5±8.3% @ 4.19 s | 37.6% @ 6.25 s | 50.7% @ 4.31 s | 33.7% @ 6.15 s | 50.2% @ 6.48 s | 40.5% @ 5.20 s | 43.4±6.1% @ 4.74 s | 42.9% @ 3.73 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0022 | 100.0% (205/205) | 13.65 s | 7.03 s |
| 205 | 204.5 | 0.1379 | 85.4% (175/205) | 8.16 s | 3.40 s |
| 95 | 94.5 | 0.4293 | 70.2% (144/205) | 4.53 s | 1.97 s |
| 41 | 40.5 | 0.6904 | 56.6% (116/205) | 3.18 s | 1.28 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0027 | 99.5% (204/205) | 13.63 s | 6.52 s |
| 205 | 204.5 | 0.1268 | 85.4% (175/205) | 8.80 s | 3.80 s |
| 95 | 94.5 | 0.4484 | 67.8% (139/205) | 5.06 s | 2.03 s |
| 41 | 40.5 | 0.7010 | 58.0% (119/205) | 3.14 s | 1.43 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0024 | 100.0% (205/205) | 13.70 s | 6.63 s |
| 205 | 204.5 | 0.1155 | 86.3% (177/205) | 8.70 s | 3.53 s |
| 95 | 94.5 | 0.4431 | 71.2% (146/205) | 4.84 s | 2.02 s |
| 41 | 40.5 | 0.6971 | 56.1% (115/205) | 3.46 s | 1.30 s |

## R2s v1 block-order — r2s_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0025 | 98.0% (201/205) | 14.02 s | 6.97 s |
| 205 | 204.5 | 0.1267 | 84.9% (174/205) | 9.57 s | 3.55 s |
| 95 | 94.5 | 0.6190 | 49.8% (102/205) | 4.64 s | 2.05 s |
| 41 | 40.5 | 0.8462 | 27.3% (56/205) | 3.76 s | 1.60 s |

## R2s v1 block-order — r2s_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0000 | 98.5% (202/205) | 13.90 s | 6.72 s |
| 205 | 204.5 | 0.0012 | 89.8% (184/205) | 11.60 s | 4.17 s |
| 95 | 94.5 | 0.0461 | 76.6% (157/205) | 8.05 s | 2.67 s |
| 41 | 40.5 | 0.8786 | 55.6% (114/205) | 4.12 s | 1.68 s |

## R2s v1 block-order — r2s_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0015 | 98.0% (201/205) | 13.57 s | 7.07 s |
| 205 | 204.5 | 0.3385 | 83.4% (171/205) | 8.45 s | 3.10 s |
| 95 | 94.5 | 0.8438 | 60.0% (123/205) | 5.46 s | 1.60 s |
| 41 | 40.5 | 0.9591 | 34.6% (71/205) | 3.71 s | 0.90 s |

## R3 v1 block-order — r3 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0084 | 98.5% (202/205) | 14.00 s | 6.43 s |
| 205 | 204.5 | 0.2362 | 84.4% (173/205) | 10.00 s | 3.40 s |
| 95 | 94.5 | 0.5556 | 60.5% (124/205) | 5.52 s | 2.10 s |
| 41 | 40.5 | 0.8152 | 47.8% (98/205) | 3.03 s | 0.67 s |

## R3 v1 block-order — r3_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0240 | 98.5% (202/205) | 13.84 s | 7.13 s |
| 205 | 204.5 | 0.3181 | 79.5% (163/205) | 9.64 s | 4.43 s |
| 95 | 94.5 | 0.6934 | 50.7% (104/205) | 7.20 s | 3.07 s |
| 41 | 40.5 | 0.9096 | 31.7% (65/205) | 7.05 s | 2.07 s |

## R3 v1 block-order — r3_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0043 | 98.5% (202/205) | 14.17 s | 6.88 s |
| 205 | 204.5 | 0.1030 | 84.9% (174/205) | 9.97 s | 4.33 s |
| 95 | 94.5 | 0.5694 | 59.0% (121/205) | 5.87 s | 2.07 s |
| 41 | 40.5 | 0.8584 | 36.1% (74/205) | 2.48 s | 1.15 s |

## R2 v3 shuffled — 20260925_003826_pose_full_v3_base_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0344 | 98.0% (201/205) | 13.98 s | 7.43 s |
| 205 | 204.5 | 0.2365 | 85.4% (175/205) | 11.72 s | 4.43 s |
| 95 | 94.5 | 0.4886 | 58.5% (120/205) | 8.22 s | 2.97 s |
| 41 | 40.5 | 0.6906 | 37.6% (77/205) | 6.25 s | 2.07 s |

## R3 v3 shuffled — 20260925_124019_pose_full_v3_onset_pure_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0144 | 99.0% (203/205) | 14.05 s | 7.23 s |
| 205 | 204.5 | 0.3002 | 89.8% (184/205) | 11.29 s | 4.38 s |
| 95 | 94.5 | 0.7328 | 70.7% (145/205) | 7.75 s | 2.27 s |
| 41 | 40.5 | 0.9019 | 50.7% (104/205) | 4.31 s | 0.57 s |

## PF control — 20260926_040728_pose_kinematics_pf_ctrl_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0006 | 99.5% (204/205) | 14.19 s | 7.35 s |
| 205 | 204.5 | 0.3907 | 88.3% (181/205) | 10.48 s | 3.97 s |
| 95 | 94.5 | 0.9002 | 58.0% (119/205) | 7.43 s | 2.80 s |
| 41 | 40.5 | 0.9750 | 33.7% (69/205) | 6.15 s | 0.57 s |

## PF +std — 20260926_104045_pose_kinematics_pf_std_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0017 | 99.0% (203/205) | 14.16 s | 7.23 s |
| 205 | 204.5 | 0.3123 | 90.7% (186/205) | 10.83 s | 5.10 s |
| 95 | 94.5 | 0.8097 | 69.8% (143/205) | 7.98 s | 2.77 s |
| 41 | 40.5 | 0.9179 | 50.2% (103/205) | 6.48 s | 2.13 s |

## PF +bs32 — 20260926_141010_pose_kinematics_pf_bs32_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0005 | 100.0% (205/205) | 14.26 s | 7.23 s |
| 205 | 204.5 | 0.0139 | 89.8% (184/205) | 11.44 s | 5.58 s |
| 95 | 94.5 | 0.5276 | 68.8% (141/205) | 8.10 s | 2.50 s |
| 41 | 40.5 | 0.9299 | 40.5% (83/205) | 5.20 s | 1.87 s |

## PF fix — 20260926_081908_pose_kinematics_pf_fix_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0003 | 99.0% (203/205) | 14.11 s | 7.47 s |
| 205 | 204.5 | 0.0194 | 84.9% (174/205) | 8.28 s | 2.80 s |
| 95 | 94.5 | 0.6086 | 66.8% (137/205) | 6.10 s | 2.00 s |
| 41 | 40.5 | 0.9754 | 37.6% (77/205) | 4.82 s | 1.13 s |

## PF fix — 20260926_164754_pose_kinematics_pf_fix_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0389 | 97.1% (199/205) | 13.57 s | 7.23 s |
| 205 | 204.5 | 0.3730 | 85.4% (175/205) | 10.58 s | 4.13 s |
| 95 | 94.5 | 0.7027 | 66.8% (137/205) | 6.70 s | 2.27 s |
| 41 | 40.5 | 0.8360 | 49.8% (102/205) | 4.80 s | 1.75 s |

## PF fix — 20260926_184323_pose_kinematics_pf_fix_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0165 | 98.5% (202/205) | 13.84 s | 7.07 s |
| 205 | 204.5 | 0.2489 | 85.9% (176/205) | 10.96 s | 4.15 s |
| 95 | 94.5 | 0.6332 | 62.9% (129/205) | 6.26 s | 2.47 s |
| 41 | 40.5 | 0.8267 | 42.9% (88/205) | 4.61 s | 1.83 s |

## PF fix onset — 20260926_203707_pose_kinematics_pf_fixonset_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0545 | 98.0% (201/205) | 13.93 s | 7.23 s |
| 205 | 204.5 | 0.2600 | 88.8% (182/205) | 10.02 s | 4.08 s |
| 95 | 94.5 | 0.7607 | 71.2% (146/205) | 6.36 s | 2.32 s |
| 41 | 40.5 | 0.9447 | 42.9% (88/205) | 3.73 s | 1.40 s |


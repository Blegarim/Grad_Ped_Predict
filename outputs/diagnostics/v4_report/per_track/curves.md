## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | GBM probe (n=3) | R2 binary (n=3) | R3 pure (n=3) | R1 aux (n=3) | R4 hedge (n=3) | R3C censored (n=3) | R2 anchored-trained (n=3) | Model A streaming (n=1) | Model A anchored-trained (n=1) |
|---|---|---|---|---|---|---|---|---|---|
| 1200 | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0±0.0% @ 14.34 s | 100.0% @ 14.34 s | 100.0% @ 14.34 s |
| 460 | 99.8±0.3% @ 13.66 s | 98.0±0.8% @ 13.84 s | 99.2±1.0% @ 14.00 s | 98.2±1.7% @ 13.68 s | 98.2±3.1% @ 13.86 s | 99.8±0.3% @ 14.10 s | 100.0±0.0% @ 14.34 s | 100.0% @ 13.93 s | 100.0% @ 14.33 s |
| 205 | 85.7±0.6% @ 8.55 s | 86.3±1.3% @ 10.27 s | 86.8±1.5% @ 10.20 s | 85.7±4.1% @ 9.45 s | 83.3±8.0% @ 9.14 s | 87.0±2.0% @ 10.52 s | 80.2±5.7% @ 15.48 s | 87.3% @ 10.82 s | 68.3% @ 16.49 s |
| 95 | 69.8±1.8% @ 4.81 s | 66.3±3.2% @ 6.59 s | 68.5±3.1% @ 5.81 s | 65.2±5.0% @ 6.55 s | 64.7±9.2% @ 6.36 s | 68.3±2.5% @ 6.31 s | 47.3±5.9% @ 18.80 s | 65.4% @ 7.63 s | 36.1% @ 20.91 s |
| 41 | 56.9±1.0% @ 3.26 s | 44.6±4.6% @ 4.88 s | 47.0±4.1% @ 3.78 s | 46.0±2.7% @ 4.78 s | 48.6±9.3% @ 4.37 s | 50.1±1.0% @ 4.25 s | 24.9±2.7% @ 22.39 s | 43.9% @ 5.67 s | 22.0% @ 23.05 s |

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

## R2 binary — 20260927_020529_pose_kinematics_c4_r2s_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0003 | 98.5% (202/205) | 14.10 s | 7.25 s |
| 205 | 204.5 | 0.0017 | 87.8% (180/205) | 9.26 s | 3.67 s |
| 95 | 94.5 | 0.3208 | 69.3% (142/205) | 6.83 s | 2.35 s |
| 41 | 40.5 | 0.9839 | 41.0% (84/205) | 5.24 s | 1.52 s |

## R2 binary — 20260927_064019_pose_kinematics_c4_r2s_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0389 | 97.1% (199/205) | 13.57 s | 7.23 s |
| 205 | 204.5 | 0.3734 | 85.4% (175/205) | 10.58 s | 4.13 s |
| 95 | 94.5 | 0.7021 | 66.8% (137/205) | 6.70 s | 2.27 s |
| 41 | 40.5 | 0.8358 | 49.8% (102/205) | 4.80 s | 1.75 s |

## R2 binary — 20260927_103113_pose_kinematics_c4_r2s_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0164 | 98.5% (202/205) | 13.84 s | 7.07 s |
| 205 | 204.5 | 0.2476 | 85.9% (176/205) | 10.96 s | 4.15 s |
| 95 | 94.5 | 0.6327 | 62.9% (129/205) | 6.26 s | 2.47 s |
| 41 | 40.5 | 0.8268 | 42.9% (88/205) | 4.61 s | 1.83 s |

## R3 pure — 20260927_044252_pose_kinematics_c4_r3_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0545 | 98.0% (201/205) | 13.93 s | 7.23 s |
| 205 | 204.5 | 0.2616 | 88.3% (181/205) | 9.95 s | 4.03 s |
| 95 | 94.5 | 0.7596 | 70.7% (145/205) | 6.40 s | 2.30 s |
| 41 | 40.5 | 0.9461 | 42.4% (87/205) | 3.76 s | 1.40 s |

## R3 pure — 20260927_083144_pose_kinematics_c4_r3_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0496 | 100.0% (205/205) | 14.00 s | 7.17 s |
| 205 | 204.5 | 0.4297 | 85.4% (175/205) | 10.20 s | 4.80 s |
| 95 | 94.5 | 0.8022 | 64.9% (133/205) | 5.68 s | 2.03 s |
| 41 | 40.5 | 0.9250 | 50.2% (103/205) | 4.28 s | 1.20 s |

## R3 pure — 20260927_122240_pose_kinematics_c4_r3_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0473 | 99.5% (204/205) | 14.08 s | 7.12 s |
| 205 | 204.5 | 0.2901 | 86.8% (178/205) | 10.46 s | 4.18 s |
| 95 | 94.5 | 0.7923 | 69.8% (143/205) | 5.36 s | 2.13 s |
| 41 | 40.5 | 0.9292 | 48.3% (99/205) | 3.29 s | 1.27 s |

## R1 aux — 20260927_164845_pose_kinematics_c4_r1_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0430 | 98.0% (201/205) | 13.65 s | 6.97 s |
| 205 | 204.5 | 0.3361 | 84.4% (173/205) | 9.14 s | 4.00 s |
| 95 | 94.5 | 0.6336 | 61.0% (125/205) | 6.65 s | 2.50 s |
| 41 | 40.5 | 0.8276 | 43.4% (89/205) | 4.82 s | 1.63 s |

## R1 aux — 20260927_225651_pose_kinematics_c4_r1_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0505 | 96.6% (198/205) | 13.45 s | 7.22 s |
| 205 | 204.5 | 0.3047 | 82.4% (169/205) | 9.97 s | 4.07 s |
| 95 | 94.5 | 0.6428 | 63.9% (131/205) | 6.60 s | 2.20 s |
| 41 | 40.5 | 0.8277 | 45.9% (94/205) | 4.59 s | 1.55 s |

## R1 aux — 20260928_043457_pose_kinematics_c4_r1_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0007 | 100.0% (205/205) | 13.96 s | 7.07 s |
| 205 | 204.5 | 0.0560 | 90.2% (185/205) | 9.25 s | 3.60 s |
| 95 | 94.5 | 0.6761 | 70.7% (145/205) | 6.41 s | 2.20 s |
| 41 | 40.5 | 0.9327 | 48.8% (100/205) | 4.93 s | 1.65 s |

## R4 hedge — 20260927_183811_pose_kinematics_c4_r4_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0031 | 100.0% (205/205) | 13.66 s | 7.03 s |
| 205 | 204.5 | 0.0561 | 89.3% (183/205) | 9.05 s | 3.17 s |
| 95 | 94.5 | 0.4815 | 70.2% (144/205) | 6.45 s | 2.18 s |
| 41 | 40.5 | 0.8866 | 52.2% (107/205) | 4.27 s | 1.17 s |

## R4 hedge — 20260928_004618_pose_kinematics_c4_r4_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.1688 | 94.6% (194/205) | 13.95 s | 7.52 s |
| 205 | 204.5 | 0.3443 | 74.1% (152/205) | 8.29 s | 3.08 s |
| 95 | 94.5 | 0.5686 | 54.1% (111/205) | 6.85 s | 2.60 s |
| 41 | 40.5 | 0.8174 | 38.0% (78/205) | 4.62 s | 1.50 s |

## R4 hedge — 20260928_063824_pose_kinematics_c4_r4_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0132 | 100.0% (205/205) | 13.98 s | 6.97 s |
| 205 | 204.5 | 0.1722 | 86.3% (177/205) | 10.08 s | 3.80 s |
| 95 | 94.5 | 0.7893 | 69.8% (143/205) | 5.79 s | 2.20 s |
| 41 | 40.5 | 0.9399 | 55.6% (114/205) | 4.21 s | 1.25 s |

## R3C censored — 20260927_205737_pose_kinematics_c4_r3c_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0505 | 100.0% (205/205) | 14.10 s | 7.23 s |
| 205 | 204.5 | 0.1663 | 89.3% (183/205) | 11.15 s | 4.63 s |
| 95 | 94.5 | 0.7102 | 66.8% (137/205) | 6.37 s | 2.27 s |
| 41 | 40.5 | 0.8798 | 49.8% (102/205) | 4.14 s | 1.40 s |

## R3C censored — 20260928_023544_pose_kinematics_c4_r3c_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0443 | 99.5% (204/205) | 14.13 s | 7.42 s |
| 205 | 204.5 | 0.3559 | 85.4% (175/205) | 10.32 s | 3.97 s |
| 95 | 94.5 | 0.7376 | 66.8% (137/205) | 6.39 s | 2.03 s |
| 41 | 40.5 | 0.8599 | 51.2% (105/205) | 4.92 s | 1.40 s |

## R3C censored — 20260928_084352_pose_kinematics_c4_r3c_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0465 | 100.0% (205/205) | 14.07 s | 7.23 s |
| 205 | 204.5 | 0.1732 | 86.3% (177/205) | 10.08 s | 3.87 s |
| 95 | 94.5 | 0.6351 | 71.2% (146/205) | 6.18 s | 2.42 s |
| 41 | 40.5 | 0.8659 | 49.3% (101/205) | 3.69 s | 1.30 s |

## R2 anchored-trained — 20260927_142412_pose_kinematics_c4_r2a_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0193 | 100.0% (205/205) | 14.33 s | 7.47 s |
| 205 | 204.5 | 0.9726 | 73.7% (151/205) | 16.28 s | 9.97 s |
| 95 | 94.5 | 0.9876 | 43.9% (90/205) | 18.70 s | 14.10 s |
| 41 | 40.5 | 0.9909 | 24.4% (50/205) | 19.50 s | 15.50 s |

## R2 anchored-trained — 20260927_143335_pose_kinematics_c4_r2a_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0867 | 100.0% (205/205) | 14.34 s | 7.47 s |
| 205 | 204.5 | 0.8104 | 82.9% (170/205) | 15.27 s | 8.00 s |
| 95 | 94.5 | 0.9296 | 54.1% (111/205) | 17.37 s | 9.43 s |
| 41 | 40.5 | 0.9549 | 27.8% (57/205) | 23.96 s | 22.63 s |

## R2 anchored-trained — 20260927_144045_pose_kinematics_c4_r2a_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0544 | 100.0% (205/205) | 14.34 s | 7.47 s |
| 205 | 204.5 | 0.6863 | 83.9% (172/205) | 14.88 s | 7.02 s |
| 95 | 94.5 | 0.9206 | 43.9% (90/205) | 20.32 s | 17.15 s |
| 41 | 40.5 | 0.9470 | 22.4% (46/205) | 23.71 s | 21.82 s |

## Model A streaming — 20260927_144755_pose_kinematics_c4_mA_str_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0374 | 100.0% (205/205) | 13.93 s | 7.03 s |
| 205 | 204.5 | 0.3102 | 87.3% (179/205) | 10.82 s | 4.23 s |
| 95 | 94.5 | 0.6094 | 65.4% (134/205) | 7.63 s | 2.63 s |
| 41 | 40.5 | 0.7903 | 43.9% (90/205) | 5.67 s | 1.53 s |

## Model A anchored-trained — 20260927_163520_pose_kinematics_c4_mA_anc_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_track`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 477.5 | -inf | 100.0% (205/205) | 14.34 s | 7.47 s |
| 460 | 459.2 | 0.0014 | 100.0% (205/205) | 14.33 s | 7.47 s |
| 205 | 204.5 | 0.9983 | 68.3% (140/205) | 16.49 s | 9.75 s |
| 95 | 93.6 | 0.9995 | 36.1% (74/205) | 20.91 s | 18.05 s |
| 41 | 40.5 | 0.9997 | 22.0% (45/205) | 23.05 s | 18.83 s |


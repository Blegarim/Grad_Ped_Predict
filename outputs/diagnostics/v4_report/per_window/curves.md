## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | GBM probe (n=3) | R2 binary (n=3) | R3 pure (n=3) | R1 aux (n=3) | R4 hedge (n=3) | R3C censored (n=3) | R2 anchored-trained (n=3) | Model A streaming (n=1) | Model A anchored-trained (n=1) |
|---|---|---|---|---|---|---|---|---|---|
| 1200 | 74.3±1.0% @ 5.38 s | 69.6±6.0% @ 7.33 s | 72.0±2.8% @ 7.00 s | 69.1±6.6% @ 7.27 s | 69.1±12.1% @ 7.14 s | 72.2±2.1% @ 7.02 s | 35.9±6.2% @ 20.99 s | 65.4% @ 7.63 s | 34.6% @ 21.74 s |
| 460 | 65.9±0.5% @ 3.92 s | 55.0±6.2% @ 5.84 s | 59.2±4.9% @ 4.68 s | 55.0±5.2% @ 4.88 s | 56.4±11.3% @ 5.31 s | 59.7±2.3% @ 5.40 s | 24.1±2.9% @ 22.25 s | 49.3% @ 6.14 s | 20.0% @ 22.76 s |
| 205 | 56.7±1.2% @ 3.27 s | 42.3±3.2% @ 4.66 s | 44.1±4.7% @ 3.41 s | 42.1±2.7% @ 4.73 s | 46.7±11.4% @ 3.92 s | 47.8±3.0% @ 3.99 s | 15.1±2.6% @ 21.18 s | 42.9% @ 5.61 s | 12.2% @ 23.32 s |
| 95 | 46.8±0.5% @ 2.87 s | 32.0±4.2% @ 3.77 s | 31.4±5.9% @ 3.03 s | 29.4±3.9% @ 3.92 s | 37.4±9.8% @ 3.01 s | 37.7±2.5% @ 2.84 s | 8.9±2.9% @ 20.15 s | 34.1% @ 3.75 s | 3.4% @ 21.80 s |
| 41 | 38.9±1.0% @ 2.01 s | 22.9±3.5% @ 3.03 s | 17.4±9.1% @ 2.67 s | 17.6±6.1% @ 2.81 s | 23.4±11.7% @ 2.12 s | 18.7±7.5% @ 2.28 s | 4.1±2.3% @ 18.42 s | 24.9% @ 2.43 s | 1.5% @ 19.32 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.3409 | 74.6% (153/205) | 5.17 s | 2.27 s |
| 460 | 459.2 | 0.5577 | 65.4% (134/205) | 3.74 s | 1.58 s |
| 205 | 204.5 | 0.6923 | 56.6% (116/205) | 3.17 s | 1.28 s |
| 95 | 94.5 | 0.8008 | 46.3% (95/205) | 2.82 s | 1.03 s |
| 41 | 40.5 | 0.8781 | 38.0% (78/205) | 2.00 s | 0.85 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.3391 | 73.2% (150/205) | 5.47 s | 2.37 s |
| 460 | 459.2 | 0.5462 | 65.9% (135/205) | 3.92 s | 1.80 s |
| 205 | 204.5 | 0.7010 | 58.0% (119/205) | 3.14 s | 1.43 s |
| 95 | 94.5 | 0.7945 | 46.8% (96/205) | 3.03 s | 1.05 s |
| 41 | 40.5 | 0.8723 | 38.5% (79/205) | 2.30 s | 0.87 s |

## GBM probe — probe (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.3390 | 75.1% (154/205) | 5.50 s | 2.33 s |
| 460 | 459.2 | 0.5482 | 66.3% (136/205) | 4.10 s | 1.73 s |
| 205 | 204.5 | 0.6997 | 55.6% (114/205) | 3.49 s | 1.35 s |
| 95 | 94.5 | 0.8122 | 47.3% (97/205) | 2.77 s | 0.87 s |
| 41 | 40.5 | 0.8832 | 40.0% (82/205) | 1.72 s | 0.60 s |

## R2 binary — 20260927_020529_pose_kinematics_c4_r2s_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.0359 | 74.6% (153/205) | 7.81 s | 2.87 s |
| 460 | 459.2 | 0.8296 | 59.0% (121/205) | 6.49 s | 2.17 s |
| 205 | 204.5 | 0.9787 | 44.4% (91/205) | 5.37 s | 1.57 s |
| 95 | 94.5 | 0.9911 | 36.6% (75/205) | 4.43 s | 0.90 s |
| 41 | 40.5 | 0.9955 | 23.9% (49/205) | 3.53 s | 0.57 s |

## R2 binary — 20260927_064019_pose_kinematics_c4_r2s_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6428 | 71.2% (146/205) | 7.92 s | 2.37 s |
| 460 | 459.2 | 0.7829 | 58.0% (119/205) | 5.68 s | 2.00 s |
| 205 | 204.5 | 0.8667 | 43.9% (90/205) | 4.52 s | 1.58 s |
| 95 | 94.5 | 0.9180 | 28.3% (58/205) | 3.89 s | 1.32 s |
| 41 | 40.5 | 0.9437 | 19.0% (39/205) | 3.14 s | 1.07 s |

## R2 binary — 20260927_103113_pose_kinematics_c4_r2s_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6306 | 62.9% (129/205) | 6.26 s | 2.47 s |
| 460 | 459.2 | 0.7768 | 47.8% (98/205) | 5.36 s | 1.90 s |
| 205 | 204.5 | 0.8527 | 38.5% (79/205) | 4.10 s | 1.70 s |
| 95 | 94.5 | 0.8934 | 31.2% (64/205) | 2.98 s | 1.50 s |
| 41 | 40.5 | 0.9145 | 25.9% (53/205) | 2.41 s | 1.33 s |

## R3 pure — 20260927_044252_pose_kinematics_c4_r3_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6861 | 73.7% (151/205) | 7.12 s | 2.50 s |
| 460 | 459.2 | 0.8691 | 58.5% (120/205) | 4.54 s | 1.73 s |
| 205 | 204.5 | 0.9413 | 44.9% (92/205) | 3.72 s | 1.40 s |
| 95 | 94.5 | 0.9781 | 29.3% (60/205) | 3.42 s | 0.92 s |
| 41 | 40.5 | 0.9951 | 13.7% (28/205) | 3.03 s | 0.57 s |

## R3 pure — 20260927_083144_pose_kinematics_c4_r3_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.7426 | 68.8% (141/205) | 7.09 s | 2.23 s |
| 460 | 459.2 | 0.8958 | 54.6% (112/205) | 4.78 s | 1.50 s |
| 205 | 204.5 | 0.9578 | 39.0% (80/205) | 3.22 s | 0.97 s |
| 95 | 94.5 | 0.9852 | 26.8% (55/205) | 2.60 s | 0.60 s |
| 41 | 40.5 | 0.9963 | 10.7% (22/205) | 2.94 s | 0.52 s |

## R3 pure — 20260927_122240_pose_kinematics_c4_r3_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6830 | 73.7% (151/205) | 6.78 s | 2.50 s |
| 460 | 459.2 | 0.8526 | 64.4% (132/205) | 4.71 s | 1.93 s |
| 205 | 204.5 | 0.9267 | 48.3% (99/205) | 3.30 s | 1.30 s |
| 95 | 94.5 | 0.9700 | 38.0% (78/205) | 3.06 s | 0.82 s |
| 41 | 40.5 | 0.9900 | 27.8% (57/205) | 2.03 s | 0.33 s |

## R1 aux — 20260927_164845_pose_kinematics_c4_r1_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6022 | 64.4% (132/205) | 6.86 s | 2.72 s |
| 460 | 459.2 | 0.7534 | 52.2% (107/205) | 4.99 s | 1.73 s |
| 205 | 204.5 | 0.8354 | 42.0% (86/205) | 4.52 s | 1.65 s |
| 95 | 94.5 | 0.8879 | 33.7% (69/205) | 3.89 s | 1.53 s |
| 41 | 40.5 | 0.9231 | 23.9% (49/205) | 2.46 s | 1.40 s |

## R1 aux — 20260927_225651_pose_kinematics_c4_r1_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5720 | 66.3% (136/205) | 7.42 s | 2.65 s |
| 460 | 459.2 | 0.7525 | 51.7% (106/205) | 4.64 s | 1.98 s |
| 205 | 204.5 | 0.8621 | 39.5% (81/205) | 4.74 s | 1.47 s |
| 95 | 94.5 | 0.9175 | 25.9% (53/205) | 4.53 s | 1.30 s |
| 41 | 40.5 | 0.9501 | 17.1% (35/205) | 3.46 s | 1.20 s |

## R1 aux — 20260928_043457_pose_kinematics_c4_r1_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.4961 | 76.6% (157/205) | 7.52 s | 2.57 s |
| 460 | 459.2 | 0.8567 | 61.0% (125/205) | 5.01 s | 1.80 s |
| 205 | 204.5 | 0.9468 | 44.9% (92/205) | 4.92 s | 1.43 s |
| 95 | 94.5 | 0.9764 | 28.8% (59/205) | 3.33 s | 0.77 s |
| 41 | 40.5 | 0.9888 | 11.7% (24/205) | 2.51 s | 0.42 s |

## R4 hedge — 20260927_183811_pose_kinematics_c4_r4_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.2635 | 76.1% (156/205) | 7.41 s | 2.50 s |
| 460 | 459.2 | 0.6793 | 62.0% (127/205) | 5.70 s | 1.70 s |
| 205 | 204.5 | 0.9071 | 51.2% (105/205) | 4.11 s | 1.10 s |
| 95 | 94.5 | 0.9738 | 44.9% (92/205) | 3.39 s | 0.42 s |
| 41 | 40.5 | 0.9939 | 35.6% (73/205) | 2.05 s | 0.07 s |

## R4 hedge — 20260928_004618_pose_kinematics_c4_r4_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5446 | 55.1% (113/205) | 6.88 s | 2.60 s |
| 460 | 459.2 | 0.7091 | 43.4% (89/205) | 4.96 s | 1.93 s |
| 205 | 204.5 | 0.8537 | 33.7% (69/205) | 3.46 s | 1.30 s |
| 95 | 94.5 | 0.9195 | 26.3% (54/205) | 2.90 s | 1.10 s |
| 41 | 40.5 | 0.9734 | 12.2% (25/205) | 3.58 s | 0.97 s |

## R4 hedge — 20260928_063824_pose_kinematics_c4_r4_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6317 | 76.1% (156/205) | 7.13 s | 2.48 s |
| 460 | 459.2 | 0.8741 | 63.9% (131/205) | 5.26 s | 1.87 s |
| 205 | 204.5 | 0.9456 | 55.1% (113/205) | 4.20 s | 1.23 s |
| 95 | 94.5 | 0.9791 | 41.0% (84/205) | 2.75 s | 0.63 s |
| 41 | 40.5 | 0.9970 | 22.4% (46/205) | 0.73 s | 0.07 s |

## R3C censored — 20260927_205737_pose_kinematics_c4_r3c_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6084 | 73.2% (150/205) | 7.31 s | 2.48 s |
| 460 | 459.2 | 0.8114 | 57.1% (117/205) | 5.75 s | 1.70 s |
| 205 | 204.5 | 0.9143 | 45.4% (93/205) | 3.66 s | 0.97 s |
| 95 | 94.5 | 0.9666 | 35.1% (72/205) | 2.84 s | 0.80 s |
| 41 | 40.5 | 0.9974 | 10.2% (21/205) | 1.76 s | 0.33 s |

## R3C censored — 20260928_023544_pose_kinematics_c4_r3c_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6735 | 69.8% (143/205) | 7.36 s | 2.50 s |
| 460 | 459.2 | 0.8048 | 60.5% (124/205) | 5.65 s | 1.72 s |
| 205 | 204.5 | 0.8701 | 51.2% (105/205) | 4.89 s | 1.40 s |
| 95 | 94.5 | 0.9246 | 38.0% (78/205) | 2.92 s | 0.90 s |
| 41 | 40.5 | 0.9600 | 21.5% (44/205) | 2.67 s | 0.68 s |

## R3C censored — 20260928_084352_pose_kinematics_c4_r3c_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5694 | 73.7% (151/205) | 6.38 s | 2.50 s |
| 460 | 459.2 | 0.7798 | 61.5% (126/205) | 4.80 s | 1.63 s |
| 205 | 204.5 | 0.8850 | 46.8% (96/205) | 3.42 s | 1.02 s |
| 95 | 94.5 | 0.9375 | 40.0% (82/205) | 2.77 s | 0.63 s |
| 41 | 40.5 | 0.9791 | 24.4% (50/205) | 2.41 s | 0.13 s |

## R2 anchored-trained — 20260927_142412_pose_kinematics_c4_r2a_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.9880 | 40.0% (82/205) | 19.19 s | 14.62 s |
| 460 | 459.2 | 0.9909 | 24.4% (50/205) | 19.50 s | 15.50 s |
| 205 | 204.5 | 0.9926 | 13.2% (27/205) | 15.66 s | 13.33 s |
| 95 | 94.5 | 0.9933 | 6.8% (14/205) | 15.36 s | 12.35 s |
| 41 | 40.5 | 0.9945 | 1.5% (3/205) | 9.26 s | 11.17 s |

## R2 anchored-trained — 20260927_143335_pose_kinematics_c4_r2a_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.9469 | 39.0% (80/205) | 21.14 s | 17.42 s |
| 460 | 459.2 | 0.9557 | 26.8% (55/205) | 24.05 s | 22.40 s |
| 205 | 204.5 | 0.9596 | 18.0% (37/205) | 25.40 s | 23.93 s |
| 95 | 94.5 | 0.9622 | 12.2% (25/205) | 24.89 s | 22.10 s |
| 41 | 40.5 | 0.9652 | 5.4% (11/205) | 23.68 s | 17.20 s |

## R2 anchored-trained — 20260927_144045_pose_kinematics_c4_r2a_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.9404 | 28.8% (59/205) | 22.65 s | 18.83 s |
| 460 | 459.2 | 0.9490 | 21.0% (43/205) | 23.19 s | 23.37 s |
| 205 | 204.5 | 0.9532 | 14.1% (29/205) | 22.48 s | 22.30 s |
| 95 | 94.5 | 0.9560 | 7.8% (16/205) | 20.18 s | 18.73 s |
| 41 | 40.5 | 0.9579 | 5.4% (11/205) | 22.31 s | 20.93 s |

## Model A streaming — 20260927_144755_pose_kinematics_c4_mA_str_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6104 | 65.4% (134/205) | 7.63 s | 2.60 s |
| 460 | 459.2 | 0.7381 | 49.3% (101/205) | 6.14 s | 1.93 s |
| 205 | 204.5 | 0.7985 | 42.9% (88/205) | 5.61 s | 1.50 s |
| 95 | 94.5 | 0.8471 | 34.1% (70/205) | 3.75 s | 1.45 s |
| 41 | 40.5 | 0.8731 | 24.9% (51/205) | 2.43 s | 1.27 s |

## Model A anchored-trained — 20260927_163520_pose_kinematics_c4_mA_anc_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.9996 | 34.6% (71/205) | 21.74 s | 19.10 s |
| 460 | 459.2 | 0.9997 | 20.0% (41/205) | 22.76 s | 18.77 s |
| 205 | 202.6 | 0.9998 | 12.2% (25/205) | 23.32 s | 21.00 s |
| 95 | 94.5 | 0.9998 | 3.4% (7/205) | 21.80 s | 18.73 s |
| 41 | 40.5 | 0.9998 | 1.5% (3/205) | 19.32 s | 18.77 s |


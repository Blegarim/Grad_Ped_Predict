## Detection rate @ mean lead time, at matched false-alarm budgets

| FA/hr | GBM probe (n=3) | R2s v1 block-order (n=3) | R3 v1 block-order (n=3) | R2 v3 shuffled (n=1) | R3 v3 shuffled (n=1) | PF control (n=1) | PF +std (n=1) | PF +bs32 (n=1) | PF fix (n=3) | PF fix onset (n=1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1200 | 74.3±1.0% @ 5.38 s | 62.0±19.0% @ 6.49 s | 59.7±10.4% @ 6.67 s | 53.7% @ 7.36 s | 77.1% @ 8.40 s | 74.6% @ 8.85 s | 78.0% @ 9.56 s | 68.8% @ 7.97 s | 70.9±7.8% @ 7.06 s | 73.7% @ 7.17 s |
| 460 | 65.9±0.5% @ 3.92 s | 45.2±22.0% @ 5.01 s | 45.9±11.3% @ 5.17 s | 38.0% @ 6.22 s | 59.0% @ 5.11 s | 54.6% @ 6.86 s | 65.4% @ 6.81 s | 48.8% @ 5.88 s | 55.0±6.2% @ 5.69 s | 58.0% @ 4.56 s |
| 205 | 56.7±1.2% @ 3.27 s | 32.4±18.5% @ 3.88 s | 35.0±12.3% @ 4.25 s | 32.2% @ 5.32 s | 48.3% @ 3.80 s | 33.7% @ 6.15 s | 48.8% @ 6.13 s | 33.7% @ 5.11 s | 40.3±3.1% @ 4.54 s | 45.4% @ 3.70 s |
| 95 | 46.8±0.5% @ 2.87 s | 21.3±16.2% @ 2.91 s | 25.4±13.6% @ 3.77 s | 22.9% @ 3.17 s | 38.0% @ 2.81 s | 14.6% @ 7.42 s | 31.7% @ 4.64 s | 19.5% @ 4.03 s | 29.4±1.6% @ 3.98 s | 29.8% @ 3.37 s |
| 41 | 38.9±1.0% @ 2.01 s | 11.9±7.9% @ 1.82 s | 19.5±12.8% @ 3.59 s | 15.6% @ 2.68 s | 30.2% @ 0.79 s | 4.9% @ 15.97 s | 18.0% @ 5.20 s | 3.4% @ 2.16 s | 20.2±4.7% @ 4.23 s | 13.7% @ 3.02 s |

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

## R2s v1 block-order — r2s_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.7030 | 40.5% (83/205) | 4.51 s | 2.17 s |
| 460 | 459.2 | 0.8741 | 21.5% (44/205) | 4.13 s | 1.77 s |
| 205 | 204.5 | 0.9251 | 15.6% (32/205) | 3.38 s | 1.20 s |
| 95 | 94.5 | 0.9526 | 10.7% (22/205) | 2.87 s | 1.15 s |
| 41 | 40.5 | 0.9678 | 6.8% (14/205) | 1.70 s | 1.18 s |

## R2s v1 block-order — r2s_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.0381 | 76.6% (157/205) | 8.08 s | 2.67 s |
| 460 | 459.2 | 0.4838 | 64.9% (133/205) | 5.60 s | 2.00 s |
| 205 | 204.5 | 0.9379 | 52.2% (107/205) | 4.16 s | 1.60 s |
| 95 | 94.5 | 0.9868 | 40.0% (82/205) | 3.40 s | 1.27 s |
| 41 | 40.5 | 0.9985 | 21.0% (43/205) | 2.00 s | 0.73 s |

## R2s v1 block-order — r2s_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.7097 | 68.8% (141/205) | 6.88 s | 2.23 s |
| 460 | 459.2 | 0.9093 | 49.3% (101/205) | 5.30 s | 1.20 s |
| 205 | 204.5 | 0.9660 | 29.3% (60/205) | 4.11 s | 1.03 s |
| 95 | 94.5 | 0.9857 | 13.2% (27/205) | 2.47 s | 0.57 s |
| 41 | 40.5 | 0.9929 | 7.8% (16/205) | 1.76 s | 0.40 s |

## R3 v1 block-order — r3 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.4574 | 66.8% (137/205) | 6.52 s | 2.50 s |
| 460 | 459.2 | 0.6786 | 56.6% (116/205) | 4.40 s | 1.58 s |
| 205 | 204.5 | 0.8031 | 48.8% (100/205) | 3.05 s | 0.68 s |
| 95 | 94.5 | 0.8762 | 41.0% (84/205) | 1.84 s | 0.17 s |
| 41 | 40.5 | 0.9215 | 34.1% (70/205) | 0.72 s | 0.07 s |

## R3 v1 block-order — r3_s43 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.7229 | 47.8% (98/205) | 6.91 s | 2.75 s |
| 460 | 459.2 | 0.8958 | 34.1% (70/205) | 6.96 s | 2.07 s |
| 205 | 204.5 | 0.9594 | 25.4% (52/205) | 7.63 s | 1.83 s |
| 95 | 94.5 | 0.9865 | 16.1% (33/205) | 7.96 s | 4.27 s |
| 41 | 40.5 | 0.9943 | 13.7% (28/205) | 8.83 s | 8.13 s |

## R3 v1 block-order — r3_s44 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.4603 | 64.4% (132/205) | 6.58 s | 2.35 s |
| 460 | 459.2 | 0.7726 | 46.8% (96/205) | 4.14 s | 1.52 s |
| 205 | 204.5 | 0.9090 | 30.7% (63/205) | 2.06 s | 0.90 s |
| 95 | 94.5 | 0.9667 | 19.0% (39/205) | 1.52 s | 0.77 s |
| 41 | 40.5 | 0.9895 | 10.7% (22/205) | 1.23 s | 0.68 s |

## R2 v3 shuffled — 20260925_003826_pose_full_v3_base_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5422 | 53.7% (110/205) | 7.36 s | 2.63 s |
| 460 | 459.2 | 0.6806 | 38.0% (78/205) | 6.22 s | 2.05 s |
| 205 | 204.5 | 0.7528 | 32.2% (66/205) | 5.32 s | 1.53 s |
| 95 | 94.5 | 0.8151 | 22.9% (47/205) | 3.17 s | 1.27 s |
| 41 | 40.5 | 0.8577 | 15.6% (32/205) | 2.68 s | 0.95 s |

## R3 v3 shuffled — 20260925_124019_pose_full_v3_onset_pure_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6765 | 77.1% (158/205) | 8.40 s | 2.47 s |
| 460 | 459.2 | 0.8383 | 59.0% (121/205) | 5.11 s | 1.60 s |
| 205 | 204.5 | 0.9210 | 48.3% (99/205) | 3.80 s | 0.57 s |
| 95 | 94.5 | 0.9678 | 38.0% (78/205) | 2.81 s | 0.07 s |
| 41 | 40.5 | 0.9912 | 30.2% (62/205) | 0.79 s | 0.07 s |

## PF control — 20260926_040728_pose_kinematics_pf_ctrl_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.7346 | 74.6% (153/205) | 8.85 s | 3.17 s |
| 460 | 459.2 | 0.9205 | 54.6% (112/205) | 6.86 s | 2.38 s |
| 205 | 204.5 | 0.9751 | 33.7% (69/205) | 6.15 s | 0.57 s |
| 95 | 94.5 | 0.9907 | 14.6% (30/205) | 7.42 s | 0.87 s |
| 41 | 40.5 | 0.9955 | 4.9% (10/205) | 15.97 s | 6.37 s |

## PF +std — 20260926_104045_pose_kinematics_pf_std_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6894 | 78.0% (160/205) | 9.56 s | 3.22 s |
| 460 | 459.2 | 0.8540 | 65.4% (134/205) | 6.81 s | 2.37 s |
| 205 | 204.5 | 0.9252 | 48.8% (100/205) | 6.13 s | 2.07 s |
| 95 | 94.5 | 0.9582 | 31.7% (65/205) | 4.64 s | 1.00 s |
| 41 | 40.5 | 0.9743 | 18.0% (37/205) | 5.20 s | 1.47 s |

## PF +bs32 — 20260926_141010_pose_kinematics_pf_bs32_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.5380 | 68.8% (141/205) | 7.97 s | 2.50 s |
| 460 | 459.2 | 0.8811 | 48.8% (100/205) | 5.88 s | 2.12 s |
| 205 | 204.5 | 0.9553 | 33.7% (69/205) | 5.11 s | 1.53 s |
| 95 | 94.5 | 0.9774 | 19.5% (40/205) | 4.03 s | 0.98 s |
| 41 | 40.5 | 0.9905 | 3.4% (7/205) | 2.16 s | 1.47 s |

## PF fix — 20260926_081908_pose_kinematics_pf_fix_s42 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.0835 | 78.5% (161/205) | 7.00 s | 2.40 s |
| 460 | 459.2 | 0.8290 | 59.0% (121/205) | 6.03 s | 1.73 s |
| 205 | 204.5 | 0.9721 | 38.5% (79/205) | 5.02 s | 1.20 s |
| 95 | 94.5 | 0.9869 | 28.8% (59/205) | 5.07 s | 0.97 s |
| 41 | 40.5 | 0.9935 | 16.1% (33/205) | 7.21 s | 1.10 s |

## PF fix — 20260926_164754_pose_kinematics_pf_fix_s43 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6433 | 71.2% (146/205) | 7.92 s | 2.37 s |
| 460 | 459.2 | 0.7831 | 58.0% (119/205) | 5.68 s | 2.00 s |
| 205 | 204.5 | 0.8667 | 43.9% (90/205) | 4.52 s | 1.58 s |
| 95 | 94.5 | 0.9180 | 28.3% (58/205) | 3.89 s | 1.32 s |
| 41 | 40.5 | 0.9437 | 19.0% (39/205) | 3.14 s | 1.07 s |

## PF fix — 20260926_184323_pose_kinematics_pf_fix_s44 (p_frame)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6305 | 62.9% (129/205) | 6.26 s | 2.47 s |
| 460 | 459.2 | 0.7768 | 47.8% (98/205) | 5.36 s | 1.90 s |
| 205 | 204.5 | 0.8526 | 38.5% (79/205) | 4.10 s | 1.70 s |
| 95 | 94.5 | 0.8934 | 31.2% (64/205) | 2.98 s | 1.50 s |
| 41 | 40.5 | 0.9147 | 25.4% (52/205) | 2.34 s | 1.32 s |

## PF fix onset — 20260926_203707_pose_kinematics_pf_fixonset_s42 (p_readout)

205 crossing pedestrians, alarms counted `per_window`.

| FA/hr budget | realised FA/hr | threshold | detection rate | mean lead | median lead |
|---|---|---|---|---|---|
| 1200 | 1199.2 | 0.6863 | 73.7% (151/205) | 7.17 s | 2.50 s |
| 460 | 459.2 | 0.8699 | 58.0% (119/205) | 4.56 s | 1.70 s |
| 205 | 204.5 | 0.9407 | 45.4% (93/205) | 3.70 s | 1.40 s |
| 95 | 94.5 | 0.9785 | 29.8% (61/205) | 3.37 s | 0.90 s |
| 41 | 40.5 | 0.9957 | 13.7% (28/205) | 3.02 s | 0.57 s |


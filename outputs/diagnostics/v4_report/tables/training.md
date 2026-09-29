# Training summary per run (validation split)

| arm | seed | epochs | selected epoch (metric) | val AUC selected | val AUC last | train loss first→last | collapse epochs | readout spread selected / last |
|---|---|---|---|---|---|---|---|---|
| R2 binary | 42 | 23 | 8 (crosses_auc) | 0.849 | 0.814 | 0.54→0.03 | none | – |
| R2 binary | 43 | 16 | 1 (crosses_auc) | 0.870 | 0.751 | 0.54→0.06 | none | – |
| R2 binary | 44 | 16 | 1 (crosses_auc) | 0.854 | 0.813 | 0.54→0.06 | none | – |
| R3 pure hazard | 42 | 17 | 2 (crosses_auc) | 0.850 | 0.757 | 2.35→0.27 | none | 0.50 / 0.00 |
| R3 pure hazard | 43 | 17 | 2 (crosses_auc) | 0.842 | 0.689 | 2.35→0.26 | none | 0.57 / 0.00 |
| R3 pure hazard | 44 | 17 | 2 (crosses_auc) | 0.839 | 0.745 | 2.37→0.26 | none | 0.55 / 0.01 |
| R1 auxiliary | 42 | 16 | 1 (crosses_auc) | 0.858 | 0.755 | 0.85→0.11 | none | 0.08 / 0.00 |
| R1 auxiliary | 43 | 16 | 1 (crosses_auc) | 0.860 | 0.763 | 0.85→0.12 | none | 0.09 / 0.00 |
| R1 auxiliary | 44 | 18 | 3 (crosses_auc) | 0.854 | 0.760 | 0.85→0.10 | none | 0.25 / 0.00 |
| R4 hedge | 42 | 20 | 5 (crosses_auc) | 0.861 | 0.711 | 2.72→0.26 | none | 0.13 / 0.00 |
| R4 hedge | 43 | 16 | 1 (crosses_auc) | 0.843 | 0.719 | 2.70→0.33 | none | 0.22 / 0.00 |
| R4 hedge | 44 | 18 | 3 (crosses_auc) | 0.837 | 0.733 | 2.74→0.28 | none | 0.44 / 0.00 |
| R3C censored | 42 | 17 | 2 (crosses_auc) | 0.844 | 0.743 | 2.24→0.25 | none | 0.42 / 0.00 |
| R3C censored | 43 | 17 | 2 (crosses_auc) | 0.836 | 0.678 | 2.24→0.26 | none | 0.48 / 0.01 |
| R3C censored | 44 | 17 | 2 (crosses_auc) | 0.841 | 0.729 | 2.26→0.24 | none | 0.38 / 0.00 |
| R2 anchored-trained | 42 | 21 | 6 (crosses_auc) | 0.844 | 0.806 | 0.77→0.11 | n/a (anchored val) | – |
| R2 anchored-trained | 43 | 17 | 2 (crosses_auc) | 0.849 | 0.817 | 0.78→0.13 | n/a (anchored val) | – |
| R2 anchored-trained | 44 | 17 | 2 (crosses_auc) | 0.835 | 0.820 | 0.77→0.15 | n/a (anchored val) | – |
| Model A streaming-trained | 42 | 16 | 1 (crosses_auc) | 0.848 | 0.785 | 1.41→0.32 | none | – |
| Model A anchored-trained | 42 | 30 | 22 (crosses_auc) | 0.849 | 0.840 | 1.85→0.51 | n/a (anchored val) | – |

# Separability by time to onset (AUC vs never-crossers; streaming test; mean ± sd)

| arm | 0-1s | 1-2s | 2-3s | 3-5s | 5-10s | >=10s |
|---|---|---|---|---|---|---|
| R2 binary | 0.809 ± 0.011 | 0.777 ± 0.009 | 0.723 ± 0.013 | 0.648 ± 0.029 | 0.641 ± 0.028 | 0.519 ± 0.002 |
| R3 pure hazard | 0.820 ± 0.010 | 0.778 ± 0.017 | 0.732 ± 0.016 | 0.668 ± 0.017 | 0.657 ± 0.013 | 0.539 ± 0.010 |
| R1 auxiliary | 0.824 ± 0.010 | 0.788 ± 0.002 | 0.736 ± 0.014 | 0.670 ± 0.012 | 0.661 ± 0.007 | 0.525 ± 0.008 |
| R4 hedge | 0.813 ± 0.009 | 0.772 ± 0.014 | 0.718 ± 0.032 | 0.647 ± 0.033 | 0.635 ± 0.027 | 0.512 ± 0.031 |
| R3C censored | 0.825 ± 0.009 | 0.780 ± 0.011 | 0.734 ± 0.011 | 0.671 ± 0.013 | 0.660 ± 0.007 | 0.553 ± 0.011 |
| R2 anchored-trained | 0.600 ± 0.018 | 0.621 ± 0.043 | 0.633 ± 0.040 | 0.623 ± 0.039 | 0.629 ± 0.036 | 0.708 ± 0.002 |
| Model A streaming-trained | 0.816 | 0.779 | 0.744 | 0.681 | 0.665 | 0.537 |
| Model A anchored-trained | 0.581 | 0.566 | 0.575 | 0.572 | 0.576 | 0.689 |

# Within-track smoothing (k=15) and the who/when split (streaming test; mean ± sd)

Causal = trailing mean (deployable). Centered = looks ahead (offline only; comparable to the paper's earlier 'k=15 moving average').

| arm | window AUC | Δ causal | Δ centered | who crosses (per pedestrian) | when (crossers only, <32 frames) |
|---|---|---|---|---|---|
| R2 binary | 0.783 ± 0.010 | -0.004 ± 0.005 | 0.011 ± 0.009 | 0.766 ± 0.034 | 0.753 ± 0.008 |
| R3 pure hazard | 0.793 ± 0.010 | -0.013 ± 0.001 | 0.011 ± 0.002 | 0.766 ± 0.016 | 0.766 ± 0.010 |
| R1 auxiliary | 0.797 ± 0.010 | -0.006 ± 0.003 | 0.009 ± 0.005 | 0.764 ± 0.036 | 0.765 ± 0.007 |
| R4 hedge | 0.789 ± 0.008 | -0.008 ± 0.002 | 0.015 ± 0.009 | 0.768 ± 0.042 | 0.765 ± 0.010 |
| R3C censored | 0.795 ± 0.009 | -0.017 ± 0.002 | 0.011 ± 0.001 | 0.779 ± 0.016 | 0.765 ± 0.009 |
| R2 anchored-trained | 0.516 ± 0.010 | -0.000 ± 0.023 | -0.003 ± 0.014 | 0.799 ± 0.010 | 0.414 ± 0.012 |
| Model A streaming-trained | 0.785 | -0.013 | 0.009 | 0.763 | 0.753 |
| Model A anchored-trained | 0.511 | -0.016 | -0.016 | 0.805 | 0.418 |

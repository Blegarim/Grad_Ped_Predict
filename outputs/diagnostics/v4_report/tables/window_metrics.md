# Window metrics (test split, val-tuned thresholds; mean ± sd over seeds)

## Tested on streaming

| arm | n | score | AUC | raw F1 | tuned F1 | tuned precision | tuned recall |
|---|---|---|---|---|---|---|---|
| R2 binary | 3 | p_frame | 0.783 ± 0.010 | 0.237 ± 0.023 | 0.243 ± 0.008 | 0.174 ± 0.012 | 0.402 ± 0.023 |
| R3 pure hazard | 3 | p_readout | 0.793 ± 0.010 | 0.234 ± 0.012 | 0.277 ± 0.025 | 0.218 ± 0.047 | 0.396 ± 0.048 |
| R1 auxiliary | 3 | p_frame | 0.797 ± 0.010 | 0.254 ± 0.025 | 0.255 ± 0.032 | 0.192 ± 0.045 | 0.400 ± 0.040 |
| R4 hedge | 3 | p_readout | 0.789 ± 0.008 | 0.255 ± 0.026 | 0.245 ± 0.059 | 0.188 ± 0.069 | 0.413 ± 0.097 |
| R3C censored | 3 | p_readout | 0.795 ± 0.009 | 0.258 ± 0.027 | 0.271 ± 0.028 | 0.209 ± 0.040 | 0.400 ± 0.044 |
| R2 anchored-trained | 3 | p_frame | 0.516 ± 0.010 | 0.063 ± 0.002 | 0.062 ± 0.002 | 0.032 ± 0.001 | 0.820 ± 0.150 |
| Model A streaming-trained | 1 | p_frame | 0.785 | 0.225 | 0.265 | 0.216 | 0.342 |
| Model A anchored-trained | 1 | p_frame | 0.511 | 0.062 | 0.064 | 0.033 | 0.924 |

## Tested on anchored

| arm | n | score | AUC | raw F1 | tuned F1 | tuned precision | tuned recall |
|---|---|---|---|---|---|---|---|
| R2 binary | 3 | p_frame | 0.696 ± 0.028 | 0.388 ± 0.071 | 0.459 ± 0.031 | 0.453 ± 0.165 | 0.582 ± 0.232 |
| R3 pure hazard | 3 | p_readout | 0.708 ± 0.022 | 0.411 ± 0.005 | 0.517 ± 0.029 | 0.399 ± 0.046 | 0.746 ± 0.053 |
| R1 auxiliary | 3 | p_frame | 0.694 ± 0.032 | 0.383 ± 0.019 | 0.496 ± 0.020 | 0.406 ± 0.070 | 0.667 ± 0.114 |
| R4 hedge | 3 | p_readout | 0.698 ± 0.048 | 0.357 ± 0.060 | 0.505 ± 0.039 | 0.411 ± 0.075 | 0.703 ± 0.164 |
| R3C censored | 3 | p_readout | 0.734 ± 0.020 | 0.412 ± 0.020 | 0.536 ± 0.010 | 0.434 ± 0.017 | 0.707 ± 0.081 |
| R2 anchored-trained | 3 | p_frame | 0.889 ± 0.007 | 0.730 ± 0.017 | 0.729 ± 0.015 | 0.664 ± 0.049 | 0.813 ± 0.041 |
| Model A streaming-trained | 1 | p_frame | 0.711 | 0.397 | 0.510 | 0.423 | 0.643 |
| Model A anchored-trained | 1 | p_frame | 0.869 | 0.710 | 0.711 | 0.701 | 0.722 |

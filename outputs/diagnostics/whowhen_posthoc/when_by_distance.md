# POST HOC: when split by how far ahead the negative window's crossing starts (not pre-registered)

AUC of all positives (onset within H) against the hard temporal negatives of each bin; mean over seeds.

| arm | n | 1.07-2s | 2-3s | 3-5s | 5-10s | >=10s | pooled |
|---|---|---|---|---|---|---|---|
| share of hard temporal negatives | | 6.3% | 6.1% | 10.0% | 18.9% | 58.7% | |
| trees, pedestrian only | 5 | 0.603 | 0.684 | 0.770 | 0.776 | 0.855 | 0.805 |
| trees + scene | 5 | 0.635 | 0.734 | 0.805 | 0.809 | 0.897 | 0.845 |
| neural net, pedestrian only | 3 | 0.571 | 0.642 | 0.717 | 0.712 | 0.804 | 0.753 |
| neural net + scene | 3 | 0.612 | 0.679 | 0.728 | 0.727 | 0.853 | 0.791 |
| trees, anchored-trained | 5 | 0.472 | 0.458 | 0.472 | 0.438 | 0.286 | 0.356 |

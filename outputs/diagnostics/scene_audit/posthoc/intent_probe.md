# POST HOC: intent_probe (not pre-registered; computed 2026-10-06T07:54:37Z from the dumps in this directory tree)

Script: `/workspace/setup_logs/v5/intent_probe.py` (copied alongside as intent_probe.py.txt)

```
test: 205 crossing / 495 never-crossing pedestrians (all 700 found in attributes)
PIE 'crossing' attribute of our never-crossers: {'1': 0, '0': 362, '-1': 133}
human intention_prob: crossers median 0.93; never-crossers median 0.82; never-crossers with intention >= 0.5: 373 of 495
HUMAN 'who' AUC (intention_prob vs actually crossed, per pedestrian): 0.739
never-crosser WINDOWS on high-intention pedestrians: 70.0% of 37316
  tree B s42 @205/hr: 80% of its 212 false-alarm windows fall on high-intention never-crossers
  tree B s42 @41/hr: 90% of its 42 false-alarm windows fall on high-intention never-crossers
  tree B s42 detection @205/95/41: all never-crossers as negatives [54.6, 42.9, 36.1] -> only low-intention never-crossers [56.1, 51.7, 42.9]
  deep R2 s42 @205/hr: 79% of its 212 false-alarm windows fall on high-intention never-crossers
  deep R2 s42 @41/hr: 71% of its 42 false-alarm windows fall on high-intention never-crossers
  deep R2 s42 detection @205/95/41: all never-crossers as negatives [44.4, 36.6, 23.9] -> only low-intention never-crossers [49.8, 36.6, 23.9]
```

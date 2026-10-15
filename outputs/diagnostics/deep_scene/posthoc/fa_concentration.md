# POST HOC: fa_concentration, deep scene arm (not pre-registered; computed 2026-10-10T06:06:46Z from the pulled dumps)

Script: `fa_concentration.py` (this directory). Same method as `scene_audit/posthoc/fa_concentration`.

```
never-crossers: 495 pedestrians, 37316 windows; budget @205/hr = 212 alarm windows
deep base     alarms from  35.0 pedestrians (seeds [43, 29, 33]); top-3 hold  51.7 of 212 (seeds [45, 62, 48]); top-1 holds  21.0; det % per seed [44.4, 43.9, 38.5]
deep + scene  alarms from  15.0 pedestrians (seeds [16, 20, 9]); top-3 hold 145.7 of 212 (seeds [131, 122, 184]); top-1 holds  90.0; det % per seed [34.1, 38.5, 24.9]
_c6_r2s_scene_s42 | 3_19_994: 87 alarms / 199 windows, signalized=3, intent=0.82; 3_19_1001: 23 alarms / 55 windows, signalized=3, intent=0.92; 3_12_756: 21 alarms / 269 windows, signalized=3, intent=0.60
_c6_r2s_scene_s43 | 3_19_994: 67 alarms / 199 windows, signalized=3, intent=0.82; 3_8_536: 39 alarms / 146 windows, signalized=3, intent=0.87; 3_10_653: 16 alarms / 154 windows, signalized=3, intent=0.98
_c6_r2s_scene_s44 | 3_10_669: 116 alarms / 241 windows, signalized=3, intent=0.92; 3_10_653: 40 alarms / 154 windows, signalized=3, intent=0.98; 3_10_652: 28 alarms / 418 windows, signalized=3, intent=0.75
```

signalized: 0 n/a, 1 C, 2 S, 3 CS (crosswalk + signal). intent = PIE intention_prob.

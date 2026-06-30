# Backbone Candidate Study (RQ1 / A1–A2) — design note

**Status:** audit complete; **implementation pending confirmation.** This note picks concrete pretrained
backbones to drop in for the visual stream and records *why*. It is the WP2 deliverable that
[RESEARCH_PLAN.md](RESEARCH_PLAN.md) §WP2 calls for. Numbers below are **measured against the installed
`timm` 1.0.20** (built offline, `pretrained=False`, `num_classes=0`, `global_pool='avg'`), not recalled —
re-run the query in [Appendix](#appendix-reproduce-the-numbers) to reproduce.

## Why swap (the motivation, from the hole audit)

The v1 visual stream is a from-scratch hierarchical ViT with three documented defects (A1/A2):
- **Dimension collapse** — features pass through 288 dims then are crushed to **36** before `frame_proj`
  ([vit.py](../src/pedpredict/models/vit.py)); the per-frame visual representation bottlenecks at 36 floats.
- **2×2 attention windows** in the expensive stage — most attention FLOPs buy ~no receptive field.
- **From-scratch on ~96k PIE crops** — ViTs are data-hungry; it has never seen anything else.
- **(new, from the post-benchmark slop pass)** the final `x.mean([2,3])` **global-average-pools the 3×
  context crop**, diluting the pedestrian ~9:1 into background. Any replacement must not re-inherit this.

The benchmark symptom that implicates it: AUC ~0.60 across *all* tasks, including `visual_only`. RQ1 asks
whether a modern pretrained hierarchical backbone fixes this **at a matched param/FLOP budget**.

## The drop-in contract

The visual stream is a pure `[B, T, 3, H, W] context crops → [B, T, d_model=128]` map. The current
`ViT_Hierarchical` does `stem → stages → pool → norm → frame_proj`. A timm backbone slots in behind the
**same contract** with a thin wrapper — no modeling, just API:

```python
# sketch (implement on confirm): src/pedpredict/models/timm_backbone.py
class TimmBackbone(nn.Module):
    def __init__(self, name, img_size, d_model, pretrained=True):
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                      global_pool="avg", in_chans=3)        # -> [B, feat_dim]
        self.frame_proj = nn.Linear(self.net.num_features, d_model)         # feat_dim -> 128
    def forward(self, x):              # x: [B, T, 3, H, W]
        b, t = x.shape[:2]
        f = self.net(x.flatten(0, 1))                                       # [B*T, feat_dim]
        return self.frame_proj(f.view(b, t, -1))                            # [B, T, d_model]
```

Selected by config (`model.vit_backbone: "legacy" | "<timm_name>"`), registered as a backbone option so
`full` / `visual_only` build it via the existing factory. Compatibility notes:
- **Norm matches.** The read pipeline already applies **ImageNet** normalization (`norm_mean/std` in
  `data.yaml` = `0.485…/0.229…`), exactly what these ImageNet-pretrained backbones expect. No change.
- **`frame_proj` absorbs the feature width.** Each backbone has a different pooled `feat_dim`
  (320/256/576…); `Linear(feat_dim, 128)` handles it, identical to how the legacy ViT projects `36→128`.
- **Pooling caveat (the A1 finding).** `global_pool='avg'` re-introduces the context-crop dilution. For a
  first pass use the default avg-pool (simplest, honest baseline for the swap); **flag a follow-up** to
  pool the `forward_features` map with attention / a pedestrian-centered weighting. Keep that a separate
  one-axis spoke, not bundled into the swap.

## Candidate table (measured, `timm` 1.0.20)

| model | params (M) | feat_dim | input | hierarchical (reductions) | pretrain | notes |
|---|---|---|---|---|---|---|
| **tiny_vit_5m_224** | **5.07** | 320 | **224** | ✅ 4-stage [4,8,16,32] | **dist_in22k** | Swin-style window attn; exact res match |
| tiny_vit_11m_224 | 10.55 | 448 | 224 | ✅ [4,8,16,32] | dist_in22k | mid-point of the family Pareto |
| **tiny_vit_21m_224** | **20.62** | 576 | **224** | ✅ [4,8,16,32] | **dist_in22k** | the ~20M "punch-above" |
| **pvt_v2_b0** | **3.41** | 256 | **224** | ✅ pyramid SRA | in1k | non-window mechanism (diversity) |
| pvt_v2_b1 | 13.50 | 512 | 224 | ✅ pyramid SRA | in1k | — |
| fastvit_t8 | 3.26 | 768 | 256 | ✅ [4,8,16,32] | apple_dist | reparam, latency-king; **256 input** |
| fastvit_sa12 | 10.56 | 1024 | 256 | ✅ | apple_dist | latency-optimized ~10M; **256 input** |
| mobilevitv2_100 | 4.39 | 512 | 256 | ✅ 5-stage | cvnets_in1k | hybrid; **256 input** |
| mobilevit_s | 4.94 | 640 | 256 | ✅ 5-stage | cvnets_in1k | hybrid; **256 input** |
| edgenext_small | 5.28 | 304 | 256 | ✅ | usi_in1k | **256 input** |
| efficientformerv2_s1 | 5.74 | 224 | 224 | ✅ | snap_dist | on-device-latency design |
| efficientformerv2_s2 | 12.13 | 288 | 224 | ✅ | snap_dist | — |
| xcit_tiny_12_p16_224 | 6.52 | 192 | 224 | ❌ isotropic (all /16) | fb_dist | breaks the factorized-hierarchy story → **excluded** |

Selection criteria applied: (a) pretrained weights in `timm` ✓ for all; (b) **224 input = zero-surgery**
match to the current `read_context=224` (256-input models need `read_context→256`, a config change at read
time, no rebuild — but a complication); (c) **hierarchical/windowed** to keep the factorized space-time
narrative; (d) clean `num_features → frame_proj → 128` drop-in ✓ for all.

## The picks

**Lightweight #1 — primary: `tiny_vit_5m_224.dist_in22k` (5.07M).** The cleanest drop-in on every axis:
**224 input matches the pipeline exactly** (no resolution surgery), genuinely **Swin-style window
attention** so the architecture narrative survives the swap, and the **strongest pretraining** here
(ImageNet-**22k** distillation). At 5.07M it is **param-matched to the current ViT (~5.6M)** → the *fair*
RQ1 comparison (same budget, pretrained-hierarchical vs from-scratch).

**Lightweight #2 — mechanism diversity: `pvt_v2_b0.in1k` (3.41M).** A *different* hierarchical mechanism
(spatial-reduction attention, not windows) at 224. Including it stops RQ1 from being a single-family
result: if TinyViT wins, PVTv2 tells you whether it's *pretraining+hierarchy* generally or *windows*
specifically. Smallest and fastest of the set.

**Punch-above — `tiny_vit_21m_224.dist_in22k` (20.62M).** Hits the ~20M cutoff to **stress real-time
deployment** (RQ5), and — the decisive feature — it's the **same family** as the 5M pick, so 5M→11M→21M is
a clean **3-point accuracy/latency Pareto from one architecture** (a strong thesis figure), confounded by
nothing but capacity. If 21M proves too slow on the A4500 / edge target, the **latency-optimized fallback**
is `fastvit_sa12` (10.56M, reparameterizable, Apple-designed for on-device speed) — note its 256 input.

## Suggested experiment order (on confirm)

1. Wrapper + `model.vit_backbone` config option + a small build/shape test (no modeling).
2. Hub spoke: `full` with `tiny_vit_5m_224` vs the legacy ViT hub — the RQ1 headline (param-matched).
3. If it wins, promote to the carried-forward reference; add `tiny_vit_21m_224` (+ optionally 11M) for the
   RQ5 Pareto, and `pvt_v2_b0` for the mechanism check.
4. Separate later spoke: attention/pedestrian-centered pooling vs `global_pool='avg'` (the A1 dilution fix).

Budget: 1 backbone is a config + a thin wrapper; each comparison is one training run on the lab PC.

## Appendix: reproduce the numbers

```python
import timm, torch
for name in ["tiny_vit_5m_224", "tiny_vit_21m_224", "pvt_v2_b0", "fastvit_sa12"]:
    m = timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
    print(name, sum(p.numel() for p in m.parameters())/1e6, "M",
          m.num_features, m.default_cfg["input_size"], [f["reduction"] for f in m.feature_info])
```
`timm.list_models("tiny_vit*", pretrained=True)` lists the available pretrained tags (e.g. `.dist_in22k`).

# Backbone Candidate Study (A1/A2)

**Status:** desk research (no code changed). Deliverable per [HOLE_AUDIT](HOLE_AUDIT.md) A2: rank
pretrained drop-in backbones for `ViT_Hierarchical`, name a primary + fallback. Scoped early in WP2.

## The drop-in contract (traced from code)

| Constraint | Value | Source |
|---|---|---|
| Model input | 224×224×3, **ImageNet-normalized already** | `data.yaml` `read_context_*=224`, `norm_mean/std` |
| Temporal fold | backbone runs **per-frame**, `[B,T,…]→[B·T,…]`, **T=20** | `vit.py:308-320` |
| Output contract | `[B, T, 128]` via *pooled feat → `frame_proj` Linear(feat→128)* | `vit.py:265-320`, `registry.py:93` |
| Budget anchor | current ViT params/FLOPs ⚠️ **measure on GPU box** | needs `torch` |

**Binding constraint = per-frame latency, not param count.** The backbone is called 20× per sample
(`[B,T,C,H,W]→[B·T,C,H,W]`), so FLOPs/frame multiply by 20 before batch.

**Integration is uniform across candidates** via timm's feature API — collapses criterion (f):
```python
backbone = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
feat = backbone(x_bt)                  # [B·T, num_features]
out  = frame_proj(feat).view(B,T,128)  # frame_proj = Linear(num_features, 128)
```
The only per-candidate wiring variable is `num_features` (the `frame_proj` input width).

## Candidates (verified against `timm==1.0.20`)

| Model | timm id (pretrained) | Params | ~GFLOPs/frame | Pretrain | Hierarchical/windowed | feat dim | Native res |
|---|---|---|---|---|---|---|---|
| **TinyViT-5M** | `tiny_vit_5m_224.dist_in22k_ft_in1k` | ~5.4M | ~1.3 | **IN-22k distilled**→1k | ✅ Swin-style windows `[7,7,14,7]` | 320 | 224 |
| **FastViT-T8** | `fastvit_t8.apple_in1k` | ~3.6M | ~0.7 | IN-1k | ✅ hierarchical, conv/RepMixer mixing | 1024 | 256* |
| **PVTv2-B0** | `pvt_v2_b0.in1k` | ~3.4M | ~0.6 | IN-1k | ✅ pyramid, SRA (not windows) | 256 | 224 |
| **MobileViT-S** | `mobilevit_s.cvnets_in1k` | ~5.6M | ~2.0 | IN-1k | hybrid CNN-ViT | 640 | 256 |
| **DeiT-Tiny** | `deit_tiny_patch16_224.fb_in1k` | ~5.7M | ~1.3 | IN-1k | ❌ plain, single-scale | 192 | 224 |

\* FastViT is fully conv/reparameterizable → resolution-flexible; runs at 224 without surgery.

## Scoring vs A2 criteria

- **(a) timm + pretrained, ideally 21k:** only **TinyViT-5M** ships IN-22k (distilled) weights. Decisive.
- **(b) accepts input res without surgery:** TinyViT/PVTv2/DeiT native 224; FastViT res-agnostic;
  MobileViT native 256 (but see Resolution Headroom — no longer a hard penalty).
- **(c) hierarchical/windowed (keeps the factorized-space-time thesis narrative):** TinyViT is genuinely
  Swin-style → narrative survives intact. PVTv2 (SRA) and FastViT (conv) weaken it slightly.
  DeiT-Tiny breaks it (single-scale) → ablation rung only, not the WP2 backbone.
- **(d) budget matched to current ViT:** all 3.4–5.7M, same class. ⚠️ measure the current ViT to anchor.
- **(e) A4500 latency (×20/frame):** FastViT purpose-built for on-device latency (reparam to plain conv);
  lowest FLOPs. TinyViT throughput-strong but heavier. ⚠️ benchmark on real A4500 — can flip 1↔2.
- **(f) clean drop-in:** uniform via `num_classes=0` — non-differentiating.

## Recommendation

- **Primary: TinyViT-5M** (`tiny_vit_5m_224.dist_in22k_ft_in1k`). Only candidate winning (a) **and** (c):
  IN-22k distilled pretraining + genuine Swin-style windowed hierarchy. Largest expected jump over the
  from-scratch ViT, and the windowed-attention thesis story survives the swap. `frame_proj = Linear(320,128)`.
  A1's 36→288→36 collapse + 2×2 windows become the *motivation* for the swap.
- **Fallback: FastViT-T8** (`fastvit_t8.apple_in1k`). If A4500 latency at T=20×batch makes TinyViT too
  slow, ~half the FLOPs, latency-optimized, still hierarchical. Trades 22k-pretrain + pure-windowed story
  for speed. `frame_proj = Linear(1024,128)`. PVTv2-B0 = third option if absolute smallest params needed.
- **Drop:** MobileViT-S (256-native fights the pipeline; IN-1k; 2.0 GFLOPs) and DeiT-Tiny (non-hierarchical
  → breaks framing; keep only as a deliberate "does hierarchy matter?" rung).

## Resolution Headroom (free WP2 lever — recorded so it isn't folklore)

The data layer **stores context crops at 384** (`img_height × context_scale = 128 × 3.0`,
`transforms.py:174`) but the runtime **downsamples to 224** at read time
(`Resize((read_context_height, read_context_width))`, `transforms.py:196`) before the ViT. This
store-384/read-224 gap is an **inherited legacy inefficiency** (OLD `train.py:355-366`; flagged in
`MIGRATION.md` as "a legacy inefficiency preserved this phase") — OLD already trained and inferred at 224.

Consequence: the stored 384 is **latent resolution headroom already paid for on disk**. Reclaiming it is a
one-line config bump (`read_context_height: 256`), **no re-extraction, no rebuild** — capped at **384**
(beyond that needs a full re-extraction).

Caveats:
1. **Free to enable, not to run** — higher read res costs FLOPs ×20 frames (real A4500 latency hit).
2. **Worthless to the current ViT** — its 36-dim/2×2-window architecture (A1) can't exploit the detail;
   384 just burns ~3× compute. The headroom becomes *value* only behind a pretrained backbone.
3. Bounded by the 384 store.

**→ WP2 spoke:** input-resolution ablation **224 vs 256** (both free from the 384 store), run only on the
swapped pretrained backbone, compared to the hub.

## Open items before WP2 implementation (need `torch` on the GPU box)

1. **Measure the current ViT** (the missing budget anchor):
   ```python
   from pedpredict.models.vit import ViT_Hierarchical
   from pedpredict.config import ModelCfg
   m = ViT_Hierarchical.from_config(ModelCfg(), 224)
   print(sum(p.numel() for p in m.parameters()))   # + fvcore FLOPs @ 1 frame
   ```
2. **A4500 latency bench** TinyViT-5M vs FastViT-T8 at realistic `B·T` (e.g. 16×20=320 imgs/forward) —
   the only number that can flip primary↔fallback.
3. **`frame_proj` resize** (320 or 1024 → 128) is the entire model-code delta; reuse the existing
   `freeze_backbone` partition for warmup.
4. **`rebuild_position_bias` becomes moot** — timm backbones own their pos-embedding; the benchmark/export
   resolution path simplifies (doc-sync when this lands).

## Sources

- [timm/tiny_vit_5m_224.dist_in22k_ft_in1k](https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k_ft_in1k)
- [timm tiny_vit.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/tiny_vit.py)
- [timm/fastvit_t8.apple_in1k](https://huggingface.co/timm/fastvit_t8.apple_in1k)
- [timm pvt_v2.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/pvt_v2.py)
- [timm mobilevit.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilevit.py)
- [TinyViT (ECCV 2022)](https://arxiv.org/pdf/2207.10666)
- [FastViT (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.pdf)

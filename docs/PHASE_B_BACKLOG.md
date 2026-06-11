# Phase B — Architecture redesign backlog

> **SUPERSEDED (2026-06-11):** the answered [HOLE_AUDIT.md](HOLE_AUDIT.md) is now the working setlist
> (see its Final attack order). Item mapping: 1 → A1/A2 · 2 → A3/A5 · 4 → A4 · 8 → M1/C4 · 9 → M8
> (resolved: fix) · 10 → Q7 · 11 → absorbed by WP0 baselines on the v2 dataset. Items **3** (unified
> crosses head), **5** (online augmentation), and **6** (variable-length sequences) are not audit holes
> and remain live in [RESEARCH_PLAN.md](RESEARCH_PLAN.md) WP1/WP2. Kept for reference only.

Phase A (the behavior-preserving rebuild) is locked at the **v1.0 clean baseline**. Phase B is the
architecture-redesign phase: it may deliberately change model math and outputs. Each item below was
deferred during Phase A and flagged in the (now archived) migration ledger
([docs/archive/MIGRATION.md](archive/MIGRATION.md)). Each should get its own design doc + golden
re-baseline before it lands.

## Model architecture
1. **ViT backbone replacement.** The hierarchical ViT uses eager, resolution-bound relative-position
   tables — a 224-trained checkpoint cannot `strict`-load at another resolution by design. Evaluate a
   modern backbone (or resolution-agnostic rel-pos) as a drop-in for `ViT_Hierarchical`.
2. **Fusion redesign.** Revisit LayerNorm-before-fusion + cross-attention; consider alternatives to the
   query=motion / key=image arrangement.
3. **Single unified crosses head.** Collapse `crosses_frame` and the live-but-unsupervised
   `crosses_pooled` into one supervised head, retiring the dual-head contract (the lasting fix for the
   old dead-head smell).

## Data & motion features
4. **Motion-feature corrections** (currently preserved quirks):
   - frame-0 `dw`/`dh` (idx 6/7) hold the *raw* `w0`/`h0`, not a delta.
   - absolute `cx` (idx 0) is not reflected under horizontal flip (only `dx`, idx 2, is negated).
   - motion-noise augmentation perturbs absolute channels too.
5. **Online augmentation** replacing the offline, write-time augmentation pass.
6. **Variable-length sequences** — drop the fixed `seq_len=20` truncate-no-pad policy; add a padding +
   masking path.

## Training & imbalance
7. **Standard DataLoader sharding** replacing the custom `ChunkPrefetcher` / chunked-LMDB prefetch.
8. **Imbalance policy v2** — global (not per-chunk) sampler frequencies; remove the
   `legacy_x00_sign_bug` compatibility flag in `data/balance.py`; optional FocalLoss.
9. **Schedule semantics** — reconsider driving the LR scheduler and early-stopping on macro-F1 rather
   than val_loss.

## Tooling
10. **Hard CI coverage floor** (Phase A left `--cov` soft / advisory).
11. **End-to-end parity report** — once PIE + GPU are available, run `scripts/evaluate.py` per
    `model_type` and record clean baselines in [docs/archive/legacy_baselines.md](archive/legacy_baselines.md).

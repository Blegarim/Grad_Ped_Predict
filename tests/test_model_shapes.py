"""Prompt 2.1 — ViT_Hierarchical port tests.

  * GOLDEN PARITY: new ViT loads the OLD post-forward state_dict (strict=True) and reproduces the OLD
    output within tolerance (eval mode -> deterministic fp32 math).
  * B2 (no lazy params): the OLD state_dict loads strict with NO dummy forward; all params (incl. the
    global-stage relative-position table) exist at __init__.
  * GEOMETRY: feature_map_size matches the runtime feature maps and the derived global window.
  * B6 SMOKE: from_config(ModelCfg) forwards to [B, T, d_model].
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pedpredict.config import DataCfg, ModelCfg, RootCfg
from pedpredict.models.ablations import MotionOnlyModel, VanillaConcatModel, VisualOnlyModel
from pedpredict.models.cross_attention import CrossAttentionModule
from pedpredict.models.ensemble import EnsembleModel
from pedpredict.models.geometry import feature_map_size
from pedpredict.models.heads import (
    build_crosses_frame_head,
    build_pool_mlp,
    build_task_classifiers,
    frame_pool_reduce,
    temporal_attention_pool,
)
from pedpredict.models.motion_encoder import MotionEncoder
from pedpredict.models.registry import (
    MODEL_INPUT_SIGNATURE,
    ModelType,
    build_model,
    forward_model,
)
from pedpredict.models.vit import ViT_Hierarchical

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "vit.pt"
_MOTION_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "motion_encoder.pt"
_CROSS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "cross_attention.pt"
_ENSEMBLE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "ensemble.pt"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not _FIXTURE.exists():
        pytest.skip(f"missing golden fixture {_FIXTURE} (run tests/_capture/capture_vit_golden.py)")
    return torch.load(_FIXTURE, weights_only=False)


def _build(golden: dict) -> ViT_Hierarchical:
    return ViT_Hierarchical(img_size=golden["img_size"], **golden["vit_kwargs"])


# --------------------------------------------------------------------------- golden parity


def test_golden_vit_parity(golden: dict) -> None:
    model = _build(golden)
    model.load_state_dict(golden["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        y = model(golden["inputs"]["x"])
    tol = golden["meta"]
    torch.testing.assert_close(y, golden["outputs"]["y"], atol=tol["atol"], rtol=tol["rtol"])


# --------------------------------------------------------------------------- B2: eager params / strict load


def test_strict_load_without_dummy_forward(golden: dict) -> None:
    """B2 resolved: the OLD post-forward state_dict loads with NO missing/unexpected keys and NO forward."""
    model = _build(golden)  # never call forward first
    missing, unexpected = model.load_state_dict(golden["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.load_state_dict(golden["state_dict"], strict=True)  # must not raise


def test_global_table_exists_at_init() -> None:
    """The global (last) stage's relative-position table exists immediately after construction (B2)."""
    cfg = ModelCfg()
    model = ViT_Hierarchical.from_config(cfg, img_size=224)
    last = model.stages[-1]["block"][0].attn
    # global window at 224 is 7x7 -> table (2*7-1)^2 = 169 rows, head_nums[-1]=2 cols
    assert tuple(last.relative_position_bias_table.shape) == (169, cfg.head_nums[-1])
    assert tuple(last.relative_position_index.shape) == (49, 49)


# --------------------------------------------------------------------------- geometry


def test_feature_map_size_matches_runtime() -> None:
    """feature_map_size predicts the actual per-stage feature maps captured via forward hooks."""
    model = ViT_Hierarchical.from_config(ModelCfg(), img_size=224).eval()
    seen: list[int] = []

    def _hook(_module, args, _out) -> None:
        seen.append(args[0].shape[-1])  # H==W of the block input

    handles = [blk.register_forward_hook(_hook) for stage in model.stages for blk in stage["block"]]
    try:
        with torch.no_grad():
            model(torch.randn(1, 1, 3, 224, 224))
    finally:
        for h in handles:
            h.remove()

    # first block of each stage should match feature_map_size(224, stage_idx)
    expected_per_stage = [feature_map_size(224, i) for i in range(len(model.stages))]
    assert expected_per_stage == [56, 28, 14, 7]
    # every block input size equals its stage's predicted size
    idx = 0
    for i, stage in enumerate(model.stages):
        for _ in stage["block"]:
            assert seen[idx] == expected_per_stage[i]
            idx += 1


# --------------------------------------------------------------------------- B6 smoke + rebuild


def test_from_config_output_shape() -> None:
    cfg = ModelCfg()
    model = ViT_Hierarchical.from_config(cfg, img_size=224).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 5, cfg.in_channels, 224, 224))
    assert out.shape == (2, 5, cfg.d_model)
    assert torch.isfinite(out).all()


def test_rebuild_position_bias_changes_global_window() -> None:
    """Explicit resolution change rebuilds the global table to the new feature-map size; forward runs."""
    model = ViT_Hierarchical.from_config(ModelCfg(), img_size=224).eval()
    model.rebuild_position_bias(128)
    # 128 -> stem 32 -> 16 -> 8 -> 4; global window 4x4 -> table (2*4-1)^2 = 49 rows
    last = model.stages[-1]["block"][0].attn
    assert tuple(last.relative_position_bias_table.shape) == (49, 2)
    with torch.no_grad():
        out = model(torch.randn(1, 2, 3, 128, 128))
    assert out.shape == (1, 2, 128)


# =========================================================================== Prompt 2.2 — MotionEncoder


@pytest.fixture(scope="module")
def motion_golden() -> dict:
    if not _MOTION_FIXTURE.exists():
        pytest.skip(f"missing golden fixture {_MOTION_FIXTURE} (run tests/_capture/capture_motion_golden.py)")
    return torch.load(_MOTION_FIXTURE, weights_only=False)


def test_golden_motion_parity(motion_golden: dict) -> None:
    """New MotionEncoder loads the OLD state_dict (strict=True) and reproduces the OLD output (eval).

    Pinned to ``motion_norm="per_sequence"`` — the legacy in-forward z-norm the capture used; the
    A4 default ("image") is a deliberate, config-gated behavior change outside this parity surface.
    """
    model = MotionEncoder(**motion_golden["motion_kwargs"], motion_norm="per_sequence")
    model.load_state_dict(motion_golden["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        y = model(motion_golden["inputs"]["motion"], motion_golden["inputs"]["tight"])
    tol = motion_golden["meta"]
    torch.testing.assert_close(y, motion_golden["outputs"]["y"], atol=tol["atol"], rtol=tol["rtol"])


def test_strict_load_motion_no_lazy_params(motion_golden: dict) -> None:
    """No B2-style lazy params: the OLD state_dict loads strict with no missing/unexpected keys, no forward."""
    model = MotionEncoder(**motion_golden["motion_kwargs"])  # never call forward first
    missing, unexpected = model.load_state_dict(motion_golden["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.load_state_dict(motion_golden["state_dict"], strict=True)  # must not raise


def test_motion_from_config_output_shape() -> None:
    cfg = ModelCfg()
    model = MotionEncoder.from_config(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 5, cfg.motion_dim), torch.randn(2, 5, 3, 128, 128))
    assert out.shape == (2, 5, cfg.d_model)
    assert torch.isfinite(out).all()


def test_motion_pos_encoding_capacity_guard() -> None:
    """T == capacity forwards; T > capacity raises a clear error instead of an opaque broadcast crash."""
    model = MotionEncoder(max_positions=6).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 6, model.motion_dim), torch.randn(1, 6, 3, 128, 128))
    assert out.shape == (1, 6, model.d_model)
    with pytest.raises(ValueError, match="exceeds positional-encoding capacity"):
        model(torch.randn(1, 7, model.motion_dim), torch.randn(1, 7, 3, 128, 128))


def test_motion_conv_in_channels_matches_datacfg() -> None:
    """Coupling guard (1.2/1.4/2.2): the Conv1d input width equals the consumed motion_dim."""
    model = MotionEncoder.from_config(ModelCfg())
    assert model.motion_encoder[0].in_channels == DataCfg().motion_dim == ModelCfg().motion_dim


# --------------------------------------------------------------------------- A4: motion_norm flag


def test_motion_norm_image_scale_vector() -> None:
    """Image norm divides x-channels by W, y-channels by H, ego by ego_speed_scale (channel order)."""
    model = MotionEncoder.from_config(ModelCfg(motion_dim=9, motion_norm="image"))
    expected = [1920.0, 1080.0, 1920.0, 1080.0, 1920.0, 1080.0, 1920.0, 1080.0, 50.0]
    assert model.motion_scale.flatten().tolist() == expected
    # default 8-dim slices the same pattern
    model8 = MotionEncoder.from_config(ModelCfg())
    assert model8.motion_scale.flatten().tolist() == expected[:8]


def test_motion_norm_image_preserves_absolute_geometry() -> None:
    """Two constant-position sequences at different cx must produce different outputs under image
    norm (per-sequence z-norm erases the difference — that is exactly hole A4)."""
    torch.manual_seed(0)
    model = MotionEncoder.from_config(ModelCfg(motion_norm="image")).eval()
    tight = torch.zeros(1, 4, 3, 32, 32)
    left = torch.tensor([[100.0, 500.0, 0.0, 0.0, 50.0, 100.0, 0.0, 0.0]]).repeat(4, 1).unsqueeze(0)
    right = left.clone()
    right[..., 0] = 1800.0  # same box, opposite side of the frame
    with torch.no_grad():
        out_left = model(left, tight)
        out_right = model(right, tight)
    assert not torch.allclose(out_left, out_right)


def test_motion_norm_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="motion_norm"):
        MotionEncoder(motion_norm="zscore")


def test_motion_scale_buffer_not_in_state_dict() -> None:
    """The norm buffer is non-persistent so OLD checkpoints keep loading strict=True."""
    model = MotionEncoder.from_config(ModelCfg())
    assert "motion_scale" not in model.state_dict()


# =========================================================================== Prompt 2.3 — CrossAttention

_CROSS_DEFAULT_KEYS = {"actions", "looks", "crosses_pooled", "crosses_frame", "temporal_weights"}
_CROSS_LEGACY_KEYS = ("actions", "looks", "crosses_frame", "temporal_weights")


@pytest.fixture(scope="module")
def cross_golden() -> dict:
    if not _CROSS_FIXTURE.exists():
        pytest.skip(
            f"missing golden fixture {_CROSS_FIXTURE} (run tests/_capture/capture_cross_attention_golden.py)"
        )
    return torch.load(_CROSS_FIXTURE, weights_only=False)


def _build_cross(cross_golden: dict) -> CrossAttentionModule:
    return CrossAttentionModule(**cross_golden["cross_kwargs"])


def test_golden_cross_attention_parity(cross_golden: dict) -> None:
    """New module loads the OLD state_dict (strict=True) and reproduces every golden key (eval)."""
    model = _build_cross(cross_golden)
    model.load_state_dict(cross_golden["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(cross_golden["inputs"]["motion_feats"], cross_golden["inputs"]["image_feats"])
    tol = cross_golden["meta"]
    # All 5 keys: the 4 genuine legacy outputs + crosses_pooled (reconstructed from legacy weights, B4).
    assert set(out) == _CROSS_DEFAULT_KEYS
    for key, expected in cross_golden["outputs"].items():
        torch.testing.assert_close(out[key], expected, atol=tol["atol"], rtol=tol["rtol"])


def test_strict_load_cross_attention_no_lazy_params(cross_golden: dict) -> None:
    """B4 param parity: OLD state_dict (incl. the legacy-dead classifier.crosses) loads strict, no forward."""
    model = _build_cross(cross_golden)  # never call forward first
    missing, unexpected = model.load_state_dict(cross_golden["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert any(k.startswith("classifier.crosses") for k in cross_golden["state_dict"]), (
        "fixture should retain the legacy classifier.crosses param (param-layout parity)"
    )
    model.load_state_dict(cross_golden["state_dict"], strict=True)  # must not raise


def test_cross_attention_emit_flag_default_on() -> None:
    """B4 default: crosses_pooled IS emitted; shape [B, C]; the 4 legacy keys are byte-identical to off."""
    cfg = ModelCfg()
    motion = torch.randn(2, 5, cfg.d_model)
    image = torch.randn(2, 5, cfg.d_model)

    on = CrossAttentionModule.from_config(cfg).eval()
    off = CrossAttentionModule(**cfg.cross_kwargs(), emit_crosses_pooled=False)
    off.load_state_dict(on.state_dict(), strict=True)  # same weights
    off.eval()

    with torch.no_grad():
        out_on = on(motion, image)
        out_off = off(motion, image)

    assert set(out_on) == _CROSS_DEFAULT_KEYS
    assert "crosses_pooled" not in out_off
    assert set(out_off) == set(_CROSS_LEGACY_KEYS)
    assert out_on["crosses_pooled"].shape == (2, cfg.num_classes["crosses"])
    for key in _CROSS_LEGACY_KEYS:  # gating crosses_pooled must not perturb the legacy keys
        torch.testing.assert_close(out_on[key], out_off[key])


def test_cross_attention_output_shapes() -> None:
    cfg = ModelCfg()
    b, t = 3, 7
    model = CrossAttentionModule.from_config(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(b, t, cfg.d_model), torch.randn(b, t, cfg.d_model))
    assert out["actions"].shape == (b, cfg.num_classes["actions"])
    assert out["looks"].shape == (b, cfg.num_classes["looks"])
    assert out["crosses_frame"].shape == (b, cfg.num_classes["crosses"])
    assert out["crosses_pooled"].shape == (b, cfg.num_classes["crosses"])
    assert out["temporal_weights"].shape == (b, t)
    assert torch.isfinite(out["crosses_frame"]).all()


def test_cross_attn_heads_from_config() -> None:
    """get_model wired num_heads=4 (NOT the legacy class default 8); config must reproduce that."""
    model = CrossAttentionModule.from_config(ModelCfg())
    assert model.cross_attn.num_heads == 4


# --------------------------------------------------------------------------- heads.py in isolation


def test_heads_builders_shapes() -> None:
    d, dropout = 128, 0.1
    num_classes = {"actions": 2, "looks": 2, "crosses": 2}
    pool_mlp = build_pool_mlp(d)
    classifiers = build_task_classifiers(num_classes, d, dropout)
    frame_head = build_crosses_frame_head(d, num_classes["crosses"])
    assert set(classifiers) == set(num_classes)  # incl. crosses (param-layout parity)
    feats = torch.randn(4, d)
    assert pool_mlp(torch.randn(4, 6, d)).shape == (4, 6, 1)
    assert classifiers["actions"](feats).shape == (4, 2)
    assert frame_head(torch.randn(4, 6, d)).shape == (4, 6, 2)


def test_temporal_attention_pool_weights_normalized() -> None:
    feats = torch.randn(3, 8, 16)
    pooled, weights = temporal_attention_pool(feats, build_pool_mlp(16))
    assert pooled.shape == (3, 16)
    assert weights.shape == (3, 8)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))  # softmax over time


def test_frame_pool_reduce_modes() -> None:
    x = torch.randn(2, 5, 2)
    torch.testing.assert_close(frame_pool_reduce(x, "mean"), x.mean(dim=1))
    torch.testing.assert_close(frame_pool_reduce(x, "max"), x.max(dim=1).values)
    torch.testing.assert_close(frame_pool_reduce(x, "logsumexp"), torch.logsumexp(x, dim=1))
    with pytest.raises(ValueError, match="Unsupported frame_pool"):
        frame_pool_reduce(x, "median")  # type: ignore[arg-type]


# =========================================================================== Prompt 2.4 — Ensemble + registry

_FULL_KEYS = {"actions", "looks", "crosses_pooled", "crosses_frame", "temporal_weights"}
_ABLATION_TYPES = (ModelType.MOTION_ONLY, ModelType.VISUAL_ONLY, ModelType.VANILLA_CONCAT)


@pytest.fixture(scope="module")
def ensemble_golden() -> dict:
    if not _ENSEMBLE_FIXTURE.exists():
        pytest.skip(
            f"missing golden fixture {_ENSEMBLE_FIXTURE} (run tests/_capture/capture_ensemble_golden.py)"
        )
    return torch.load(_ENSEMBLE_FIXTURE, weights_only=False)


def _dummy_full_inputs(cfg: ModelCfg, img_size: int = 224, b: int = 2, t: int = 3) -> tuple[torch.Tensor, ...]:
    tight = torch.randn(b, t, cfg.in_channels, 128, 128)
    context = torch.randn(b, t, cfg.in_channels, img_size, img_size)
    motions = torch.randn(b, t, cfg.motion_dim)
    return tight, context, motions


# --------------------------------------------------------------------------- golden parity (full model)


def test_golden_ensemble_full_parity(ensemble_golden: dict) -> None:
    """New EnsembleModel loads the OLD full-model state_dict (strict=True) and reproduces every key (eval).

    Pinned to the legacy per-sequence motion norm (see test_golden_motion_parity).
    """
    entry = ensemble_golden["full"]
    model = EnsembleModel.from_config(ModelCfg(motion_norm="per_sequence"), img_size=entry["img_size"])
    model.load_state_dict(entry["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(entry["inputs"]["images_tight"], entry["inputs"]["images_context"], entry["inputs"]["motions"])
    tol = entry["meta"]
    assert set(out) == _FULL_KEYS  # 4 legacy keys + crosses_pooled (B4, recomputed from legacy weights)
    for key, expected in entry["outputs"].items():
        torch.testing.assert_close(out[key], expected, atol=tol["atol"], rtol=tol["rtol"])


def test_strict_load_ensemble_full_no_missing_unexpected(ensemble_golden: dict) -> None:
    """The OLD full state_dict loads strict with zero missing/unexpected keys and NO forward (eager ViT, 2.1)."""
    entry = ensemble_golden["full"]
    model = EnsembleModel.from_config(ModelCfg(), img_size=entry["img_size"])  # never forward first
    missing, unexpected = model.load_state_dict(entry["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    model.load_state_dict(entry["state_dict"], strict=True)  # must not raise


def test_ensemble_return_feats_path(ensemble_golden: dict) -> None:
    """return_feats yields the post-LayerNorm fusion features (the viz path, 6.2) alongside the logits."""
    entry = ensemble_golden["full"]
    model = EnsembleModel.from_config(ModelCfg(), img_size=entry["img_size"])
    model.load_state_dict(entry["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits, image_feats, motion_feats = model(
            entry["inputs"]["images_tight"],
            entry["inputs"]["images_context"],
            entry["inputs"]["motions"],
            return_feats=True,
        )
    b, t = entry["inputs"]["motions"].shape[:2]
    assert set(logits) == _FULL_KEYS
    assert image_feats.shape == (b, t, ModelCfg().d_model)
    assert motion_feats.shape == (b, t, ModelCfg().d_model)


# --------------------------------------------------------------------------- registry: typed factory (B10)


def test_modeltype_coerce_valid_and_invalid() -> None:
    assert ModelType.coerce("full") is ModelType.FULL
    assert ModelType.coerce(ModelType.MOTION_ONLY) is ModelType.MOTION_ONLY
    with pytest.raises(ValueError, match="Unknown model type: 'ful'"):
        ModelType.coerce("ful")  # B10: a typo is a clear error, not a silent wrong-branch


def test_model_input_signature_covers_all_types() -> None:
    assert set(MODEL_INPUT_SIGNATURE) == set(ModelType)


def test_build_model_full_from_root() -> None:
    model = build_model(RootCfg(), "full")
    assert isinstance(model, EnsembleModel)
    assert model.model_type is ModelType.FULL


def test_build_model_defaults_to_eval_model_type() -> None:
    """No explicit type -> read cfg.eval.model_type (default 'full')."""
    cfg = RootCfg()
    assert cfg.eval.model_type == "full"
    assert build_model(cfg).model_type is ModelType.FULL


_ABLATION_CLASS = {
    ModelType.MOTION_ONLY: MotionOnlyModel,
    ModelType.VISUAL_ONLY: VisualOnlyModel,
    ModelType.VANILLA_CONCAT: VanillaConcatModel,
}


@pytest.mark.parametrize("mt", _ABLATION_TYPES, ids=lambda m: m.value)
def test_build_model_ablations(mt: ModelType) -> None:
    """Prompt 2.5: ablation classes now build through the registry to their concrete types (B10)."""
    model = build_model(RootCfg(), mt)
    assert isinstance(model, _ABLATION_CLASS[mt])
    assert model.model_type is mt


# --------------------------------------------------------------------------- registry: forward adapter


def test_forward_model_full_shapes() -> None:
    cfg = ModelCfg()
    model = build_model(RootCfg(), "full").eval()
    tight, context, motions = _dummy_full_inputs(cfg)
    with torch.no_grad():
        out = forward_model(model, tight, context, motions)
    assert set(out) == _FULL_KEYS
    assert out["actions"].shape == (2, cfg.num_classes["actions"])
    assert out["crosses_frame"].shape == (2, cfg.num_classes["crosses"])
    assert out["temporal_weights"].shape == (2, 3)


def test_forward_model_unpacks_collate_triple() -> None:
    """forward_model(model, *batch[:3]) is the intended call form (collate returns the triple + labels)."""
    cfg = ModelCfg()
    model = build_model(RootCfg(), "full").eval()
    batch = (*_dummy_full_inputs(cfg), {"actions": torch.zeros(2, dtype=torch.long)})  # +labels
    with torch.no_grad():
        out = forward_model(model, *batch[:3])
    assert set(out) == _FULL_KEYS


def test_forward_model_return_feats_full_only() -> None:
    cfg = ModelCfg()
    model = build_model(RootCfg(), "full").eval()
    tight, context, motions = _dummy_full_inputs(cfg)
    with torch.no_grad():
        result = forward_model(model, tight, context, motions, return_feats=True)
    assert isinstance(result, tuple) and len(result) == 3


def test_forward_model_dispatch_is_intrinsic() -> None:
    """forward_model needs no type argument — it reads model.model_type set by build_model (B10)."""
    model = build_model(RootCfg(), "full")
    assert model.model_type is ModelType.FULL


# =========================================================================== Prompt 2.5 — Ablation models

_ABLATION_KEYS_ON = {"actions", "looks", "crosses_frame", "crosses_pooled"}
_ABLATION_KEYS_OFF = {"actions", "looks", "crosses_frame"}
# Per-type registry forward call, from the collate triple (images_tight, images_context, motions).
_ABLATION_FORWARD = {
    ModelType.MOTION_ONLY: lambda m, ti, ctx, mo: m(mo, ti),
    ModelType.VISUAL_ONLY: lambda m, ti, ctx, mo: m(ctx),
    ModelType.VANILLA_CONCAT: lambda m, ti, ctx, mo: m(ti, ctx, mo),
}


def _build_ablation(mt: ModelType, entry: dict) -> nn.Module:
    # per-sequence norm: the legacy capture's motion normalization (A4 parity pin)
    return _ABLATION_CLASS[mt].from_config(ModelCfg(motion_norm="per_sequence"), entry["img_size"])


def _rebuild_ablation_with_gate(mt: ModelType, ref: nn.Module, emit: bool) -> nn.Module:
    """Rebuild an ablation sharing ``ref``'s sub-encoders but with ``emit_crosses_pooled=emit``; load ref weights."""
    cfg = ModelCfg()
    kw = {
        "d_model": cfg.d_model,
        "num_classes_dict": dict(cfg.num_classes),
        "dropout": cfg.head_dropout,
        "use_frame_crosses": cfg.use_frame_crosses,
        "frame_pool": cfg.frame_pool,
        "emit_crosses_pooled": emit,
    }
    if mt is ModelType.MOTION_ONLY:
        model = MotionOnlyModel(motion_enc=ref.motion_enc, **kw)
    elif mt is ModelType.VISUAL_ONLY:
        model = VisualOnlyModel(vit=ref.vit, **kw)
    else:
        model = VanillaConcatModel(motion_enc=ref.motion_enc, vit=ref.vit, **kw)
    model.load_state_dict(ref.state_dict(), strict=True)
    return model.eval()


# --------------------------------------------------------------------------- golden parity (per ablation)


@pytest.mark.parametrize("mt", _ABLATION_TYPES, ids=lambda m: m.value)
def test_golden_ablation_parity(ensemble_golden: dict, mt: ModelType) -> None:
    """Each new ablation loads its OLD state_dict (strict=True) and reproduces every golden key (eval)."""
    entry = ensemble_golden[mt.value]
    model = _build_ablation(mt, entry)
    model.load_state_dict(entry["state_dict"], strict=True)
    model.eval()
    inp = entry["inputs"]
    with torch.no_grad():
        out = _ABLATION_FORWARD[mt](model, inp["images_tight"], inp["images_context"], inp["motions"])
    tol = entry["meta"]
    # 3 genuine legacy keys + crosses_pooled (B4, recomputed from legacy weights); no temporal_weights.
    assert set(out) == _ABLATION_KEYS_ON
    for key, expected in entry["outputs"].items():
        torch.testing.assert_close(out[key], expected, atol=tol["atol"], rtol=tol["rtol"])


@pytest.mark.parametrize("mt", _ABLATION_TYPES, ids=lambda m: m.value)
def test_strict_load_ablation_no_missing_unexpected(ensemble_golden: dict, mt: ModelType) -> None:
    """OLD ablation state_dict loads strict with zero missing/unexpected keys and NO forward (eager, 2.1)."""
    entry = ensemble_golden[mt.value]
    model = _build_ablation(mt, entry)  # never forward first
    missing, unexpected = model.load_state_dict(entry["state_dict"], strict=False)
    assert not missing, f"missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert any(k.startswith("classifier.crosses") for k in entry["state_dict"]), (
        "fixture should retain the legacy-dead classifier.crosses param (B4 param-layout parity)"
    )
    model.load_state_dict(entry["state_dict"], strict=True)  # must not raise


# --------------------------------------------------------------------------- B4 gate + output contract


@pytest.mark.parametrize("mt", _ABLATION_TYPES, ids=lambda m: m.value)
def test_ablation_emit_flag_gate(mt: ModelType) -> None:
    """emit_crosses_pooled toggles ONLY crosses_pooled; ablations never emit temporal_weights; keys stable."""
    on = _ABLATION_CLASS[mt].from_config(ModelCfg(), 224).eval()
    off = _rebuild_ablation_with_gate(mt, on, emit=False)  # same weights, gate off
    tight, context, motions = _dummy_full_inputs(ModelCfg())
    fwd = _ABLATION_FORWARD[mt]
    with torch.no_grad():
        out_on = fwd(on, tight, context, motions)
        out_off = fwd(off, tight, context, motions)
    assert set(out_on) == _ABLATION_KEYS_ON
    assert set(out_off) == _ABLATION_KEYS_OFF
    assert "temporal_weights" not in out_on  # structurally full-model-only
    for key in _ABLATION_KEYS_OFF:  # gating crosses_pooled must not perturb the other keys
        torch.testing.assert_close(out_on[key], out_off[key])


# --------------------------------------------------------------------------- consolidated: all four types


@pytest.mark.parametrize("mt", list(ModelType), ids=lambda m: m.value)
def test_all_model_types_build_and_forward(mt: ModelType) -> None:
    """Every model_type builds via the registry and runs forward_model on a dummy batch (per-type contract)."""
    cfg = ModelCfg()
    model = build_model(RootCfg(), mt).eval()
    tight, context, motions = _dummy_full_inputs(cfg)
    with torch.no_grad():
        out = forward_model(model, tight, context, motions)
    expected = _FULL_KEYS if mt is ModelType.FULL else _ABLATION_KEYS_ON
    assert set(out) == expected
    # crosses_frame is the ONLY supervised key — present and finite for all four types.
    assert out["crosses_frame"].shape == (2, cfg.num_classes["crosses"])
    assert torch.isfinite(out["crosses_frame"]).all()
    # temporal_weights is structurally full-model-only.
    assert ("temporal_weights" in out) is (mt is ModelType.FULL)

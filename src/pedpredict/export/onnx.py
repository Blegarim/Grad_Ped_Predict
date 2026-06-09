"""ONNX export + onnxruntime parity check (Prompt 7.1).

Port of OLD ``onnx/onnx_export.py``. Changes over legacy:

* All four model types exported via the typed registry (legacy: full only, hardcoded path).
* Dynamic axes on batch (dim 0) AND seq_len (dim 1) for every temporal input, not batch-only.
* ``check_onnx_parity`` is a first-class public function, not an afterthought.
* All knobs flow from ``ExportCfg``; no hardcoded paths or constants.
* No dummy-forward before export — **B2 resolved** in Prompt 2.1: ``ViT_Hierarchical``
  creates all parameters (including the global-stage relative-position table) in
  ``__init__``, so ``torch.onnx.export`` traces deterministically on the first call.
* ``crosses_pooled`` is NOT exported: live-but-unsupervised per B4/CLAUDE.md; downstream
  consumers must only see the three supervised outputs (actions, looks, crosses_frame).
* ``temporal_weights`` is opt-in via ``ExportCfg.include_temporal_weights`` (full model
  only; default False preserves the legacy 3-key output contract).
"""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager

import torch
import torch.nn as nn

from pedpredict.config import RootCfg
from pedpredict.models.registry import MODEL_INPUT_SIGNATURE, ModelType, ModelTypeLike

__all__ = ["check_onnx_parity", "export_onnx", "get_dynamic_axes"]

_SUPERVISED_KEYS: tuple[str, ...] = ("actions", "looks", "crosses_frame")
_SEQ_INPUT_NAMES: frozenset[str] = frozenset({"images_tight", "images_context", "motions"})


@contextmanager
def _mha_no_fastpath():
    """Disable the nn.MultiheadAttention C++ fast path within the context.

    In eval mode, nn.MultiheadAttention routes through aten::_native_multi_head_attention
    (a fused C++ kernel). The TorchScript ONNX exporter in torch >= 2.12 has no ONNX
    lowering for that op. Disabling the fast path forces F.multi_head_attention_forward,
    which decomposes into linear + bmm + softmax — all fully traceable at opset 17.

    Both export and the parity-check PyTorch forward use this context so that ONNX and
    PyTorch compute through the exact same code path, giving sub-1e-6 absolute differences
    (vs ~5e-3 when the fused kernel mismatches the decomposed ONNX ops).
    """
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)


# ── per-type ONNX wrappers ─────────────────────────────────────────────────────
# Each wrapper carries exactly the positional inputs that model type expects
# (matching MODEL_INPUT_SIGNATURE). ONNX graph outputs cannot be dicts; each
# wrapper unpacks the logits dict to a flat tuple. ``crosses_pooled`` is excluded.


class _FullWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = output_keys

    def forward(
        self,
        images_tight: torch.Tensor,
        images_context: torch.Tensor,
        motions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out: dict[str, torch.Tensor] = self.model(
            images_tight, images_context, motions, return_feats=False
        )
        return tuple(out[k] for k in self.output_keys)


class _VanillaConcatWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = output_keys

    def forward(
        self,
        images_tight: torch.Tensor,
        images_context: torch.Tensor,
        motions: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out: dict[str, torch.Tensor] = self.model(images_tight, images_context, motions)
        return tuple(out[k] for k in self.output_keys)


class _MotionOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = output_keys

    def forward(
        self,
        motions: torch.Tensor,
        images_tight: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out: dict[str, torch.Tensor] = self.model(motions, images_tight)
        return tuple(out[k] for k in self.output_keys)


class _VisualOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = output_keys

    def forward(
        self,
        images_context: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out: dict[str, torch.Tensor] = self.model(images_context)
        return tuple(out[k] for k in self.output_keys)


# ── internal helpers ───────────────────────────────────────────────────────────


def _resolve_output_keys(model_type: ModelType, include_temporal_weights: bool) -> tuple[str, ...]:
    keys = list(_SUPERVISED_KEYS)
    if include_temporal_weights and model_type is ModelType.FULL:
        keys.append("temporal_weights")
    return tuple(keys)


def _make_wrapper(model: nn.Module, model_type: ModelType, output_keys: tuple[str, ...]) -> nn.Module:
    if model_type is ModelType.FULL:
        return _FullWrapper(model, output_keys)
    if model_type is ModelType.VANILLA_CONCAT:
        return _VanillaConcatWrapper(model, output_keys)
    if model_type is ModelType.MOTION_ONLY:
        return _MotionOnlyWrapper(model, output_keys)
    if model_type is ModelType.VISUAL_ONLY:
        return _VisualOnlyWrapper(model, output_keys)
    raise ValueError(f"Unhandled model type: {model_type!r}")  # unreachable; coerce validated


def _make_dummy_inputs(
    model_type: ModelType,
    cfg: RootCfg,
    *,
    batch_size: int | None = None,
    seq_len: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Deterministic dummy tensors for ONNX tracing and parity checks."""
    B = batch_size if batch_size is not None else cfg.export.parity_batch_size
    T = seq_len if seq_len is not None else cfg.export.parity_seq_len
    H, W = cfg.data.img_height, cfg.data.img_width
    Hc, Wc = cfg.data.read_context_height, cfg.data.read_context_width
    mdim = cfg.data.motion_dim
    gen = torch.Generator().manual_seed(seed)
    tight = torch.randn(B, T, 3, H, W, generator=gen)
    context = torch.randn(B, T, 3, Hc, Wc, generator=gen)
    motions = torch.randn(B, T, mdim, generator=gen)
    if model_type in (ModelType.FULL, ModelType.VANILLA_CONCAT):
        return (tight, context, motions)
    if model_type is ModelType.MOTION_ONLY:
        return (motions, tight)
    if model_type is ModelType.VISUAL_ONLY:
        return (context,)
    raise ValueError(f"Unhandled model type: {model_type!r}")  # unreachable


# ── public API ─────────────────────────────────────────────────────────────────


def get_dynamic_axes(
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> dict[str, dict[int, str]]:
    """Build the ``dynamic_axes`` dict for ``torch.onnx.export``.

    Every temporal input (images_tight, images_context, motions) gets both batch (0)
    and seq_len (1) as dynamic axes — an upgrade over the legacy batch-only spec.
    All outputs get batch (0) dynamic; temporal_weights additionally gets seq_len (1).
    """
    axes: dict[str, dict[int, str]] = {}
    for name in input_names:
        axes[name] = {0: "batch"}
        if name in _SEQ_INPUT_NAMES:
            axes[name][1] = "seq_len"
    for name in output_names:
        axes[name] = {0: "batch"}
        if name == "temporal_weights":
            axes[name][1] = "seq_len"
    return axes


def _make_dynamic_shapes(input_names: tuple[str, ...]) -> dict:
    """Build ``dynamic_shapes`` for the dynamo-based ONNX exporter (torch >= 2.0).

    Keys are the wrapper ``forward`` argument names; values mark batch (0) and
    seq_len (1) as dynamic for every temporal input.
    """
    from torch.export import Dim  # noqa: PLC0415

    batch = Dim("batch", min=1)
    seq_len = Dim("seq_len", min=1)
    temporal: dict[str, dict] = {
        "images_tight": {0: batch, 1: seq_len},
        "images_context": {0: batch, 1: seq_len},
        "motions": {0: batch, 1: seq_len},
    }
    return {name: temporal[name] for name in input_names if name in temporal}


def _do_export(
    wrapper: nn.Module,
    dummy: tuple[torch.Tensor, ...],
    out_path: str,
    input_names: tuple[str, ...],
    output_keys: tuple[str, ...],
    opset: int,
) -> None:
    """Run ``torch.onnx.export``, trying the legacy TorchScript exporter first and
    falling back to the dynamo-based exporter only if TorchScript raises an
    unsupported-op error that is NOT solved by ``_mha_no_fastpath``.

    The MHA fast path (``aten::_native_multi_head_attention``) is disabled for the
    TorchScript attempt so the exporter traces ``F.multi_head_attention_forward``
    instead — a pure-Python path with full opset-17 ONNX lowerings.
    """
    # TorchScript path — MHA fast path disabled so _native_multi_head_attention
    # is never reached and the standard F.multi_head_attention_forward is traced.
    try:
        with _mha_no_fastpath(), torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                out_path,
                input_names=list(input_names),
                output_names=list(output_keys),
                dynamic_axes=get_dynamic_axes(input_names, output_keys),
                opset_version=opset,
                dynamo=False,
            )
        return
    except Exception as exc:  # noqa: BLE001
        # Only fall through on missing ONNX symbolic / unsupported op errors.
        # Re-raise everything else (shape errors, type errors, etc.) immediately.
        _msg = str(exc).lower()
        if "unsupportedoperatorerror" not in type(exc).__name__.lower() and "not supported" not in _msg:
            raise

    # Dynamo fallback — last resort for ops that have no TorchScript ONNX lowering
    # even after the MHA fix (e.g. future torch versions dropping additional symbols).
    dyn_shapes = _make_dynamic_shapes(input_names)
    effective_opset = max(opset, 18)  # dynamo exporter requires opset ≥ 18
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            input_names=list(input_names),
            output_names=list(output_keys),
            dynamic_shapes=dyn_shapes,
            opset_version=effective_opset,
            dynamo=True,
        )


def export_onnx(
    model: nn.Module,
    cfg: RootCfg,
    output_path: str | os.PathLike,
    *,
    model_type: ModelTypeLike | None = None,
) -> pathlib.Path:
    """Export ``model`` to ONNX at ``output_path``; return the resolved path.

    ``model_type`` defaults to ``cfg.eval.model_type``. All export knobs come from
    ``cfg.export`` (opset, output_dir, include_temporal_weights).

    B2 note: no dummy-forward is needed before calling this function.
    ``ViT_Hierarchical`` (Prompt 2.1) creates ALL parameters at ``__init__``, so
    ``torch.onnx.export`` traces deterministically on the first call — the OLD
    ``lazy relative_position_bias init → param created mid-trace`` path no longer exists.

    Compatibility: uses the legacy TorchScript exporter (correct for the pinned
    ``torch==2.7.1`` production environment and verified on torch ≥ 2.12). The MHA
    fast path (``aten::_native_multi_head_attention``) is disabled during tracing so
    all torch versions export the same ``F.multi_head_attention_forward`` ops — fully
    lowerable to ONNX opset 17. A dynamo fallback exists for any future op that still
    lacks a TorchScript lowering after the MHA fix.

    Raises ``ImportError`` if the ``onnx`` package is not installed (add ``[export]`` extra).
    """
    import onnx as onnx_lib  # [export] optional dep; lazy-imported so core is usable without it

    mt = ModelType.coerce(cfg.eval.model_type if model_type is None else model_type)
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    output_keys = _resolve_output_keys(mt, cfg.export.include_temporal_weights)
    input_names: tuple[str, ...] = MODEL_INPUT_SIGNATURE[mt]
    wrapper = _make_wrapper(model, mt, output_keys)
    wrapper.eval()

    dummy = _make_dummy_inputs(mt, cfg)
    _do_export(wrapper, dummy, str(out), input_names, output_keys, cfg.export.opset)

    onnx_lib.checker.check_model(str(out))
    return out


def check_onnx_parity(
    model: nn.Module,
    onnx_path: str | os.PathLike,
    cfg: RootCfg,
    *,
    model_type: ModelTypeLike | None = None,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Run PyTorch + onnxruntime on identical dummy inputs; raise if any output drifts.

    Tolerances come from ``cfg.export.parity_atol`` / ``parity_rtol``.
    Dummy input sizes come from ``cfg.export.parity_batch_size`` / ``parity_seq_len``.

    Returns ``{output_name: (max_abs_diff, max_rel_diff)}`` for logging.

    Raises ``ImportError`` if ``onnxruntime`` is not installed (add ``[export]`` extra).
    Raises ``AssertionError`` if any output exceeds the configured tolerances.
    """
    import numpy as np
    import onnxruntime as ort  # [export] optional dep

    mt = ModelType.coerce(cfg.eval.model_type if model_type is None else model_type)
    output_keys = _resolve_output_keys(mt, cfg.export.include_temporal_weights)
    input_names: tuple[str, ...] = MODEL_INPUT_SIGNATURE[mt]

    dummy = _make_dummy_inputs(mt, cfg, seed=seed)

    wrapper = _make_wrapper(model, mt, output_keys)
    wrapper.eval()
    # Disable MHA fast path to match the code path traced during export.
    with _mha_no_fastpath(), torch.no_grad():
        torch_outs: tuple[torch.Tensor, ...] = wrapper(*dummy)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {name: t.numpy() for name, t in zip(input_names, dummy, strict=True)}
    ort_outs: list[np.ndarray] = sess.run(list(output_keys), feed)

    atol, rtol = cfg.export.parity_atol, cfg.export.parity_rtol
    diffs: dict[str, tuple[float, float]] = {}
    for key, torch_t, ort_arr in zip(output_keys, torch_outs, ort_outs, strict=True):
        torch_np = torch_t.numpy()
        abs_diff = float(np.abs(torch_np - ort_arr).max())
        rel_diff = float((np.abs(torch_np - ort_arr) / (np.abs(torch_np) + 1e-8)).max())
        diffs[key] = (abs_diff, rel_diff)
        np.testing.assert_allclose(
            torch_np,
            ort_arr,
            atol=atol,
            rtol=rtol,
            err_msg=f"ONNX parity failed for output '{key}': max_abs={abs_diff:.2e}",
        )
    return diffs

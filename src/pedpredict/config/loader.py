"""Config loader: yaml -> dataclass -> argparse-override merge -> validate -> dump.

Pipeline (precedence: yaml defaults < file load < CLI overrides; ``validate`` runs last):

    load_config(config_dir, overrides) ->
        read configs/*.yaml -> build *Cfg -> apply dotted overrides -> validate -> RootCfg

Override forms accepted: ``section.field=value`` and ``--section.field value``. Container
overrides are parsed with ``yaml.safe_load`` then coerced to the field's declared type.
Dict overrides REPLACE the whole dict (Q1), they do not deep-merge.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from collections.abc import Sequence
from pathlib import Path

import yaml

from ..models.geometry import feature_map_size, is_global
from .schema import (
    AugmentCfg,
    BalanceCfg,
    DataCfg,
    EvalCfg,
    ExportCfg,
    InferenceCfg,
    ModelCfg,
    PathsCfg,
    RootCfg,
    ScheduleCfg,
    TrainCfg,
)

__all__ = [
    "ConfigError",
    "load_config",
    "load_resolved_config",
    "parse_overrides",
    "apply_overrides",
    "validate_config",
    "dump_config",
    "build_argparser",
]

# section name -> (dataclass, yaml filename)
_SECTIONS: dict[str, tuple[type, str]] = {
    "paths": (PathsCfg, "paths.yaml"),
    "data": (DataCfg, "data.yaml"),
    "model": (ModelCfg, "model.yaml"),
    "train": (TrainCfg, "train.yaml"),
    "eval": (EvalCfg, "eval.yaml"),
    "infer": (InferenceCfg, "infer.yaml"),
    "balance": (BalanceCfg, "balance.yaml"),
    "augment": (AugmentCfg, "augment.yaml"),
    "schedule": (ScheduleCfg, "schedule.yaml"),
    "export": (ExportCfg, "export.yaml"),
}

_TASK_KEYS = frozenset({"actions", "looks", "crosses"})
_FRAME_POOLS = frozenset({"logsumexp", "max", "mean"})  # CrossAttentionModule frame-pool modes (2.3)


class ConfigError(ValueError):
    """Raised on unknown keys, type-coercion failures, or validation violations."""


# --------------------------------------------------------------------------- coercion


def _is_union(origin: object) -> bool:
    return origin is typing.Union or origin is getattr(types, "UnionType", object())


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Cannot parse boolean from {value!r}")


def _coerce(declared: object, value: object) -> object:
    """Coerce ``value`` to the field's declared type. Strings (from CLI) are yaml-parsed first."""
    origin = typing.get_origin(declared)

    if _is_union(declared) or _is_union(origin):
        members = [a for a in typing.get_args(declared) if a is not type(None)]
        if value is None:
            return None
        for member in members:
            try:
                return _coerce(member, value)
            except (TypeError, ValueError):
                continue
        raise ConfigError(f"Value {value!r} does not match {declared!r}")

    if origin is tuple:
        seq = yaml.safe_load(value) if isinstance(value, str) else value
        if not isinstance(seq, list | tuple):
            raise ConfigError(f"Expected a sequence for {declared!r}, got {value!r}")
        args = typing.get_args(declared)
        if len(args) == 2 and args[1] is Ellipsis:          # tuple[X, ...]
            return tuple(_coerce(args[0], x) for x in seq)
        return tuple(_coerce(t, x) for t, x in zip(args, seq, strict=False))  # fixed-length

    if origin is dict:
        mapping = yaml.safe_load(value) if isinstance(value, str) else value
        if not isinstance(mapping, dict):
            raise ConfigError(f"Expected a mapping for {declared!r}, got {value!r}")
        key_t, val_t = typing.get_args(declared) or (str, object)
        return {_coerce(key_t, k): _coerce(val_t, v) for k, v in mapping.items()}

    if declared is bool:
        return _to_bool(value)
    if declared is int:
        return int(value)
    if declared is float:
        return float(value)
    if declared is str:
        return str(value)
    # Nested frozen dataclass (e.g. PhaseCfg inside tuple[PhaseCfg, ...])
    if isinstance(declared, type) and dataclasses.is_dataclass(declared):
        if isinstance(value, declared):
            return value
        if isinstance(value, dict):
            return _build_section(declared, value)
        raise ConfigError(f"Expected a mapping for {declared.__name__}; got {value!r}")
    return value


def _build_section(cls: type, data: dict) -> object:
    """Instantiate one section dataclass from a yaml mapping, coercing + rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict[str, object] = {}
    for key, raw in (data or {}).items():
        if key not in known:
            raise ConfigError(f"Unknown field '{key}' in section '{cls.__name__}'")
        kwargs[key] = _coerce(hints[key], raw)
    return cls(**kwargs)


def _build_root(nested: dict) -> RootCfg:
    sections = {name: _build_section(cls, nested.get(name, {})) for name, (cls, _) in _SECTIONS.items()}
    return RootCfg(**sections)


# --------------------------------------------------------------------------- overrides


def parse_overrides(tokens: Sequence[str]) -> dict[str, str]:
    """Accept ``section.field=value`` and ``--section.field value`` -> ``{"section.field": "value"}``."""
    flat: dict[str, str] = {}
    items = list(tokens)
    i = 0
    while i < len(items):
        token = items[i]
        if token.startswith("--"):
            token = token[2:]
        if "=" in token:
            key, val = token.split("=", 1)
            flat[key.strip()] = val
            i += 1
        else:
            if i + 1 >= len(items):
                raise ConfigError(f"Override '{token}' is missing a value")
            flat[token.strip()] = items[i + 1]
            i += 2
    return flat


def apply_overrides(root: RootCfg, flat: dict[str, str]) -> RootCfg:
    """Apply a flat ``{'section.field': value}`` map, rebuilding the frozen tree immutably."""
    sections = {name: getattr(root, name) for name in _SECTIONS}
    for dotted, raw in flat.items():
        parts = dotted.split(".")
        if len(parts) != 2:
            raise ConfigError(f"Override key '{dotted}' must have the form 'section.field'")
        section, field_name = parts
        if section not in sections:
            raise ConfigError(f"Unknown config section '{section}' in override '{dotted}'")
        cfg = sections[section]
        hints = typing.get_type_hints(type(cfg))
        if field_name not in hints:
            raise ConfigError(f"Unknown field '{field_name}' in section '{section}'")
        coerced = _coerce(hints[field_name], raw)
        sections[section] = dataclasses.replace(cfg, **{field_name: coerced})
    return dataclasses.replace(root, **sections)


# --------------------------------------------------------------------------- validation


def validate_config(root: RootCfg) -> None:
    """Structural invariants. Raises ``ConfigError`` on violation (no silent passthrough)."""
    m, d, t, e, b, a = root.model, root.data, root.train, root.eval, root.balance, root.augment

    lengths = {
        "stage_dims": len(m.stage_dims),
        "layer_nums": len(m.layer_nums),
        "head_nums": len(m.head_nums),
        "window_size": len(m.window_size),
        "mlp_ratio": len(m.mlp_ratio),
    }
    if len(set(lengths.values())) != 1:
        raise ConfigError(f"ViT stage lists must share one length; got {lengths}")

    for i, (dim, heads) in enumerate(zip(m.stage_dims, m.head_nums, strict=False)):
        if heads <= 0 or dim % heads != 0:
            raise ConfigError(f"stage_dims[{i}]={dim} not divisible by head_nums[{i}]={heads}")

    if m.motion_num_heads <= 0 or m.motion_hidden_dim % m.motion_num_heads != 0:
        raise ConfigError(
            f"motion_hidden_dim={m.motion_hidden_dim} not divisible by motion_num_heads={m.motion_num_heads}"
        )

    # CrossAttentionModule: MultiheadAttention requires d_model divisible by num_heads.
    if m.cross_attn_num_heads <= 0 or m.d_model % m.cross_attn_num_heads != 0:
        raise ConfigError(
            f"model.d_model={m.d_model} not divisible by model.cross_attn_num_heads={m.cross_attn_num_heads}"
        )
    if m.frame_pool not in _FRAME_POOLS:
        raise ConfigError(f"model.frame_pool must be one of {sorted(_FRAME_POOLS)}; got {m.frame_pool!r}")

    if d.motion_dim != m.motion_dim:  # B7: writer-channel / model-input agreement
        raise ConfigError(f"data.motion_dim ({d.motion_dim}) != model.motion_dim ({m.motion_dim})")

    if set(m.num_classes) != _TASK_KEYS:
        raise ConfigError(f"model.num_classes keys must be {sorted(_TASK_KEYS)}; got {sorted(m.num_classes)}")

    positives = {
        "model.d_model": m.d_model,
        "data.max_seq_len": d.max_seq_len,
        "data.motion_dim": d.motion_dim,
        "data.chunk_size": d.chunk_size,
        "data.seq_len": d.seq_len,
        "data.stride": d.stride,
        "data.min_track_size": d.min_track_size,
        "train.batch_size": t.batch_size,
        "train.num_epochs": t.num_epochs,
        "eval.batch_size": e.batch_size,
    }
    for name, value in positives.items():
        if value <= 0:
            raise ConfigError(f"{name} must be a positive integer; got {value}")

    # writer geometry/encode invariants
    if d.context_scale <= 0.0:
        raise ConfigError(f"data.context_scale must be > 0; got {d.context_scale}")
    if not (1 <= d.jpeg_quality <= 100):
        raise ConfigError(f"data.jpeg_quality must be in [1, 100]; got {d.jpeg_quality}")
    if d.img_height <= 0 or d.img_width <= 0:
        raise ConfigError(f"data.img_height/img_width must be positive; got {d.img_height}x{d.img_width}")
    if d.read_context_height <= 0 or d.read_context_width <= 0:  # read-time context model input (1.5)
        raise ConfigError(
            f"data.read_context_height/width must be positive; "
            f"got {d.read_context_height}x{d.read_context_width}"
        )

    # ViT window tiling: the context-crop resolution must tile every stage's window so the
    # eager relative-position tables (B2) are buildable. The ViT is built from a scalar img_size, so the
    # context crop must be square; global windows (None) always tile (window == feature map).
    if d.read_context_height != d.read_context_width:
        raise ConfigError(
            f"ViT requires a square context crop; got "
            f"{d.read_context_height}x{d.read_context_width} (data.read_context_height/width)"
        )
    for i, win in enumerate(m.window_size):
        side = feature_map_size(d.read_context_height, i)
        if side <= 0:
            raise ConfigError(f"ViT stage {i} feature map collapses to {side} at img_size={d.read_context_height}")
        if not is_global(win) and side % int(win) != 0:
            raise ConfigError(
                f"ViT stage {i}: feature map {side} not divisible by window {win} "
                f"(img_size={d.read_context_height})"
            )

    # training-loop invariants
    if t.grad_clip_max_norm <= 0.0:
        raise ConfigError(f"train.grad_clip_max_norm must be > 0; got {t.grad_clip_max_norm}")

    # chunk prefetch loader invariants
    if t.chunk_preload_depth < 1:
        raise ConfigError(f"train.chunk_preload_depth must be >= 1; got {t.chunk_preload_depth}")
    if not (0.0 < t.chunk_warm_ram_threshold <= 100.0):
        raise ConfigError(
            f"train.chunk_warm_ram_threshold must be in (0, 100]; got {t.chunk_warm_ram_threshold}"
        )
    if t.chunk_warm_mem_interval <= 0.0:
        raise ConfigError(f"train.chunk_warm_mem_interval must be > 0; got {t.chunk_warm_mem_interval}")
    if t.chunk_warm_mem_timeout is not None and t.chunk_warm_mem_timeout <= 0.0:
        raise ConfigError(
            f"train.chunk_warm_mem_timeout must be > 0 or null; got {t.chunk_warm_mem_timeout}"
        )
    if t.chunk_queue_timeout <= 0.0:
        raise ConfigError(f"train.chunk_queue_timeout must be > 0; got {t.chunk_queue_timeout}")
    if t.dataloader_prefetch_factor < 1:
        raise ConfigError(
            f"train.dataloader_prefetch_factor must be >= 1; got {t.dataloader_prefetch_factor}"
        )

    # online sampler invariants
    if t.sampler_min_weight <= 0.0:
        raise ConfigError(f"train.sampler_min_weight must be > 0; got {t.sampler_min_weight}")
    if set(t.sampler_powers) != _TASK_KEYS:
        raise ConfigError(
            f"train.sampler_powers keys must be {sorted(_TASK_KEYS)}; got {sorted(t.sampler_powers)}"
        )
    for task, power in t.sampler_powers.items():
        if power < 0.0:
            raise ConfigError(f"train.sampler_powers[{task}] must be >= 0; got {power}")

    if not (0.0 <= e.threshold_sweep_lo < e.threshold_sweep_hi <= 1.0):
        raise ConfigError(
            f"require 0 <= threshold_sweep_lo < threshold_sweep_hi <= 1; "
            f"got lo={e.threshold_sweep_lo}, hi={e.threshold_sweep_hi}"
        )

    # video-inference invariants
    inf = root.infer
    if not (0.0 <= inf.detector_conf <= 1.0):
        raise ConfigError(f"infer.detector_conf must be in [0, 1]; got {inf.detector_conf}")
    if inf.detector_class_idx < 0:
        raise ConfigError(f"infer.detector_class_idx must be >= 0; got {inf.detector_class_idx}")
    if inf.smooth_window < 0:
        raise ConfigError(f"infer.smooth_window must be >= 0; got {inf.smooth_window}")
    if inf.window_stride < 1:
        raise ConfigError(f"infer.window_stride must be >= 1; got {inf.window_stride}")
    if inf.batch_size <= 0:
        raise ConfigError(f"infer.batch_size must be a positive integer; got {inf.batch_size}")
    if inf.default_fps <= 0.0:
        raise ConfigError(f"infer.default_fps must be > 0; got {inf.default_fps}")

    # offline balance invariants
    if not (0.0 < b.cross_pos_ratio < 1.0):
        raise ConfigError(f"balance.cross_pos_ratio must be in (0, 1); got {b.cross_pos_ratio}")
    for name, rate in {"target_action_rate": b.target_action_rate, "target_look_rate": b.target_look_rate}.items():
        if not (0.0 <= rate <= 1.0):
            raise ConfigError(f"balance.{name} must be in [0, 1]; got {rate}")
    if b.x11_select not in {"lower", "upper"}:
        raise ConfigError(f"balance.x11_select must be 'lower' or 'upper'; got {b.x11_select!r}")
    if b.on_infeasible not in {"raise", "empty"}:
        raise ConfigError(f"balance.on_infeasible must be 'raise' or 'empty'; got {b.on_infeasible!r}")

    # offline augmentation invariants
    for name, prob in {"p_flip": a.p_flip, "p_color": a.p_color, "p_noise": a.p_noise, "p_erase": a.p_erase}.items():
        if not (0.0 <= prob <= 1.0):
            raise ConfigError(f"augment.{name} must be in [0, 1]; got {prob}")
    if not (1 <= a.n_augs_min <= a.n_augs_max <= 4):
        raise ConfigError(f"require 1 <= augment.n_augs_min <= n_augs_max <= 4; got {a.n_augs_min}, {a.n_augs_max}")
    if a.crosses_multiplier < 1 or a.looks_multiplier < 1:
        raise ConfigError(
            f"augment.crosses_multiplier/looks_multiplier must be >= 1; "
            f"got {a.crosses_multiplier}, {a.looks_multiplier}"
        )
    if a.motion_noise_std < 0.0:
        raise ConfigError(f"augment.motion_noise_std must be >= 0; got {a.motion_noise_std}")
    if a.erase_n_frames < 0:
        raise ConfigError(f"augment.erase_n_frames must be >= 0; got {a.erase_n_frames}")

    # ONNX export invariants
    ex = root.export
    if ex.opset < 1:
        raise ConfigError(f"export.opset must be >= 1; got {ex.opset}")
    if ex.parity_atol <= 0.0:
        raise ConfigError(f"export.parity_atol must be > 0; got {ex.parity_atol}")
    if ex.parity_rtol < 0.0:
        raise ConfigError(f"export.parity_rtol must be >= 0; got {ex.parity_rtol}")
    if ex.parity_batch_size < 1:
        raise ConfigError(f"export.parity_batch_size must be >= 1; got {ex.parity_batch_size}")
    if ex.parity_seq_len < 1:
        raise ConfigError(f"export.parity_seq_len must be >= 1; got {ex.parity_seq_len}")

    # phase schedule
    for i, phase in enumerate(root.schedule.phases):
        if not phase.name:
            raise ConfigError(f"schedule.phases[{i}].name must be non-empty")
        if not phase.data_source:
            raise ConfigError(f"schedule.phases[{i}].data_source must be non-empty")
        if phase.lr <= 0.0:
            raise ConfigError(f"schedule.phases[{i}].lr must be > 0; got {phase.lr}")
        if phase.max_epochs <= 0:
            raise ConfigError(f"schedule.phases[{i}].max_epochs must be > 0; got {phase.max_epochs}")
        if phase.early_stop_patience <= 0:
            raise ConfigError(
                f"schedule.phases[{i}].early_stop_patience must be > 0; got {phase.early_stop_patience}"
            )


# --------------------------------------------------------------------------- public load / dump


def load_config(
    config_dir: str | Path = "configs",
    overrides: Sequence[str] | None = None,
    *,
    validate: bool = True,
) -> RootCfg:
    """Read ``configs/*.yaml``, apply CLI overrides, validate, return the resolved tree."""
    base = Path(config_dir)
    nested: dict[str, dict] = {}
    for name, (_, filename) in _SECTIONS.items():
        path = base / filename
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                nested[name] = yaml.safe_load(handle) or {}
        else:
            nested[name] = {}  # fall back to dataclass defaults for a missing section file
    root = _build_root(nested)
    if overrides:
        root = apply_overrides(root, parse_overrides(overrides))
    if validate:
        validate_config(root)
    return root


def load_resolved_config(path: str | Path, *, validate: bool = False) -> RootCfg:
    """Rebuild a ``RootCfg`` from a single nested yaml (e.g. a ``dump_config`` artifact)."""
    with open(path, encoding="utf-8") as handle:
        nested = yaml.safe_load(handle) or {}
    root = _build_root(nested)
    if validate:
        validate_config(root)
    return root


def _to_plain(obj: object) -> object:
    """Dataclass tree -> json/yaml-safe primitives (tuples -> lists)."""
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def dump_config(root: RootCfg, out_dir: str | Path) -> Path:
    """Write the resolved config to ``<out_dir>/resolved_config.yaml`` for reproducibility."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_plain(root), handle, sort_keys=False, default_flow_style=False)
    return path


def build_argparser(existing: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add ``--config-dir`` and a repeatable ``--set section.field=value`` override channel.

    A dedicated ``--set`` channel (rather than argparse.REMAINDER) keeps overrides from
    swallowing real subcommand flags.
    """
    parser = existing or argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs", help="Directory holding the *.yaml config files.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.field=value",
        help="Override a config value (repeatable).",
    )
    return parser

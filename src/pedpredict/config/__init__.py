"""Typed config: yaml -> dataclass schema -> argparse-override loader (Prompt 0.2)."""

from .loader import (
    ConfigError,
    apply_overrides,
    build_argparser,
    dump_config,
    load_config,
    load_resolved_config,
    parse_overrides,
    validate_config,
)
from .schema import DataCfg, EvalCfg, ModelCfg, PathsCfg, RootCfg, TrainCfg

__all__ = [
    "ConfigError",
    "DataCfg",
    "EvalCfg",
    "ModelCfg",
    "PathsCfg",
    "RootCfg",
    "TrainCfg",
    "apply_overrides",
    "build_argparser",
    "dump_config",
    "load_config",
    "load_resolved_config",
    "parse_overrides",
    "validate_config",
]

"""Prompt 0.2 — config system tests.

Two kinds of checks:
  * PARITY: ModelCfg adapters reproduce OLD config.py dicts byte-for-byte
    (golden fixture tests/fixtures/golden/legacy_config.json, captured from the OLD repo).
  * MECHANICS: yaml<->dataclass equality, override coercion, validation, dump roundtrip.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pedpredict.config import (
    ConfigError,
    ModelCfg,
    RootCfg,
    apply_overrides,
    build_argparser,
    dump_config,
    load_config,
    load_resolved_config,
    parse_overrides,
    validate_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "configs"
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "legacy_config.json"


@pytest.fixture(scope="module")
def legacy() -> dict:
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- parity


def test_vit_kwargs_parity(legacy: dict) -> None:
    assert ModelCfg().vit_kwargs() == legacy["vit"]


def test_motion_kwargs_parity(legacy: dict) -> None:
    assert ModelCfg().motion_kwargs() == legacy["motion"]


def test_d_model_parity(legacy: dict) -> None:
    assert ModelCfg().d_model == legacy["d_model"] == 128


# --------------------------------------------------------------------------- yaml <-> dataclass


def test_yaml_defaults_match_dataclass() -> None:
    """Loading configs/*.yaml with no overrides reproduces the dataclass defaults exactly."""
    assert load_config(_CONFIG_DIR) == RootCfg()


def test_default_config_is_valid() -> None:
    validate_config(load_config(_CONFIG_DIR))  # must not raise


# --------------------------------------------------------------------------- overrides


def test_override_scalar() -> None:
    cfg = load_config(_CONFIG_DIR, overrides=["train.lr=5e-5"])
    assert cfg.train.lr == 5e-5
    assert isinstance(cfg.train.lr, float)


def test_override_dashed_form() -> None:
    cfg = load_config(_CONFIG_DIR, overrides=["--train.batch_size", "8"])
    assert cfg.train.batch_size == 8
    assert isinstance(cfg.train.batch_size, int)


def test_override_container() -> None:
    cfg = load_config(_CONFIG_DIR, overrides=["model.stage_dims=[48,96,168,96]"], validate=False)
    assert cfg.model.stage_dims == (48, 96, 168, 96)
    assert isinstance(cfg.model.stage_dims, tuple)


def test_override_window_size_with_null() -> None:
    cfg = load_config(_CONFIG_DIR, overrides=["model.window_size=[8,4,null]"], validate=False)
    assert cfg.model.window_size == (8, 4, None)


def test_override_dict_replaces_not_merges() -> None:
    """Q1: a dict override REPLACES the whole dict (no deep-merge)."""
    cfg = load_config(_CONFIG_DIR, overrides=["train.loss_weight={crosses: 2.0}"])
    assert cfg.train.loss_weight == {"crosses": 2.0}


def test_override_bool() -> None:
    cfg = load_config(_CONFIG_DIR, overrides=["train.use_amp=false"])
    assert cfg.train.use_amp is False


def test_parse_overrides_forms() -> None:
    flat = parse_overrides(["train.lr=5e-5", "--model.d_model", "128"])
    assert flat == {"train.lr": "5e-5", "model.d_model": "128"}


def test_unknown_field_override_raises() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["train.lrr=1"])


def test_unknown_section_override_raises() -> None:
    with pytest.raises(ConfigError):
        apply_overrides(RootCfg(), {"nope.field": "1"})


# --------------------------------------------------------------------------- validation


def test_validation_head_divisibility() -> None:
    """stage_dims[i] must be divisible by head_nums[i] (36 % 5 != 0)."""
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["model.head_nums=[5,2,16,2]"])


def test_validation_motion_head_divisibility() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["model.motion_num_heads=5"])


def test_validation_stage_list_lengths() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["model.mlp_ratio=[4,4,4]"])


def test_validation_motion_dim_consistency() -> None:
    """B7: data.motion_dim and model.motion_dim must agree."""
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["data.motion_dim=7"])


def test_validation_num_classes_keys() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["model.num_classes={actions: 2, looks: 2}"])


def test_validation_vit_window_tiling() -> None:
    """Prompt 2.1: a context resolution that doesn't tile a stage window is rejected.

    225 -> stem 57 (57 % 8 != 0 at stage 0) is not tileable by window 8.
    """
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["data.read_context_height=225", "data.read_context_width=225"])


def test_validation_vit_requires_square_context() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["data.read_context_width=256"])


def test_validation_threshold_sweep_order() -> None:
    with pytest.raises(ConfigError):
        load_config(_CONFIG_DIR, overrides=["eval.threshold_sweep_lo=0.95"])


# --------------------------------------------------------------------------- dump / immutability / argparse


def test_dump_roundtrip(tmp_path: Path) -> None:
    root = load_config(_CONFIG_DIR)
    out = dump_config(root, tmp_path)
    assert out.name == "resolved_config.yaml"
    assert load_resolved_config(out) == root


def test_frozen_immutable() -> None:
    cfg = ModelCfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.d_model = 256  # type: ignore[misc]


def test_argparser_set_channel() -> None:
    parser = build_argparser()
    ns = parser.parse_args(["--set", "train.lr=5e-5", "--set", "train.batch_size=8"])
    assert ns.overrides == ["train.lr=5e-5", "train.batch_size=8"]
    cfg = load_config(_CONFIG_DIR, overrides=ns.overrides, validate=False)
    assert cfg.train.lr == 5e-5
    assert cfg.train.batch_size == 8

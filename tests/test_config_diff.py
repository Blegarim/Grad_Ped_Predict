"""Config diff — the campaign's comparability check (an accidental difference invalidates a comparison)."""

from __future__ import annotations

from pathlib import Path

from pedpredict.config import load_config
from pedpredict.config.diff import config_diff, unexpected_diffs
from pedpredict.config.loader import dump_config

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_identical_configs_have_no_diff() -> None:
    assert config_diff(load_config(_CONFIG_DIR), load_config(_CONFIG_DIR)) == {}


def test_intended_flags_are_the_only_diff() -> None:
    ref = load_config(_CONFIG_DIR, ["train.seed=42"])
    cand = load_config(_CONFIG_DIR, ["train.seed=43", "model.onset_head=true", "train.onset_hazard_weight=0.1"])
    diff = config_diff(ref, cand)
    assert diff["train.seed"] == (42, 43)
    assert set(diff) == {"train.seed", "model.onset_head", "train.onset_hazard_weight"}
    assert unexpected_diffs(diff, ["train.seed", "model.onset_", "train.onset_"]) == []
    assert unexpected_diffs(diff, ["train.seed"]) == ["model.onset_head", "train.onset_hazard_weight"]


def test_result_neutral_keys_are_ignored_but_model_type_is_not() -> None:
    ref = load_config(_CONFIG_DIR)
    cand = load_config(_CONFIG_DIR, ["train.num_workers=2", "eval.num_workers=1", "eval.threshold_sweep_lo=0.2"])
    assert config_diff(ref, cand) == {}
    cand = load_config(_CONFIG_DIR, ["eval.model_type=visual_only"])
    assert set(config_diff(ref, cand)) == {"eval.model_type"}


def test_an_accidental_difference_is_caught() -> None:
    ref = load_config(_CONFIG_DIR, ["train.seed=42"])
    cand = load_config(_CONFIG_DIR, ["train.seed=43", "train.use_weighted_sampler=false"])
    assert unexpected_diffs(config_diff(ref, cand), ["train.seed"]) == ["train.use_weighted_sampler"]


def test_cli_against_a_dumped_snapshot(tmp_path: Path) -> None:
    from scripts.check_run_config import main

    ref = dump_config(load_config(_CONFIG_DIR, ["train.seed=42"]), tmp_path)
    base = ["--reference", str(ref), "--config-dir", str(_CONFIG_DIR)]
    assert main([*base, "--allow", "train.seed", "--set", "train.seed=44"]) == 0
    assert main([*base, "--set", "train.seed=44"]) == 1
    assert main([*base, "--set", "train.lr=0.5", "--set", "train.batch_size=-3"]) in (1, 2)

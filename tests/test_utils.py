"""Tests for the shared utils + paths helpers (Prompt 0.3).

CPU-only and CI-safe. These are infrastructure modules, so there is no saved golden tensor
(MIGRATION row 0.3: fixture = n/a). The one real behavior-parity surface is ``to_float_logits``
(band-aid B8); everything else asserts side-effects / semantics. Tests import the *installed*
``pedpredict`` package (src-layout editable install), matching ``test_smoke.py``.
"""

from __future__ import annotations

import csv
import re

import torch

from pedpredict.config.schema import PathsCfg
from pedpredict.paths import find_project_root, resolve_paths
from pedpredict.utils import (
    CsvLogger,
    create_run_dir,
    get_device,
    make_run_id,
    resolve_amp,
    set_seed,
    to_float_logits,
    wait_for_memory,
)
from pedpredict.utils import device as device_mod
from pedpredict.utils import memory as memory_mod

# --------------------------------------------------------------------------- amp / B8


def test_to_float_logits_upcasts_floats() -> None:
    """fp16 floats -> fp32; int/bool tensors and non-tensors untouched; input not mutated."""
    src = {
        "actions": torch.randn(2, 2, dtype=torch.float16),
        "crosses_frame": torch.randn(2, 2, dtype=torch.float16),
        "labels": torch.zeros(2, dtype=torch.long),
        "mask": torch.ones(2, dtype=torch.bool),
        "model_type": "full",
    }
    out = to_float_logits(src)

    assert out["actions"].dtype == torch.float32
    assert out["crosses_frame"].dtype == torch.float32
    assert out["labels"].dtype == torch.long          # integer tensor passes through
    assert out["mask"].dtype == torch.bool            # bool tensor passes through
    assert out["model_type"] == "full"                # non-tensor passes through
    assert src["actions"].dtype == torch.float16      # original dict not mutated


def test_to_float_logits_matches_legacy_cast() -> None:
    """Each upcast value equals the OLD per-key ``logits.float()`` and is idempotent on fp32."""
    half = {"crosses_frame": torch.randn(3, 2, dtype=torch.float16)}
    out = to_float_logits(half)
    torch.testing.assert_close(out["crosses_frame"], half["crosses_frame"].float())

    again = to_float_logits(out)                       # idempotent: fp32 -> fp32, same values
    torch.testing.assert_close(again["crosses_frame"], out["crosses_frame"])


def test_resolve_amp_gating() -> None:
    """AMP only resolves true on CUDA, regardless of the request (Q2)."""
    cpu = torch.device("cpu")
    assert resolve_amp(True, cpu) is False
    assert resolve_amp(False, cpu) is False
    assert resolve_amp(False, torch.device("cuda")) is False


# --------------------------------------------------------------------------- device


def test_get_device_cpu_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"
    assert get_device(prefer_cuda=False).type == "cpu"


def test_enable_perf_flags_cpu_noop() -> None:
    """CPU device must not raise and must not flip cudnn.benchmark on."""
    before = torch.backends.cudnn.benchmark
    device_mod.enable_perf_flags(torch.device("cpu"))
    assert torch.backends.cudnn.benchmark == before


# --------------------------------------------------------------------------- seed


def test_set_seed_reproducible() -> None:
    set_seed(0)
    a = torch.rand(4)
    set_seed(0)
    b = torch.rand(4)
    torch.testing.assert_close(a, b)


def test_set_seed_deterministic_flags() -> None:
    try:
        set_seed(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        # restore defaults so later tests / other suites aren't forced deterministic
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


# --------------------------------------------------------------------------- memory


def test_wait_for_memory_returns_immediately() -> None:
    # threshold above any possible RAM% -> loop body never entered, returns at once
    wait_for_memory(threshold=100.0, interval=0.01)


def test_wait_for_memory_times_out(monkeypatch) -> None:
    """High RAM + finite timeout returns instead of hanging forever."""

    class _FakeVM:
        percent = 99.9

    monkeypatch.setattr(memory_mod.psutil, "virtual_memory", lambda: _FakeVM())
    wait_for_memory(threshold=50.0, interval=0.01, timeout=0.05)  # must return


# --------------------------------------------------------------------------- logging


def test_make_run_id_format() -> None:
    rid = make_run_id("full", "lrsched")
    assert re.fullmatch(r"\d{8}_\d{6}_full_lrsched", rid)
    # empty tag -> no trailing underscore
    assert re.fullmatch(r"\d{8}_\d{6}_motion_only", make_run_id("motion_only"))
    # unsafe tag characters collapse to underscores
    assert " " not in make_run_id("full", "two phase/run")
    assert "/" not in make_run_id("full", "two phase/run")


def test_create_run_dir_layout(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path / "runs", "20260603_010203_full_tag")
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "plots").is_dir()


def test_csv_logger_roundtrip(tmp_path) -> None:
    path = tmp_path / "train_log.csv"
    fields = ["epoch", "loss"]

    with CsvLogger(path, fields) as logger:
        logger.log({"epoch": 1, "loss": 0.5})
        logger.log({"epoch": 2, "loss": 0.25})

    # re-open existing file: append a row, header must NOT be duplicated
    with CsvLogger(path, fields) as logger:
        logger.log({"epoch": 3, "loss": 0.1})

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == fields                       # single header
    assert rows.count(fields) == 1
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]


# --------------------------------------------------------------------------- paths


def test_find_project_root_has_pyproject() -> None:
    root = find_project_root()
    assert (root / "pyproject.toml").is_file()


def test_resolve_paths_roots_relative_and_keeps_absolute(tmp_path) -> None:
    abs_test = (tmp_path / "abs_test").resolve()
    cfg = PathsCfg(lmdb_test=str(abs_test))
    resolved = resolve_paths(cfg, root=tmp_path)

    assert resolved.root == tmp_path.resolve()
    assert isinstance(resolved.lmdb_train, tuple)
    assert resolved.lmdb_train[0] == tmp_path.resolve() / "preprocessed_train"
    assert resolved.runs_dir == tmp_path.resolve() / "outputs" / "runs"
    assert resolved.lmdb_test == abs_test           # absolute entry passes through unchanged

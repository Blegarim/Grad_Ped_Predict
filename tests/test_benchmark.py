"""Prompt 5.2 — efficiency benchmark (params/FLOPs/latency/FPS/peak VRAM).

No golden fixture: these are wall-clock/structural metrics, not model math. Tests run on CPU at a small
resolution (overridden config) so the warmup+timed forwards stay fast; they assert the methodology wiring,
the output schema, and finiteness. FLOPs is ``nan`` when fvcore is absent (an accepted optional dep).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import torch

from pedpredict.config import DataCfg, EvalCfg, RootCfg
from pedpredict.eval.benchmark import (
    BENCHMARK_COLUMNS,
    EFFICIENCY_KEYS,
    benchmark_model,
    count_params,
    measure_efficiency,
    run_benchmark,
)
from pedpredict.models.registry import build_model

_CPU = torch.device("cpu")


def _fast_cfg() -> RootCfg:
    """Small resolution + 1-warmup/2-trial timing so a CPU benchmark forward is cheap."""
    return dataclasses.replace(
        RootCfg(),
        data=dataclasses.replace(
            DataCfg(), img_height=64, img_width=64, read_context_height=64, read_context_width=64,
            max_seq_len=2,
        ),
        eval=dataclasses.replace(EvalCfg(), bench_batch_size=1, bench_warmup=1, latency_trials=2),
    )


def test_count_params_matches_manual() -> None:
    cfg = _fast_cfg()
    model = build_model(cfg, "motion_only")
    assert count_params(model) == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_benchmark_result_finite_on_cpu() -> None:
    result = benchmark_model(_fast_cfg(), "motion_only", device=_CPU)
    assert result.model_type == "motion_only"
    assert result.params > 0
    assert result.latency_ms_per_frame > 0 and math.isfinite(result.latency_ms_per_frame)
    assert result.fps > 0 and math.isfinite(result.fps)
    assert result.peak_vram_mb == 0.0                       # CPU -> no VRAM
    # flops is finite (fvcore present) or nan (absent) — both acceptable.
    assert result.flops_per_frame > 0 or math.isnan(result.flops_per_frame)


def test_efficiency_and_row_schemas() -> None:
    result = benchmark_model(_fast_cfg(), "visual_only", device=_CPU)
    assert tuple(result.efficiency().keys()) == EFFICIENCY_KEYS
    assert tuple(result.as_row().keys()) == BENCHMARK_COLUMNS


def test_measure_efficiency_matches_eval_columns() -> None:
    # The keys handed to evaluate's eval-log row must equal its efficiency columns (5.1 contract).
    from pedpredict.eval.evaluate import _EFFICIENCY_COLUMNS

    eff = measure_efficiency(_fast_cfg(), device=_CPU)
    assert tuple(eff.keys()) == _EFFICIENCY_COLUMNS


def test_run_benchmark_writes_csv(tmp_path: Path) -> None:
    cfg = _fast_cfg()
    csv_path = tmp_path / "benchmark.csv"
    results = run_benchmark(cfg, ["motion_only", "visual_only"], device=_CPU, csv_path=csv_path)

    assert [r.model_type for r in results] == ["motion_only", "visual_only"]
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == ",".join(BENCHMARK_COLUMNS)           # header == schema
    assert len(rows) == 3                                   # header + 2 model rows

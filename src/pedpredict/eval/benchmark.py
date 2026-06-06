"""Efficiency benchmarking (Prompt 5.2): params, FLOPs, latency, FPS, peak VRAM per model_type.

Consolidates the OLD efficiency code that lived inline in ``test.py`` (``compute_flops`` /
``inference_latency``) and the ``Vision_Transformer.__main__`` fvcore snippet — there was no shared
benchmark module. Models are built + dispatched through the typed registry (2.4), so all four model types
are benchmarked identically.

Methodology (all from config — no literals):

* **Input shapes** come from ``DataCfg`` (the REAL inference resolution): tight ``img_height×img_width``,
  context ``read_context_height×read_context_width``, ``T = max_seq_len``, ``motion_dim`` channels. The
  rebuilt ViT is resolution-bound to ``read_context_height`` (B2, 2.1), so — unlike the OLD lazy ViT that
  was fed a synthetic 384-px context — we benchmark at the resolution the model actually runs at.
* **params**: trainable parameter count (OLD ``test.py:521`` ``requires_grad`` sum).
* **FLOPs**: ``fvcore.FlopCountAnalysis`` total / ``T`` -> per-frame FLOPs (OLD ``compute_flops``). fvcore is
  optional at runtime: if unavailable the value is ``nan`` (params/latency still report).
* **latency / FPS**: ``EvalCfg.bench_warmup`` warmup iters, then ``EvalCfg.latency_trials`` timed iters with
  CUDA sync around the timed region (OLD ``inference_latency``). ``fps = 1 / avg_seq_latency``;
  ``latency_ms_per_frame = avg_seq_latency / T * 1000``.
* **peak VRAM**: ``torch.cuda.max_memory_allocated`` over the run, in MB (``0.0`` on CPU).

Deterministic: inputs are seeded; ``model.eval()`` + ``inference_mode`` make the forward dropout-free.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from pedpredict.config.schema import RootCfg
from pedpredict.models.registry import MODEL_INPUT_SIGNATURE, ModelType, build_model
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.logging import CsvLogger

__all__ = [
    "EFFICIENCY_KEYS",
    "BENCHMARK_COLUMNS",
    "BenchmarkResult",
    "benchmark_model",
    "measure_efficiency",
    "run_benchmark",
]

#: The five efficiency metrics (== ``eval.evaluate._EFFICIENCY_COLUMNS``, folded into the eval-log row).
EFFICIENCY_KEYS: tuple[str, ...] = (
    "params",
    "flops_per_frame",
    "latency_ms_per_frame",
    "fps",
    "peak_vram_mb",
)
#: Standalone benchmark-CSV schema: one row per model_type.
BENCHMARK_COLUMNS: tuple[str, ...] = ("model_type", *EFFICIENCY_KEYS)

_BENCH_SEED = 0


@dataclass(frozen=True)
class BenchmarkResult:
    """One model_type's efficiency numbers."""

    model_type: str
    params: int
    flops_per_frame: float
    latency_ms_per_frame: float
    fps: float
    peak_vram_mb: float

    def efficiency(self) -> dict[str, float]:
        """The five :data:`EFFICIENCY_KEYS` (folded into the eval-log row by 5.1)."""
        return {
            "params": self.params,
            "flops_per_frame": self.flops_per_frame,
            "latency_ms_per_frame": self.latency_ms_per_frame,
            "fps": self.fps,
            "peak_vram_mb": self.peak_vram_mb,
        }

    def as_row(self) -> dict[str, object]:
        """A :data:`BENCHMARK_COLUMNS` row."""
        return {"model_type": self.model_type, **self.efficiency()}


# --------------------------------------------------------------------------- measurement primitives


def _dummy_inputs(cfg: RootCfg, device: torch.device) -> dict[str, torch.Tensor]:
    """Seeded random inputs at the REAL inference shapes (DataCfg); keyed by registry input name."""
    torch.manual_seed(_BENCH_SEED)
    d, b, t = cfg.data, cfg.eval.bench_batch_size, cfg.data.max_seq_len
    return {
        "images_tight": torch.randn(b, t, 3, d.img_height, d.img_width, device=device),
        "images_context": torch.randn(b, t, 3, d.read_context_height, d.read_context_width, device=device),
        "motions": torch.randn(b, t, d.motion_dim, device=device),
    }


def _input_tuple(model_type: ModelType, dummies: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Positional inputs for ``model_type`` in its forward order (registry's MODEL_INPUT_SIGNATURE)."""
    return tuple(dummies[name] for name in MODEL_INPUT_SIGNATURE[model_type])


def count_params(model: nn.Module) -> int:
    """Trainable parameter count (OLD ``test.py:521``)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flops_per_frame(model: nn.Module, inputs: tuple[torch.Tensor, ...], seq_len: int) -> float:
    """fvcore total FLOPs / ``seq_len`` (OLD ``compute_flops``). ``nan`` if fvcore is unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return float("nan")
    with torch.inference_mode():
        analysis = FlopCountAnalysis(model, inputs)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = analysis.total()
    return total / seq_len


def _measure_latency(
    model: nn.Module, inputs: tuple[torch.Tensor, ...], *, warmup: int, trials: int, seq_len: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(fps, latency_ms_per_frame)`` (OLD ``inference_latency``: warmup, then CUDA-synced timing)."""
    is_cuda = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(trials):
            model(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    avg_seq = elapsed / trials
    return 1.0 / avg_seq, (avg_seq / seq_len) * 1000.0


# --------------------------------------------------------------------------- public API


def benchmark_model(
    cfg: RootCfg,
    model_type: str | ModelType | None = None,
    *,
    device: torch.device | None = None,
    model: nn.Module | None = None,
) -> BenchmarkResult:
    """Benchmark one model_type (built from ``cfg`` unless ``model`` is supplied)."""
    device = device if device is not None else get_device()
    enable_perf_flags(device)
    mt = ModelType.coerce(model_type if model_type is not None else cfg.eval.model_type)
    if model is None:
        model = build_model(cfg, mt)
    model = model.to(device).eval()

    seq_len = cfg.data.max_seq_len
    inputs = _input_tuple(mt, _dummy_inputs(cfg, device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    flops = _flops_per_frame(model, inputs, seq_len)
    fps, ms_per_frame = _measure_latency(
        model, inputs, warmup=cfg.eval.bench_warmup, trials=cfg.eval.latency_trials,
        seq_len=seq_len, device=device,
    )
    vram = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    return BenchmarkResult(
        model_type=mt.value,
        params=count_params(model),
        flops_per_frame=flops,
        latency_ms_per_frame=ms_per_frame,
        fps=fps,
        peak_vram_mb=vram,
    )


def measure_efficiency(
    cfg: RootCfg, *, device: torch.device | None = None, model: nn.Module | None = None
) -> dict[str, float]:
    """The five efficiency metrics for ``cfg.eval.model_type`` — the 5.1 eval-log hook (``--benchmark``)."""
    return benchmark_model(cfg, device=device, model=model).efficiency()


def run_benchmark(
    cfg: RootCfg,
    model_types: Sequence[str | ModelType] | None = None,
    *,
    device: torch.device | None = None,
    csv_path: str | Path | None = None,
) -> list[BenchmarkResult]:
    """Benchmark several model_types (default: all four); optionally write a ``BENCHMARK_COLUMNS`` CSV."""
    types = list(model_types) if model_types is not None else list(ModelType)
    results = [benchmark_model(cfg, mt, device=device) for mt in types]
    if csv_path is not None:
        with CsvLogger(Path(csv_path), BENCHMARK_COLUMNS) as logger:
            for result in results:
                logger.log(result.as_row())
    return results

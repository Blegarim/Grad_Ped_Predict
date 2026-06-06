"""Prompt 5.1 — evaluation pipeline parity + orchestration + tiny-chunk smoke test.

Parity here is COMPOSITIONAL — ``evaluate_model`` adds no math. The two pieces it wires are each already
golden-locked, so this suite REUSES their fixtures rather than re-running OLD ``test.py``:

  * metric oracle — ``tests/fixtures/golden/metrics_cases.pt`` (OLD ``test.py:74-100`` / ``train.py:580-595``
    transcription, 3.2). Driven through ``evaluate_model`` via a fake full model that replays the fixture's
    per-batch output dicts; the aggregated metrics must equal the fixture's expected values.
  * real full-model outputs incl. ``temporal_weights`` — ``tests/fixtures/golden/ensemble.pt`` (2.4/2.5).
    The rebuilt model strict-loads the captured weights; collected ``temporal_weights`` must equal the
    captured forward.

Plus: the output contract (B4 — crosses scored on ``crosses_frame``; ``temporal_weights`` full-only),
B10 (all four model types run), B2 (strict load, no forward), cross-chunk aggregation, the WIDE
``eval_log.csv`` / ``index.csv`` schema, and an end-to-end smoke over a tiny real LMDB chunk.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from pedpredict.config import DataCfg, EvalCfg, PathsCfg, RootCfg, TrainCfg
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.eval.evaluate import (
    EVAL_LOG_COLUMNS,
    evaluate_model,
    load_eval_weights,
    run_evaluation,
)
from pedpredict.models.registry import ModelType, build_model
from pedpredict.training.metrics import MetricAccumulator

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "golden"
_METRICS_FX = _FIXTURES / "metrics_cases.pt"
_ENSEMBLE_FX = _FIXTURES / "ensemble.pt"
_TASKS = ("actions", "looks", "crosses")
_CPU = torch.device("cpu")
_TOL = 1e-6


@pytest.fixture(scope="module")
def metrics_golden() -> dict:
    return torch.load(_METRICS_FX, weights_only=False)


@pytest.fixture(scope="module")
def ensemble_golden() -> dict:
    return torch.load(_ENSEMBLE_FX, weights_only=False)


# --------------------------------------------------------------------------- fakes / helpers


class _ReplayModel(nn.Module):
    """A fake model whose forward replays pre-baked per-batch output dicts (drives metric-wiring tests)."""

    def __init__(self, outputs_batches: list[dict[str, torch.Tensor]], model_type=ModelType.FULL) -> None:
        super().__init__()
        self.model_type = model_type
        self._queue = list(outputs_batches)
        self._i = 0

    def forward(self, images_tight, images_context, motions, return_feats: bool = False):  # noqa: ARG002
        out = self._queue[self._i]
        self._i += 1
        return out


def _replay_loader(outputs_batches: list[dict], targets_batches: list[dict]) -> list:
    """One chunk = a list of collate tuples; image/motion tensors are dummies (the model replays outputs)."""
    batches = []
    for targets in targets_batches:
        n = targets["actions"].shape[0]
        dummy = torch.zeros(n, 1)
        batches.append((dummy, dummy, dummy, {k: targets[k].clone() for k in _TASKS}))
    return batches


# --------------------------------------------------------------------------- metric-wiring parity


def test_evaluate_metrics_match_golden_oracle(metrics_golden: dict) -> None:
    scenario = metrics_golden["main"]
    outputs_batches = scenario["inputs"]["outputs_batches"]
    targets_batches = scenario["inputs"]["targets_batches"]
    model = _ReplayModel(outputs_batches)
    loader = _replay_loader(outputs_batches, targets_batches)

    artifacts = evaluate_model(model, [loader], _CPU, EvalCfg())

    expected = scenario["expected"]
    for task in _TASKS:
        got = artifacts.metrics.per_task[task]
        exp = expected["per_task"][task]
        assert got.accuracy == pytest.approx(exp["acc"], abs=_TOL)
        assert got.f1 == pytest.approx(exp["f1"], abs=_TOL)
        assert got.precision == pytest.approx(exp["precision"], abs=_TOL)
        assert got.recall == pytest.approx(exp["recall"], abs=_TOL)
        assert got.auc == pytest.approx(exp["auc"], abs=_TOL)
    assert artifacts.metrics.macro_f1 == pytest.approx(expected["macro_f1"], abs=_TOL)
    assert artifacts.metrics.overall_accuracy == pytest.approx(expected["overall_acc"], abs=_TOL)


def test_optimal_thresholds_wired(metrics_golden: dict) -> None:
    scenario = metrics_golden["main"]
    ob, tb = scenario["inputs"]["outputs_batches"], scenario["inputs"]["targets_batches"]
    cfg = EvalCfg()
    artifacts = evaluate_model(_ReplayModel(ob), [_replay_loader(ob, tb)], _CPU, cfg)

    # Same accumulator + EvalCfg -> identical sweep (this is the eval-only enrichment, owned by 3.2).
    ref_acc = MetricAccumulator()
    for outputs, targets in zip(ob, tb, strict=True):
        ref_acc.update(outputs, targets)
    expected = ref_acc.optimal_threshold_metrics(cfg)
    assert artifacts.optimal == pytest.approx(expected)


def test_aggregation_equals_concatenation(metrics_golden: dict) -> None:
    ob = metrics_golden["main"]["inputs"]["outputs_batches"]
    tb = metrics_golden["main"]["inputs"]["targets_batches"]
    # 2 separate single-batch chunks vs 1 two-batch chunk -> identical global metrics.
    a_multi = evaluate_model(_ReplayModel(ob), [[b] for b in _replay_loader(ob, tb)], _CPU, EvalCfg())
    a_single = evaluate_model(_ReplayModel(ob), [_replay_loader(ob, tb)], _CPU, EvalCfg())
    assert a_multi.metrics.as_flat_dict() == pytest.approx(a_single.metrics.as_flat_dict(), nan_ok=True)


# --------------------------------------------------------------------------- output contract (B4)


def test_crosses_scored_on_frame_not_pooled(metrics_golden: dict) -> None:
    ob = metrics_golden["main"]["inputs"]["outputs_batches"]
    tb = metrics_golden["main"]["inputs"]["targets_batches"]
    base = evaluate_model(_ReplayModel(ob), [_replay_loader(ob, tb)], _CPU, EvalCfg())

    perturbed = [{**o, "crosses_pooled": torch.randn_like(o["crosses_pooled"])} for o in ob]
    after = evaluate_model(_ReplayModel(perturbed), [_replay_loader(perturbed, tb)], _CPU, EvalCfg())
    assert after.metrics.as_flat_dict() == pytest.approx(base.metrics.as_flat_dict(), nan_ok=True)


def test_predictions_extracted_and_argmax(metrics_golden: dict) -> None:
    ob = metrics_golden["main"]["inputs"]["outputs_batches"]
    tb = metrics_golden["main"]["inputs"]["targets_batches"]
    artifacts = evaluate_model(
        _ReplayModel(ob), [_replay_loader(ob, tb)], _CPU, EvalCfg(), collect_predictions=True
    )
    preds = artifacts.predictions
    assert preds is not None
    n = sum(t["actions"].shape[0] for t in tb)
    for task in _TASKS:
        assert preds[f"{task}_true"].shape == (n,)
        probs = np.stack([preds[f"{task}_prob_0"], preds[f"{task}_prob_1"]], axis=1)
        assert np.array_equal(preds[f"{task}_pred"], probs.argmax(axis=1))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_predictions_none_when_not_requested(metrics_golden: dict) -> None:
    ob = metrics_golden["main"]["inputs"]["outputs_batches"]
    tb = metrics_golden["main"]["inputs"]["targets_batches"]
    artifacts = evaluate_model(_ReplayModel(ob), [_replay_loader(ob, tb)], _CPU, EvalCfg())
    assert artifacts.predictions is None
    assert artifacts.temporal_weights is None


# --------------------------------------------------------------------------- real models (ensemble.pt)


def _build_loaded(entry: dict, model_type: str) -> nn.Module:
    model = build_model(RootCfg(), model_type)
    model.load_state_dict(entry["state_dict"], strict=True)   # B2: strict, no forward
    return model.eval()


def _labels_for(n: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {k: (torch.rand(n, generator=gen) < 0.5).long() for k in _TASKS}


def test_temporal_weights_collected_full_model(ensemble_golden: dict) -> None:
    entry = ensemble_golden["full"]
    model = _build_loaded(entry, "full")
    inputs = entry["inputs"]
    n = inputs["images_tight"].shape[0]
    batch = (inputs["images_tight"], inputs["images_context"], inputs["motions"], _labels_for(n))

    artifacts = evaluate_model(model, [[batch]], _CPU, EvalCfg(), collect_temporal_weights=True)

    assert artifacts.temporal_weights is not None
    torch.testing.assert_close(
        torch.from_numpy(artifacts.temporal_weights),
        entry["outputs"]["temporal_weights"].float(),
        atol=1e-6, rtol=1e-5,
    )


def test_temporal_weights_none_for_ablations(ensemble_golden: dict) -> None:
    for model_type in ("motion_only", "visual_only", "vanilla_concat"):
        entry = ensemble_golden[model_type]
        model = _build_loaded(entry, model_type)
        inputs = entry["inputs"]
        n = inputs["images_tight"].shape[0]
        batch = (inputs["images_tight"], inputs["images_context"], inputs["motions"], _labels_for(n))
        artifacts = evaluate_model(model, [[batch]], _CPU, EvalCfg(), collect_temporal_weights=True)
        assert artifacts.temporal_weights is None, model_type


def test_runs_for_all_four_model_types(ensemble_golden: dict) -> None:
    for model_type in ("full", "motion_only", "visual_only", "vanilla_concat"):
        entry = ensemble_golden[model_type]
        model = _build_loaded(entry, model_type)
        inputs = entry["inputs"]
        n = inputs["images_tight"].shape[0]
        batch = (inputs["images_tight"], inputs["images_context"], inputs["motions"], _labels_for(n))
        artifacts = evaluate_model(model, [[batch]], _CPU, EvalCfg())
        for task in _TASKS:
            assert np.isfinite(artifacts.metrics.per_task[task].accuracy)


# --------------------------------------------------------------------------- weight loading (B2)


def test_load_eval_weights_strict_roundtrip(tmp_path: Path) -> None:
    model = build_model(RootCfg())
    ckpt = tmp_path / "weights.pth"
    torch.save(model.state_dict(), ckpt)             # bare state_dict (no version key)

    fresh = build_model(RootCfg())
    load_eval_weights(fresh, ckpt, device=_CPU, strict=True)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)


def test_load_eval_weights_accepts_versioned_payload(tmp_path: Path) -> None:
    model = build_model(RootCfg())
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"pedpredict_ckpt_version": 1, "model_state_dict": model.state_dict()}, ckpt)
    fresh = build_model(RootCfg())
    load_eval_weights(fresh, ckpt, device=_CPU, strict=True)   # extracts model_state_dict; no error


# --------------------------------------------------------------------------- end-to-end smoke


def _write_tiny_chunk(tmp_path: Path) -> Path:
    """Write one tiny LMDB chunk with the real writer (mirrors test_lmdb_roundtrip)."""
    seq_len, n = 4, 3
    cfg_data = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=64 * 1024 * 1024)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    records = []
    for idx in range(n):
        paths = []
        for t in range(seq_len):
            yy, xx = np.mgrid[0:200, 0:200]
            arr = np.stack([(xx + t) % 256, (yy + idx) % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
            p = frames_dir / f"s{idx}_f{t}.png"
            Image.fromarray(arr).save(p)
            paths.append(str(p))
        bboxes = [[10.0 + t, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(seq_len)]
        records.append({"images": paths, "bboxes": bboxes, "actions": idx % 2, "looks": 0, "crosses": idx % 2})
    out_dir = tmp_path / "test_lmdb"
    write_dataset_chunks(records, out_dir, cfg_data, num_workers=0)
    return out_dir


def test_run_evaluation_smoke_tiny_chunk(tmp_path: Path) -> None:
    test_dir = _write_tiny_chunk(tmp_path)
    runs_dir = tmp_path / "outputs" / "runs"
    cfg = dataclasses.replace(
        RootCfg(),
        paths=dataclasses.replace(PathsCfg(), lmdb_test=str(test_dir), runs_dir=str(runs_dir)),
        eval=dataclasses.replace(EvalCfg(), batch_size=2, num_workers=0),
        train=dataclasses.replace(TrainCfg(), use_amp=False),
    )
    model = build_model(cfg)
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    before = set(mp.active_children())
    report = run_evaluation(
        cfg, checkpoint=ckpt, device=_CPU, save_predictions=True, save_temporal_weights=True
    )

    # metrics finite + artifacts written
    assert np.isfinite(report.artifacts.metrics.overall_accuracy)
    assert report.artifacts.n_samples == 3
    rows = report.eval_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == ",".join(EVAL_LOG_COLUMNS)       # header == WIDE schema
    assert len(rows) == 2                               # header + 1 aggregate row
    assert report.predictions_path is not None and report.predictions_path.exists()
    assert report.temporal_weights_path is not None and report.temporal_weights_path.exists()
    assert (runs_dir / "index.csv").exists()           # cross-run index row appended
    assert set(mp.active_children()) == before          # no leaked processes

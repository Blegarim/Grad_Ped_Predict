"""``scripts/train.py`` path-wiring regression (guards the ``gather_lmdb_chunks`` call shapes).

``scripts/train.py`` is the training entry point but had no automated coverage, so a bug where
``gather_lmdb_chunks`` was called with a splatted tuple (``*cfg.paths.lmdb_train`` -> ``TypeError``,
two positional args) and a bare string (``cfg.paths.lmdb_val`` -> the function iterated the string
character-by-character -> ``FileNotFoundError``) shipped while the full suite stayed green.

These tests pin the path-gathering wiring for BOTH dispatch branches: they build tiny on-disk LMDB
chunk dirs, point an overridden config at them, and assert the single-phase
(``ChunkPrefetcher.from_config``) and schedule (``_build_chunk_builders``) wiring each produce
providers whose ``train_lmdb_paths`` / ``val_lmdb_paths`` match the configured dirs. No training pass
is run — only the path resolution + gather call shapes are exercised (the exact regression surface).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pickle
from pathlib import Path

import lmdb
import pytest
import torch

from pedpredict.config.schema import PathsCfg, RootCfg
from pedpredict.data.sampler import LabelScanCache
from pedpredict.training.chunk_loader import ChunkPrefetcher

# scripts/ is not an importable package; load the entry-point module by path.
_TRAIN_PY = Path(__file__).resolve().parent.parent / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("_train_script", _TRAIN_PY)
assert _spec is not None and _spec.loader is not None
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _write_chunk_dir(dir_path: Path) -> None:
    """Create ``dir_path`` holding one tiny ``chunk_*.lmdb`` env (enough for gather + a scan)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(dir_path / "chunk_000000.lmdb"), map_size=8 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"0_meta", pickle.dumps({"actions": 0, "looks": 0, "crosses": 0}))
    finally:
        env.close()


def _cfg_with_paths(tmp_path: Path) -> RootCfg:
    """A ``RootCfg`` whose LMDB dirs are absolute tmp dirs (base+aug train, val, balanced)."""
    train, aug = tmp_path / "train", tmp_path / "aug"
    val, balanced = tmp_path / "val", tmp_path / "balanced"
    for d in (train, aug, val, balanced):
        _write_chunk_dir(d)
    paths = dataclasses.replace(
        PathsCfg(),
        lmdb_train=(str(train), str(aug)),       # tuple -> must NOT be splatted into positional args
        lmdb_val=str(val),                       # str -> must be wrapped, not iterated char-by-char
        lmdb_train_balanced=(str(balanced),),
        runs_dir=str(tmp_path / "runs"),         # keep run-dir scaffolding out of the repo tree
    )
    return dataclasses.replace(RootCfg(), paths=paths)


def test_single_phase_wiring_gathers_train_and_val(tmp_path: Path) -> None:
    cfg = _cfg_with_paths(tmp_path)
    chunks = ChunkPrefetcher.from_config(cfg, pin_memory=False)
    try:
        assert len(chunks.train_lmdb_paths) == 2      # base + aug
        assert len(chunks.val_lmdb_paths) == 1
    finally:
        chunks.close()


def test_build_chunk_builders_resolves_both_sources(tmp_path: Path) -> None:
    cfg = _cfg_with_paths(tmp_path)
    builders = train_script._build_chunk_builders(
        cfg, device=torch.device("cpu"), scan_cache=LabelScanCache()
    )
    assert set(builders) == {"augmented", "balanced"}
    augmented = builders["augmented"]()
    balanced = builders["balanced"]()
    try:
        assert len(augmented.train_lmdb_paths) == 2 and len(augmented.val_lmdb_paths) == 1
        assert len(balanced.train_lmdb_paths) == 1 and len(balanced.val_lmdb_paths) == 1
    finally:
        augmented.close()
        balanced.close()


def test_schedule_branch_wiring_runs(tmp_path: Path, monkeypatch) -> None:
    """The schedule branch wires loss + model + chunks + a placeholder CheckpointManager and reaches
    ``run_phase_schedule`` without crashing (guards the gather shapes AND ``CheckpointManager(None)``,
    which previously raised ``TypeError`` for missing run_id/model_type). Training itself is stubbed."""
    cfg = dataclasses.replace(
        _cfg_with_paths(tmp_path),
        schedule=dataclasses.replace(RootCfg().schedule, enabled=True),
    )
    captured: dict[str, object] = {}

    def _fake_schedule(cfg_, trainer, schedule, chunk_builders, *, run_dir=None):
        captured["trainer"] = trainer
        captured["sources"] = set(chunk_builders)
        trainer.chunks.close()        # the schedule normally owns/closes providers via fit()
        return []

    monkeypatch.setattr(train_script, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(train_script, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(train_script, "run_phase_schedule", _fake_schedule)

    assert train_script.main([]) == 0
    assert captured["sources"] == {"augmented", "balanced"}


# --------------------------------------------------------------------------- --resume CLI wiring


def test_resume_passes_checkpoint_and_its_original_run_dir(tmp_path: Path, monkeypatch) -> None:
    """``--resume <run>/checkpoints/last.pth`` must reach ``build_trainer`` with BOTH the checkpoint
    and the run dir it came from. Without the second, every restart on a preemptible instance opens
    a fresh run dir and the epochs of one logical run scatter across several train_log.csv files."""
    cfg = _cfg_with_paths(tmp_path)
    run_dir = tmp_path / "runs" / "20260910_000000_pose_full_onset_aux"
    ckpt = run_dir / "checkpoints" / "last.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")                      # only the PATH is consumed on this code path

    captured: dict[str, object] = {}
    stub_dir = run_dir

    class _StubTrainer:
        run_dir = stub_dir

        def fit(self):
            return []

    def _fake_build_trainer(_cfg, chunks, **kwargs):
        captured.update(kwargs)
        chunks.close()
        return _StubTrainer()

    monkeypatch.setattr(train_script, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(train_script, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(train_script, "build_trainer", _fake_build_trainer)

    assert train_script.main(["--resume", str(ckpt)]) == 0
    assert captured["resume_from"] == ckpt
    assert captured["resume_run_dir"] == run_dir


def test_no_resume_flag_leaves_both_resume_kwargs_none(tmp_path: Path, monkeypatch) -> None:
    """The default path must stay exactly as before: a fresh run dir, no warm state."""
    cfg = _cfg_with_paths(tmp_path)
    captured: dict[str, object] = {}

    class _StubTrainer:
        run_dir = tmp_path

        def fit(self):
            return []

    def _fake_build_trainer(_cfg, chunks, **kwargs):
        captured.update(kwargs)
        chunks.close()
        return _StubTrainer()

    monkeypatch.setattr(train_script, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(train_script, "get_device", lambda *a, **k: torch.device("cpu"))
    monkeypatch.setattr(train_script, "build_trainer", _fake_build_trainer)

    assert train_script.main([]) == 0
    assert captured["resume_from"] is None and captured["resume_run_dir"] is None


def test_resume_rejects_multi_phase_schedule(tmp_path: Path, monkeypatch) -> None:
    """A checkpoint records an epoch, not which PHASE it belongs to, so resuming a schedule run would
    silently restart at phase 1 with warm weights. Refuse instead."""
    cfg = dataclasses.replace(
        _cfg_with_paths(tmp_path),
        schedule=dataclasses.replace(RootCfg().schedule, enabled=True),
    )
    ckpt = tmp_path / "runs" / "r" / "checkpoints" / "last.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    monkeypatch.setattr(train_script, "load_config", lambda *a, **k: cfg)

    with pytest.raises(SystemExit):
        train_script.main(["--resume", str(ckpt)])


def test_resume_with_missing_checkpoint_fails_before_any_training(tmp_path: Path, monkeypatch) -> None:
    """Fail on the typo immediately, not after warming chunks and building a model."""
    monkeypatch.setattr(train_script, "load_config", lambda *a, **k: _cfg_with_paths(tmp_path))

    with pytest.raises(SystemExit):
        train_script.main(["--resume", str(tmp_path / "nope.pth")])

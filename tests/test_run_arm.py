"""``scripts/run_arm.py`` orchestration regression.

``run_arm.py`` chains one train.py call + the four-cell anchored×streaming evaluate.py matrix into a
single command with train.py's exact ``--set`` surface. These tests pin the orchestration contract
without running any real training/eval: subprocess is stubbed to record every argv, so we assert

  * the training ``--set`` overrides are forwarded verbatim to train.py AND every eval,
  * each eval appends ``--set data.protocol=<cell>`` AFTER the user's overrides (so it wins),
  * the fixed matrix order (val-before-test per protocol; anchored before streaming),
  * the checkpoint is discovered from train.py's ``run dir:`` stdout and pointed at ``best.pth``,
  * fail-fast: a non-zero eval exit aborts the remaining cells,
  * ``--skip-train`` skips train.py and evaluates the passed ``--checkpoint``,
  * eval passthrough flags (``--save-predictions`` …) reach every eval call.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from pedpredict.config.schema import PathsCfg, RootCfg

# scripts/ is not an importable package; load the entry-point module by path (as test_train_script does).
_RUN_ARM_PY = Path(__file__).resolve().parent.parent / "scripts" / "run_arm.py"
_spec = importlib.util.spec_from_file_location("_run_arm_script", _RUN_ARM_PY)
assert _spec is not None and _spec.loader is not None
run_arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_arm)


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclasses.dataclass
class _Harness:
    """Records every subprocess argv and lets a test script the run dir / exit codes."""

    runs_dir: Path
    calls: list[list[str]]
    run_dir_name: str = "20260101_000000_pose_full"
    train_rc: int = 0
    eval_rc_by_cell: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)


def _install(monkeypatch, tmp_path: Path, harness: _Harness) -> None:
    """Point run_arm at a tmp runs_dir and stub subprocess.run + load_config."""
    cfg = dataclasses.replace(
        RootCfg(), paths=dataclasses.replace(PathsCfg(), runs_dir=str(harness.runs_dir))
    )
    monkeypatch.setattr(run_arm, "load_config", lambda *a, **k: cfg)

    def _fake_run(cmd, *, check=False, capture_output=False, text=False):  # noqa: ARG001
        harness.calls.append(list(cmd))
        script = Path(cmd[1]).name
        if script == "train.py":
            # Materialize the run dir so the runner's existence checks pass, and echo the marker line.
            run_dir = harness.runs_dir / harness.run_dir_name
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints" / "best.pth").write_bytes(b"x")
            return _FakeProc(harness.train_rc, stdout=f"Training complete. 1 epoch(s), run dir: {run_dir}\n")
        # evaluate.py — look up this cell's scripted return code.
        split = cmd[cmd.index("--split") + 1]
        protocol = next(c.split("=")[1] for c in cmd if c.startswith("data.protocol="))
        return _FakeProc(harness.eval_rc_by_cell.get((split, protocol), 0))

    monkeypatch.setattr(run_arm.subprocess, "run", _fake_run)


_ARM_OVERRIDES = ["eval.model_type=pose_full", "data.protocol=anchored", "train.num_epochs=30"]


def _argv(overrides, *extra):
    out: list[str] = []
    for o in overrides:
        out += ["--set", o]
    return [*out, *extra]


def _eval_calls(calls):
    return [c for c in calls if Path(c[1]).name == "evaluate.py"]


def _effective_protocol(cmd):
    """The protocol the config loader resolves for an eval cmd: the LAST ``data.protocol=`` --set wins."""
    protos = [t.split("=")[1] for t in cmd if t.startswith("data.protocol=")]
    return protos[-1]


def _train_calls(calls):
    return [c for c in calls if Path(c[1]).name == "train.py"]


def test_full_matrix_order_and_protocol_override(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 0

    # exactly one train + four evals in the fixed 2×2 order.
    assert len(_train_calls(harness.calls)) == 1
    evals = _eval_calls(harness.calls)
    cells = [(c[c.index("--split") + 1], _effective_protocol(c)) for c in evals]
    assert cells == [("val", "anchored"), ("test", "anchored"), ("val", "streaming"), ("test", "streaming")]


def test_training_overrides_forwarded_and_protocol_wins(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES))

    train_cmd = _train_calls(harness.calls)[0]
    # every training override reaches train.py verbatim (one --set token each).
    for o in _ARM_OVERRIDES:
        assert o in train_cmd

    for cell_cmd in _eval_calls(harness.calls):
        # The user's data.protocol=anchored is forwarded, but the cell appends its own protocol override
        # LAST, and parse_overrides is last-wins — so the cell protocol is the final data.protocol= token.
        proto_indices = [i for i, t in enumerate(cell_cmd) if t.startswith("data.protocol=")]
        user_idx = cell_cmd.index("data.protocol=anchored")     # the forwarded training override
        assert proto_indices[-1] >= user_idx                    # the winning cell override is last
        assert _effective_protocol(cell_cmd) in {"anchored", "streaming"}
        # non-protocol training overrides are forwarded to eval too.
        assert "train.num_epochs=30" in cell_cmd


def test_checkpoint_discovered_from_run_dir(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES))

    expected = harness.runs_dir / harness.run_dir_name / "checkpoints" / "best.pth"
    for cell_cmd in _eval_calls(harness.calls):
        assert cell_cmd[cell_cmd.index("--checkpoint") + 1] == str(expected)


def test_fail_fast_aborts_matrix_on_eval_error(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(
        runs_dir=tmp_path / "runs", calls=[],
        eval_rc_by_cell={("test", "anchored"): 2},   # 2nd cell fails
    )
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 2
    # only the first two cells (val/anchored, test/anchored) ran; streaming cells were skipped.
    assert len(_eval_calls(harness.calls)) == 2


def test_train_failure_aborts_before_eval(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[], train_rc=1)
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 1
    assert _eval_calls(harness.calls) == []       # no eval attempted after a failed train


def test_skip_train_evaluates_existing_checkpoint(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    ckpt = tmp_path / "existing" / "checkpoints" / "best.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    rc = run_arm.main(_argv(_ARM_OVERRIDES, "--skip-train", "--checkpoint", str(ckpt)))
    assert rc == 0
    assert _train_calls(harness.calls) == []          # train.py never invoked
    assert len(_eval_calls(harness.calls)) == 4
    for cell_cmd in _eval_calls(harness.calls):
        assert cell_cmd[cell_cmd.index("--checkpoint") + 1] == str(ckpt)


def test_skip_train_requires_checkpoint(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    with pytest.raises(SystemExit):        # argparse parser.error -> SystemExit
        run_arm.main(_argv(_ARM_OVERRIDES, "--skip-train"))


def test_passthrough_eval_flags_reach_every_eval(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES, "--save-predictions", "--benchmark"))

    for cell_cmd in _eval_calls(harness.calls):
        assert "--save-predictions" in cell_cmd
        assert "--benchmark" in cell_cmd


def test_dry_run_executes_nothing(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES, "--dry-run"))
    assert rc == 0
    assert harness.calls == []          # subprocess.run never called in dry-run

"""``scripts/run_arm.py`` orchestration regression.

``run_arm.py`` chains the full cross-protocol matrix for one model arm into a single command with
train.py's exact ``--set`` surface: TWO training runs (anchored-trained, streaming-trained), each
followed by the fixed 4-cell (val/test × anchored/streaming) evaluate.py matrix. These tests pin the
orchestration contract without running any real training/eval: subprocess is stubbed to record every
argv, so we assert

  * two train arms in canonical order (anchored, then streaming), 4 evals after each (10 steps),
  * the runner OWNS ``data.protocol``: each train/eval appends its own override LAST (last-wins),
    and a user-passed ``data.protocol`` only triggers a warning,
  * the training ``--set`` overrides are forwarded verbatim to every train AND eval call,
  * each arm's evals point at THAT arm's ``<run_dir>/checkpoints/best.pth``,
  * per-arm ``--tag`` (``[{tag}_]train{protocol}``) reaches train.py,
  * fail-fast: a non-zero train or eval exit aborts everything after it,
  * ``--protocols streaming`` restricts to one arm; ``--skip-train --checkpoint`` runs eval-only,
  * eval passthrough flags (``--save-predictions`` …) reach every eval call; ``--dry-run`` runs nothing.
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
    """Records every subprocess argv and lets a test script exit codes per step."""

    runs_dir: Path
    calls: list[list[str]]
    train_rc_by_protocol: dict[str, int] = dataclasses.field(default_factory=dict)
    #: (train_protocol_of_ckpt, split, eval_protocol) -> rc for evaluate.py calls.
    eval_rc_by_cell: dict[tuple[str, str, str], int] = dataclasses.field(default_factory=dict)


def _train_protocol_of(cmd: list[str]) -> str:
    """The protocol the config loader would resolve for a train cmd (LAST data.protocol= wins)."""
    return [t.split("=")[1] for t in cmd if t.startswith("data.protocol=")][-1]


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
            protocol = _train_protocol_of(cmd)
            rc = harness.train_rc_by_protocol.get(protocol, 0)
            # Materialize the run dir so the runner's existence checks pass; echo the marker line.
            run_dir = harness.runs_dir / f"20260101_000000_pose_full_train{protocol}"
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints" / "best.pth").write_bytes(b"x")
            return _FakeProc(rc, stdout=f"Training complete. 1 epoch(s), run dir: {run_dir}\n")
        # evaluate.py — recover which arm's checkpoint this cell targets from the ckpt path.
        ckpt = cmd[cmd.index("--checkpoint") + 1]
        arm = "anchored" if "trainanchored" in ckpt else ("streaming" if "trainstreaming" in ckpt else "ckpt")
        split = cmd[cmd.index("--split") + 1]
        protocol = [t.split("=")[1] for t in cmd if t.startswith("data.protocol=")][-1]
        return _FakeProc(harness.eval_rc_by_cell.get((arm, split, protocol), 0))

    monkeypatch.setattr(run_arm.subprocess, "run", _fake_run)


_ARM_OVERRIDES = ["eval.model_type=pose_full", "train.num_epochs=30"]

_FULL_MATRIX = [
    ("val", "anchored"), ("test", "anchored"), ("val", "streaming"), ("test", "streaming"),
]


def _argv(overrides, *extra):
    out: list[str] = []
    for o in overrides:
        out += ["--set", o]
    return [*out, *extra]


def _eval_calls(calls):
    return [c for c in calls if Path(c[1]).name == "evaluate.py"]


def _train_calls(calls):
    return [c for c in calls if Path(c[1]).name == "train.py"]


def _cell_of(eval_cmd):
    """(split, effective eval protocol) of one evaluate.py argv (LAST data.protocol= wins)."""
    protos = [t.split("=")[1] for t in eval_cmd if t.startswith("data.protocol=")]
    return eval_cmd[eval_cmd.index("--split") + 1], protos[-1]


def test_two_arms_ten_steps_in_canonical_order(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 0

    # 10 steps: train(anchored), 4 evals, train(streaming), 4 evals — in that global order.
    kinds = [Path(c[1]).name for c in harness.calls]
    assert kinds == ["train.py"] + ["evaluate.py"] * 4 + ["train.py"] + ["evaluate.py"] * 4

    trains = _train_calls(harness.calls)
    assert [_train_protocol_of(c) for c in trains] == ["anchored", "streaming"]

    evals = _eval_calls(harness.calls)
    assert [_cell_of(c) for c in evals] == _FULL_MATRIX + _FULL_MATRIX


def test_runner_owns_protocol_and_forwards_overrides(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    # user passes data.protocol=anchored — the runner must still train BOTH arms (its own override wins).
    run_arm.main(_argv([*_ARM_OVERRIDES, "data.protocol=anchored"]))

    trains = _train_calls(harness.calls)
    assert [_train_protocol_of(c) for c in trains] == ["anchored", "streaming"]
    for cmd in trains + _eval_calls(harness.calls):
        # every training override reaches every subprocess verbatim (one --set token each) ...
        for o in _ARM_OVERRIDES:
            assert o in cmd
        # ... and the runner's protocol override sits AFTER the user's (last-wins).
        proto_idx = [i for i, t in enumerate(cmd) if t.startswith("data.protocol=")]
        assert cmd.index("data.protocol=anchored") <= proto_idx[-1]


def test_each_arm_evaluates_its_own_checkpoint(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES))

    evals = _eval_calls(harness.calls)
    anchored_ckpt = harness.runs_dir / "20260101_000000_pose_full_trainanchored" / "checkpoints" / "best.pth"
    streaming_ckpt = harness.runs_dir / "20260101_000000_pose_full_trainstreaming" / "checkpoints" / "best.pth"
    ckpts = [c[c.index("--checkpoint") + 1] for c in evals]
    assert ckpts == [str(anchored_ckpt)] * 4 + [str(streaming_ckpt)] * 4


def test_tag_flows_to_train_with_protocol_suffix(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES, "--tag", "posev2"))

    tags = [c[c.index("--tag") + 1] for c in _train_calls(harness.calls)]
    assert tags == ["posev2_trainanchored", "posev2_trainstreaming"]


def test_default_tag_is_protocol_only(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv(_ARM_OVERRIDES))

    tags = [c[c.index("--tag") + 1] for c in _train_calls(harness.calls)]
    assert tags == ["trainanchored", "trainstreaming"]


def test_fail_fast_on_eval_aborts_everything_after(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(
        runs_dir=tmp_path / "runs", calls=[],
        eval_rc_by_cell={("anchored", "test", "anchored"): 2},   # 2nd cell of the FIRST arm fails
    )
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 2
    # 1 train + only the first two eval cells ran; the streaming arm never started.
    assert len(_train_calls(harness.calls)) == 1
    assert len(_eval_calls(harness.calls)) == 2


def test_fail_fast_on_second_train(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(
        runs_dir=tmp_path / "runs", calls=[], train_rc_by_protocol={"streaming": 1},
    )
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES))
    assert rc == 1
    # anchored arm completed (1 train + 4 evals), streaming train failed, no further evals.
    assert len(_train_calls(harness.calls)) == 2
    assert len(_eval_calls(harness.calls)) == 4


def test_protocols_flag_restricts_arms(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES, "--protocols", "streaming"))
    assert rc == 0
    trains = _train_calls(harness.calls)
    assert [_train_protocol_of(c) for c in trains] == ["streaming"]
    assert len(_eval_calls(harness.calls)) == 4


def test_user_protocol_override_warns(monkeypatch, tmp_path: Path, capsys) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    run_arm.main(_argv([*_ARM_OVERRIDES, "data.protocol=streaming"]))

    err = capsys.readouterr().err
    assert "data.protocol" in err and "ignored" in err


def test_skip_train_evaluates_existing_checkpoint(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)
    ckpt = tmp_path / "existing" / "checkpoints" / "best.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")

    rc = run_arm.main(_argv(_ARM_OVERRIDES, "--skip-train", "--checkpoint", str(ckpt)))
    assert rc == 0
    assert _train_calls(harness.calls) == []          # train.py never invoked
    evals = _eval_calls(harness.calls)
    assert len(evals) == 4                            # eval-only mode: ONE 4-cell matrix
    for cell_cmd in evals:
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

    evals = _eval_calls(harness.calls)
    assert len(evals) == 8
    for cell_cmd in evals:
        assert "--save-predictions" in cell_cmd
        assert "--benchmark" in cell_cmd


def test_dry_run_executes_nothing(monkeypatch, tmp_path: Path) -> None:
    harness = _Harness(runs_dir=tmp_path / "runs", calls=[])
    _install(monkeypatch, tmp_path, harness)

    rc = run_arm.main(_argv(_ARM_OVERRIDES, "--dry-run"))
    assert rc == 0
    assert harness.calls == []          # subprocess.run never called in dry-run

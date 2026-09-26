"""Re-run campaign go/no-go gate (docs/RERUN_PLAN_2026-09-26.md §2).

Pins what the gate must tell apart: a curve that learns and then holds (GO) from the measured control-arm
pattern — validation AUC peaking at epoch 4 and declining while train loss falls — and from collapse epochs;
plus the seed-level checks and the CLI's three exit codes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pedpredict.eval.rerun_gate import (
    GateThresholds,
    curve_checks,
    seed_checks,
    selected_epoch_auc,
)

_TH = GateThresholds()
# pf_ctrl_s42, measured 2026-09-26: epochs 1-10 val AUC — the memorization pattern the gate must reject.
_CTRL_AUC = [0.8204, 0.8314, 0.8204, 0.8352, 0.8247, 0.8098, 0.7830, 0.8200, 0.7906, 0.8026]


def _rows(aucs: list[float], *, recall: float = 0.3, val_loss: float = 0.2) -> list[dict[str, float]]:
    return [{"epoch": float(i + 1), "crosses_auc": a, "crosses_f1": a / 4, "crosses_recall": recall,
             "val_loss": val_loss} for i, a in enumerate(aucs)]


def _learning(n: int = 20) -> list[float]:
    """Rises to a plateau at epoch 12 and holds within 0.005 of it."""
    return [0.80 + 0.004 * min(i, 11) - (0.003 if i % 2 else 0.0) * (i > 11) for i in range(n)]


def _passed(checks) -> dict[str, bool]:
    return {c.name.split()[0]: c.passed for c in checks}


def test_learning_curve_passes_l1_to_l3() -> None:
    assert _passed(curve_checks("fix", _rows(_learning()), _TH)) == {"L1": True, "L2": True, "L3": True}


def test_control_arm_pattern_fails_decline_and_early_peak() -> None:
    got = _passed(curve_checks("ctrl", _rows(_CTRL_AUC), _TH))
    assert got == {"L1": False, "L2": False, "L3": True}


def test_collapse_epoch_fails_l3() -> None:
    rows = _rows(_learning())
    rows[6]["crosses_recall"] = 0.95          # an "everything crosses" epoch
    assert _passed(curve_checks("fix", rows, _TH))["L3"] is False
    rows = _rows(_learning())
    rows[3]["val_loss"] = 1.4                 # the paper's val-loss collapse definition
    assert _passed(curve_checks("fix", rows, _TH))["L3"] is False


def test_too_few_epochs_fails() -> None:
    (check,) = curve_checks("short", _rows(_learning(6)), _TH)
    assert not check.passed and "only 6 epochs" in check.detail


def test_selected_epoch_is_first_max_of_selection_metric() -> None:
    rows = _rows([0.80, 0.84, 0.82])
    rows[2]["crosses_f1"] = rows[1]["crosses_f1"]            # tie -> the earlier epoch, as the Trainer keeps
    assert selected_epoch_auc(rows) == pytest.approx(0.84)


def test_selected_epoch_follows_the_runs_own_metric() -> None:
    rows = _rows([0.80, 0.84, 0.82])
    rows[0]["crosses_f1"] = 0.9                               # F1 would pick epoch 1 ...
    assert selected_epoch_auc(rows, "crosses_f1") == pytest.approx(0.80)
    assert selected_epoch_auc(rows, "crosses_auc") == pytest.approx(0.84)   # ... AUC selection picks epoch 2
    rows[2]["val_loss"] = 0.05
    assert selected_epoch_auc(rows, "val_loss") == pytest.approx(0.82)      # val_loss is minimized


def test_seed_checks_pass_when_tight_and_good() -> None:
    checks = seed_checks([0.85, 0.86, 0.855], [0.79, 0.80, 0.795], [0.55, 0.57, 0.56], [0.30, 0.32, 0.31], _TH)
    assert all(c.passed for c in checks)


def test_seed_checks_fail_on_the_v1_spread() -> None:
    # v1 binary per-seed detection at 205/hr: 15.6 / 52.2 / 29.3 %
    checks = seed_checks([0.83, 0.83, 0.83], [0.79, 0.79, 0.79], [0.156, 0.522, 0.293], [0.2, 0.2, 0.2], _TH)
    by_name = {c.name: c.passed for c in checks}
    assert by_name["L4 detection spread"] is False
    assert by_name["L4 val AUC range"] is True


def test_single_seed_cannot_pass() -> None:
    (check,) = seed_checks([0.85], [0.8], [0.5], [0.3], _TH)
    assert not check.passed


def _write_run(root: Path, name: str, aucs: list[float], test_auc: float) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True)
    rows = _rows(aucs)
    with open(run / "train_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(run / "eval_log.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "protocol", "crosses_auc"])
        writer.writeheader()
        writer.writerows([{"split": "test", "protocol": "streaming", "crosses_auc": 0.5},
                          {"split": "test", "protocol": "anchored", "crosses_auc": 0.9},
                          {"split": "test", "protocol": "streaming", "crosses_auc": test_auc}])
    return run


def _write_dump(path: Path, seed: int) -> None:
    """Crossers score high before onset, non-crossers low: detection well above the level floor."""
    rng = np.random.default_rng(seed)
    tid, onset, score = [], [], []
    for t in range(60):
        cross = t < 12
        for w in range(10):
            tid.append(f"t{t}")
            onset.append(90 - 9 * w if cross else -1)
            score.append((0.9 if cross and w > 3 else 0.1) + 0.05 * rng.random())
    n = len(tid)
    path.parent.mkdir(parents=True)
    np.savez(path, track_id=np.asarray(tid), onset_offset=np.asarray(onset), p_frame=np.asarray(score),
             future_observed=np.full(n, 200), track_crosses=(np.asarray(onset) >= 0).astype(int),
             crosses=np.zeros(n, dtype=int), meta=json.dumps({}))


def test_cli_exit_codes(tmp_path: Path) -> None:
    from scripts.rerun_gate import main

    runs = [_write_run(tmp_path, f"r_s{s}", _learning(), 0.80) for s in (42, 43, 44)]
    args = ["--dumps-root", str(tmp_path / "diag"), "--out", str(tmp_path / "gate")]
    for r in runs:
        args += ["--run", str(r)]
    assert main(args) == 2                                   # dumps not there yet
    for s, r in zip((42, 43, 44), runs, strict=True):
        _write_dump(tmp_path / "diag" / r.name / "onset_test.npz", s)
    assert main(args) == 0
    verdict = json.loads((tmp_path / "gate" / "verdict.json").read_text())
    assert verdict["verdict"] == "GO"
    assert verdict["mean_selected_val_auc"] == pytest.approx(max(_learning()), abs=0.01)
    bad = _write_run(tmp_path, "r_ctrl", _CTRL_AUC + [0.80] * 5, 0.80)
    _write_dump(tmp_path / "diag" / bad.name / "onset_test.npz", 7)
    assert main([*args, "--run", str(bad)]) == 1

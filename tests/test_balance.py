"""Prompt 1.3 — offline-balance parity + count-solver invariant tests.

Three kinds of checks (same shape as the 1.1 / 1.2 suites):
  * GOLDEN parity: ``balance_indices`` reproduces OLD ``balance_dataset`` (``BALANCE_EQUAL``) and
    ``balance_split`` (``BALANCE_RATIO_30_70``) selected-index lists EXACTLY (integer indices, tol=0),
    plus the three OLD solvers, from ``tests/fixtures/golden/balance_cases.json``
    (see ``tests/_capture/capture_balance_golden.py``).
  * COUNT-SOLVER invariants: whenever the corrected solver returns a solution it satisfies the four
    balance constraints; the OLD ``solve_exact`` sign bug is shown to reject a solvable case.
  * CONFIG: the new ``balance`` section loads, validates, and overrides via the 2-level channel.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pedpredict.config import BalanceCfg, ConfigError, RootCfg, load_config, validate_config
from pedpredict.data.balance import (
    BALANCE_EQUAL,
    BALANCE_RATIO_30_70,
    X11Select,
    balance_indices,
    balance_records,
    solve_cross0_counts,
    solve_cross0_counts_approx,
    summarize,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "balance_cases.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(_FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- golden index parity


def test_golden_balance_equal(golden):
    """balance_indices(BALANCE_EQUAL) == OLD balance_dataset, exactly."""
    got = balance_indices(golden["data"], BALANCE_EQUAL)
    assert got == golden["cases"]["balance_equal"]["selected"]


def test_golden_balance_ratio_30_70(golden):
    """balance_indices(BALANCE_RATIO_30_70) == OLD balance_split(0.30), exactly (incl. the sign bug)."""
    got = balance_indices(golden["data"], BALANCE_RATIO_30_70)
    assert got == golden["cases"]["balance_ratio_30_70"]["selected"]


def test_golden_balance_ratio_50_50(golden):
    """Same RATIO solver path at ratio=0.50 reproduces OLD balance_split(0.50)."""
    cfg = dataclasses.replace(BALANCE_RATIO_30_70, cross_pos_ratio=0.50)
    got = balance_indices(golden["data"], cfg)
    assert got == golden["cases"]["balance_ratio_50_50"]["selected"]


def test_golden_solver_cases(golden):
    """Each OLD solver output is reproduced by the unified solver with the matching flags."""
    for case in golden["solver_cases"]:
        out = None if case["out"] is None else tuple(case["out"])
        if case["solver"] == "balance_sequences._solve_cross0_counts":
            got = solve_cross0_counts(
                *case["derived"], x11_select=X11Select.UPPER, legacy_x00_sign_bug=False
            )
        elif case["solver"] == "split.solve_exact":
            got = solve_cross0_counts(
                *case["args"], x11_select=X11Select.LOWER, legacy_x00_sign_bug=True
            )
        elif case["solver"] == "split.solve_approx":
            got = solve_cross0_counts_approx(*case["args"])
        else:
            pytest.fail(f"unknown solver {case['solver']!r}")
        assert got == out, case


# --------------------------------------------------------------------------- count-solver invariants


def test_solver_invariants_hold_when_feasible():
    """Whenever the corrected solver returns a tuple, it satisfies all four balance constraints."""
    checked = 0
    for n0 in range(0, 9):
        for a_target in range(0, n0 + 1):
            for l_target in range(0, n0 + 1):
                for c00 in range(0, 4):
                    for c11 in range(0, 4):
                        c01 = c10 = 3
                        for sel in (X11Select.LOWER, X11Select.UPPER):
                            sol = solve_cross0_counts(
                                n0, a_target, l_target, c00, c01, c10, c11, x11_select=sel
                            )
                            if sol is None:
                                continue
                            x00, x01, x10, x11 = sol
                            assert x00 + x01 + x10 + x11 == n0
                            assert x10 + x11 == a_target
                            assert x01 + x11 == l_target
                            assert 0 <= x00 <= c00
                            assert 0 <= x01 <= c01
                            assert 0 <= x10 <= c10
                            assert 0 <= x11 <= c11
                            checked += 1
    assert checked > 0


def test_solver_x11_select_picks_interval_ends():
    """LOWER/UPPER pick the two ends of the feasible x11 interval (here [0, 4])."""
    args = (10, 4, 4, 10, 10, 10, 10)  # feasible: lower=0, upper=min(c11,a,l,a+l-n0+c00)=4
    lo = solve_cross0_counts(*args, x11_select=X11Select.LOWER)
    hi = solve_cross0_counts(*args, x11_select=X11Select.UPPER)
    assert lo is not None and hi is not None
    assert lo[3] == 0 and hi[3] == 4


def test_legacy_sign_bug_rejects_a_solvable_case():
    """The corrected solver finds (10,5,5,0); the OLD solve_exact sign bug returns None."""
    args = (20, 5, 5, 10, 5, 5, 8)
    corrected = solve_cross0_counts(*args, x11_select=X11Select.LOWER, legacy_x00_sign_bug=False)
    buggy = solve_cross0_counts(*args, x11_select=X11Select.LOWER, legacy_x00_sign_bug=True)
    assert corrected == (10, 5, 5, 0)
    assert buggy is None


def test_solver_infeasible_returns_none():
    """a_target exceeds available cross=0 action-positives -> infeasible."""
    assert solve_cross0_counts(5, 5, 0, c00=0, c01=0, c10=1, c11=0, x11_select=X11Select.LOWER) is None


# --------------------------------------------------------------------------- output properties


def test_balanced_ratio_and_rates(golden):
    """The 30/70 balanced subset hits ~30% crosses and ~50% actions/looks (corrected summarize)."""
    cfg = dataclasses.replace(BALANCE_RATIO_30_70, legacy_x00_sign_bug=False)  # the shipped default behavior
    idx = balance_indices(golden["data"], cfg)
    s = summarize(golden["data"], idx)
    assert s["crosses_pos_rate"] == pytest.approx(0.30, abs=0.02)
    assert s["actions_pos_rate"] == pytest.approx(0.50, abs=0.05)
    assert s["looks_pos_rate"] == pytest.approx(0.50, abs=0.05)


def test_summarize_clamps_crosses(golden):
    """EQUAL keeps all cross=1 with n0==n1, so corrected (clamped) crosses rate is exactly 0.5.

    The OLD balance_sequences.summarize counted raw -1 and reported ~0.46 — this is the fix.
    """
    idx = balance_indices(golden["data"], BALANCE_EQUAL)
    s = summarize(golden["data"], idx)
    assert s["crosses_pos_rate"] == pytest.approx(0.5)
    assert golden["cases"]["balance_equal"]["summary"]["crosses_pos_rate"] < 0.5  # the OLD unclamped bug


def test_determinism_and_seed_sensitivity(golden):
    """Same seed -> identical indices; a different seed -> same size, (generally) different membership."""
    a = balance_indices(golden["data"], BALANCE_RATIO_30_70)
    b = balance_indices(golden["data"], BALANCE_RATIO_30_70)
    assert a == b
    other = balance_indices(golden["data"], dataclasses.replace(BALANCE_RATIO_30_70, seed=999))
    assert len(other) == len(a)
    assert other != a


def test_balance_records_matches_indices(golden):
    """balance_records returns the records at balance_indices, same order."""
    idx = balance_indices(golden["data"], BALANCE_RATIO_30_70)
    recs = balance_records(golden["data"], BALANCE_RATIO_30_70)
    assert recs == [golden["data"][i] for i in idx]


def test_empty_when_no_positives():
    """No cross=1 -> empty selection (both presets), never a crash."""
    data = [{"actions": 0, "looks": 0, "crosses": 0} for _ in range(10)]
    assert balance_indices(data, BALANCE_RATIO_30_70) == []
    assert balance_indices(data, BALANCE_EQUAL) == []


# --------------------------------------------------------------------------- config wiring


def test_balance_section_loads_and_defaults_off():
    cfg = load_config("configs")
    assert isinstance(cfg, RootCfg)
    assert cfg.balance.enabled is False
    assert cfg.balance.cross_pos_ratio == pytest.approx(0.30)


def test_balance_override_two_level():
    cfg = load_config("configs", overrides=["balance.enabled=true", "balance.cross_pos_ratio=0.25"])
    assert cfg.balance.enabled is True
    assert cfg.balance.cross_pos_ratio == pytest.approx(0.25)


@pytest.mark.parametrize(
    "field, value",
    [
        ("cross_pos_ratio", 0.0),
        ("cross_pos_ratio", 1.0),
        ("x11_select", "middle"),
        ("on_infeasible", "explode"),
        ("target_action_rate", 1.5),
    ],
)
def test_validate_rejects_bad_balance(field, value):
    root = dataclasses.replace(RootCfg(), balance=dataclasses.replace(BalanceCfg(), **{field: value}))
    with pytest.raises(ConfigError):
        validate_config(root)


def test_presets_are_valid_configs():
    """Both legacy presets pass schema validation when dropped into the tree."""
    for preset in (BALANCE_EQUAL, BALANCE_RATIO_30_70):
        validate_config(dataclasses.replace(RootCfg(), balance=preset))

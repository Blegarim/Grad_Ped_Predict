"""Capture golden fixtures for Prompt 1.3 by running the OLD code (provenance, not a test).

Runs OLD ``scripts/balance_sequences.py`` and ``scripts/split_balance_sequences_all.py`` on a
single, fully deterministic synthetic label list and snapshots their *selected index lists* and
*solver outputs* so ``data/balance.py`` can be diffed against them. Not collected by pytest
(filename is ``capture_*``, not ``test_*``); rerun manually only if the OLD code or the chosen
inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_balance_golden.py

Parity notes:
  * Both OLD balancers are pure stdlib (argparse/os/pickle/random/collections) — no torch, no PIE,
    no venv needed. They read only ``item["actions"|"looks"|"crosses"]``.
  * Determinism is twofold: the *data* is generated once with a fixed seed AND saved verbatim into
    the fixture (so the test is independent of regeneration); the *balancer* RNG is the OLD
    ``random.Random(seed)``, whose output depends on group/pick/shuffle call order — that order is
    the parity contract the new code must preserve. Parity class is EXACT (integer indices, tol=0).
  * The data deliberately contains ``crosses=-1`` (exercises the clamp) and populates all eight
    (crosses, actions, looks) groups so the count solvers are feasible.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

OLD_SCRIPTS = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project" / "scripts"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "balance_cases.json"

N_RECORDS = 300
GEN_SEED = 12345
BAL_SEED = 0


def _make_data() -> list[dict]:
    """Deterministic label-only records; ~25% crosses positive, a few raw -1, all groups populated."""
    rng = random.Random(GEN_SEED)
    data: list[dict] = []
    for _ in range(N_RECORDS):
        roll = rng.random()
        crosses = 1 if roll < 0.25 else (-1 if roll < 0.30 else 0)  # -1 must clamp to 0
        data.append(
            {
                "actions": rng.randint(0, 1),
                "looks": rng.randint(0, 1),
                "crosses": crosses,
            }
        )
    return data


def _solver_cases(bal, split) -> list[dict]:
    """Snapshot the three OLD solvers on hand-picked tuples (incl. a sign-bug-divergent case)."""
    # balance_sequences._solve_cross0_counts(n1, a1_pos, l1_pos, c00, c01, c10, c11)
    eq_args = [
        (20, 5, 5, 10, 5, 5, 8),
        (10, 8, 8, 2, 3, 3, 5),
        (12, 6, 4, 4, 4, 4, 4),
    ]
    # split_*.solve_exact / solve_approx(n0, a_target, l_target, c00, c01, c10, c11)
    sp_args = [
        (20, 5, 5, 10, 5, 5, 8),   # correct: feasible (x11=0); buggy solve_exact: None
        (10, 8, 8, 2, 3, 3, 5),
        (14, 7, 7, 5, 5, 5, 5),
    ]
    cases: list[dict] = []
    for a in eq_args:
        out = bal._solve_cross0_counts(*a)
        cases.append(
            {
                "solver": "balance_sequences._solve_cross0_counts",
                "args": list(a),                                   # (n1, a1_pos, l1_pos, c00..c11)
                "derived": [a[0], a[0] - a[1], a[0] - a[2], *a[3:]],  # (n0, a_target, l_target, c00..c11)
                "out": list(out) if out is not None else None,
            }
        )
    for a in sp_args:
        ex = split.solve_exact(*a)
        ap = split.solve_approx(*a)
        cases.append(
            {"solver": "split.solve_exact", "args": list(a), "out": list(ex) if ex is not None else None}
        )
        cases.append(
            {"solver": "split.solve_approx", "args": list(a), "out": list(ap) if ap is not None else None}
        )
    return cases


def main() -> None:
    sys.path.insert(0, str(OLD_SCRIPTS))
    import balance_sequences as bal  # noqa: E402  (path injected above)
    import split_balance_sequences_all as split  # noqa: E402

    data = _make_data()
    n = len(data)

    equal_sel = bal.balance_dataset(data, seed=BAL_SEED)
    ratio_30 = split.balance_split(data, list(range(n)), cross_pos_ratio=0.30, seed=BAL_SEED)
    ratio_50 = split.balance_split(data, list(range(n)), cross_pos_ratio=0.50, seed=BAL_SEED)
    tr, va, te = split.split_indices(n, 0.75, 0.10, BAL_SEED)  # test_pct is the remainder

    fixture = {
        "data": data,
        "seed": BAL_SEED,
        "tol": 0,
        "cases": {
            "balance_equal": {"selected": list(equal_sel), "summary": bal.summarize(data, equal_sel)},
            "balance_ratio_30_70": {
                "cross_pos_ratio": 0.30,
                "selected": list(ratio_30),
                "summary": split.summarize(data, ratio_30),
            },
            "balance_ratio_50_50": {
                "cross_pos_ratio": 0.50,
                "selected": list(ratio_50),
                "summary": split.summarize(data, ratio_50),
            },
            "split_indices": {
                "train_pct": 0.75,
                "val_pct": 0.10,
                "test_pct": 0.15,
                "train": list(tr),
                "val": list(va),
                "test": list(te),
            },
        },
        "solver_cases": _solver_cases(bal, split),
        "meta": {
            "src_equal": "scripts/balance_sequences.py::balance_dataset",
            "src_ratio": "scripts/split_balance_sequences_all.py::balance_split",
            "n_records": n,
            "gen_seed": GEN_SEED,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2)

    print(f"wrote {OUT}")
    print(f"  data n={n}")
    print(f"  balance_equal:        {len(equal_sel)} selected | {fixture['cases']['balance_equal']['summary']}")
    print(f"  balance_ratio_30_70:  {len(ratio_30)} selected | {fixture['cases']['balance_ratio_30_70']['summary']}")
    print(f"  balance_ratio_50_50:  {len(ratio_50)} selected | {fixture['cases']['balance_ratio_50_50']['summary']}")
    print(f"  split_indices:        train={len(tr)} val={len(va)} test={len(te)}")
    print(f"  solver_cases:         {len(fixture['solver_cases'])}")


if __name__ == "__main__":
    main()

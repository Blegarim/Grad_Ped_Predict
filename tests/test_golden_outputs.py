"""Golden-output coverage gate (Prompt 8.1, B12).

These are *meta-tests* over ``tests/_golden.GOLDEN_MANIFEST`` — they do not re-run per-module parity
(those assertions live co-located with each module's test, e.g. ``test_model_shapes`` /
``test_losses``). Instead they guarantee the safety net itself stays intact:

  1. every registered fixture exists on disk;
  2. no fixture sits in ``fixtures/golden/`` unregistered (orphan / forgotten capture);
  3. every fixture with an in-repo regenerator has that ``_capture/`` script present;
  4. every fixture is actually referenced by at least one test module (i.e. it guards something).

Together these make "a module was ported without a golden guard" a failing test, which is the
structural intent of B12 beyond merely "tests exist".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._golden import CAPTURE_DIR, GOLDEN_DIR, GOLDEN_MANIFEST

_TESTS_DIR = Path(__file__).resolve().parent
# Test modules that may legitimately not reference any fixture by name.
_NON_FIXTURE_TESTS = {"test_golden_outputs.py", "conftest.py", "_golden.py"}


@pytest.mark.parametrize("name", sorted(GOLDEN_MANIFEST), ids=str)
def test_manifest_fixture_present(name: str) -> None:
    """Every registered fixture must exist (the captured reference is committed)."""
    spec = GOLDEN_MANIFEST[name]
    path = GOLDEN_DIR / spec.fixture
    assert path.exists(), f"missing golden fixture {path} — guards: {spec.note}"


def test_no_orphan_golden_fixtures() -> None:
    """Every file in fixtures/golden/ must be registered (no forgotten / stale captures)."""
    on_disk = {p.name for p in GOLDEN_DIR.iterdir() if p.is_file()}
    registered = {spec.fixture for spec in GOLDEN_MANIFEST.values()}
    orphans = on_disk - registered
    assert not orphans, f"unregistered fixtures in {GOLDEN_DIR}: {sorted(orphans)} (add to GOLDEN_MANIFEST)"


@pytest.mark.parametrize("name", sorted(GOLDEN_MANIFEST), ids=str)
def test_capture_script_present(name: str) -> None:
    """A fixture that declares a regenerator must ship it (None = captured offline, allowed)."""
    spec = GOLDEN_MANIFEST[name]
    if spec.capture is None:
        pytest.skip(f"{name} captured offline (no in-repo regenerator by design)")
    script = CAPTURE_DIR / spec.capture
    assert script.exists(), f"missing capture script {script} for fixture {spec.fixture}"


@pytest.mark.parametrize("name", sorted(GOLDEN_MANIFEST), ids=str)
def test_fixture_is_referenced_by_a_test(name: str) -> None:
    """Each fixture must be loaded by at least one test module — i.e. it actually guards code."""
    fixture = GOLDEN_MANIFEST[name].fixture
    referencing = [
        p.name
        for p in _TESTS_DIR.glob("test_*.py")
        if p.name not in _NON_FIXTURE_TESTS and fixture in p.read_text(encoding="utf-8")
    ]
    assert referencing, f"fixture {fixture} ({name}) is registered but no test references it"

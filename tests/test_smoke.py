"""Scaffold smoke test (Prompt 0.1) — the CI gate's first green test (B12 baseline).

No numerical golden fixtures this phase; those land in Prompt 8.1. These tests only
prove the package layout imports cleanly and the interpreter floor is met.
"""

import importlib
import sys

import pedpredict

_SUBPACKAGES = (
    "config",
    "utils",
    "data",
    "models",
    "losses",
    "training",
    "eval",
    "viz",
    "export",
)


def test_package_imports() -> None:
    """Top-level package imports and exposes its version marker."""
    assert pedpredict.__version__ == "0.0.0"


def test_subpackages_import() -> None:
    """All nine subpackages import (catches a missing __init__ / bad pyproject)."""
    for name in _SUBPACKAGES:
        module = importlib.import_module(f"pedpredict.{name}")
        assert module is not None


def test_python_version() -> None:
    """Interpreter meets the numpy 2.2 / torch 2.7 floor."""
    assert sys.version_info >= (3, 10)

"""Pytest rootdir/pathing anchor.

The presence of this file fixes pytest's ``rootdir`` to the repository root so
``pyproject.toml`` config is picked up regardless of the invoking cwd. Tests
import the *installed* ``pedpredict`` package (``pip install -e .``); we
deliberately do not inject ``src/`` onto ``sys.path`` here, so the smoke test
also proves the src-layout editable install works.
"""

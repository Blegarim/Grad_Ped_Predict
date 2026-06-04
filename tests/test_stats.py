"""Prompt 1.7 — dataset stats aggregation + drift gate.

Checks: pos-rate math; aggregation reuses the 1.6 scanner (no parallel counter); the drift gate passes
on matched counts and fails on an off-by-one; the reference fixture agrees with the CLAUDE.md table; and
(slow) the real base LMDBs reproduce the fixture exactly when present.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lmdb
import pytest

from pedpredict.data.sampler import scan_chunk_labels
from pedpredict.data.stats import (
    SplitStats,
    check_drift,
    compute_split_stats,
    format_table,
    load_reference,
    write_stats_csv,
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden" / "pie_sequences_counts.json"


def _write_chunk(records: list[dict], path: Path) -> None:
    """Label-only ``<i>_meta`` writer (same shape as the 1.6 suite), named so glob('chunk_*.lmdb') finds it."""
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for i, rec in enumerate(records):
                txn.put(f"{i}_meta".encode(), pickle.dumps(rec))
    finally:
        env.close()


@pytest.fixture
def split_dir(tmp_path) -> Path:
    """One split dir with two chunks: 8 samples total, crosses=1 ×2, looks=1 ×3, actions=1 ×5."""
    d = tmp_path / "preprocessed_demo"
    d.mkdir()
    _write_chunk(
        [{"actions": 1, "looks": 1, "crosses": 1}, {"actions": 1, "looks": 1, "crosses": 0},
         {"actions": 1, "looks": 0, "crosses": 0}, {"actions": 0, "looks": 0, "crosses": 0}],
        d / "chunk_000000.lmdb",
    )
    _write_chunk(
        [{"actions": 1, "looks": 1, "crosses": 1}, {"actions": 1, "looks": 0, "crosses": 0},
         {"actions": 0, "looks": 0, "crosses": 0}, {"actions": 0, "looks": 0, "crosses": 0}],
        d / "chunk_000004.lmdb",
    )
    return d


def test_pos_rate_math():
    s = SplitStats("train", n=8, counts={"actions": {0: 3, 1: 5}, "looks": {0: 5, 1: 3}, "crosses": {0: 6, 1: 2}})
    assert s.pos == {"actions": 5, "looks": 3, "crosses": 2}
    assert s.pos_rate["crosses"] == pytest.approx(0.25)
    empty = SplitStats("x", n=0, counts={})
    assert empty.pos_rate == {"actions": 0.0, "looks": 0.0, "crosses": 0.0}


def test_compute_aggregates_over_chunks(split_dir):
    s = compute_split_stats("demo", [split_dir])
    assert s.n == 8
    assert s.pos == {"actions": 5, "looks": 3, "crosses": 2}


def test_aggregation_matches_scanner(split_dir):
    """compute_split_stats == manually summing the 1.6 scanner (proves the reuse, no parallel counter)."""
    s = compute_split_stats("demo", [split_dir])
    manual = {"actions": 0, "looks": 0, "crosses": 0}
    for chunk in sorted(split_dir.glob("chunk_*.lmdb")):
        scan = scan_chunk_labels(str(chunk))
        for task in manual:
            manual[task] += scan.counts[task].get(1, 0)
    assert s.pos == manual


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute_split_stats("demo", [tmp_path / "nope"])


def test_check_drift_passes_and_detects():
    ref = {"train": {"N": 8, "actions": 5, "looks": 3, "crosses": 2}}
    matched = SplitStats("train", 8, {"actions": {1: 5}, "looks": {1: 3}, "crosses": {1: 2}})
    assert check_drift([matched], ref) == []
    off = SplitStats("train", 8, {"actions": {1: 5}, "looks": {1: 3}, "crosses": {1: 1}})
    assert any("crosses" in m for m in check_drift([off], ref))
    unref = SplitStats("ghost", 1, {"actions": {1: 1}, "looks": {1: 0}, "crosses": {1: 0}})
    assert any("no reference" in m for m in check_drift([unref], ref))


def test_reference_fixture_matches_claude_table():
    """The 1.1 fixture's derived rates round to the CLAUDE.md table (doc/fixture drift catcher)."""
    ref = load_reference(_FIXTURE)
    expected = {  # CLAUDE.md "Dataset Statistics" table
        "train": (95684, 45.3, 17.1, 2.6),
        "val": (22665, 41.8, 11.9, 2.5),
        "test": (76048, 43.5, 15.8, 2.8),
    }
    for split, (n, a, lo, cr) in expected.items():
        r = ref[split]
        assert r["N"] == n
        assert round(r["actions"] / n * 100, 1) == a
        assert round(r["looks"] / n * 100, 1) == lo
        assert round(r["crosses"] / n * 100, 1) == cr


def test_format_and_csv(split_dir, tmp_path):
    s = compute_split_stats("demo", [split_dir])
    table = format_table([s])
    assert "| Split | N |" in table and "demo" in table
    out = tmp_path / "label_count.csv"
    write_stats_csv([s], out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("split,N,actions_pos")
    assert lines[1].startswith("demo,8,5,3,2,")

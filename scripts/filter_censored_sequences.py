#!/usr/bin/env python
"""Keep ONLY the M4-censored windows from a ``data.emit_censored`` sequence pkl.

``data.emit_censored`` keeps censored windows *in addition* to the ordinary ones, so a generation pass with
that flag produces the union. The censored-window experiment wants them in their OWN LMDB dir, added to
``paths.lmdb_train`` alongside the existing (pinned) dirs -- so the union would double-count the ~88k
windows those dirs already hold. This script extracts the difference.

A window is M4-censored exactly when its future was truncated AND no crossing was ever observed:

    future_observed < future_offset + tol   (the future window did not fit)  AND  onset_offset < 0

which is the same rule ``data/onset_stats.is_usable`` states, and the same condition
``pie_sequences.window_track`` counts as ``censored`` / ``censored_emitted``. Determined positives
(``onset_offset >= 0`` inside a truncated remainder) are NOT censored and are deliberately excluded --
they carry a real label and belong with the ordinary windows.

Usage:
    python scripts/filter_censored_sequences.py IN.pkl OUT.pkl --future-offset 30 --tol 2 [--expect 7470]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pedpredict.data.pie_sequences import load_sequences, save_sequences


def is_censored(record: dict, horizon: int) -> bool:
    """True when the window's future was truncated and no crossing was seen in what remained."""
    return int(record["future_observed"]) < horizon and int(record["onset_offset"]) < 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="sequence pkl generated with data.emit_censored=true")
    ap.add_argument("dst", type=Path, help="output pkl holding only the censored windows")
    ap.add_argument("--future-offset", type=int, required=True, help="data.future_offset used to generate")
    ap.add_argument("--tol", type=int, required=True, help="data.tol used to generate")
    ap.add_argument("--expect", type=int, default=0, help="fail unless this many are kept (0 = no check)")
    args = ap.parse_args(argv)

    horizon = args.future_offset + args.tol
    records = load_sequences(args.src)
    kept = [r for r in records if is_censored(r, horizon)]

    missing = [f for f in ("future_observed", "onset_offset") if records and f not in records[0]]
    if missing:
        print(f"ERROR: {args.src} has no {missing} -- it predates S1 and cannot be filtered", file=sys.stderr)
        return 2
    if not kept:
        print(f"ERROR: no censored windows in {args.src}; was it generated with data.emit_censored=true?",
              file=sys.stderr)
        return 2
    if args.expect and len(kept) != args.expect:
        print(f"ERROR: kept {len(kept)} censored windows, expected {args.expect}", file=sys.stderr)
        return 1

    # Every kept window must be unusable at the generation horizon -- that is the whole point of the arm.
    assert all(is_censored(r, horizon) for r in kept)
    assert all(int(r["crosses"]) == 0 for r in kept), "a censored window must carry the placeholder 0"

    save_sequences(kept, args.dst)
    print(f"[filter_censored] {len(records)} -> {len(kept)} censored windows (horizon {horizon}) -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

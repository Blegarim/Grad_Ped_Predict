"""Backfill the S1 onset keys into already-built LMDB chunks (metadata-only, no rebuild).

Thin wrapper: load config -> resolve the split's chunk dir + ``<sequences_dir>/sequences_<split>.pkl``
-> write ``onset_offset`` / ``future_observed`` / ``track_crosses`` into every ``{j}_meta``. Image blobs
are never read or rewritten. Idempotent — re-running reports the keys as already present.

Every sample is verified against its record (``track_id`` + ``crosses``) before anything is written, so a
mismatched pkl aborts instead of corrupting. Run ``--dry-run`` first: it does the full read and the full
verification and writes nothing.

    python scripts/backfill_onset_meta.py --split train --dry-run
    python scripts/backfill_onset_meta.py --split train
    python scripts/backfill_onset_meta.py --split val --split test
    python scripts/backfill_onset_meta.py --split test_benchmark      # M5 anchored chunks

Offline-augmented dirs (``preprocessed_train_aug``) cannot be backfilled — augmentation breaks the
positional sample-to-record map, and the verification will say so. Backfill the base train dir, then
re-run ``scripts/augment_dataset.py``; the copies pick the keys up through the normal write path.
"""

from __future__ import annotations

from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.onset_backfill import backfill_dir, format_reports
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.paths import resolve_paths

#: split name -> (attribute on ResolvedPaths, sequence-pkl stem). Train resolves to the BASE dir only
#: (``lmdb_train[0]``); the augmented sibling is out of scope, see the module docstring.
_SPLIT_DIRS: dict[str, tuple[str, str]] = {
    "train": ("lmdb_train", "sequences_train"),
    "val": ("lmdb_val", "sequences_val"),
    "test": ("lmdb_test", "sequences_test"),
    "train_benchmark": ("lmdb_train_benchmark", "sequences_train_benchmark"),
    "val_benchmark": ("lmdb_val_benchmark", "sequences_val_benchmark"),
    "test_benchmark": ("lmdb_test_benchmark", "sequences_test_benchmark"),
}


ONSET_MISSING_HINT = (
    "{pkl} has no onset fields — it predates S1. Re-run scripts/make_sequences.py to regenerate it "
    "(the fields are pure per-window functions; regeneration does not change labels or window count)."
)


def _resolve_split(paths, split: str) -> tuple[Path, Path]:
    """(chunk dir, sequence pkl) for one split name."""
    attr, stem = _SPLIT_DIRS[split]
    value = getattr(paths, attr)
    chunk_dir = value[0] if isinstance(value, tuple) else value
    return chunk_dir, paths.sequences_dir / f"{stem}.pkl"


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", action="append", choices=sorted(_SPLIT_DIRS),
                        help="Split to backfill; repeatable. Default: train, val, test.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Read and verify every sample, write nothing.")
    parser.add_argument("--chunk-dir", default=None,
                        help="Override the resolved chunk dir (requires exactly one --split).")
    parser.add_argument("--pkl", default=None,
                        help="Override the resolved sequence pkl (requires exactly one --split).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    splits = args.split or ["train", "val", "test"]
    if (args.chunk_dir or args.pkl) and len(splits) != 1:
        raise SystemExit("--chunk-dir/--pkl override a single split; pass exactly one --split.")

    for split in splits:
        chunk_dir, pkl = _resolve_split(paths, split)
        chunk_dir = Path(args.chunk_dir) if args.chunk_dir else chunk_dir
        pkl = Path(args.pkl) if args.pkl else pkl
        if not chunk_dir.is_dir():
            raise SystemExit(f"Missing LMDB dir {chunk_dir} — build the split first.")
        if not pkl.is_file():
            raise SystemExit(f"Missing sequence pkl {pkl} — run scripts/make_sequences.py first.")

        records = load_sequences(pkl)
        missing = ONSET_MISSING_HINT if not records or "onset_offset" not in records[0] else None
        if missing:
            raise SystemExit(missing.format(pkl=pkl))
        reports = backfill_dir(chunk_dir, records, dry_run=args.dry_run)
        print(f"[backfill] {split}: {chunk_dir}  ({len(records)} records in {pkl.name})")
        print(format_reports(reports, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

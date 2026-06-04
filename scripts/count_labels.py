"""Per-split label statistics + drift gate (Prompt 1.7 CLI; replaces OLD ``label_count.py``).

Thin wrapper: load config -> resolve paths -> scan the base split LMDBs (1.6 scanner) -> print the
canonical table -> write ``<log_dir>/label_count.csv`` -> diff against the documented stat table and
exit nonzero on drift (CI-friendly). Missing LMDBs are skipped (exit 0) so CI without built data stays
green; the real gate lives in ``tests/test_stats.py`` (slow).

    python scripts/count_labels.py                 # base LMDBs, print + CSV + drift gate
    python scripts/count_labels.py --include-aug    # include preprocessed_train_aug (skips the gate)
    python scripts/count_labels.py --no-check       # report only, no drift gate
"""

from __future__ import annotations

from pedpredict.config import build_argparser, load_config
from pedpredict.data.stats import (
    check_drift,
    compute_dataset_stats,
    format_table,
    load_reference,
    write_stats_csv,
)
from pedpredict.paths import find_project_root, resolve_paths

_REFERENCE = "tests/fixtures/golden/pie_sequences_counts.json"


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--include-aug", action="store_true",
                        help="include preprocessed_train_aug in train (changes the distribution; skips drift gate)")
    parser.add_argument("--no-check", action="store_true", help="report only; skip the drift gate")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)

    stats = compute_dataset_stats(paths, include_aug=args.include_aug, skip_missing=True)
    if not stats:
        print("No LMDB chunks found — nothing to count (build them with scripts/build_lmdb.py).")
        return
    print(format_table(stats))

    csv_path = paths.log_dir / "label_count.csv"
    write_stats_csv(stats, csv_path)
    print(f"\nWrote {csv_path}")

    if args.no_check or args.include_aug:
        return
    reference = load_reference(find_project_root() / _REFERENCE)
    drift = check_drift(stats, reference)
    if drift:
        raise SystemExit("Label-count drift vs documented table:\n  " + "\n  ".join(drift))
    print("No drift vs documented stat table.")


if __name__ == "__main__":
    main()

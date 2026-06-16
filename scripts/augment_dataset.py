"""Build the offline minority-class augmentation LMDB from the built base-train LMDB.

Thin wrapper: load config -> read the ``chunk_*.lmdb`` under ``preprocessed_train`` (``paths.lmdb_train[0]``)
-> oversample minority records into single-transform copies -> write ``preprocessed_train_aug``
(``paths.lmdb_train[1]``), unioned with ``preprocessed_train`` at train time. Source crops come from the
built LMDB, NOT the PIE frames — so this runs AFTER ``build_lmdb_incremental.py`` (which deletes those
frames). All augmentation params flow from ``AugmentCfg`` (no literals here).

    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --in-dir preprocessed_train_balanced   # augment a balanced base
    python scripts/augment_dataset.py --set augment.crosses_multiplier=8
"""

from __future__ import annotations

from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.augment import augment_lmdb_dir
from pedpredict.paths import resolve_paths


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--in-dir", default=None,
                        help="Base train LMDB dir to augment from (default: lmdb_train[0]).")
    parser.add_argument("--out-dir", default=None, help="Output LMDB dir (default: lmdb_train[1]).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    if not cfg.augment.enabled:
        raise SystemExit(
            "augment.enabled is false - offline augmentation is opt-out here. "
            "Re-run with '--set augment.enabled=true' to confirm."
        )

    paths = resolve_paths(cfg.paths)
    in_dir = Path(args.in_dir) if args.in_dir else paths.lmdb_train[0]
    out_dir = Path(args.out_dir) if args.out_dir else paths.lmdb_train[1]
    if not in_dir.is_dir():
        raise SystemExit(
            f"Missing base train LMDB {in_dir} — build it first "
            f"(scripts/build_lmdb_incremental.py --split train)."
        )

    chunks = augment_lmdb_dir(in_dir, out_dir, cfg.data, cfg.augment)
    print(f"[augment] wrote {len(chunks)} chunk(s)")


if __name__ == "__main__":
    main()

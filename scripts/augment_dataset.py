"""Build the offline minority-class augmentation LMDB.

Thin wrapper: load config -> read ``<sequences_dir>/sequences_train.pkl`` -> oversample minority
records into single-transform copies -> write ``preprocessed_train_aug`` (``paths.lmdb_train[1]``).
This LMDB is unioned with ``preprocessed_train`` at train time (the legacy imbalance policy). All
augmentation params flow from ``AugmentCfg`` (no literals here).

    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --pkl data/sequences/sequences_train_balanced.pkl
    python scripts/augment_dataset.py --set augment.crosses_multiplier=8
"""

from __future__ import annotations

from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.augment import AugmentedCropSequenceDataset, plan_oversample, summarize_plan
from pedpredict.data.lmdb_writer import write_dataset_chunks_from
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.paths import resolve_paths


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--pkl", default=None, help="Input train sequence pkl (default: sequences_train.pkl).")
    parser.add_argument("--out-dir", default=None, help="Output LMDB dir (default: lmdb_train[1]).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    if not cfg.augment.enabled:
        raise SystemExit(
            "augment.enabled is false - offline augmentation is opt-out here. "
            "Re-run with '--set augment.enabled=true' to confirm."
        )

    paths = resolve_paths(cfg.paths)
    pkl = Path(args.pkl) if args.pkl else paths.sequences_dir / "sequences_train.pkl"
    out_dir = Path(args.out_dir) if args.out_dir else paths.lmdb_train[1]
    if not pkl.exists():
        raise SystemExit(f"Missing {pkl}")

    records = load_sequences(pkl)
    items = plan_oversample(records, cfg.augment)
    print(f"[augment] {pkl.name} -> {out_dir} | plan: {summarize_plan(records, items)}")
    dataset = AugmentedCropSequenceDataset(records, items, cfg.data, cfg.augment)
    chunks = write_dataset_chunks_from(dataset, out_dir, cfg.data)
    print(f"[augment] wrote {len(chunks)} chunk(s)")


if __name__ == "__main__":
    main()

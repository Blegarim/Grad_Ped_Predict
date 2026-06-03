"""Build per-split LMDB chunk datasets from sequence pickles (Prompt 1.2 CLI).

Thin wrapper: load config -> resolve paths -> read ``<sequences_dir>/sequences_<split>.pkl`` ->
``write_dataset_chunks`` into the split's configured LMDB dir. All geometry/encode params flow from
``DataCfg`` (no literals here).

    python scripts/build_lmdb.py --split val
    python scripts/build_lmdb.py --split all --set data.context_scale=3.0
    python scripts/build_lmdb.py --split train --pkl data/sequences/sequences_train_aug.pkl \
        --out-dir preprocessed_train_aug
"""

from __future__ import annotations

from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.lmdb_writer import write_dataset_chunks
from pedpredict.data.pie_sequences import load_sequences
from pedpredict.paths import ResolvedPaths, resolve_paths

_SPLITS = ("train", "val", "test")


def _out_dir_for(split: str, paths: ResolvedPaths) -> Path:
    """Map a split to its configured LMDB dir (train -> first dir; balanced/aug variants in 1.3/1.4)."""
    if split == "train":
        return paths.lmdb_train[0]
    return paths.lmdb_val if split == "val" else paths.lmdb_test


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", choices=(*_SPLITS, "all"), default="all")
    parser.add_argument("--pkl", default=None, help="Override the input sequence pickle path.")
    parser.add_argument("--out-dir", default=None, help="Override the output LMDB directory.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)

    splits = _SPLITS if args.split == "all" else (args.split,)
    if (args.pkl or args.out_dir) and len(splits) != 1:
        raise SystemExit("--pkl / --out-dir require a single --split")

    for split in splits:
        pkl = Path(args.pkl) if args.pkl else paths.sequences_dir / f"sequences_{split}.pkl"
        out_dir = Path(args.out_dir) if args.out_dir else _out_dir_for(split, paths)
        records = load_sequences(pkl)
        chunks = write_dataset_chunks(records, out_dir, cfg.data)
        print(f"[{split}] {len(records)} sequences -> {len(chunks)} chunk(s) in {out_dir}")


if __name__ == "__main__":
    main()

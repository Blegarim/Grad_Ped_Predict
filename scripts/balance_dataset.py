"""Offline-balance sequence pickles.

Thin wrapper: load config -> for each split, balance ``<sequences_dir>/sequences_<split>.pkl`` ->
write ``<sequences_dir>/sequences_<split>_balanced.pkl``. All balance params flow from
``BalanceCfg`` (no literals here). This is the OPT-IN majority-downsample lever; the balanced pkl is
an ordinary sequence pkl that feeds the LMDB writer (1.2) like any other.

    python scripts/balance_dataset.py --split train --set balance.enabled=true
    python scripts/balance_dataset.py --split all --set balance.enabled=true --set balance.cross_pos_ratio=0.25
"""

from __future__ import annotations

from pedpredict.config import build_argparser, load_config
from pedpredict.data.balance import balance_sequence_file
from pedpredict.paths import resolve_paths

_SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", choices=(*_SPLITS, "all"), default="all")
    parser.add_argument("--suffix", default="_balanced", help="appended to the input pkl stem")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    if not cfg.balance.enabled:
        raise SystemExit(
            "balance.enabled is false - offline balancing is opt-in. "
            "Re-run with '--set balance.enabled=true' to confirm."
        )

    paths = resolve_paths(cfg.paths)
    splits = _SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        in_path = paths.sequences_dir / f"sequences_{split}.pkl"
        if not in_path.exists():
            print(f"[{split}] MISSING {in_path} — skipped")
            continue
        out_path = paths.sequences_dir / f"sequences_{split}{args.suffix}.pkl"
        summary = balance_sequence_file(in_path, out_path, cfg.balance)
        print(f"[{split}] {in_path.name} -> {out_path.name} | {summary}")


if __name__ == "__main__":
    main()

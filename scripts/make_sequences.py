"""Generate PIE sequence-window pickles.

Thin wrapper: load config -> build a PIE instance rooted at ``configs/paths.yaml`` ``pie_root`` ->
window each split -> pickle to ``<sequences_dir>/sequences_<split>.pkl``. All window/source params
flow from ``DataCfg`` (no literals here).

    python scripts/make_sequences.py --split val
    python scripts/make_sequences.py --split all --set data.stride=5
"""

from __future__ import annotations

import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.pie_sequences import generate_sequences, save_sequences
from pedpredict.paths import find_project_root, resolve_paths

_SPLITS = ("train", "val", "test")


def _build_pie(pie_root: Path):
    """Import the PIE toolkit (cloned into this repo root) and construct an instance at ``pie_root``."""
    root = find_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from PIE.utilities.pie_data import PIE
    except ImportError as exc:  # PIE toolkit not cloned yet
        raise SystemExit(
            "PIE toolkit not found. Clone it into the repo root so 'PIE/utilities/pie_data.py' is "
            f"importable, and ensure the dataset lives at '{pie_root}'."
        ) from exc
    return PIE(data_path=str(pie_root))


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--split", choices=(*_SPLITS, "all"), default="all")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    imdb = _build_pie(paths.pie_root)

    splits = _SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        records = generate_sequences(imdb, split, cfg.data)
        out = save_sequences(records, paths.sequences_dir / f"sequences_{split}.pkl")
        print(f"[{split}] {len(records)} windows -> {out}")


if __name__ == "__main__":
    main()

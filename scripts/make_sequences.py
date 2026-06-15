"""Generate PIE sequence-window pickles (v2 labeling contract).

Thin wrapper: load config -> build a PIE instance rooted at ``configs/paths.yaml`` ``pie_root`` ->
window each split -> pickle to ``<sequences_dir>/sequences_<split>.pkl``. All window/source params
flow from ``DataCfg`` (no literals here).

Each standard split also writes ``sequences_<split>_stats.json`` — the M4 windowing accounting
(emitted / right-censored / observation-crossing / short-track counts). The ``censored`` number is
the thesis-reportable "N windows excluded as right-censored".

``--benchmark`` generates the M5 benchmark-protocol eval set instead (test split only by default):
fixed-TTE windows labeled by the crossing event -> ``sequences_test_benchmark.pkl``.

    python scripts/make_sequences.py --split all
    python scripts/make_sequences.py --split val --set data.stride=5
    python scripts/make_sequences.py --benchmark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.pie_sequences import WindowStats, generate_sequences, save_sequences
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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Generate the M5 benchmark-protocol eval windows (TTE-sampled, event-labeled) instead "
        "of the standard sliding windows. Default split: test.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    imdb = _build_pie(paths.pie_root)

    if args.benchmark:
        splits = ("test",) if args.split == "all" else (args.split,)
        for split in splits:
            records = generate_sequences(imdb, split, cfg.data, mode="benchmark")
            out = save_sequences(records, paths.sequences_dir / f"sequences_{split}_benchmark.pkl")
            n_pos = sum(r["crosses"] for r in records)
            print(f"[{split}/benchmark] {len(records)} windows ({n_pos} event-positive) -> {out}")
        return

    splits = _SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        stats = WindowStats()
        records = generate_sequences(imdb, split, cfg.data, stats=stats)
        out = save_sequences(records, paths.sequences_dir / f"sequences_{split}.pkl")
        stats_path = out.with_name(f"sequences_{split}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(stats.as_dict(), handle, indent=2)
        print(
            f"[{split}] {len(records)} windows -> {out}\n"
            f"[{split}] windowing: {stats.emitted} emitted | {stats.censored} right-censored (M4) | "
            f"{stats.obs_crossing} obs-crossing (filter #2) | {stats.short_tracks} short tracks "
            f"-> {stats_path.name}"
        )


if __name__ == "__main__":
    main()

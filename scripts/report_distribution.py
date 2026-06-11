"""Report the effective training distribution (the M1 instrument) without training.

Computes, per task, the positive fraction an epoch of ``WeightedRandomSampler`` draws is expected
to contain over the configured train chunks, next to the stored base rate — so any lever
combination (``--set train.use_weighted_sampler=...`` / ``train.sampler_powers=...`` /
``train.use_class_weights=...``) can be inspected before a run burns GPU hours. The same report is
written automatically into every run dir as ``train_distribution.json`` by ``scripts/train.py``.

Usage:
    python scripts/report_distribution.py
    python scripts/report_distribution.py --set train.use_weighted_sampler=false
    python scripts/report_distribution.py --out dist.json
"""

from __future__ import annotations

import json
import sys

from pedpredict.config import build_argparser, load_config
from pedpredict.paths import resolve_paths
from pedpredict.training.chunk_loader import gather_lmdb_chunks
from pedpredict.training.distribution import effective_distribution


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--out", default=None, help="Also write the report json to this path.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)

    resolved = resolve_paths(cfg.paths)
    train_paths = gather_lmdb_chunks(resolved.lmdb_train)
    report = effective_distribution(train_paths, cfg.train)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SCRATCH (not committed): sweep sampler levers against the M1 instrument with ONE disk scan.

Scans the train LMDB metadata once into a shared LabelScanCache, then evaluates the expected
per-task effective positive rate for many (use_weighted_sampler, sampler_powers) combos in memory.
Goal: find a config whose effective `crosses` rate lands ~20-30% (vs the current 89%).
"""

from __future__ import annotations

import dataclasses

from pedpredict.config import build_argparser, load_config
from pedpredict.data.sampler import LabelScanCache
from pedpredict.paths import resolve_paths
from pedpredict.training.chunk_loader import gather_lmdb_chunks
from pedpredict.training.distribution import effective_distribution


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = load_config(args.config_dir, args.overrides)
    resolved = resolve_paths(cfg.paths)
    train_paths = gather_lmdb_chunks(resolved.lmdb_train)

    cache = LabelScanCache()  # scanned once, reused for every combo below
    base = cfg.train

    # (label, use_weighted_sampler, crosses_pow, actions_pow, looks_pow)
    combos = [
        ("sampler OFF (=base rate)",      False, 0.0,  0.0, 0.0),
        ("current default",               True,  1.5,  0.3, 0.7),
        ("crosses^1.0",                   True,  1.0,  0.3, 0.7),
        ("crosses^0.75",                  True,  0.75, 0.3, 0.7),
        ("crosses^0.5",                   True,  0.5,  0.3, 0.7),
        ("crosses^0.5 looks^0.3",         True,  0.5,  0.3, 0.3),
        ("crosses^0.35",                  True,  0.35, 0.2, 0.3),
        ("crosses^0.25",                  True,  0.25, 0.2, 0.3),
        ("crosses^0.5 actions/looks off", True,  0.5,  0.0, 0.0),
    ]

    print(f"{'config':<32} {'crosses':>9} {'looks':>9} {'actions':>9}")
    print("-" * 64)
    base_printed = False
    for label, use_sampler, cp, ap, lp in combos:
        train_cfg = dataclasses.replace(
            base,
            use_weighted_sampler=use_sampler,
            sampler_powers={"crosses": cp, "actions": ap, "looks": lp},
        )
        rep = effective_distribution(train_paths, train_cfg, scan_cache=cache)
        eff = rep["effective_rate"]
        if not base_printed:
            b = rep["base_rate"]
            print(f"{'(stored base rate)':<32} {b['crosses']:>8.1%} {b['looks']:>8.1%} {b['actions']:>8.1%}")
            print("-" * 64)
            base_printed = True
        print(f"{label:<32} {eff['crosses']:>8.1%} {eff['looks']:>8.1%} {eff['actions']:>8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Refuse a run whose config differs from the reference run by anything but its intended flags.

Run BEFORE training, with the exact ``--set`` flags the run will get (the candidate is resolved through the
same ``load_config`` path train.py uses, so config validation errors surface here too), or AFTER, on a
run dir's snapshot via ``--candidate``:

    python scripts/check_run_config.py --reference outputs/runs/<ref>/resolved_config.yaml \\
        --allow train.seed --allow model.onset_ --allow train.onset_ --allow train.loss_weight \\
        --set train.seed=43 --set model.onset_head=true ...

``--allow`` takes an exact key, or a prefix ending in ``.`` or ``_``. Exit 0 = only allowed differences
(listed), 1 = an unexpected difference (listed), 2 = the candidate config does not load.
"""

from __future__ import annotations

import argparse
import sys

from pedpredict.config import load_config
from pedpredict.config.diff import config_diff, unexpected_diffs
from pedpredict.config.loader import ConfigError, load_resolved_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Reference run's resolved_config.yaml.")
    parser.add_argument("--candidate", default="", help="Candidate resolved_config.yaml (else built from --set).")
    parser.add_argument("--allow", action="append", default=[], help="Allowed key or prefix (repeatable).")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.field=value")
    args = parser.parse_args(argv)

    reference = load_resolved_config(args.reference)
    try:
        candidate = (load_resolved_config(args.candidate) if args.candidate
                     else load_config(args.config_dir, args.overrides))
    except ConfigError as exc:
        print(f"[config-check] candidate config does not load: {exc}")
        return 2
    diff = config_diff(reference, candidate)
    bad = unexpected_diffs(diff, args.allow)
    for key, (ref, cand) in diff.items():
        print(f"[config-check] {'UNEXPECTED' if key in bad else 'allowed'}  {key}: {ref!r} -> {cand!r}")
    print(f"[config-check] {len(diff)} difference(s), {len(bad)} unexpected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""Rewrite the training LMDB dirs as ONE globally-shuffled dir (fixes block-ordered chunk visits).

Thin wrapper over :mod:`pedpredict.data.reshuffle`. Reads every sample of ``paths.lmdb_train`` (base +
augmented, plus any extra dir you pass), permutes them with ``--seed``, and writes fresh chunks whose
class mix matches the global rate. Image blobs and metas are copied **verbatim** — no decode, no
re-encode, no label change, no Dataset Statistics re-pin. Source dirs are never modified.

Why: the loader does a full pass over one chunk before the next, and the augmented dir holds only
minority records, so ~1 chunk in 3 is nearly all-positive. The model's output bias tracks whichever
chunk ran last, which is the epoch boundary where ``best.pth`` is selected.

    python scripts/reshuffle_train_lmdb.py --dry-run        # enumerate + report, write nothing
    python scripts/reshuffle_train_lmdb.py                  # write preprocessed_train_shuffled
    python scripts/reshuffle_train_lmdb.py --verify-only     # re-scan an existing output dir

After it succeeds, point training at the new dir:

    --set "paths.lmdb_train=[preprocessed_train_shuffled]"
"""

from __future__ import annotations

import json
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.data.reshuffle import (
    chunk_positive_rates,
    enumerate_dir,
    enumerate_dirs,
    plan_shuffle,
    write_shuffled,
)
from pedpredict.paths import resolve_paths

#: A shuffled dir passes when the per-chunk crossing rates span less than this.
#: Sampling noise alone spreads them: at chunk_size 5000 and p ~ 0.146 one chunk's sd is ~0.005, so the
#: range over ~27 chunks runs to ~0.025. This sits at 2x that — loose enough never to fail on a lucky
#: permutation, tight enough that real block structure (which reaches a spread of ~1.0) can never pass.
SPREAD_TOLERANCE = 0.05


def _report(label: str, rates: list[float]) -> tuple[float, float]:
    """Print the per-chunk crossing-rate envelope for one ordering; return (min, max)."""
    lo, hi = min(rates), max(rates)
    print(f"  {label:<10} {len(rates):>3} chunks | rate min {lo:.4f} max {hi:.4f} spread {hi - lo:.4f}")
    return lo, hi


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--out", default="preprocessed_train_shuffled",
                        help="Output dir name, resolved next to the source dirs.")
    parser.add_argument("--seed", type=int, default=42, help="Permutation seed (recorded in the manifest).")
    parser.add_argument("--extra-dir", action="append", default=[],
                        help="Additional source dir to fold in (e.g. preprocessed_train_censored).")
    parser.add_argument("--dry-run", action="store_true", help="Enumerate and report; write nothing.")
    parser.add_argument("--verify-only", action="store_true", help="Re-scan an existing output dir and exit.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    sources = [Path(p) for p in paths.lmdb_train]
    data_root = sources[0].parent
    out_dir = data_root / args.out

    if args.verify_only:
        rates = chunk_positive_rates(enumerate_dir(out_dir), cfg.data.chunk_size)
        print(f"[verify] {out_dir}")
        lo, hi = _report("shuffled", rates)
        raise SystemExit(0 if hi - lo <= SPREAD_TOLERANCE else 1)

    sources += [data_root / d for d in args.extra_dir]
    for src in sources:
        if not src.is_dir():
            raise SystemExit(f"Missing LMDB dir {src} — build it first.")
    print(f"[reshuffle] sources: {', '.join(s.name for s in sources)}")

    refs = enumerate_dirs(sources)
    n_pos = sum(r.crosses for r in refs)
    print(f"[reshuffle] {len(refs)} samples | {n_pos} crossing ({n_pos / len(refs):.4f} global rate)")

    before = chunk_positive_rates(refs, cfg.data.chunk_size)
    shuffled = plan_shuffle(refs, args.seed)
    after = chunk_positive_rates(shuffled, cfg.data.chunk_size)
    print("[reshuffle] per-chunk crossing rate:")
    _report("source", before)
    lo, hi = _report("shuffled", after)
    if hi - lo > SPREAD_TOLERANCE:
        raise SystemExit(f"shuffled spread {hi - lo:.4f} exceeds tolerance {SPREAD_TOLERANCE} — aborting")

    if args.dry_run:
        print("[reshuffle] --dry-run: nothing written")
        return

    print(f"[reshuffle] writing {out_dir} (verbatim blob copy)...")
    written = write_shuffled(shuffled, out_dir, cfg.data)
    manifest = {
        "sources": [s.name for s in sources],
        "seed": args.seed,
        "n_samples": len(refs),
        "global_crosses_rate": n_pos / len(refs),
        "chunk_size": cfg.data.chunk_size,
        "chunks_written": len(written),
        "chunk_crosses_rate": {"min": lo, "max": hi},
    }
    (out_dir / "reshuffle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[reshuffle] wrote {len(written)} chunks -> {out_dir}")
    print(f'[reshuffle] train with: --set "paths.lmdb_train=[{args.out}]"')


if __name__ == "__main__":
    main()

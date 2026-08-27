"""Report what the streaming negatives are made of, and how that moves with the horizon.

Thin wrapper over :mod:`pedpredict.data.onset_stats`. Answers two questions that size the method
work and need no GPU, no LMDB, and no frames:

1. how much of the imbalance is the *hard* case (someone who will cross, just not yet);
2. what a longer anticipation horizon would actually buy — base rate, confusable band, censoring cost.

Two input sources, same numbers:

    # lab PC — the sequence pkls written by make_sequences.py
    python scripts/report_negative_composition.py

    # laptop — straight from PIE's annotation XMLs (~25 MB, no frames needed)
    python scripts/report_negative_composition.py --annotations PIE/annotations/annotations.zip

``--annotations`` re-windows the tracks through ``window_track``, so it reproduces the pkls exactly;
it exists because the XML set fits on a machine that the frames and LMDBs do not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pedpredict.config import build_argparser, load_config
from pedpredict.data.onset_stats import (
    compose,
    format_composition_table,
    format_sweep_table,
    horizon_sweep,
)
from pedpredict.data.pie_sequences import load_sequences, window_track
from pedpredict.paths import resolve_paths

_SPLITS = ("train", "val", "test")
_DEFAULT_HORIZONS = "32,45,60,90,150,300"


def _records_from_annotations(annotations: Path, split: str, cfg) -> list[dict[str, Any]]:
    """Re-window PIE's annotation XMLs into v2 records (labels only — see ``pie_annotations``)."""
    from pedpredict.data.pie_annotations import load_annotation_tracks

    records: list[dict[str, Any]] = []
    for track in load_annotation_tracks(annotations, split):
        records += window_track(track.images, track.bboxes, track.actions, track.looks, track.crosses,
                                cfg, track_id=track.track_id, ego_speed=track.ego_speed)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    parser.add_argument("--annotations", default=None,
                        help="PIE annotations.zip or unpacked dir; re-windows from XML instead of the pkls")
    parser.add_argument("--splits", default=",".join(_SPLITS), help="comma-separated splits to report")
    parser.add_argument("--horizons", default=_DEFAULT_HORIZONS,
                        help="comma-separated horizons (frames) for the sweep; sweep runs on the first split")
    parser.add_argument("--band-frames", type=int, default=60,
                        help="width of the 'confusable band' just past the horizon (frames)")
    parser.add_argument("--out", default=None, help="also write the report as json to this path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, overrides=args.overrides)
    paths = resolve_paths(cfg.paths)
    horizon = cfg.data.future_offset + cfg.data.tol
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    per_split: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        if args.annotations:
            per_split[split] = _records_from_annotations(Path(args.annotations), split, cfg.data)
        else:
            pkl = paths.sequences_dir / f"sequences_{split}.pkl"
            if not pkl.exists():
                print(f"skipping {split}: {pkl} not found (generate it with scripts/make_sequences.py)")
                continue
            per_split[split] = load_sequences(pkl)
    if not per_split:
        print("Nothing to report — pass --annotations, or build the sequence pkls first.")
        return

    compositions = [compose(split, records, horizon) for split, records in per_split.items()]
    print(f"Streaming negative composition at the canonical horizon H={horizon} frames "
          f"({horizon / 30:.2f} s at 30 fps)\n")
    print(format_composition_table(compositions))

    sweep_split = splits[0] if splits[0] in per_split else next(iter(per_split))
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    points = horizon_sweep(per_split[sweep_split], horizons, band_frames=args.band_frames)
    print(f"\nHorizon sweep on {sweep_split} (re-labelled from the same windows)\n")
    print(format_sweep_table(points))

    if args.out:
        report = {"horizon": horizon,
                  "composition": [c.as_dict() for c in compositions],
                  "sweep_split": sweep_split,
                  "sweep": [p.as_dict() for p in points]}
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

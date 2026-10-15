#!/usr/bin/env python
"""Scene sidecar for the deep scene arm (``outputs/diagnostics/deep_scene/PREREG.md``). Research PC (needs the
window exports and the scene audit's ``scene_{split}.npz``, both under ``outputs/features/pose58``):

    python scripts/build_scene_sidecar.py build --groups R,T,S --out outputs/features/scene/scene_rts.npz
    python scripts/build_scene_sidecar.py check --set data.scene_features=outputs/features/scene/scene_rts.npz \\
        --set data.scene_dim=K --set data.motion_dim=58+K --set model.motion_dim=58+K ...   # the arm's flags
    python scripts/build_scene_sidecar.py same-dump --a old/onset_test.npz --b new/onset_test.npz

``build`` encodes the scene features the trees saw (``data.scene_lookup.encode_scene``, statistics fitted on train)
and keys every row by pedestrian + onset fields + last-frame box centre. Gates: each scene file has its export's row
count, and key collisions (PIE pedestrians split into segments) are separable by box, flipped copies included.
``check`` reads every window of the arm's train, val and test dirs through the production read path and fails
unless each one resolves to a scene row and comes out ``data.motion_dim`` wide. ``same-dump`` is the scene-off
reproduction gate: an existing checkpoint re-dumped by the new code must score every window as before.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_EXPORTS = Path("outputs/features/pose58")
_SPLITS = ("train", "val", "test")
_WIDTH, _HEIGHT = 1920.0, 1080.0


def _load_split(exports: Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(keys + last-frame centre from the export, raw scene columns) for one split; row counts must agree."""
    from pedpredict.eval.onset_timing import load_dump

    with np.load(exports / f"{split}.npz", allow_pickle=False) as z:
        last = z["X"][:, -1, :2].astype(np.float64)
        keys = {"track_id": z["track_id"].astype(str), "future_observed": z["future_observed"].astype(np.int64),
                "onset_offset": z["onset_offset"].astype(np.int64),
                "cx": last[:, 0] * _WIDTH, "cy": last[:, 1] * _HEIGHT}
    scene, _ = load_dump(exports / f"scene_{split}.npz")
    n = len(keys["track_id"])
    lengths = {k: len(v) for k, v in scene.items()}
    if any(m != n for m in lengths.values()):
        raise SystemExit(f"[sidecar] {split}: scene rows {set(lengths.values())} != export rows {n} - stopping")
    return keys, scene


def _collisions(keys: dict[str, np.ndarray], tol: float) -> tuple[int, list[str]]:
    """(# keys shared by several windows, the ones whose boxes - or one box's mirror - lie within ``tol``)."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(zip(keys["track_id"], keys["future_observed"], keys["onset_offset"], strict=True)):
        groups[key].append(i)
    shared, bad = 0, []
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        shared += 1
        cx, cy = keys["cx"][rows], keys["cy"][rows]
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                direct = abs(cx[a] - cx[b]) + abs(cy[a] - cy[b])
                mirror = abs(cx[a] - (_WIDTH - cx[b])) + abs(cy[a] - cy[b])
                if min(direct, mirror) <= tol:
                    bad.append(f"{key}: rows {rows[a]}/{rows[b]} {direct:.1f}/{mirror:.1f} px")
    return shared, bad


def cmd_build(args: argparse.Namespace) -> int:
    from pedpredict.data.scene_lookup import encode_scene
    from pedpredict.eval.scene_features import SCENE_GROUPS

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    names = [n for g in groups for n in SCENE_GROUPS[g]]
    parts = [_load_split(args.exports, s) for s in _SPLITS]
    keys = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    raw = {n: np.concatenate([p[1][n] for p in parts]) for n in names}
    sizes = [len(p[0]["track_id"]) for p in parts]
    split = np.concatenate([np.full(n, s) for s, n in zip(_SPLITS, sizes, strict=True)])
    shared, bad = _collisions(keys, tol=4.0)
    print(f"[sidecar] rows {dict(zip(_SPLITS, sizes, strict=True))}; {shared} keys shared by several windows")
    if bad:
        print("[sidecar] AMBIGUOUS collisions (boxes too close to tell apart):\n  " + "\n  ".join(bad[:20]))
        return 1
    block, spec = encode_scene(raw, names, fit_rows=split == "train", clip=args.clip)
    spec.update(groups=groups, source_width=_WIDTH, fit_split="train", rows=dict(zip(_SPLITS, sizes, strict=True)),
                shared_keys=shared, exports=str(args.exports))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=block, split=split, spec=np.asarray(json.dumps(spec)), **keys)
    print(f"[sidecar] wrote {args.out}: {block.shape[0]} rows x {block.shape[1]} columns (data.scene_dim)")
    print(f"[sidecar] columns: {spec['names']}")
    return 0


def _widths(batch: list[dict]) -> list[int]:
    return [int(b["motions"].shape[1]) for b in batch]


def cmd_check(args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader

    from pedpredict.config import load_config
    from pedpredict.data.lmdb_dataset import LMDBChunkDataset
    from pedpredict.data.pose import pose_motion_transform
    from pedpredict.paths import protocol_lmdb_dirs, resolve_paths
    from pedpredict.training.chunk_loader import gather_lmdb_chunks

    cfg = load_config(args.config_dir, args.overrides)
    if not cfg.data.scene_features:
        raise SystemExit("[sidecar] check needs the arm's flags, including data.scene_features")
    train, val, test = protocol_lmdb_dirs(resolve_paths(cfg.paths), cfg.data.protocol)
    transform = pose_motion_transform(cfg)
    for name, dirs in (("train", train), ("val", (val,)), ("test", (test,))):
        n, wrong = 0, 0
        for path in gather_lmdb_chunks(dirs):
            ds = LMDBChunkDataset.from_config(path, cfg.data, pose_transform=transform)
            loader = DataLoader(ds, batch_size=256, num_workers=args.num_workers, collate_fn=_widths)
            for widths in loader:            # a window without a scene row raises KeyError here - loudly
                n += len(widths)
                wrong += sum(w != cfg.data.motion_dim for w in widths)
            ds.close()
        print(f"[sidecar] {name}: {n} windows resolved, {wrong} with a width other than {cfg.data.motion_dim}")
        if wrong or n == 0:
            return 1
    print("[sidecar] check passed: every window of the arm's train, val and test dirs has its scene row")
    return 0


def cmd_same_dump(args: argparse.Namespace) -> int:
    from pedpredict.eval.onset_timing import load_dump

    a, _ = load_dump(args.a)
    b, _ = load_dump(args.b)
    if len(a["p_frame"]) != len(b["p_frame"]) or (a["track_id"] != b["track_id"]).any():
        print("[sidecar] same-dump: the two dumps cover different windows")
        return 1
    diff = float(np.max(np.abs(a["p_frame"].astype(np.float64) - b["p_frame"].astype(np.float64))))
    print(f"[sidecar] same-dump: {len(a['p_frame'])} windows, max |p_frame difference| = {diff:.2e}")
    return 0 if diff <= args.atol else 1


def main(argv: list[str] | None = None) -> int:
    from pedpredict.config import build_argparser

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exports", type=Path, default=_EXPORTS)
    b.add_argument("--groups", default="R,T,S")
    b.add_argument("--clip", type=float, default=5.0)
    b.add_argument("--out", type=Path, default=Path("outputs/features/scene/scene_rts.npz"))
    c = build_argparser(sub.add_parser("check"))
    c.add_argument("--num-workers", type=int, default=6)
    s = sub.add_parser("same-dump")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    s.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check, "same-dump": cmd_same_dump}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

"""Capture a golden fixture for Prompt 1.5 by running the OLD runtime dataset (provenance, not a test).

Builds a tiny LMDB chunk with the **new (1.2-parity-locked) writer**, then reads it back with the OLD
``scripts/lmdb_dataset.py::LMDBChunkDataset`` + OLD ``train_utils.collate_fn`` and the OLD read-time
transforms (``train.py:355-366``: tight 128, context 224, ImageNet normalize). The new dataset/collate
(1.5) is diffed against the saved per-item dicts + batched tuple. Not collected by pytest (``capture_*``);
rerun manually only if the OLD code or inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_lmdb_dataset_golden.py

Determinism / parity notes:
  * Source frames are PNG (lossless) and the LMDB is written by the *same* deterministic writer the test
    uses, so OLD and new read byte-identical JPEG blobs; the only float ops are Resize/ToTensor/Normalize
    (identical torchvision code) -> parity holds at atol=1e-6 (motions/labels exact).
  * Labels vary per sample so the collate's label stacking is meaningful.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pedpredict.config import DataCfg
from pedpredict.data.lmdb_writer import write_dataset_chunks

OLD_SCRIPTS = Path(__file__).resolve().parents[2] / "OLD" / "Undergrad_thesis_project" / "scripts"
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "lmdb_dataset_cases.pt"

_N = 3
_SEQ_LEN = 5
_IMG_H, _IMG_W = 200, 200
_SMALL_MAP = 64 * 1024 * 1024
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _frame(sample: int, t: int) -> np.ndarray:
    """Deterministic non-flat RGB frame (no RNG)."""
    yy, xx = np.mgrid[0:_IMG_H, 0:_IMG_W]
    r = (xx * 3 + t * 11 + sample * 5) % 256
    g = (yy * 5 + t * 7 + sample * 9) % 256
    b = ((xx + yy) * 7 + t * 13) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _bboxes(sample: int) -> list[list[float]]:
    return [[10.0 + t + sample, 10.0 + t, 60.0 + 2 * t, 90.0 + 3 * t] for t in range(_SEQ_LEN)]


def _labels(sample: int) -> dict[str, int]:
    return {"actions": sample % 2, "looks": (sample + 1) % 2, "crosses": sample % 2}


def main() -> None:
    cfg = dataclasses.replace(DataCfg(), lmdb_map_size_bytes=_SMALL_MAP)
    tmp = OUT.parent / "_tmp_capture_dataset"
    frames_dir = tmp / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames_store: list[list[torch.Tensor]] = []
    records = []
    for s in range(_N):
        paths, fr = [], []
        for t in range(_SEQ_LEN):
            arr = _frame(s, t)
            p = frames_dir / f"s{s}_f{t}.png"
            Image.fromarray(arr).save(p)
            paths.append(str(p))
            fr.append(torch.from_numpy(arr))
        frames_store.append(fr)
        records.append({"images": paths, "bboxes": _bboxes(s), **_labels(s)})

    chunk_paths = write_dataset_chunks(records, tmp / "lmdb", cfg, num_workers=0)
    assert len(chunk_paths) == 1, chunk_paths

    sys.path.insert(0, str(OLD_SCRIPTS))
    from lmdb_dataset import LMDBChunkDataset as OldDataset  # noqa: E402
    from train_utils import collate_fn as old_collate  # noqa: E402

    tt = transforms.Compose(
        [transforms.Resize((cfg.img_height, cfg.img_width)), transforms.ToTensor(),
         transforms.Normalize(_MEAN, _STD)]
    )
    tc = transforms.Compose(
        [transforms.Resize((cfg.read_context_height, cfg.read_context_width)), transforms.ToTensor(),
         transforms.Normalize(_MEAN, _STD)]
    )
    ds = OldDataset(str(chunk_paths[0]), transform_tight=tt, transform_context=tc)
    per_item = [{k: v.clone() for k, v in ds[i].items()} for i in range(len(ds))]
    it, ic, mo, lab = old_collate([ds[i] for i in range(len(ds))])

    fixture = {
        "inputs": {
            "frames": frames_store,                        # [N][T] uint8 [H,W,3]
            "bboxes": [_bboxes(s) for s in range(_N)],
            "labels": [_labels(s) for s in range(_N)],
        },
        "seq_ids": list(ds.seq_ids),
        "per_item": per_item,
        "batch": {
            "images_tight": it.clone(),
            "images_context": ic.clone(),
            "motions": mo.clone(),
            "labels": {k: v.clone() for k, v in lab.items()},
        },
        "meta": {
            "src": "scripts/lmdb_dataset.py::LMDBChunkDataset + train_utils.collate_fn",
            "img_atol": 1e-6, "exact": "motions,labels",
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "PIL": Image.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    for p in frames_dir.iterdir():
        p.unlink()
    print(f"wrote {OUT}")
    print(f"  seq_ids = {fixture['seq_ids']}")
    print(f"  images_tight   {tuple(it.shape)} {it.dtype}")
    print(f"  images_context {tuple(ic.shape)} {ic.dtype}")
    print(f"  motions        {tuple(mo.shape)} {mo.dtype}")
    print(f"  labels         {{k: v.tolist() for k,v}} -> {{a:{lab['actions'].tolist()}, "
          f"l:{lab['looks'].tolist()}, c:{lab['crosses'].tolist()}}}")


if __name__ == "__main__":
    main()

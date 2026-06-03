"""Capture a golden fixture for Prompt 1.2 by running the OLD code (provenance, not a test).

Runs OLD ``PIE_sequence_Dataset_1.PIESequenceDataset._process_sequence`` on a tiny, fully
deterministic synthetic track and saves its outputs so ``data/transforms.process_record`` can be
diffed against them. Not collected by pytest (filename is ``capture_*``, not ``test_*``); rerun
manually only if the OLD code or the chosen inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_lmdb_golden.py

Determinism / parity notes:
  * Source frames are written as **PNG** (lossless) so the decoded pixels equal the stored arrays
    exactly — the geometry/motion parity is then independent of any JPEG-source variability.
  * TurboJPEG is force-disabled (``old.jpeg = None``) so the capture uses the PIL decode path,
    which the rebuild standardizes on (the hardcoded ``C:\\libjpeg-turbo64`` DLL path is dropped).
  * ``context_scale = 3.0`` — the uniform rebuild value (== what OLD ``preprocess_data_lmdb.__main__``
    actually ran). The clamped-context case (frame y-extent exceeds image height) is exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# --- OLD repo (read-only reference) ---------------------------------------------------------------
OLD_SCRIPTS = Path(r"c:/Users/LENOVO/Desktop/Undergrad_Project/Undergrad_thesis_project/scripts")
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "lmdb_process_record.pt"

# --- deterministic synthetic track ----------------------------------------------------------------
IMG_W, IMG_H = 240, 180          # W != H to catch any axis transposition
CONTEXT_SCALE = 3.0
RESIZE_TIGHT = (128, 128)
RESIZE_CONTEXT = (int(128 * CONTEXT_SCALE), int(128 * CONTEXT_SCALE))   # (384, 384)
# moving + growing bboxes -> non-zero dx/dy and dw/dh; frame0 context clamps on the y-axis.
BBOXES = [
    [50.0, 40.0, 90.0, 120.0],
    [55.0, 45.0, 99.0, 127.0],
    [60.0, 50.0, 108.0, 134.0],
    [62.0, 52.0, 112.0, 140.0],
]
LABELS = {"actions": 1, "looks": 0, "crosses": 1}


def _synthetic_frame(t: int) -> np.ndarray:
    """A deterministic, non-flat RGB frame (no RNG) so resize/crop are meaningful and reproducible."""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    r = (xx * 3 + t * 11) % 256
    g = (yy * 5 + t * 7) % 256
    b = ((xx + yy) * 7 + t * 13) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def main() -> None:
    sys.path.insert(0, str(OLD_SCRIPTS))
    import PIE_sequence_Dataset_1 as old  # noqa: E402  (path injected above)

    old.jpeg = None  # force the PIL decode path (drop TurboJPEG)

    frames = [_synthetic_frame(t) for t in range(len(BBOXES))]
    tmp = OUT.parent / "_tmp_capture_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for t, arr in enumerate(frames):
        p = tmp / f"frame_{t}.png"
        Image.fromarray(arr).save(p)
        image_paths.append(str(p))

    record = {"images": image_paths, "bboxes": [list(b) for b in BBOXES], **LABELS}

    ds = old.PIESequenceDataset(
        [record],
        transform_tight=transforms.Compose([transforms.Resize(RESIZE_TIGHT), transforms.ToTensor()]),
        transform_context=transforms.Compose([transforms.Resize(RESIZE_CONTEXT), transforms.ToTensor()]),
        crop=True,
        preload=False,
        context_scale=CONTEXT_SCALE,
    )
    out = ds[0]

    fixture = {
        "inputs": {
            "frames": [torch.from_numpy(a) for a in frames],   # [H,W,3] uint8 — re-materialized by the test
            "bboxes": [list(b) for b in BBOXES],
            **LABELS,
        },
        "cfg": {
            "img_height": 128, "img_width": 128, "context_scale": CONTEXT_SCALE,
            "resize_tight": list(RESIZE_TIGHT), "resize_context": list(RESIZE_CONTEXT),
        },
        "outputs": {
            "images_tight": out["images_tight"].clone(),       # [T,3,128,128] float[0,1]
            "images_context": out["images_context"].clone(),   # [T,3,384,384] float[0,1]
            "motions": out["motions"].clone(),                 # [T,8] float32
            "actions": out["actions"].clone(),
            "looks": out["looks"].clone(),
            "crosses": out["crosses"].clone(),
        },
        "meta": {
            "src": "scripts/PIE_sequence_Dataset_1.py::PIESequenceDataset._process_sequence",
            "decode": "PIL (turbojpeg force-disabled)",
            "tol": 1e-6,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "PIL": Image.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)

    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()

    o = fixture["outputs"]
    print(f"wrote {OUT}")
    print(f"  images_tight   {tuple(o['images_tight'].shape)} {o['images_tight'].dtype}")
    print(f"  images_context {tuple(o['images_context'].shape)} {o['images_context'].dtype}")
    print(f"  motions        {tuple(o['motions'].shape)} {o['motions'].dtype}")
    print(f"  motions[0] = {o['motions'][0].tolist()}")
    print(f"  motions[1] = {o['motions'][1].tolist()}")


if __name__ == "__main__":
    main()

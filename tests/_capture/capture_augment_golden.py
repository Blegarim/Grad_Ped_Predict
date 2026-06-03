"""Capture a golden fixture for Prompt 1.4 by running OLD ``SequenceAugmenter`` (provenance, not a test).

The OLD augmenter operates on a *tensor* sequence dict (``images_tight``/``images_context``/``motions``)
— the format it assumed but the real pipeline never produced (see ``data/augment.py`` docstring). We feed
it a tiny deterministic synthetic dict and snapshot each transform's output so the re-homed
``SequenceAugmenter`` (operating on ``ProcessedSample``) can be diffed against it. Rerun manually only if
the OLD code or chosen inputs change::

    .venv/Scripts/python.exe tests/_capture/capture_augment_golden.py

Determinism: each stochastic transform is captured under a fixed seed using the SAME seeding the new
``SequenceAugmenter.apply`` uses (``torch.manual_seed`` for color/noise, ``random`` for erase), so parity
is exact, not distributional.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import torch

OLD_SCRIPTS = Path(r"c:/Users/LENOVO/Desktop/Undergrad_Project/Undergrad_thesis_project/scripts")
OUT = Path(__file__).resolve().parents[1] / "fixtures" / "golden" / "augment_cases.pt"

T, C, HT, WT, HC, WC = 4, 3, 8, 6, 10, 7  # H != W to catch any axis transposition
SEED_COLOR, SEED_NOISE, SEED_ERASE = 7, 11, 123
NOISE_STD, ERASE_N = 0.02, 2


def _synthetic_seq() -> dict:
    """Deterministic [0,1] image tensors + a [T,8] motion tensor (no RNG)."""
    tight = (torch.arange(T * C * HT * WT, dtype=torch.float32).reshape(T, C, HT, WT) % 97) / 97.0
    context = (torch.arange(T * C * HC * WC, dtype=torch.float32).reshape(T, C, HC, WC) % 89) / 89.0
    motions = (torch.arange(T * 8, dtype=torch.float32).reshape(T, 8) - 13.0)  # mix of signs
    return {"images_tight": tight, "images_context": context, "motions": motions}


def _snapshot(seq: dict) -> dict:
    return {
        "images_tight": seq["images_tight"].clone(),
        "images_context": seq["images_context"].clone(),
        "motions": seq["motions"].clone(),
    }


def main() -> None:
    sys.path.insert(0, str(OLD_SCRIPTS))
    import augment_sequences as old  # noqa: E402  (path injected above)

    aug = old.SequenceAugmenter()  # defaults match AugmentCfg defaults
    base = _synthetic_seq()

    out_flip = aug.horizontal_flip(copy.deepcopy(base))

    torch.manual_seed(SEED_COLOR)
    out_color = aug.color_augment(copy.deepcopy(base))

    torch.manual_seed(SEED_NOISE)
    out_noise = aug.motion_noise(copy.deepcopy(base), noise_std=NOISE_STD)

    random.seed(SEED_ERASE)
    out_erase = aug.random_erase_frames(copy.deepcopy(base), n_frames=ERASE_N)

    fixture = {
        "input": _snapshot(base),
        "seeds": {"color": SEED_COLOR, "noise": SEED_NOISE, "erase": SEED_ERASE},
        "params": {"noise_std": NOISE_STD, "erase_n": ERASE_N},
        "outputs": {
            "flip": _snapshot(out_flip),
            "color": _snapshot(out_color),
            "noise": _snapshot(out_noise),
            "erase": _snapshot(out_erase),
        },
        "meta": {
            "src": "scripts/augment_sequences.py::SequenceAugmenter",
            "tol": 1e-6,
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, OUT)
    print(f"wrote {OUT}")
    print(f"  flip motions[:,2] sign-flipped: {torch.equal(out_flip['motions'][:, 2], -base['motions'][:, 2])}")
    print(f"  erase frames changed: {int((out_erase['images_tight'] != base['images_tight']).any(dim=(1,2,3)).sum())}")


if __name__ == "__main__":
    main()

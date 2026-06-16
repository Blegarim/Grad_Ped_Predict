"""Extract PIE *annotated* frames as JPG, without editing PIE source.

PIE's ``extract_and_save_images('annotated')`` already selects only the annotated frames
(``get_annotated_frame_numbers`` reads the annotation XML), but it hardcodes a ``.png`` write.
This wrapper monkeypatches ``cv2.imwrite`` so that write emits ``.jpg`` instead — keeping PIE's
annotated-frame selection and decoding while shrinking the transient frame cache (~3-5x vs PNG).

The sequence records reference ``.png`` paths; the runtime loader's missing-file fallback
(``data/transforms.load_rgb``) resolves them to the same-stem ``.jpg``, so this is a drop-in for
the standard val/test/benchmark extraction step in setup.md.

Run from the repo root with ``data/`` staged. PIE only processes set folders present under
``data/PIE_clips/``, so stage one split's sets at a time (val = set05/06, test = set03), build its
LMDB, delete the frames, then move on. Not idempotent: a re-run re-extracts (PIE's skip-if-exists
check looks for the ``.png`` that this wrapper never writes).

    python scripts/extract_annotated_jpg.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import cv2  # noqa: E402

from PIE.utilities.pie_data import PIE  # noqa: E402

# extract_and_save_images iterates every set folder physically present under data/PIE_clips/, so
# staging one split's sets at a time (val = set05/06, test = set03) is how you scope extraction —
# no in-code set filter. Match the LMDB writer's encode quality envelope (jpeg_quality=90); 95 keeps
# the only real lossy step the LMDB re-encode, not this intermediate.
_JPEG_QUALITY = 95

_orig_imwrite = cv2.imwrite


def _imwrite_jpg(path, img, params=None):
    """``cv2.imwrite`` shim: redirect PIE's hardcoded ``.png`` write to ``.jpg``."""
    if path.endswith(".png"):
        path = path[:-4] + ".jpg"
    return _orig_imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])


def main() -> None:
    cv2.imwrite = _imwrite_jpg
    PIE(data_path="data").extract_and_save_images(extract_frame_type="annotated")


if __name__ == "__main__":
    main()

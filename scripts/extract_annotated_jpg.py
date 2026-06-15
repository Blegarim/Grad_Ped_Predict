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

import os
import sys

sys.path.insert(0, ".")

import cv2  # noqa: E402

import PIE.utilities.pie_data as _pie_data  # noqa: E402
from PIE.utilities.pie_data import PIE  # noqa: E402

# --- TEMP (set04-only extraction): remove this block to restore all-sets behavior. ---------------
# extract_and_save_images iterates every set folder under data/PIE_clips/ (pie_data.py: the
# `listdir(self._clips_path)` line). Until set04 is the only set staged, restrict just that one
# directory listing to set04 so extraction skips sets we haven't downloaded. Surgical: every other
# listdir call (annotation dirs, etc.) passes straight through.
_ONLY_SET = "set04"
_orig_listdir = _pie_data.listdir


def _listdir_only_set(path):
    entries = _orig_listdir(path)
    if os.path.basename(os.path.normpath(path)) == "PIE_clips":
        return [e for e in entries if e == _ONLY_SET]
    return entries


_pie_data.listdir = _listdir_only_set
# --- END TEMP ------------------------------------------------------------------------------------

# Match the LMDB writer's encode quality envelope (jpeg_quality=90); 95 keeps the only real lossy
# step the LMDB re-encode, not this intermediate.
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

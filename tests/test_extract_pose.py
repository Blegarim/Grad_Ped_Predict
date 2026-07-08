"""scripts/extract_pose.py clip-streaming path: frames come from PIE_clips, never staged images."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

# scripts/ is not an importable package; load the entry-point module by path.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "extract_pose.py"
_spec = importlib.util.spec_from_file_location("_extract_pose_script", _SCRIPT)
extract_pose = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_pose)


def _write_clip(tmp_path, n_frames: int = 10) -> Path:
    """Tiny mp4 where frame i is a flat gray level i*20 (same recipe as test_incremental)."""
    clips_dir = tmp_path / "PIE_clips" / "set01"
    clips_dir.mkdir(parents=True)
    mp4 = clips_dir / "video_0001.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available in this opencv build")
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
    writer.release()
    return tmp_path / "PIE_clips"


class _FakeExtractor:
    """Stands in for DwposeExtractor: returns keypoints filled with the frame's gray level."""

    def __call__(self, img_bgr: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        level = float(img_bgr[0, 0, 0])
        return np.full((len(boxes), 23, 3), level, dtype=np.float32)


def test_extract_video_streams_frames_from_clip(tmp_path) -> None:
    clips_dir = _write_clip(tmp_path)
    frames = {
        "2_ped_a": [1.0, 1.0, 10.0, 20.0],
        "2_ped_b": [5.0, 5.0, 15.0, 30.0],
        "7_ped_a": [2.0, 2.0, 12.0, 22.0],
    }
    out = extract_pose.extract_video(
        "set01", "video_0001", frames, clips_dir=clips_dir, extractor=_FakeExtractor()
    )
    assert set(out) == set(frames)
    # Frame alignment: each key's keypoints carry its frame's gray level (mp4 is lossy -> tolerance).
    assert abs(out["2_ped_a"][0, 0] - 40.0) < 10
    assert abs(out["7_ped_a"][0, 0] - 140.0) < 10
    assert not list(tmp_path.rglob("*.png"))  # nothing staged to disk


def test_iter_clip_frames_errors(tmp_path) -> None:
    clips_dir = _write_clip(tmp_path)
    with pytest.raises(RuntimeError, match="clip too short"):
        list(extract_pose.iter_clip_frames(clips_dir / "set01" / "video_0001.mp4", {2, 50}))
    with pytest.raises(FileNotFoundError, match="cannot open clip"):
        list(extract_pose.iter_clip_frames(clips_dir / "set01" / "video_9999.mp4", {0}))

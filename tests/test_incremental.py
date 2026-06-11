"""Incremental, disk-bounded LMDB build — the plan (pure) and the cv2 extractor (roundtrip).

The plan is the load-bearing part: each video must be extracted exactly once and deleted right after
the last chunk that references it (that bound is what keeps the disk from filling), and a resume must
skip videos already consumed. Extraction is asserted to reproduce the requested frames at the exact
on-disk paths the cropper reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from pedpredict.data.incremental import (
    BuildStep,
    extract_video_frames,
    iter_build_steps,
    next_chunk_start,
    parse_frame_path,
)

cv2 = pytest.importorskip("cv2")

_SEQ = 2


def _rec(set_id: str, video: str, start_fid: int) -> dict:
    """One record: ``_SEQ`` PIE-style frame paths in a single video + dummy bbox/labels."""
    imgs = [f"data/images/{set_id}/{video}/{start_fid + t:05d}.png" for t in range(_SEQ)]
    return {"images": imgs, "bboxes": [[0.0, 0.0, 1.0, 1.0]] * _SEQ, "actions": 0, "looks": 0, "crosses": 0}


def _records() -> list[dict]:
    """5 records in video A, 5 in B, 3 in C — ordered set -> video, as PIE emits."""
    a = [_rec("set01", "video_0001", i) for i in range(5)]
    b = [_rec("set01", "video_0002", i) for i in range(5)]
    c = [_rec("set02", "video_0001", i) for i in range(3)]
    return a + b + c


@pytest.mark.parametrize(
    "path, expected",
    [
        ("data/images/set01/video_0001/00123.png", ("set01", "video_0001", 123)),
        ("data\\images\\set04\\video_0010\\00007.png", ("set04", "video_0010", 7)),
        ("/abs/root/images/set06/video_0002/01999.png", ("set06", "video_0002", 1999)),
    ],
)
def test_parse_frame_path(path, expected):
    assert parse_frame_path(path) == expected


def test_parse_frame_path_rejects_garbage():
    with pytest.raises(ValueError):
        parse_frame_path("data/images/not_a_set/frame.png")


def test_next_chunk_start():
    assert next_chunk_start([], 5000) == 0
    assert next_chunk_start([0, 5000, 10000, 15000], 5000) == 20000


def _video_keys(step: BuildStep) -> list[tuple[str, str]]:
    return list(step.extract.keys())


def test_plan_extracts_each_video_once_and_deletes_after_last_chunk():
    steps = list(iter_build_steps(_records(), chunk_size=4, start_idx=0))
    a, b, c = ("set01", "video_0001"), ("set01", "video_0002"), ("set02", "video_0001")

    # Chunk boundaries cover all 13 records.
    assert [(s.chunk_start, s.chunk_end) for s in steps] == [(0, 4), (4, 8), (8, 12), (12, 13)]
    # Each video extracted exactly once, in dataset order.
    extracted = [k for s in steps for k in _video_keys(s)]
    assert extracted == [a, b, c]
    # A finishes at record 4 (deleted once chunk end passes 4 -> step [4,8)); B at 9 -> [8,12); C at 12 -> [12,13).
    assert [s.delete for s in steps] == [[], [a], [b], [c]]


def test_plan_resume_skips_consumed_videos():
    # Resuming at index 8 must never re-extract A (fully consumed by record 4) or B's early frames.
    steps = list(iter_build_steps(_records(), chunk_size=4, start_idx=8))
    extracted = [k for s in steps for k in _video_keys(s)]
    assert ("set01", "video_0001") not in extracted          # A skipped entirely
    assert extracted == [("set01", "video_0002"), ("set02", "video_0001")]
    assert steps[0].chunk_start == 8


def test_extract_video_frames_roundtrip(tmp_path):
    # Write a tiny mp4, then extract a sparse subset of frames to their record paths.
    clips_dir = tmp_path / "PIE_clips" / "set01"
    clips_dir.mkdir(parents=True)
    mp4 = clips_dir / "video_0001.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available in this opencv build")
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
    writer.release()

    frames = {2: "images/set01/video_0001/00002.png", 7: "images/set01/video_0001/00007.png"}
    written = extract_video_frames(tmp_path / "PIE_clips", tmp_path, "set01", "video_0001", frames)

    assert written == 2
    for path in frames.values():
        dst = tmp_path / path
        assert dst.is_file()
        assert cv2.imread(str(dst)).shape == (32, 32, 3)
    # Idempotent: a second call writes nothing.
    assert extract_video_frames(tmp_path / "PIE_clips", tmp_path, "set01", "video_0001", frames) == 0

#!/bin/bash
# Extract frames from PIE clips for ONLY the sets passed as arguments — a per-set variant of
# PIE/annotations/split_clips_to_frames.sh (which is hardcoded to all six sets). Lets you process
# the dataset incrementally on a storage-limited machine: extract a split's sets, build its LMDB,
# delete the frames, repeat. See setup.md "Incremental extraction (storage-limited PCs)".
#
# Uses RELATIVE paths, so run it from inside data/ (where PIE_clips/ and images/ live):
#   cd /d/Grad_Ped_Predict/data
#   bash ../scripts/split_sets.sh set05 set06
#
# Requirements: Git Bash (or any bash) + ffmpeg on PATH. Extracts ALL frames (large); prefer the
# Python 'annotated' extractor when you only need annotated frames (~10x smaller) — see setup.md.

set -euo pipefail

CLIPS_DIR=PIE_clips   # path to the directory with mp4 videos (relative to CWD)
FRAMES_DIR=images     # path to the directory for frames (relative to CWD)

if [ "$#" -eq 0 ]; then
    echo "usage: bash split_sets.sh set01 [set02 ...]" >&2
    exit 2
fi

for set_dir in "$@"
do
    if [ ! -d "${CLIPS_DIR}/${set_dir}" ]; then
        echo "skip: ${CLIPS_DIR}/${set_dir} not found (run from data/?)" >&2
        continue
    fi
    for video in "${CLIPS_DIR}/${set_dir}"/*
    do
        filename=$(basename "$video")
        fname="${filename%.*}"
        mkdir -p "${FRAMES_DIR}/${set_dir}/${fname}"
        # -y overwrites; ffmpeg writes 00000.png, 00001.png, ... per video
        ffmpeg -y -i "$video" -start_number 0 -f image2 -qscale:v 2 \
            "${FRAMES_DIR}/${set_dir}/${fname}/%05d.jpg"
    done
done

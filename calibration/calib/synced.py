"""Extract extrinsic placements from TTL-synchronized videos.

Because the Arduino TTL triggers every camera on the same pulse, frame index N
is the SAME instant in all four videos. So an extrinsic "placement" (one moment
where the board is static and visible to every camera) is simply a single frame
index. This module scans the synced videos, finds frame indices where all
cameras detect the full board, picks a well-spread, sharp subset, and writes
them as pose1/pose2/... folders that run_extrinsics.py consumes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import capture as cap_mod
from .board import BoardSpec


def find_camera_videos(raw_dir: Path, camera_names: list[str]) -> dict[str, Path]:
    """Locate one video per camera, named <camera>.<ext> in raw_dir."""
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"{raw} does not exist. Put one synced video per "
                                f"camera there, named e.g. Camera1.avi.")
    out = {}
    for cam in camera_names:
        match = next((p for p in sorted(raw.iterdir())
                      if p.stem == cam and p.suffix.lower() in cap_mod._VIDEO_EXTS),
                     None)
        if match is None:
            raise FileNotFoundError(
                f"No video named '{cam}.<avi/mp4/...>' in {raw}.")
        out[cam] = match
    return out


def find_shared_board_frames(videos: dict[str, Path], spec: BoardSpec,
                             scan_stride: int = 10, n_placements: int = 5,
                             sharpness_min: float = 0.0) -> list[int]:
    """Return frame indices where ALL cameras see the full board.

    Reads all videos in lockstep (relying on TTL frame-sync), tests every
    `scan_stride`-th frame, and returns up to `n_placements` indices spread
    across the timeline, preferring the sharpest frame in each time bin.
    """
    names = list(videos)
    caps = {c: cv2.VideoCapture(str(videos[c])) for c in names}
    for c in names:
        if not caps[c].isOpened():
            raise RuntimeError(f"Could not open {videos[c]}")

    candidates: list[tuple[int, float]] = []   # (frame_index, min sharpness)
    idx = 0
    try:
        while True:
            frames, ok_all = {}, True
            for c in names:
                ok, fr = caps[c].read()
                if not ok:
                    ok_all = False
                    break
                frames[c] = fr
            if not ok_all:
                break
            if idx % scan_stride == 0:
                sharps, all_found = [], True
                for c in names:
                    sh = cap_mod.sharpness(frames[c])
                    if sh < sharpness_min:
                        all_found = False
                        break
                    # classic detector here -- faster for scanning many frames
                    if board_mod.find_corners(frames[c], spec,
                                              refine=False, use_sb=False) is None:
                        all_found = False
                        break
                    sharps.append(sh)
                if all_found:
                    candidates.append((idx, float(min(sharps))))
            idx += 1
    finally:
        for c in names:
            caps[c].release()

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])
    idxs = [c[0] for c in candidates]
    if n_placements >= len(candidates):
        return idxs

    # Bin the timeline and keep the sharpest candidate per bin.
    lo, hi = idxs[0], idxs[-1]
    if hi == lo:
        return idxs[:n_placements]
    edges = np.linspace(lo, hi, n_placements + 1)
    chosen = []
    for b in range(n_placements):
        in_bin = [c for c in candidates if edges[b] <= c[0] <= edges[b + 1]]
        if in_bin:
            chosen.append(max(in_bin, key=lambda x: x[1])[0])
    return sorted(set(chosen))


def extract_placements(videos: dict[str, Path], frame_indices: list[int],
                       out_dir: Path) -> list[tuple[str, int]]:
    """Write each chosen frame as out_dir/pose{k}/<camera>.png. Returns (pose, idx)."""
    out = Path(out_dir)
    written = []
    for k, fi in enumerate(frame_indices, start=1):
        pdir = out / f"pose{k}"
        pdir.mkdir(parents=True, exist_ok=True)
        for cam, vp in videos.items():
            cv2.imwrite(str(pdir / f"{cam}.png"),
                        cap_mod.read_video_frame(vp, fi))
        written.append((f"pose{k}", fi))
    return written

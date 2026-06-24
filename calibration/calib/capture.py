"""Frame sources: live UVC capture and frame harvesting from recorded video.

The OV9281 is a UVC (USB Video Class) global-shutter camera, so it enumerates
as a plain webcam and OpenCV can open it by index. Global shutter means every
pixel is exposed at the same instant -- combined with your Arduino TTL trigger,
all four cameras capture the *same* moment, which is exactly what the extrinsic
step needs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


# --------------------------------------------------------------------------- #
# Live UVC capture
# --------------------------------------------------------------------------- #
def open_uvc(index: int, width: int = 1280, height: int = 800,
             backend: str = "dshow") -> cv2.VideoCapture:
    """Open a UVC camera by index and request a resolution.

    Raises if the camera cannot be opened. Note: the camera may silently fall
    back to a different resolution if the requested one is unsupported -- check
    the returned capture's actual width/height if it matters.
    """
    cap = cv2.VideoCapture(index, _BACKENDS.get(backend, cv2.CAP_ANY))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open UVC camera at index {index} "
                           f"(backend={backend}).")
    return cap


def grab_synchronized(caps: list[cv2.VideoCapture]) -> list[np.ndarray] | None:
    """Grab one frame from each capture as simultaneously as software allows.

    Uses the grab()/retrieve() split so all sensors are latched before any frame
    is decoded -- this minimises software skew. For true hardware sync you still
    rely on the TTL trigger; this just avoids adding extra software latency.
    Returns one frame per capture, or None if any grab failed.
    """
    if not all(cap.grab() for cap in caps):
        return None
    frames = []
    for cap in caps:
        ok, frame = cap.retrieve()
        if not ok:
            return None
        frames.append(frame)
    return frames


# --------------------------------------------------------------------------- #
# Reading from saved files
# --------------------------------------------------------------------------- #
def list_sources(folder: Path) -> list[Path]:
    """All image + video files in a folder, sorted by name."""
    folder = Path(folder)
    if not folder.exists():
        return []
    out = [p for p in sorted(folder.iterdir())
           if p.suffix.lower() in _IMAGE_EXTS | _VIDEO_EXTS]
    return out


def is_video(path: Path) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def iter_video_frames(path: Path, stride: int = 1,
                      start_frame: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) for every `stride`-th frame, from start_frame on.

    Frames before start_frame are read and discarded (sequential skip is more
    reliable than seeking across codecs). Indices are absolute (from 0).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start_frame and (idx - start_frame) % stride == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    """Read a single specific frame from a video (used for extrinsic sources)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {path}")
        return frame
    finally:
        cap.release()


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian -- higher is sharper. Used to drop motion blur."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def video_stats(path: Path, exact_count: bool = False) -> dict:
    """Return {frames, fps, width, height} for a video.

    `frames` comes from container metadata by default (fast). Some codecs report
    it inaccurately; set exact_count=True to count by decoding every frame (slow
    but exact) -- worth it for the frame-sync check.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if exact_count:
            n = 0
            while cap.grab():
                n += 1
        else:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"frames": n, "fps": float(fps), "width": w, "height": h}
    finally:
        cap.release()

"""Per-camera intrinsic calibration.

Intrinsics describe a single camera's lens + sensor, independent of where the
camera sits in the world:
    K          3x3 matrix [[fx,0,cx],[0,fy,cy],[0,0,1]] -- focal lengths and
               principal point, in pixels.
    distCoeffs [k1, k2, p1, p2, k3] -- radial (k) and tangential (p) lens
               distortion.

We show the camera a chessboard at many positions/angles and let
cv2.calibrateCamera solve for K and distCoeffs that best reproject all the
observed corners. Each camera is calibrated entirely on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import capture as cap_mod
from .board import BoardSpec


@dataclass
class IntrinsicResult:
    camera: str
    K: np.ndarray                 # 3x3, OpenCV convention
    dist: np.ndarray              # (5,) [k1,k2,p1,p2,k3]
    image_size: tuple[int, int]   # (width, height)
    rms: float                    # overall RMS reprojection error (px)
    per_view_error: list[float]   # RMS per accepted view (px)
    n_views: int
    used_files: list[str] = field(default_factory=list)


def _gather_corner_views(camera: str, source_dir: Path, spec: BoardSpec,
                         frame_stride: int, sharpness_min: float,
                         max_views: int):
    """Collect (image_points, image_size) from every image/video in a folder.

    Spreads the per-video frame budget so we don't take 40 near-identical frames
    from the start of one clip. Returns (objpoints, imgpoints, image_size).
    """
    folder = source_dir / camera
    sources = cap_mod.list_sources(folder)
    if not sources:
        raise FileNotFoundError(
            f"No images/videos for {camera} in {folder}. "
            f"Put intrinsic footage there (see README).")

    objp = board_mod.object_points(spec)
    objpoints, imgpoints = [], []
    image_size = None

    def consider(frame) -> bool:
        nonlocal image_size
        if cap_mod.sharpness(frame) < sharpness_min:
            return False
        corners = board_mod.find_corners(frame, spec)
        if corners is None:
            return False
        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            # Mixed resolutions in one camera's folder is almost always a mistake.
            raise ValueError(
                f"{camera}: inconsistent image size {(w, h)} vs {image_size}.")
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        return True

    for src in sources:
        if len(imgpoints) >= max_views:
            break
        if cap_mod.is_video(src):
            for _idx, frame in cap_mod.iter_video_frames(src, stride=frame_stride):
                if len(imgpoints) >= max_views:
                    break
                consider(frame)
        else:
            img = cv2.imread(str(src))
            if img is not None:
                consider(img)

    return objpoints, imgpoints, image_size, [str(s) for s in sources]


def calibrate_camera(camera: str, source_dir: Path, spec: BoardSpec,
                     min_views: int = 12, max_views: int = 40,
                     frame_stride: int = 5, sharpness_min: float = 60.0,
                     fix_k3: bool = True) -> IntrinsicResult:
    """Run intrinsic calibration for one camera from its source folder."""
    objpoints, imgpoints, image_size, used = _gather_corner_views(
        camera, source_dir, spec, frame_stride, sharpness_min, max_views)

    n = len(imgpoints)
    if n < min_views:
        raise RuntimeError(
            f"{camera}: only {n} usable views found (need >= {min_views}). "
            f"Record more varied board images (different angles, distances, "
            f"corners of the frame) or relax sharpness_min.")

    flags = 0
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3  # OV9281 + M12 rarely needs k3; fewer params = stabler fit

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=flags)

    per_view = _per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, dist)

    return IntrinsicResult(
        camera=camera, K=K, dist=dist.ravel(), image_size=image_size,
        rms=float(rms), per_view_error=per_view, n_views=n, used_files=used)


def _per_view_errors(objpoints, imgpoints, rvecs, tvecs, K, dist) -> list[float]:
    """RMS reprojection error (px) for each calibration view, for QC."""
    errs = []
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = proj.reshape(-1, 2) - img.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return errs

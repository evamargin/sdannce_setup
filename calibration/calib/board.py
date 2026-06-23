"""Chessboard geometry and corner detection.

A chessboard is the calibration target because OpenCV knows its true 3D
geometry exactly: the inner corners lie on a perfect planar grid. We tell
OpenCV where those corners *should* be in metric space (object points) and
where they *actually* land in the image (image points); everything else is
solved from the gap between the two.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    """Chessboard description.

    cols, rows are INNER-CORNER counts (points where 4 squares meet), NOT the
    number of squares. A 10x7-square board has cols=9, rows=6.
    square_mm is the physical edge length of one square in millimetres; this is
    what sets the metric scale (and therefore the units) of the whole rig.
    """
    cols: int
    rows: int
    square_mm: float

    @property
    def size(self) -> tuple[int, int]:
        """(cols, rows) as OpenCV's patternSize."""
        return (self.cols, self.rows)

    @property
    def n_corners(self) -> int:
        return self.cols * self.rows


def object_points(spec: BoardSpec) -> np.ndarray:
    """The board's corners in its own coordinate frame, shape (N, 3) float32.

    Z is 0 for every corner (the board is planar). X runs along `cols`, Y along
    `rows`, both scaled to millimetres. This frame becomes the WORLD frame
    during extrinsic calibration, so corner (0,0) is the world origin.
    """
    obj = np.zeros((spec.n_corners, 3), np.float32)
    grid = np.mgrid[0:spec.cols, 0:spec.rows].T.reshape(-1, 2)
    obj[:, :2] = grid * spec.square_mm
    return obj


# Subpixel corner refinement: keep iterating until we move <0.001 px or hit 30 iters.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_corners(img: np.ndarray, spec: BoardSpec,
                 refine: bool = True, use_sb: bool = True) -> np.ndarray | None:
    """Detect chessboard inner corners.

    Returns corners as (N, 1, 2) float32 (OpenCV's convention) or None if the
    full board was not found. `use_sb` tries the newer, more robust
    findChessboardCornersSB first and falls back to the classic detector.
    """
    gray = _to_gray(img)

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        # SB detector is more tolerant of blur/lighting and returns subpixel
        # corners directly, so no separate cornerSubPix step is needed.
        ok, corners = cv2.findChessboardCornersSB(
            gray, spec.size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if ok:
            return corners.astype(np.float32)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, spec.size, flags=flags)
    if not ok:
        return None
    if refine:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   _SUBPIX_CRITERIA)
    return corners.astype(np.float32)


def draw_corners(img: np.ndarray, corners: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Return a BGR copy with the detected corners drawn (for QC overlays)."""
    vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    cv2.drawChessboardCorners(vis, spec.size, corners, True)
    return vis

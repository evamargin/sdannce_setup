"""OpenCV multi-camera calibration toolkit for sDANNCE.

Modules:
    board          - chessboard geometry + corner detection
    capture        - UVC live grab and frame harvesting from videos
    intrinsics     - per-camera intrinsic calibration
    extrinsics     - shared-board extrinsic calibration (common world frame)
    dannce_export  - convert OpenCV params -> DANNCE convention and write .mat
    qc             - reprojection error, triangulation check, overlays
    config         - load/validate config.yaml
"""

from .board import BoardSpec, object_points, find_corners
from .config import load_config

__all__ = ["BoardSpec", "object_points", "find_corners", "load_config"]

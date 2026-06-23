"""Convert OpenCV calibration into DANNCE/sDANNCE format and write .mat files.

THE CONVENTION GOTCHA (verified against spoonsso/dannce and tqxli/sdannce):

DANNCE projects 3D points as ROW vectors:
        x = [X Y Z 1] @ M,   M = concatenate((R, t)) @ K
This is the transpose of OpenCV's column-vector model (x = K [R|t] X). So the
matrices stored in the .mat file are TRANSPOSES of OpenCV's:

    field       value (from OpenCV)              shape
    -----       --------------------             -----
    K           K_cv.T                           3x3
    r           R_cv.T                           3x3   (loader renames r -> R)
    t           t_cv as a row vector             1x3
    RDistort    [k1, k2, k3]                     1x3
    TDistort    [p1, p2]                         1x2

OpenCV distCoeffs order is [k1, k2, p1, p2, k3], which we split/reorder above.

sDANNCE's load_camera_params reads a struct with fields K, RDistort, TDistort,
r, t (and renames r->R). We write one hires_cam{N}_params.mat per camera; these
import directly into the Label3D GUI and are referenced by the sDANNCE config.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat


def to_dannce_params(K_cv: np.ndarray, dist_cv: np.ndarray,
                     R_cv: np.ndarray, t_cv: np.ndarray) -> dict:
    """Build a DANNCE-format param dict from OpenCV intrinsics/extrinsics.

    K_cv:   3x3 OpenCV intrinsic matrix
    dist_cv: (5,) [k1, k2, p1, p2, k3]
    R_cv:   3x3 OpenCV rotation (world->camera)
    t_cv:   (3,) or (3,1) OpenCV translation (mm)
    """
    dist = np.asarray(dist_cv).ravel()
    k1, k2, p1, p2 = dist[0], dist[1], dist[2], dist[3]
    k3 = dist[4] if dist.size > 4 else 0.0

    t_row = np.asarray(t_cv, dtype=np.float64).reshape(1, 3)

    return {
        "K": np.asarray(K_cv, dtype=np.float64).T,       # transpose -> row-vector convention
        "r": np.asarray(R_cv, dtype=np.float64).T,       # transpose; loader renames r -> R
        "t": t_row,                                      # 1x3 row vector
        "RDistort": np.array([[k1, k2, k3]], dtype=np.float64),  # 1x3
        "TDistort": np.array([[p1, p2]], dtype=np.float64),      # 1x2
    }


def write_camera_params(out_dir: Path, cam_number: int, params: dict) -> Path:
    """Write one hires_cam{N}_params.mat (N is 1-based to match DANNCE)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hires_cam{cam_number}_params.mat"
    # do_compression keeps files small; these are tiny anyway.
    savemat(str(path), params, do_compression=True)
    return path


def write_combined(out_dir: Path, prefix: str,
                   all_params: list[dict]) -> Path:
    """Write a combined <prefix>_dannce.mat with a 1xN `params` cell of structs.

    This mirrors what Label3D produces and lets you point a project straight at
    one file. `sync` and `labelData` are left out here -- those are created when
    you actually label frames in Label3D; calibration only needs `params`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build an object array (MATLAB cell) of structs.
    cell = np.empty((1, len(all_params)), dtype=object)
    for i, p in enumerate(all_params):
        cell[0, i] = p
    path = out_dir / f"{prefix}_dannce.mat"
    savemat(str(path), {"params": cell}, do_compression=True)
    return path


def load_camera_params(path: Path) -> dict:
    """Read back a hires_cam params .mat the way DANNCE does (r -> R).

    Used by qc.py to verify the SAVED file -- not OpenCV's in-memory numbers --
    reprojects correctly. Returns dict with K, R, t, RDistort, TDistort.
    """
    from scipy.io import loadmat
    m = loadmat(str(path))
    out = {
        "K": m["K"],
        "R": m["r"] if "r" in m else m["R"],
        "t": m["t"],
        "RDistort": m["RDistort"],
        "TDistort": m["TDistort"],
    }
    return out

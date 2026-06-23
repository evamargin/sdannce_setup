"""Synthetic end-to-end self-test -- no cameras or images required.

The point of this test is to prove the DANNCE convention conversion is correct:
we take known OpenCV intrinsics/extrinsics, project points with OpenCV, then
write -> read the .mat and reproject with our DANNCE-convention math. If the
transpose handling or distortion model were wrong, the two would disagree.

Run:
    python selftest.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import cv2

from calib.board import BoardSpec, object_points
from calib.dannce_export import (to_dannce_params, write_camera_params,
                                 write_combined, load_camera_params)
from calib import qc


def _make_camera(seed: int):
    """A plausible OV9281-ish camera with a known pose looking at the origin."""
    rng = np.random.default_rng(seed)
    fx = fy = 900.0 + rng.uniform(-30, 30)
    cx, cy = 640.0 + rng.uniform(-5, 5), 400.0 + rng.uniform(-5, 5)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    dist = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.05, 0.05),
                     rng.uniform(-0.002, 0.002), rng.uniform(-0.002, 0.002),
                     0.0], float)
    # Place camera ~600 mm from origin at a random-ish angle, looking at origin.
    angle = seed * (2 * np.pi / 4)
    C = np.array([600 * np.cos(angle), 600 * np.sin(angle), 500.0])
    forward = -C / np.linalg.norm(C)
    up = np.array([0, 0, 1.0])
    right = np.cross(up, forward); right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.vstack([right, true_up, forward])   # world->camera
    t = (-R @ C).reshape(3, 1)
    return K, dist, R, t, C


def test_convention_roundtrip():
    spec = BoardSpec(cols=9, rows=6, square_mm=25.0)
    world = object_points(spec).astype(np.float64)  # board at world origin (Z=0)

    tmp = Path(tempfile.mkdtemp(prefix="calib_selftest_"))
    max_err = 0.0
    all_params = []
    for cam_i in range(1, 5):
        K, dist, R, t, C = _make_camera(cam_i)

        # Ground-truth projection via OpenCV.
        rvec, _ = cv2.Rodrigues(R)
        opencv_px, _ = cv2.projectPoints(world, rvec, t, K, dist)
        opencv_px = opencv_px.reshape(-1, 2)

        # Convert -> write -> read -> reproject with DANNCE math.
        params = to_dannce_params(K, dist, R, t)
        write_camera_params(tmp, cam_i, params)
        loaded = load_camera_params(tmp / f"hires_cam{cam_i}_params.mat")
        dannce_px = qc.dannce_project(world, loaded)

        err = np.max(np.linalg.norm(dannce_px - opencv_px, axis=1))
        max_err = max(max_err, err)

        # Camera centre recovered from saved params must match the true centre.
        C_rec = qc.camera_center(loaded)
        c_err = np.linalg.norm(C_rec - C)

        # Field shapes match DANNCE expectations.
        assert loaded["K"].shape == (3, 3)
        assert loaded["R"].shape == (3, 3)
        assert loaded["t"].reshape(-1).shape == (3,)
        assert loaded["RDistort"].reshape(-1).shape == (3,)
        assert loaded["TDistort"].reshape(-1).shape == (2,)

        print(f"cam{cam_i}: reproj max err = {err:.2e} px, "
              f"centre err = {c_err:.2e} mm")
        assert err < 1e-3, f"cam{cam_i} reprojection mismatch: {err} px"
        assert c_err < 1e-6, f"cam{cam_i} centre mismatch: {c_err} mm"
        all_params.append(params)

    combined = write_combined(tmp, "rig", all_params)
    assert combined.exists()
    print(f"\nCombined file written: {combined.name}")
    print(f"ALL CHECKS PASSED. Worst reprojection error across cameras: "
          f"{max_err:.2e} px")
    print(f"(temp files in {tmp})")


if __name__ == "__main__":
    test_convention_roundtrip()

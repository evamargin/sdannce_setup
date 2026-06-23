"""Quality control: validate the SAVED .mat files, not OpenCV's in-memory numbers.

Two independent checks:

1. Reprojection error using DANNCE-convention math. We re-implement DANNCE's own
   projection (row vectors, transposed matrices, its distortion model) and feed
   it the values read back from the .mat files. If the saved transposes were
   wrong, this error explodes -- so a low number here proves the file is correct
   for sDANNCE, not just internally consistent in OpenCV.

2. Metric triangulation. We triangulate the shared board's corners from camera
   pairs and compare the reconstructed square size to the true square_mm. This
   confirms the cameras agree in 3D and that world units are millimetres.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import capture as cap_mod
from .board import BoardSpec
from .dannce_export import load_camera_params


# --------------------------------------------------------------------------- #
# DANNCE-convention projection (mirrors dannce/engine ops exactly)
# --------------------------------------------------------------------------- #
def dannce_project(pts3d: np.ndarray, p: dict) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixels using DANNCE's row-vector model.

    p holds the SAVED fields: K (transposed), R (=saved r), t (1x3), RDistort,
    TDistort. Reproduces project_to_2d followed by distortPoints.
    """
    K = np.asarray(p["K"], float)            # 3x3, transposed convention
    R = np.asarray(p["R"], float)            # 3x3 (saved r = R_cv.T)
    t = np.asarray(p["t"], float).reshape(1, 3)
    rad = np.asarray(p["RDistort"], float).ravel()
    tan = np.asarray(p["TDistort"], float).ravel()

    # --- project_to_2d: x = [X Y Z 1] @ (concat([R; t]) @ K) -----------------
    M = np.concatenate((R, t), axis=0) @ K   # 4x3
    homog = np.concatenate((pts3d, np.ones((pts3d.shape[0], 1))), axis=1)  # Nx4
    proj = homog @ M                         # Nx3
    pix = proj[:, :2] / proj[:, 2:3]         # undistorted pixels, Nx2

    # --- distortPoints (intrinsicMatrix is the transposed K) -----------------
    fx, fy = K[0, 0], K[1, 1]
    skew = K[1, 0]
    cx, cy = K[2, 0], K[2, 1]

    x = pix[:, 0] - cx
    y = pix[:, 1] - cy
    yn = y / fy
    xn = (x - skew * yn) / fx

    r2 = xn**2 + yn**2
    r4 = r2 * r2
    r6 = r2 * r4
    k1, k2 = rad[0], rad[1]
    k3 = rad[2] if rad.size > 2 else 0.0
    p1, p2 = tan[0], tan[1]

    alpha = k1 * r2 + k2 * r4 + k3 * r6
    xy = xn * yn
    xd = xn + xn * alpha + (2 * p1 * xy + p2 * (r2 + 2 * xn**2))
    yd = yn + yn * alpha + (p1 * (r2 + 2 * yn**2) + 2 * p2 * xy)

    out = np.empty_like(pix)
    out[:, 0] = xd * fx + cx + skew * yd
    out[:, 1] = yd * fy + cy
    return out


def _opencv_from_saved(p: dict):
    """Recover OpenCV-convention K, dist, R, t from the saved DANNCE params."""
    K_cv = np.asarray(p["K"], float).T
    R_cv = np.asarray(p["R"], float).T
    t_cv = np.asarray(p["t"], float).reshape(3, 1)
    rad = np.asarray(p["RDistort"], float).ravel()
    tan = np.asarray(p["TDistort"], float).ravel()
    k3 = rad[2] if rad.size > 2 else 0.0
    dist = np.array([rad[0], rad[1], tan[0], tan[1], k3], float)
    return K_cv, dist, R_cv, t_cv


def camera_center(p: dict) -> np.ndarray:
    """World-frame position of the camera: C = -R_cv^T @ t_cv."""
    _, _, R_cv, t_cv = _opencv_from_saved(p)
    return (-R_cv.T @ t_cv).ravel()


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _reference_image(ref_dir: Path, camera: str, video_frame: int):
    for q in sorted(ref_dir.iterdir()):
        if q.stem == camera and q.suffix.lower() in cap_mod._IMAGE_EXTS:
            return cv2.imread(str(q))
    for q in sorted(ref_dir.iterdir()):
        if q.stem == camera and q.suffix.lower() in cap_mod._VIDEO_EXTS:
            return cap_mod.read_video_frame(q, video_frame)
    return None


def reprojection_report(camera_names, params_by_cam, ref_dir: Path,
                        spec: BoardSpec, video_frame: int,
                        qc_dir: Path | None = None) -> dict[str, float]:
    """Reproject board corners with SAVED params; print + return per-cam RMS (px)."""
    objp = board_mod.object_points(spec)
    print("\n[QC] Reprojection error from SAVED .mat (DANNCE convention):")
    out = {}
    for cam in camera_names:
        img = _reference_image(ref_dir, cam, video_frame)
        if img is None:
            print(f"  {cam}: reference image missing, skipped")
            continue
        corners = board_mod.find_corners(img, spec)
        if corners is None:
            print(f"  {cam}: board not detected, skipped")
            continue
        proj = dannce_project(objp, params_by_cam[cam])
        diff = proj - corners.reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        out[cam] = rms
        flag = "OK" if rms < 1.0 else "HIGH -- check calibration"
        print(f"  {cam}: {rms:6.3f} px   [{flag}]")

        if qc_dir is not None:
            qc_dir = Path(qc_dir)
            qc_dir.mkdir(parents=True, exist_ok=True)
            vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis = vis.copy()
            for (u, v) in corners.reshape(-1, 2):
                cv2.circle(vis, (int(round(u)), int(round(v))), 4, (0, 255, 0), 1)
            for (u, v) in proj:
                cv2.drawMarker(vis, (int(round(u)), int(round(v))),
                               (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
            cv2.imwrite(str(qc_dir / f"reproj_{cam}.png"), vis)
    return out


def triangulation_report(camera_names, params_by_cam, ref_dir: Path,
                         spec: BoardSpec, video_frame: int) -> None:
    """Triangulate the board corners pairwise; compare spacing to square_mm."""
    # Detect + undistort corners per camera, build OpenCV projection matrices.
    obs, Ps = {}, {}
    for cam in camera_names:
        img = _reference_image(ref_dir, cam, video_frame)
        if img is None:
            continue
        corners = board_mod.find_corners(img, spec)
        if corners is None:
            continue
        K_cv, dist, R_cv, t_cv = _opencv_from_saved(params_by_cam[cam])
        # undistortPoints with P=K gives back pixel coords with distortion removed
        und = cv2.undistortPoints(corners, K_cv, dist, P=K_cv).reshape(-1, 2)
        obs[cam] = und
        Ps[cam] = K_cv @ np.hstack([R_cv, t_cv])

    cams = [c for c in camera_names if c in obs]
    if len(cams) < 2:
        print("\n[QC] Triangulation skipped (need >=2 cameras seeing the board).")
        return

    print("\n[QC] Triangulated board square size vs true "
          f"{spec.square_mm:.2f} mm (per camera pair):")
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            pts4 = cv2.triangulatePoints(Ps[a], Ps[b], obs[a].T, obs[b].T)
            X = (pts4[:3] / pts4[3]).T.reshape(spec.rows, spec.cols, 3)
            # neighbour spacing along both grid directions
            dx = np.linalg.norm(np.diff(X, axis=1), axis=2)
            dy = np.linalg.norm(np.diff(X, axis=0), axis=2)
            spacing = np.concatenate([dx.ravel(), dy.ravel()])
            err_pct = 100.0 * (spacing.mean() - spec.square_mm) / spec.square_mm
            print(f"  {a}-{b}: mean={spacing.mean():7.2f} mm  "
                  f"std={spacing.std():5.2f}  err={err_pct:+5.1f}%")


def camera_layout_report(camera_names, params_by_cam) -> None:
    """Print camera centres and pairwise distances -- compare to your rig."""
    print("\n[QC] Camera centres in world frame (mm) and pairwise distances:")
    centres = {c: camera_center(params_by_cam[c]) for c in camera_names
               if c in params_by_cam}
    for c, C in centres.items():
        print(f"  {c}: [{C[0]:8.1f} {C[1]:8.1f} {C[2]:8.1f}]")
    cams = list(centres)
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            d = np.linalg.norm(centres[cams[i]] - centres[cams[j]])
            print(f"  {cams[i]}-{cams[j]}: {d:8.1f} mm")

"""Shared-board extrinsic calibration -> all cameras in one world frame.

Extrinsics place each camera relative to a common world origin. The trick that
makes this simple for a TTL-synced rig: put ONE chessboard somewhere all four
cameras can see, capture it in the same synchronized instant, and solve each
camera's pose against that single board. Because every camera is solved against
the *same physical board*, they automatically share one coordinate system --
the board's. No pairwise stereo chaining needed.

The board's own frame becomes the world frame:
    origin = inner corner (0,0)
    +X     = along the `cols` direction, +Y along `rows`, +Z out of the board.
Place a board flat where you want world (0,0,0) to be.

We support multiple board PLACEMENTS (each its own synchronized capture). The
world origin is defined by ONE reference placement (the first by default); the
others are used only to cross-check consistency, since a single planar view can
be pose-ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import board as board_mod
from . import capture as cap_mod
from .board import BoardSpec


@dataclass
class ExtrinsicResult:
    camera: str
    R: np.ndarray       # 3x3 rotation, OpenCV convention (world->camera)
    t: np.ndarray       # 3x1 translation, OpenCV convention (mm)
    rms: float          # reprojection error of the board corners (px)
    placement: str      # which placement defined this pose


def _solve_one(image: np.ndarray, K: np.ndarray, dist: np.ndarray,
               spec: BoardSpec):
    """solvePnP for a single camera/board image. Returns (R, t, rms) or None."""
    corners = board_mod.find_corners(image, spec)
    if corners is None:
        return None
    objp = board_mod.object_points(spec)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    # Levenberg-Marquardt refinement tightens the pose beyond the linear solve.
    rvec, tvec = cv2.solvePnPRefineLM(objp, corners, K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    diff = proj.reshape(-1, 2) - corners.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return R, tvec.reshape(3, 1), rms


def _load_placement_image(placement_dir: Path, camera: str,
                          video_frame: int) -> np.ndarray | None:
    """Find this camera's frame in a placement folder (image or video)."""
    for p in sorted(placement_dir.iterdir()):
        if p.stem == camera and p.suffix.lower() in cap_mod._IMAGE_EXTS:
            return cv2.imread(str(p))
    for p in sorted(placement_dir.iterdir()):
        if p.stem == camera and p.suffix.lower() in cap_mod._VIDEO_EXTS:
            return cap_mod.read_video_frame(p, video_frame)
    return None


def calibrate_extrinsics(camera_names: list[str], source_dir: Path,
                         intrinsics: dict, spec: BoardSpec,
                         video_frame: int = 0) -> dict[str, ExtrinsicResult]:
    """Compute extrinsics for all cameras from synchronized board placements.

    intrinsics: {camera_name: IntrinsicResult}
    Layout: source_dir/<placement>/<camera>.png  (one sub-folder per placement)
    The FIRST placement (alphabetically) defines the world frame. Returns
    {camera_name: ExtrinsicResult}.
    """
    placements = [d for d in sorted(Path(source_dir).iterdir()) if d.is_dir()]
    if not placements:
        raise FileNotFoundError(
            f"No placement sub-folders in {source_dir}. Expected e.g. "
            f"{source_dir}/pose1/Camera1.png ...")

    reference = placements[0]
    results: dict[str, ExtrinsicResult] = {}

    for cam in camera_names:
        intr = intrinsics[cam]
        img = _load_placement_image(reference, cam, video_frame)
        if img is None:
            raise FileNotFoundError(
                f"No image/video for {cam} in reference placement {reference.name}. "
                f"Every camera must see the board in the reference placement.")
        solved = _solve_one(img, intr.K, intr.dist, spec)
        if solved is None:
            raise RuntimeError(
                f"Chessboard not found for {cam} in placement {reference.name}. "
                f"The reference placement must show the full board to every camera.")
        R, t, rms = solved
        results[cam] = ExtrinsicResult(camera=cam, R=R, t=t, rms=rms,
                                       placement=reference.name)

    # Cross-check against any additional placements (consistency / ambiguity).
    if len(placements) > 1:
        _consistency_report(camera_names, placements[1:], intrinsics, spec,
                            video_frame, results)
    return results


def _consistency_report(camera_names, extra_placements, intrinsics, spec,
                        video_frame, ref_results):
    """Print inter-camera distances across placements; large spread = problem.

    For a rigid rig, the distance between any two camera centres is constant
    regardless of where the board sits. We compute it from the reference frame
    and from each extra placement (re-derived to a common frame via that
    placement's board) and report the spread, which catches PnP pose flips.
    """
    def centres_from(results):
        # Camera centre in world coords: C = -R^T t
        return {c: (-r.R.T @ r.t).ravel() for c, r in results.items()}

    ref_c = centres_from(ref_results)
    print("\n[extrinsics] consistency check (camera-pair distances, mm):")
    pairs = [(camera_names[i], camera_names[j])
             for i in range(len(camera_names)) for j in range(i + 1, len(camera_names))]
    ref_d = {p: float(np.linalg.norm(ref_c[p[0]] - ref_c[p[1]])) for p in pairs}

    spreads = {p: [ref_d[p]] for p in pairs}
    for pl in extra_placements:
        pl_res = {}
        ok = True
        for cam in camera_names:
            img = _load_placement_image(pl, cam, video_frame)
            solved = _solve_one(img, intrinsics[cam].K, intrinsics[cam].dist, spec) if img is not None else None
            if solved is None:
                ok = False
                break
            R, t, _ = solved
            pl_res[cam] = ExtrinsicResult(cam, R, t, 0.0, pl.name)
        if not ok:
            print(f"  - {pl.name}: board not visible to all cameras, skipped")
            continue
        c = centres_from(pl_res)
        for p in pairs:
            spreads[p].append(float(np.linalg.norm(c[p[0]] - c[p[1]])))

    for p in pairs:
        vals = np.array(spreads[p])
        print(f"  {p[0]}-{p[1]}: mean={vals.mean():8.1f}  spread(max-min)={np.ptp(vals):7.1f}")
    print("  (large spread relative to mean suggests a PnP pose flip or a bad "
          "intrinsic fit.)\n")

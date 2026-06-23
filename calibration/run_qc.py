"""Step 3: quality control on the written .mat files.

Validates the SAVED hires_cam{N}_params.mat (DANNCE convention) by:
  - reprojecting the shared board with DANNCE's own projection math,
  - triangulating the board and checking metric square size,
  - printing the camera layout for a sanity check against your rig.

Usage:
    python run_qc.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from calib.config import load_config
from calib.dannce_export import load_camera_params
from calib import qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    names = cfg.camera_names

    params_dir = cfg.path(cfg["output"]["params_dir"])
    params_by_cam = {}
    for i, cam in enumerate(names, start=1):
        mat = params_dir / f"hires_cam{i}_params.mat"
        if not mat.exists():
            raise FileNotFoundError(f"{mat} missing. Run run_extrinsics.py first.")
        params_by_cam[cam] = load_camera_params(mat)

    # The reference placement is the first sub-folder, matching extrinsics.py.
    ext_src = cfg.path(cfg["extrinsics"]["source_dir"])
    ref_dir = sorted(d for d in ext_src.iterdir() if d.is_dir())[0]
    video_frame = cfg["extrinsics"].get("video_frame", 0)
    qc_dir = cfg.path(cfg["output"]["qc_dir"])

    qc.reprojection_report(names, params_by_cam, ref_dir, spec, video_frame,
                           qc_dir=qc_dir)
    qc.triangulation_report(names, params_by_cam, ref_dir, spec, video_frame)
    qc.camera_layout_report(names, params_by_cam)

    print(f"\nOverlay images (green=detected, red=reprojected) -> {qc_dir}")
    print("If reprojection is sub-pixel and triangulated squares match "
          f"{spec.square_mm} mm, the calibration is ready for Label3D/sDANNCE.")


if __name__ == "__main__":
    main()

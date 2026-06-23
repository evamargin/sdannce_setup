"""Step 2: extrinsic calibration + write DANNCE/sDANNCE .mat files.

Uses the intrinsics from step 1 and synchronized shared-board placements in
<extrinsics.source_dir>/<placement>/<camera>.png to place all cameras in one
world frame, then writes hires_cam{N}_params.mat (and optionally a combined
<prefix>_dannce.mat).

Usage:
    python run_extrinsics.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from calib.config import load_config
from calib.extrinsics import calibrate_extrinsics
from calib.dannce_export import (to_dannce_params, write_camera_params,
                                 write_combined)


def _load_intrinsics(npz_path: Path, camera_names):
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run run_intrinsics.py first.")
    data = np.load(npz_path, allow_pickle=True)
    intr = {}
    for cam in camera_names:
        intr[cam] = SimpleNamespace(
            K=data[f"{cam}__K"],
            dist=data[f"{cam}__dist"],
            image_size=tuple(data[f"{cam}__size"]),
        )
    return intr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    names = cfg.camera_names

    params_dir = cfg.path(cfg["output"]["params_dir"])
    intr = _load_intrinsics(params_dir / "intrinsics.npz", names)

    ext_src = cfg.path(cfg["extrinsics"]["source_dir"])
    video_frame = cfg["extrinsics"].get("video_frame", 0)
    print(f"Extrinsic placements root: {ext_src}\n")

    ext = calibrate_extrinsics(names, ext_src, intr, spec,
                               video_frame=video_frame)

    print("[extrinsics] per-camera board reprojection (reference placement):")
    for cam in names:
        print(f"  {cam}: {ext[cam].rms:.3f} px   (placement '{ext[cam].placement}')")

    # Convert -> DANNCE convention and write files. Camera number is 1-based.
    all_params = []
    for i, cam in enumerate(names, start=1):
        p = to_dannce_params(intr[cam].K, intr[cam].dist,
                             ext[cam].R, ext[cam].t)
        path = write_camera_params(params_dir, i, p)
        all_params.append(p)
        print(f"  wrote {path.name}  (-> {cam})")

    out = cfg["output"]
    if out.get("write_combined", False):
        combined = write_combined(params_dir, out.get("combined_prefix", "rig"),
                                  all_params)
        print(f"  wrote {combined.name} (combined params cell)")

    print("\nDone. Next: python run_qc.py")


if __name__ == "__main__":
    main()

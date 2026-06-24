"""Step 1: intrinsic calibration for every camera.

Reads board + camera config from config.yaml, calibrates each camera from its
footage in <intrinsics.source_dir>/<camera>/, prints reprojection errors, and
saves the results to <output.params_dir>/intrinsics.npz for the next step.

Usage:
    python run_intrinsics.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from calib.config import load_config
from calib.intrinsics import calibrate_camera


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    src = cfg.path(cfg["intrinsics"]["source_dir"])
    ic = cfg["intrinsics"]

    print(f"Board: {spec.cols}x{spec.rows} inner corners, "
          f"{spec.square_mm} mm squares")
    print(f"Intrinsic footage root: {src}\n")

    results = {}
    for cam in cfg.camera_names:
        print(f"=== {cam} ===")
        res = calibrate_camera(
            cam, src, spec,
            min_views=ic["min_views"], max_views=ic["max_views"],
            frame_stride=ic["frame_stride"], sharpness_min=ic["sharpness_min"],
            fix_k3=ic.get("fix_k3", True), fix_aspect=ic.get("fix_aspect", True))
        pv = np.array(res.per_view_error)
        print(f"  views used : {res.n_views}")
        print(f"  image size : {res.image_size}")
        print(f"  RMS reproj : {res.rms:.4f} px "
              f"(per-view min {pv.min():.3f} / max {pv.max():.3f})")
        print(f"  fx,fy      : {res.K[0,0]:.1f}, {res.K[1,1]:.1f}")
        print(f"  cx,cy      : {res.K[0,2]:.1f}, {res.K[1,2]:.1f}")
        print(f"  dist       : {np.array2string(res.dist, precision=4)}\n")
        results[cam] = res

    # Persist for the extrinsic step.
    out_dir = cfg.path(cfg["output"]["params_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "intrinsics.npz"
    payload = {}
    for cam, r in results.items():
        payload[f"{cam}__K"] = r.K
        payload[f"{cam}__dist"] = r.dist
        payload[f"{cam}__size"] = np.array(r.image_size)
        payload[f"{cam}__rms"] = np.array(r.rms)
    np.savez(npz_path, cameras=np.array(cfg.camera_names), **payload)
    print(f"Saved intrinsics -> {npz_path}")
    print("Next: python run_extrinsics.py")


if __name__ == "__main__":
    main()

"""Step 1: intrinsic calibration for every camera.

Reads board + camera config from config.yaml, calibrates each camera from its
footage in <intrinsics.source_dir>/<camera>/, prints reprojection errors, and
saves the results to <output.params_dir>/intrinsics.npz for the next step.

Usage:
    python run_intrinsics.py [--config config.yaml]
    python run_intrinsics.py --source-dir data/intrinsics_small --tag small
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
    ap.add_argument("--source-dir", default=None,
                    help="override intrinsics.source_dir (e.g. compare sets)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output npz (intrinsics_<tag>.npz)")
    ap.add_argument("--fix-principal-point", dest="fix_pp", default=None,
                    choices=["true", "false"],
                    help="override config fix_principal_point")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    src = cfg.path(args.source_dir or cfg["intrinsics"]["source_dir"])
    ic = cfg["intrinsics"]
    fix_pp = (ic.get("fix_principal_point", True) if args.fix_pp is None
              else args.fix_pp == "true")

    print(f"Board: {spec.cols}x{spec.rows} inner corners")
    print(f"Intrinsic footage root: {src}")
    print(f"start_skip_seconds={ic.get('start_skip_seconds', 0)}  "
          f"fix_principal_point={fix_pp}\n")

    results = {}
    for cam in cfg.camera_names:
        print(f"=== {cam} ===")
        res = calibrate_camera(
            cam, src, spec,
            min_views=ic["min_views"], max_views=ic["max_views"],
            frame_stride=ic["frame_stride"], sharpness_min=ic["sharpness_min"],
            fix_k3=ic.get("fix_k3", True), fix_aspect=ic.get("fix_aspect", True),
            zero_tangent=ic.get("zero_tangent", True),
            fix_principal_point=fix_pp,
            start_skip_seconds=ic.get("start_skip_seconds", 0))
        pv = np.array(res.per_view_error)
        print(f"  views used : {res.n_views}")
        print(f"  coverage   : x {res.coverage_x*100:.0f}%  y {res.coverage_y*100:.0f}%"
              f"  (board-corner span across the frame)")
        print(f"  RMS reproj : {res.rms:.4f} px "
              f"(per-view min {pv.min():.3f} / max {pv.max():.3f})")
        print(f"  fx,fy      : {res.K[0,0]:.1f}, {res.K[1,1]:.1f}")
        print(f"  cx,cy      : {res.K[0,2]:.1f}, {res.K[1,2]:.1f}")
        print(f"  dist       : {np.array2string(res.dist, precision=4)}\n")
        results[cam] = res

    # Persist for the extrinsic step.
    out_dir = cfg.path(cfg["output"]["params_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"intrinsics_{args.tag}.npz" if args.tag else "intrinsics.npz"
    npz_path = out_dir / npz_name
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

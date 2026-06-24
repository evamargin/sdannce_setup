"""Step 1.5 (only if using TTL-synced videos): build extrinsic placements.

Scans the frame-synced per-camera videos in <extrinsics.synced.raw_dir>, finds
frame indices where all cameras see the full board, and writes them as
pose1/pose2/... into <extrinsics.source_dir>. Then run run_extrinsics.py.

Usage:
    python run_extract_extrinsics.py [--config config.yaml]
"""
from __future__ import annotations

import argparse

from calib.config import load_config
from calib import synced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    sc = cfg["extrinsics"].get("synced")
    if not sc:
        raise SystemExit("config: add an 'extrinsics.synced' block to use this "
                         "(see config.yaml comments).")

    raw_dir = cfg.path(sc["raw_dir"])
    videos = synced.find_camera_videos(raw_dir, cfg.camera_names)
    print("Synced videos:")
    for cam, p in videos.items():
        print(f"  {cam}: {p.name}")

    print(f"\nScanning (stride {sc.get('scan_stride', 10)}) for frames where "
          f"all {len(videos)} cameras see the board...")
    idxs = synced.find_shared_board_frames(
        videos, spec,
        scan_stride=sc.get("scan_stride", 10),
        n_placements=sc.get("n_placements", 5),
        sharpness_min=sc.get("sharpness_min", 0.0))

    if not idxs:
        raise SystemExit(
            "No frame found where ALL cameras see the full board. Check that a "
            "static board was visible to every camera at the same time, or lower "
            "extrinsics.synced.sharpness_min / scan_stride.")

    out_dir = cfg.path(cfg["extrinsics"]["source_dir"])
    written = synced.extract_placements(videos, idxs, out_dir)
    print(f"\nWrote {len(written)} placements to {out_dir}:")
    for pose, fi in written:
        print(f"  {pose}  <- frame {fi}")
    print("\nNext: python run_extrinsics.py")


if __name__ == "__main__":
    main()

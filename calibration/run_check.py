"""Step 0: sanity-check the setup BEFORE calibrating.

Catches the common foot-guns early:
  - the TTL-synced videos don't all have the SAME frame count (a dropped frame
    means frame N is no longer the same instant across cameras -> bad extrinsics),
  - videos can't be opened, or resolutions disagree,
  - the board isn't actually detectable in a sample frame,
  - intrinsic footage folders are missing or empty.

Exits non-zero if it finds a hard problem, so you can gate calibration on it.

Usage:
    python run_check.py [--config config.yaml] [--exact]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from calib.config import load_config
from calib import capture as cap_mod
from calib import board as board_mod
from calib import synced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--exact", action="store_true",
                    help="count frames by decoding (slow, exact) instead of metadata")
    ap.add_argument("--tol", type=int, default=2,
                    help="tolerated frame-count difference across cameras. Small "
                         "diffs are usually start/stop boundary artifacts and are "
                         "harmless for static-hold extrinsics (default 2).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    spec = cfg.board_spec
    names = cfg.camera_names
    problems, warnings = [], []

    print(f"Board: {spec.cols}x{spec.rows} inner corners, {spec.square_mm} mm\n")

    # --- intrinsic footage present? ----------------------------------------
    intr_root = cfg.path(cfg["intrinsics"]["source_dir"])
    print("Intrinsic footage:")
    for cam in names:
        folder = intr_root / cam
        files = cap_mod.list_sources(folder) if folder.exists() else []
        if not files:
            problems.append(f"intrinsics: no footage in {folder}")
            print(f"  {cam}: MISSING ({folder})")
        else:
            print(f"  {cam}: {len(files)} file(s)")

    # --- TTL-synced videos: the key frame-count check ----------------------
    sc = cfg["extrinsics"].get("synced")
    if sc:
        raw_dir = cfg.path(sc["raw_dir"])
        print(f"\nTTL-synced videos ({raw_dir}):")
        try:
            videos = synced.find_camera_videos(raw_dir, names)
            stats = {c: cap_mod.video_stats(p, exact_count=args.exact)
                     for c, p in videos.items()}
            mode = "exact" if args.exact else "metadata"
            print(f"  {'camera':<10}{'frames':>9}{'fps':>7}{'resolution':>14}  ({mode})")
            counts, sizes = [], set()
            for c in names:
                s = stats[c]
                counts.append(s["frames"])
                sizes.add((s["width"], s["height"]))
                print(f"  {c:<10}{s['frames']:>9}{s['fps']:>7.0f}"
                      f"{str(s['width'])+'x'+str(s['height']):>14}")

            # Frame-count agreement = the sync sanity check.
            if len(set(counts)) == 1:
                print(f"  -> all cameras have {counts[0]} frames. FRAME-SYNC OK.")
            else:
                spread = max(counts) - min(counts)
                if spread <= args.tol:
                    warnings.append(
                        f"frame counts differ by {spread} (min {min(counts)}, max "
                        f"{max(counts)}). Within tolerance ({args.tol}); almost "
                        f"always a start/stop boundary artifact and harmless for "
                        f"static-hold extrinsics. Proceed, but pick placements "
                        f"during the still holds.")
                    print(f"  -> minor diff of {spread} frame(s) (<= tol {args.tol}). OK.")
                else:
                    problems.append(
                        f"frame counts differ across cameras (min {min(counts)}, "
                        f"max {max(counts)}, diff {spread}). A dropped frame breaks "
                        f"frame-sync -> extrinsics will be wrong. Re-record, or trim "
                        f"the videos to a common start/length. (Try --exact to rule "
                        f"out a metadata quirk; --tol to allow small boundary diffs.)")
                    print(f"  -> MISMATCH: frame counts vary by {spread}.")

            if len(sizes) > 1:
                problems.append(f"synced videos have mixed resolutions: {sizes}")
            cap_w, cap_h = cfg["capture"]["width"], cfg["capture"]["height"]
            if sizes and (cap_w, cap_h) not in sizes:
                warnings.append(f"video resolution {sizes} != config capture "
                                f"{cap_w}x{cap_h} (not fatal, just FYI)")

            # Board detectable in a sample (middle) frame of each video?
            print("\nBoard detection (sample frame per camera):")
            for c in names:
                mid = max(0, stats[c]["frames"] // 2)
                frame = cap_mod.read_video_frame(videos[c], mid)
                found = board_mod.find_corners(frame, spec, refine=False) is not None
                print(f"  {c}: frame {mid} -> {'board found' if found else 'no board'}")
                if not found:
                    warnings.append(f"{c}: board not found in sample frame "
                                    f"(fine if the board isn't in view there)")
        except FileNotFoundError as e:
            problems.append(str(e))
            print(f"  {e}")
    else:
        print("\n(no extrinsics.synced block -> skipping synced-video checks)")

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 60)
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        print("\nFix the PROBLEM(s) above before calibrating.")
        raise SystemExit(1)
    print("All checks passed. Safe to calibrate.")


if __name__ == "__main__":
    main()

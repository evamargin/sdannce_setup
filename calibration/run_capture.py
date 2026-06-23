"""Optional helper: grab calibration frames live from the UVC cameras.

Two modes:

  intrinsics  -- live preview per camera; press SPACE to save a frame, 'n' for
                 the next camera, 'q' to quit. Saves to
                 <intrinsics.source_dir>/<camera>/NNN.png. Move the board around
                 between captures (varied angle/distance/position).

  extrinsics  -- live preview of ALL cameras at once; press SPACE to save one
                 synchronized multi-camera placement to
                 <extrinsics.source_dir>/<placement>/<camera>.png, 'q' to quit.
                 Hold the board static, visible to all cameras, for each capture.

This uses software-synchronized grab(); true frame sync comes from your Arduino
TTL trigger. If you already record to video, you can skip this and point the
config at those files instead.

Usage:
    python run_capture.py intrinsics [--config config.yaml]
    python run_capture.py extrinsics [--config config.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calib.config import load_config
from calib import capture as cap_mod
from calib import board as board_mod


def _annotate(frame, spec, text):
    vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    corners = board_mod.find_corners(frame, spec, use_sb=False)  # fast preview
    found = corners is not None
    if found:
        cv2.drawChessboardCorners(vis, spec.size, corners, True)
    color = (0, 200, 0) if found else (0, 0, 200)
    cv2.putText(vis, text + ("  BOARD OK" if found else "  no board"),
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def capture_intrinsics(cfg) -> None:
    spec = cfg.board_spec
    root = cfg.path(cfg["intrinsics"]["source_dir"])
    cw = cfg["capture"]
    for cam in cfg["cameras"]:
        name, idx = cam["name"], cam["uvc_index"]
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(cap_mod.list_sources(out_dir))
        cap = cap_mod.open_uvc(idx, cw["width"], cw["height"], cw["backend"])
        saved = existing
        print(f"\n{name} (index {idx}). SPACE=save, n=next camera, q=quit. "
              f"{existing} frames already present.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print(f"  {name}: read failed"); break
                cv2.imshow("intrinsics", _annotate(frame, spec, f"{name}  saved={saved}"))
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    path = out_dir / f"{saved:03d}.png"
                    cv2.imwrite(str(path), frame)
                    saved += 1
                    print(f"  saved {path.name}")
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return
        finally:
            cap.release()
    cv2.destroyAllWindows()


def capture_extrinsics(cfg) -> None:
    spec = cfg.board_spec
    root = cfg.path(cfg["extrinsics"]["source_dir"])
    cw = cfg["capture"]
    caps, names = [], []
    for cam in cfg["cameras"]:
        caps.append(cap_mod.open_uvc(cam["uvc_index"], cw["width"], cw["height"],
                                     cw["backend"]))
        names.append(cam["name"])

    placement = 1
    print("\nExtrinsics: hold board static, visible to ALL cameras. "
          "SPACE=save placement, q=quit.")
    try:
        while True:
            frames = cap_mod.grab_synchronized(caps)
            if frames is None:
                print("  grab failed"); continue
            tiles = [_annotate(f, spec, n) for f, n in zip(frames, names)]
            h = min(t.shape[0] for t in tiles)
            row = cv2.hconcat([cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
                               for t in tiles])
            cv2.imshow("extrinsics (all cameras)", row)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                pdir = root / f"pose{placement}"
                pdir.mkdir(parents=True, exist_ok=True)
                for f, n in zip(frames, names):
                    cv2.imwrite(str(pdir / f"{n}.png"), f)
                print(f"  saved placement pose{placement}")
                placement += 1
            elif key == ord("q"):
                break
    finally:
        for c in caps:
            c.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["intrinsics", "extrinsics"])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.mode == "intrinsics":
        capture_intrinsics(cfg)
    else:
        capture_extrinsics(cfg)


if __name__ == "__main__":
    main()

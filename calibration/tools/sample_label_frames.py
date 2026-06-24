"""Pick diverse frames to label and assemble a Label3D project.

sDANNCE wants ~250-400 hand-labeled 3D poses, chosen NON-contiguous (spread
across the recording) for diversity. This script:
  1. finds the synced behavior videos (one per camera, named <camera>.<ext>),
  2. selects N well-spread, non-contiguous frame indices,
  3. optionally extracts those frames as images per camera (easy to inspect),
  4. writes a Label3D project .mat with `params` (calibration), `sync` (the frame
     index map), and `camnames`. Label3D adds `labelData` when you save labels.

Run AFTER calibration (needs output/params/hires_cam*_params.mat) and AFTER you
record the mouse.

Usage:
    python tools/sample_label_frames.py --videos-dir data/behavior_raw --n 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calib.config import load_config          # noqa: E402
from calib import synced as synced_mod         # noqa: E402
from calib import capture as cap_mod           # noqa: E402


def pick_indices(n_frames: int, n: int, seed_jitter: bool = True) -> list[int]:
    """N non-contiguous indices spread across [0, n_frames)."""
    if n >= n_frames:
        return list(range(n_frames))
    base = np.linspace(0, n_frames - 1, n).round().astype(int)
    spacing = max(1, n_frames // n)
    if seed_jitter and spacing >= 3:
        # deterministic, index-derived jitter (no RNG -> reproducible)
        jit = ((base * 2654435761) % spacing) - spacing // 2
        base = np.clip(base + jit, 0, n_frames - 1)
    return sorted(set(int(x) for x in base))


def n_keypoints_from_skeleton(path: Path | None) -> int:
    if path and Path(path).exists():
        m = loadmat(str(path))
        return int(m["joint_names"].shape[0] * m["joint_names"].shape[1])
    return 20


def load_param_struct(params_dir: Path, cam_index: int) -> dict:
    m = loadmat(str(params_dir / f"hires_cam{cam_index}_params.mat"))
    return {k: m[k] for k in ("K", "r", "t", "RDistort", "TDistort")}


def build_sync(indices: list[int], n_kp: int) -> dict:
    n = len(indices)
    frame = np.array(indices, dtype=np.float64).reshape(n, 1)
    sid = np.arange(1, n + 1, dtype=np.float64).reshape(n, 1)
    return {
        "data_frame": frame,                       # 0-based frame index in the video
        "data_sampleID": sid,                      # 1..N sample id
        "data_2d": np.zeros((n, 2 * n_kp)),        # filled by Label3D on labeling
        "data_3d": np.zeros((n, 3 * n_kp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--videos-dir", default="data/behavior_raw",
                    help="folder with one synced video per camera (<camera>.avi)")
    ap.add_argument("--n", type=int, default=300, help="frames to label")
    ap.add_argument("--skeleton", default="skeletons/mouse.mat")
    ap.add_argument("--out-dir", default="labeling")
    ap.add_argument("--prefix", default="mouse")
    ap.add_argument("--extract", action="store_true",
                    help="also extract the chosen frames as images per camera")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = cfg.camera_names
    params_dir = cfg.path(cfg["output"]["params_dir"])
    videos_dir = cfg.path(args.videos_dir)
    out_dir = cfg.path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = synced_mod.find_camera_videos(videos_dir, names)
    counts = {c: cap_mod.video_stats(videos[c])["frames"] for c in names}
    n_frames = min(counts.values())
    print("Behavior videos:")
    for c in names:
        print(f"  {c}: {counts[c]} frames  ({videos[c].name})")
    if len(set(counts.values())) > 1:
        print(f"  (using common length {n_frames}; check frame-sync with run_check.py)")

    idx = pick_indices(n_frames, args.n)
    print(f"\nSelected {len(idx)} frames spanning 0..{n_frames-1} "
          f"(min gap {min(np.diff(idx)) if len(idx) > 1 else 0} frames).")

    n_kp = n_keypoints_from_skeleton(cfg.path(args.skeleton))

    # Assemble the Label3D project: params + sync + camnames.
    ncam = len(names)
    params_cell = np.empty((1, ncam), dtype=object)
    sync_cell = np.empty((1, ncam), dtype=object)
    cam_cell = np.empty((1, ncam), dtype=object)
    for i, cam in enumerate(names):
        params_cell[0, i] = load_param_struct(params_dir, i + 1)
        sync_cell[0, i] = build_sync(idx, n_kp)
        cam_cell[0, i] = cam

    proj = out_dir / f"{args.prefix}_label3d.mat"
    savemat(str(proj), {"params": params_cell, "sync": sync_cell,
                        "camnames": cam_cell}, do_compression=True)
    print(f"Wrote Label3D project: {proj}  ({n_kp} keypoints in sync placeholders)")

    # Frame index list for reference / re-mapping labels back to video frames.
    csv = out_dir / f"{args.prefix}_frames.csv"
    csv.write_text("sampleID,frame_index\n" +
                   "\n".join(f"{i+1},{f}" for i, f in enumerate(idx)),
                   encoding="utf-8")
    print(f"Wrote frame list: {csv}")

    if args.extract:
        import cv2
        for cam in names:
            d = out_dir / "frames" / cam
            d.mkdir(parents=True, exist_ok=True)
            for f in idx:
                cv2.imwrite(str(d / f"frame{f:06d}.png"),
                            cap_mod.read_video_frame(videos[cam], f))
        print(f"Extracted {len(idx)} frames/camera to {out_dir/'frames'}")

    print("\nNext: open the project in Label3D (see LABELING.md) and label the frames.")


if __name__ == "__main__":
    main()

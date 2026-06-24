"""Assemble a self-contained Label3D labeling folder to transfer to a MATLAB PC.

Pre-extracts the frames-to-label as PNGs (so MATLAB never has to decode the mp4,
which the user's player can't open either), copies the calibration params and a
skeleton, and writes a ready-to-run MATLAB launcher + README.

Usage:
    python tools/build_label3d_project.py \
        --mouse-dir data/sorted/mouse/2026-06-24T23_47_40 --n 100 --out label3d_mouse1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from calib.config import load_config                  # noqa: E402
from calib import capture as cap_mod                   # noqa: E402
from calib import synced as synced_mod                 # noqa: E402
from tools.sample_label_frames import pick_indices     # noqa: E402


LAUNCHER = r"""%% Label3D launcher (auto-generated). 1) set LABEL3D_PATH  2) Run.
LABEL3D_PATH = 'C:\path\to\Label3D';   % <-- EDIT: your Label3D clone folder

addpath(genpath(LABEL3D_PATH));           % pulls in Animator submodule too
here = fileparts(mfilename('fullpath'));  cd(here);

camNames = {__CAM_LIST__};

% --- calibration params: cell of structs (DANNCE format) ---
params = cell(1, numel(camNames));
for i = 1:numel(camNames)
    params{i} = load(sprintf('hires_cam%d_params.mat', i));
end

% --- videos: stack the PNG frames into (H,W,3,N) uint8 per camera ---
videos = cell(1, numel(camNames));
for i = 1:numel(camNames)
    d = dir(fullfile('frames', camNames{i}, 'frame*.png'));
    [~, order] = sort({d.name});  d = d(order);
    info = imfinfo(fullfile(d(1).folder, d(1).name));
    vid = zeros(info.Height, info.Width, 3, numel(d), 'uint8');
    for k = 1:numel(d)
        img = imread(fullfile(d(k).folder, d(k).name));
        if size(img,3) == 1, img = repmat(img, [1 1 3]); end
        vid(:,:,:,k) = img;
    end
    videos{i} = vid;
end

% --- skeleton + launch ---
skeleton = load('skeleton.mat');
framesToLabel = 1:size(videos{1}, 4);   % label all loaded frames
labelGui = Label3D(params, videos, skeleton, 'framesToLabel', framesToLabel, 'savePath', here);
% If your Label3D version rejects the name-value args, use:
%   labelGui = Label3D(params, videos, skeleton);
"""

README = """Label3D labeling — {cam_n} cameras, {n} frames (mouse take {take})

ON THE MATLAB PC (MATLAB 2019b/2020a/2020b):
1. Make sure you cloned Label3D WITH submodules:
     git clone --recursive https://github.com/diegoaldarondo/Label3D
2. Open launch_label3d.m, set LABEL3D_PATH to your Label3D folder, press Run.
3. The GUI opens with {cam_n} synced views. Click a keypoint in >=2 views; it
   triangulates to 3D and reprojects into the others. Work through the joints.
   - You do NOT have to label all {n} frames — even 20-30 walks the pipeline.
   - Save often (S / the save button). Label3D writes a *_dannce.mat here with
     your labelData + the camera params.
4. Copy the saved *_dannce.mat back to the main PC — that's the label file for
   sDANNCE training (mouse 1). Mouse 2 is for prediction later.

frames_index.csv maps each labeled frame (sampleID) to the original video frame.
Skeleton: see skeleton.mat (edit upstream and rebuild to change keypoints).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mouse-dir", required=True,
                    help="folder with the take's CameraN.mp4 (the take to label)")
    ap.add_argument("--n", type=int, default=100, help="frames to extract for labeling")
    ap.add_argument("--out", default="label3d_mouse1")
    ap.add_argument("--skeleton", default="skeletons/mouse.mat")
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = cfg.camera_names
    mouse_dir = cfg.path(args.mouse_dir)
    out = cfg.path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    videos = synced_mod.find_camera_videos(mouse_dir, names)
    counts = {c: cap_mod.video_stats(videos[c])["frames"] for c in names}
    n_frames = min(counts.values())
    idx = pick_indices(n_frames, args.n)
    print(f"Take: {mouse_dir.name}  (min {n_frames} frames across cams)")
    print(f"Extracting {len(idx)} frames per camera...")

    # extract PNGs
    for cam in names:
        d = out / "frames" / cam
        d.mkdir(parents=True, exist_ok=True)
        for f in idx:
            cv2.imwrite(str(d / f"frame{f:06d}.png"),
                        cap_mod.read_video_frame(videos[cam], f))

    # copy params
    params_dir = cfg.path(cfg["output"]["params_dir"])
    for i in range(1, len(names) + 1):
        src = params_dir / f"hires_cam{i}_params.mat"
        if not src.exists():
            raise FileNotFoundError(f"{src} missing — run calibration first.")
        shutil.copy(src, out / src.name)

    # copy skeleton
    shutil.copy(cfg.path(args.skeleton), out / "skeleton.mat")

    # frames index
    (out / "frames_index.csv").write_text(
        "sampleID,frame_index\n" + "\n".join(f"{i+1},{f}" for i, f in enumerate(idx)),
        encoding="utf-8")

    # launcher + readme
    cam_list = ",".join(f"'{c}'" for c in names)
    (out / "launch_label3d.m").write_text(
        LAUNCHER.replace("__CAM_LIST__", cam_list), encoding="utf-8")
    (out / "README.txt").write_text(
        README.format(cam_n=len(names), n=len(idx), take=mouse_dir.name),
        encoding="utf-8")

    print(f"\nBuilt {out}")
    print(f"  params: hires_cam1..{len(names)}_params.mat")
    print(f"  skeleton.mat, frames/<cam>/*.png ({len(idx)} each), launch_label3d.m, README.txt")
    print(f"Zip '{out.name}' and transfer it to the MATLAB PC.")


if __name__ == "__main__":
    main()

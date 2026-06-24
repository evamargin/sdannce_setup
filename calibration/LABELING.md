# Labeling (Label3D) — runs on a laptop, no GPU

You label ~250–400 diverse 3D poses of the mouse. This is manual work in the
**Label3D** MATLAB GUI; **no GPU needed**. A laptop with MATLAB is ideal. (Over
SSH the GUI needs X-forwarding and is laggy — prefer MATLAB installed locally.)

## Prerequisites
- **MATLAB 2019b / 2020a / 2020b**.
- Clone **Label3D**: `git clone --recursive https://github.com/diegoaldarondo/Label3D`
  (it pulls in the `Animator` submodule — keep `--recursive`).
- Finished calibration: `output/params/hires_cam{1..4}_params.mat` exist.

## Step 1 — Record behavior and stage the videos
- Record ~20–30 min of varied free behavior (4 cameras, TTL-synced, same settings
  as calibration). Don't move the cameras.
- Put the synced videos in `data/behavior_raw/` named `Camera1.avi … Camera4.avi`.
- Verify frame-sync:
  ```powershell
  python run_check.py --config config.yaml
  ```
  (point `extrinsics.synced.raw_dir` at `data/behavior_raw` first, or just confirm
  the four videos have equal frame counts.)

## Step 2 — Make the skeleton (once)
```powershell
python tools/make_mouse_skeleton.py        # writes skeletons/mouse.mat
```
Open `tools/make_mouse_skeleton.py` and edit `JOINTS` / `EDGES` if you want a
different keypoint set. **Fix this before labeling — it can't change later.**

## Step 3 — Pick the frames to label
```powershell
python tools/sample_label_frames.py --videos-dir data/behavior_raw --n 300 --extract
```
This writes `labeling/mouse_label3d.mat` (calibration `params` + `sync` of the
chosen frames + `camnames`), `labeling/mouse_frames.csv` (sampleID → video frame),
and (with `--extract`) the chosen frames as images per camera. Frames are spread
out and **non-contiguous** for diversity.

## Step 4 — Label in MATLAB
Copy `labeling/`, `skeletons/mouse.mat`, and the 4 videos to the MATLAB machine,
then in MATLAB:

```matlab
addpath(genpath('path/to/Label3D'));

L = load('labeling/mouse_label3d.mat');         % params, sync, camnames
S = load('skeletons/mouse.mat');                % joint_names, joints_idx, color
skeleton.joint_names = S.joint_names;
skeleton.joints_idx  = S.joints_idx;
skeleton.color       = S.color;

videos = { 'data/behavior_raw/Camera1.avi', ...
           'data/behavior_raw/Camera2.avi', ...
           'data/behavior_raw/Camera3.avi', ...
           'data/behavior_raw/Camera4.avi' };

labelGui = Label3D(L.params, videos, skeleton); % opens the GUI
```

> Label3D's exact constructor differs slightly between versions. If the line above
> errors, check `Label3D`'s own README/`examples/` — some versions take
> `Label3D(labelGui_file)` or a struct. The pieces it needs are always the same:
> camera **params** (cell of structs), the **videos**, and the **skeleton**.

Labeling tips:
- Click each keypoint in **2+ views**; Label3D triangulates to 3D. Adjust in the
  other views until the 3D point sits right.
- Label **all** keypoints, estimating occluded ones as best you can.
- Keep frames **diverse** (the sampler already spread them out).
- **Save often** (Label3D writes a `*_dannce.mat` with your `labelData`).

## Output
A `*_dannce.mat` containing `labelData` (your 3D labels) + `params` + `sync` +
`camnames`. That file + the videos + params go to the GPU server for training
(see `RUNNING_SDANNCE.md`).

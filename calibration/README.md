# Multi-camera calibration for sDANNCE (OpenCV)

Calibrate a 4-camera (ArduCam OV9281, TTL-synced) rig and export camera
parameters in the exact format **sDANNCE / DANNCE / Label3D** expects -- no
MATLAB required.

The output is one `hires_cam{N}_params.mat` per camera (plus an optional combined
`rig_dannce.mat`), each holding `K, RDistort, TDistort, r, t` in DANNCE's
row-vector convention.

---

## Get the code

```powershell
# SSH (if you've added an SSH key to GitHub)
git clone git@github.com:evamargin/sdannce_setup.git

# or HTTPS
git clone https://github.com/evamargin/sdannce_setup.git

cd sdannce_setup/calibration
```

Then follow `DOCUMENT.md` (step-by-step) or the install section below.

---

## 0. Concepts (read once)

- **Intrinsics** = a camera's lens model: focal lengths + principal point (`K`)
  and lens distortion (`k1,k2,k3,p1,p2`). Calibrated **per camera, alone**.
- **Extrinsics** = where each camera sits in a shared **world frame** (`R`, `t`).
  We define that world frame with **one chessboard seen by all cameras at once**;
  every camera is solved against that same board, so they share one coordinate
  system automatically.
- **Units**: the chessboard `square_mm` sets the scale. Use millimetres -> all 3D
  output is in mm (DANNCE convention).
- **The convention gotcha**: DANNCE projects points as *row vectors*
  (`x = [X Y Z 1] @ M`), so the stored `K`, `r`, `t` are **transposes** of
  OpenCV's. `dannce_export.py` handles this; `run_qc.py` verifies it.

---

## 1. Install

```powershell
pip install -r requirements.txt
```

## 2. Configure

Edit `config.yaml`:
- `board.cols/rows` = **inner-corner** counts (a 10x7-square board -> 9x6), and
  `board.square_mm` = real square size in mm.
- `cameras` = name + UVC index for each camera (order is fixed everywhere; camera
  `i` -> `hires_cam{i}_params.mat`).

## 3. Get calibration footage

You can either **record videos** (e.g. your TTL-triggered acquisition) and point
the config at them, or **capture live** with the helper.

**Folder layout the tools expect:**

```
calibration/
  data/
    intrinsics/                 # per-camera, independent
      Camera1/  *.png or *.avi/*.mp4   (20-40 varied board views)
      Camera2/  ...
      Camera3/  ...
      Camera4/  ...
    extrinsics/                 # synchronized, shared board
      pose1/  Camera1.png Camera2.png Camera3.png Camera4.png
      pose2/  ...               # optional extra placements (consistency check)
```

**Intrinsics footage**: per camera, ~20-40 views of the board at varied angles,
distances, and tilts, with the board reaching into the corners of the frame.
Keep it sharp (no motion blur). Easiest: wave the board in front of all four
cameras and let each harvest its own good frames from the synced recording.

**Extrinsics footage**: hold the board **static**, fully visible to **all four**
cameras, and capture one synchronized frame -> that's `pose1`. Repeat 2-4 more
times at different board positions for a consistency check. **`pose1` defines the
world origin** (board inner corner (0,0)); place it flat where you want world
(0,0,0).

> After extrinsic capture, do **not** move the cameras -- any bump invalidates
> the extrinsics.

**Optional live capture helper:**

```powershell
python run_capture.py intrinsics    # SPACE=save, n=next camera, q=quit
python run_capture.py extrinsics    # SPACE=save a synced placement, q=quit
```

## 4. Run the pipeline

```powershell
python run_intrinsics.py     # -> output/params/intrinsics.npz, prints RMS per cam
python run_extrinsics.py     # -> output/params/hires_cam{1..4}_params.mat
python run_qc.py             # validation + overlays in output/qc/
```

## 5. What "good" looks like

- **Intrinsic RMS**: sub-pixel (< ~0.5 px ideal, < 1 px acceptable).
- **QC reprojection** (from the *saved* .mat): < 1 px.
- **Triangulated square size**: within ~1% of your `square_mm`.
- **Camera layout**: the printed camera centres / pairwise distances match your
  physical rig.

If any camera is off, recapture more/better board views for it and rerun.

---

## 6. Hand-off to sDANNCE / Label3D

The `hires_cam{N}_params.mat` files are the calibration inputs to **Label3D**,
which you use to create the labeled `*_dannce.mat` project (calibration + `sync`
+ `labelData`) that sDANNCE trains on. Your TTL sync makes Label3D's frame-sync
step trivial. See the sDANNCE guide:
- https://github.com/tqxli/sdannce  (`GUIDE.md`, `DEMO.md`)
- Calibration background: https://github.com/spoonsso/dannce/tree/master/calibration

The combined `rig_dannce.mat` (a `params` cell of all four cameras) is provided
for convenience if your workflow prefers a single calibration file.

---

## Module map

| File | Role |
|------|------|
| `calib/board.py` | chessboard geometry + corner detection |
| `calib/capture.py` | UVC live grab + frame harvesting from videos |
| `calib/intrinsics.py` | per-camera intrinsic calibration |
| `calib/extrinsics.py` | shared-board `solvePnP` -> common world frame |
| `calib/dannce_export.py` | OpenCV -> DANNCE convention, write `.mat` |
| `calib/qc.py` | reprojection / triangulation / layout checks |
| `run_*.py` | CLI entry points |
| `selftest.py` | synthetic end-to-end test (no hardware needed) |

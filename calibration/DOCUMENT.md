# Calibration walkthrough (do this in order)

Goal: turn your 4 TTL-synced ArduCam OV9281 cameras into 4 calibration files
(`hires_cam{1..4}_params.mat`) that sDANNCE can use.

You do this **once per rig**. Redo it only if a camera moves or focus changes.

---

## Get the code (new machine)

```powershell
git clone git@github.com:evamargin/sdannce_setup.git   # SSH
# or: git clone https://github.com/evamargin/sdannce_setup.git
cd sdannce_setup/calibration
```

(Need Git? `winget install --id Git.Git -e`. Need an SSH key? see the SSH steps you used to set this up.)

---

## Requirements

You need Python 3.9–3.11 and 4 libraries: `opencv-python`, `numpy`, `scipy`, `pyyaml`.

**Easy install on a new laptop — pick ONE:**

```powershell
# Option A: conda (recommended, self-contained)
conda env create -f environment.yml
conda activate sdannce-calib

# Option B: plain pip / venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Test it works (no cameras needed):

```powershell
python selftest.py        # must print "ALL CHECKS PASSED"
```

---

## The chessboard

- Print a chessboard, **glue it to something rigid and flat** (foam board, acrylic).
  A bent board ruins everything.
- Count the **inner corners** (where 4 squares meet), not the squares.
  A 10×7-squares board = **9×6** inner corners.
- Measure one square edge in **mm** with calipers.
- Put those 3 numbers in `config.yaml` under `board:` (`cols`, `rows`, `square_mm`).
- Also set each camera's `uvc_index` (0,1,2,3) under `cameras:`.

---

## Step 1 — Intrinsics (lens of each camera)

**What it is:** each camera's focal length + distortion. Cameras are calibrated
**independently** here — sync does not matter yet.

**Physically:** hold the board in front of the cameras and move it around so you
cover:
- the whole field of view (centre AND all 4 corners of the image),
- several distances (near + far),
- several tilts (angle the board ±30–45° in different directions).
Keep it **sharp** — move slowly, good light. Global shutter helps, but blur still hurts.

**Acquire** (either way works):
- **Live:** `python run_capture.py intrinsics`
  Preview shows "BOARD OK" when detected. Press **SPACE** to save, **n** for next
  camera, **q** to quit. Grab ~20–40 per camera.
- **From video:** record each camera, drop the files into
  `data/intrinsics/Camera1/`, `data/intrinsics/Camera2/`, … (one folder per camera).

**Run:**
```powershell
python run_intrinsics.py
```
Look for **RMS < 1 px** (ideally < 0.5) per camera. If high, get more/sharper views
for that camera and rerun.

---

## Step 2 — Extrinsics (where the cameras are)

**What it is:** the position/orientation of all cameras in ONE shared world frame.
This is the step that needs synchronization.

**Physically:**
- Place the board **flat where you want the world origin** (e.g. arena floor centre).
  Corner (0,0) of the board = world (0,0,0); the board surface is the Z=0 plane.
- Hold/prop it **completely still**, fully visible to **all 4 cameras at once**.
- This is where the **TTL trigger matters**: one trigger = all 4 cameras capture the
  same instant.
- Repeat for **3–5 different board placements** (move it around the arena). The first
  placement defines the world; the rest are used to auto-check consistency.

**Acquire** (either way):
- **Live:** `python run_capture.py extrinsics`
  All 4 previews shown together; **SPACE** saves one synchronized placement, **q** quits.
- **From files:** save the 4 synced frames as
  `data/extrinsics/pose1/Camera1.png` … `Camera4.png`, then `pose2/…`, etc.

**Run:**
```powershell
python run_extrinsics.py
```
Writes `output/params/hires_cam{1..4}_params.mat`.

> After this, **do not move or bump the cameras.** Any movement invalidates extrinsics
> and you must redo Step 2.

---

## Step 3 — Check it

```powershell
python run_qc.py
```
Good calibration:
- reprojection error **< 1 px** per camera,
- triangulated square size **within ~1%** of your `square_mm`,
- printed camera positions match your real rig.

Overlay images land in `output/qc/` (green = detected corners, red = reprojected).

---

## Step 4 — Use it in sDANNCE

Feed the `hires_cam{N}_params.mat` files + your videos into the **Label3D** GUI to build
the labeled `*_dannce.mat` project that sDANNCE trains on. Your TTL sync makes Label3D's
frame-sync trivial.

---

## Alternative: calibrate from ONE TTL-synced video set (Bonsai)

If you record with Bonsai + Arduino TTL (one pulse triggers a frame in every
camera, so **frame N is the same instant in all cameras**), you can do both
calibrations from a single recording session.

**Record one session per rig** where you:
1. **Wave the board around** in front of the cameras for ~30–60 s — fill each
   camera's view, near + far, tilted, into the corners. (This is the intrinsics
   material; the board does NOT need to be seen by all cameras at once here.)
2. Then **hold the board still** in 4–6 spots, each visible to all cameras,
   ~1 s each. (This is the extrinsics material.)

**Lay the files out like this** (one video per camera):

```
data/
  intrinsics/
    Camera1/session.avi     # each camera's whole video (harvested independently)
    Camera2/session.avi
    Camera3/session.avi
    Camera4/session.avi
  extrinsics_raw/
    Camera1.avi             # the SAME frame-synced videos, named by camera
    Camera2.avi
    Camera3.avi
    Camera4.avi
```

(The `extrinsics_raw` files can be copies of the `intrinsics` videos — same
recording.) Then run:

```powershell
python run_check.py                # Step 0: verify all 4 videos are frame-synced
python run_extract_extrinsics.py   # scans synced videos -> data/extrinsics/poseK/
python run_intrinsics.py           # harvests board views from each camera's video
python run_extrinsics.py
python run_qc.py
```

`run_check.py` confirms every camera's video has the **same frame count** (a
dropped frame in Bonsai would break frame-sync and silently ruin extrinsics),
that resolutions match, and that the board is detectable. It exits with an error
if something's wrong, so always run it first. Add `--exact` to count frames by
decoding if you suspect the container metadata is off.

`run_extract_extrinsics.py` automatically finds the frames where **all** cameras
see the full board and turns them into placements. Tune
`extrinsics.synced.{scan_stride, n_placements, sharpness_min}` in `config.yaml`.

> **Most common failure:** high reprojection / wrong camera distances in QC almost
> always means the **intrinsic** part of the recording didn't cover each camera's
> field of view well enough. Wave the board more — fill the frame, vary depth and
> tilt, reach the corners — and rerun. The static holds only fix extrinsics, not
> intrinsics.

---

## When do the cameras acquire? (summary)

| Step | What the cameras record | Sync needed? |
|------|-------------------------|--------------|
| Intrinsics | each camera, many board views, independently | No |
| Extrinsics | all cameras, same board, same instant | **Yes (TTL)** |
| Experiment | all cameras, the animals | **Yes (TTL)** |

## If something's off

- **"only N usable views"** → record more board images, or lower `sharpness_min` in `config.yaml`.
- **Board never detected** → wrong `cols`/`rows` (use inner-corner counts), or board not fully in frame.
- **High reprojection / weird camera positions** → a camera moved between steps, or a placement didn't show the full board to every camera. Redo that step.

# 3D Animal Pose Estimation with Multi-Camera Calibration & sDANNCE
### Theory + project overview (talk-prep notes)

---

## 1. The problem

We want to track a mouse's body — snout, ears, spine, limbs, tail — as **3D coordinates over
time**. A single camera can't do this: a photo is a flat projection, so depth is ambiguous (a
small near object and a big far object can look identical). The trick is **several cameras
filming the same scene at the same instant** — then geometry recovers the missing depth, the
same way two eyes give you stereo vision.

To turn multi-camera video into 3D, two ingredients are needed:
1. **Calibration** — know exactly how each camera maps the 3D world to its pixels.
2. **A pose estimator** (sDANNCE) — a neural network that, given calibrated multi-view video,
   outputs the animal's 3D keypoints.

---

## 2. The whole pipeline at a glance

```
   Synced cameras            Calibration              Ground truth            Network
  ┌───────────────┐   ┌──────────────────────┐   ┌────────────────┐   ┌──────────────────┐
  │ 4 cams, TTL-  │ → │ intrinsics +         │ → │ Label3D: hand- │ → │ train COM +      │ → 3D pose
  │ synchronized  │   │ extrinsics (geometry)│   │ label 3D poses │   │ train sDANNCE    │   over time
  └───────────────┘   └──────────────────────┘   └────────────────┘   └──────────────────┘
```

Calibration is the foundation: without it, the cameras are just four unrelated videos. With it,
they become one 3D measuring instrument.

---

## 3. The pinhole camera model (the core idea)

Every camera turns a 3D point **X** into a pixel **(u, v)** in two steps:

**Step 1 — Extrinsics: move the world into the camera's point of view.**
The world point is rotated and translated into coordinates centered on *that* camera:
```
X_cam = R · X_world + t
```
- **R** (3×3 rotation) and **t** (3×1 translation) describe **where the camera is and which way
  it points**. This is the *extrinsic* part — it's about the camera's *pose in the world*.

**Step 2 — Intrinsics: project the camera-frame point onto the sensor.**
```
        ⎡ fx  0  cx ⎤
  s·⎡u⎤ = ⎢  0 fy  cy ⎥ · X_cam          (then apply lens distortion)
    ⎣v⎦   ⎣  0  0   1 ⎦
        =       K
```
- **K** is the *intrinsic matrix*: `fx, fy` are focal lengths (how strongly the lens magnifies),
  `cx, cy` the **principal point** (where the optical axis hits the sensor, ~image center).
- **Distortion** (`k1, k2, k3` radial + `p1, p2` tangential) bends straight lines near the edges
  — real lenses aren't perfect pinholes.

Put together: `pixel = K · (R · X + t)`, plus distortion. **Calibration = finding K, distortion,
R, and t for each camera.** Intrinsics and extrinsics are the two halves.

---

## 4. Intrinsic calibration — "what is this lens?"

**Definition:** the camera's *internal* properties — focal length, principal point, lens
distortion (`K` + distortion coefficients). These describe how rays of light become pixels, and
they **do not depend on where the camera sits**. Move the camera across the room and the
intrinsics are unchanged (as long as you don't refocus or zoom).

**Why we need it:** to convert a pixel back into a *direction* (a ray) in the camera's own frame.

**How it's measured:** show the camera a **chessboard** — an object whose geometry we know
exactly (corners on a perfect grid, known square size). The detector finds where each corner
lands in the image; we know where each corner *should* be in metric space. Capturing the board
at **many positions, angles, and distances** gives enough equations to solve for `K` and the
distortion that best explain all the observations (`cv2.calibrateCamera`).

> **Intuition:** known object + where it appears on the sensor → work backwards to the lens.

**The practical catch (we learned this the hard way):** the board must **fill the whole frame,
including the corners**, at varied tilts. If it stays in the center, the math can't pin down the
principal point or the edge distortion, and the focal length drifts to wrong values — even
though the error *looks* low. Coverage is everything.

---

## 5. Extrinsic calibration — "where is each camera?"

**Definition:** each camera's **pose** — its rotation `R` and translation `t` — in a single
**shared world coordinate frame**. This is what lets four separate cameras describe *one* 3D
space.

**Why we need it:** triangulating a 3D point requires knowing the cameras' relative positions
and orientations. Intrinsics alone tell you the *direction* of a ray within each camera;
extrinsics place those rays in a common world so they can intersect.

**How it's measured (shared-board method):** put **one chessboard where all cameras can see it
at the same moment**. For each camera, solve for the pose that explains where it sees the board
(`cv2.solvePnP`, using the already-known intrinsics). Because every camera is solved against the
**same physical board**, they all end up in **one coordinate frame — the board's**. The board
defines the world origin and axes; units come from the square size (we use millimetres).

> **Intuition:** if everyone in a room photographs the *same landmark*, you can work out where
> each person is standing relative to that landmark — and therefore relative to each other.

**Why a *static* board is robust:** if the board isn't moving, it doesn't matter if the cameras
are a few milliseconds out of sync — they all see the same motionless target. (A *moving* board
would require perfect synchronization.)

---

## 6. From 2D back to 3D: triangulation

Once calibrated, each camera turns a pixel into a **ray** in the shared world (a line from the
camera center through that pixel's direction). The **same physical point** seen by two cameras
gives two rays — and **where the rays intersect is the 3D point**.

```
   cam A  •─────────╲           ╱─────────• cam B
                     ╲         ╱
                      ╲       ╱
                       ╲     ╱
                        ╲   ╱
                         ╲ ╱
                          ✦   ← the 3D point (rays meet here)
```

More cameras → more rays → a more robust, less noisy intersection (and you still get a point
even if one view is occluded). **Reprojection error** — how far the reconstructed 3D point lands
from the originally-clicked pixels when projected back — is the standard quality score. Sub-pixel
is excellent; a few pixels is rough.

---

## 7. A convention footnote (worth a sentence in a talk)

DANNCE stores the matrices in a **transposed, row-vector convention** (`x = [X Y Z 1] · M`),
which is the transpose of the more common OpenCV column-vector form. It's purely bookkeeping, but
getting it wrong silently corrupts everything — so our pipeline converts and then *verifies* by
reprojecting from the saved files. Units are millimetres throughout.

---

## 8. sDANNCE — how the network produces 3D pose

Classical approach: detect 2D keypoints in each view independently, then triangulate. This breaks
when a keypoint is occluded or mis-detected in a view. **DANNCE/sDANNCE is smarter — it reasons
in 3D directly:**

1. **Top-down (COM first):** a small network finds the animal's **center of mass (COM)** — its
   rough 3D location — each frame. This crops a **3D cube** around the animal so the heavy
   network only works where the animal is.
2. **Volumetric triangulation:** image features from all cameras are **un-projected into a shared
   3D grid** (using the calibration). The network sees the *fused 3D evidence* and predicts where
   each keypoint sits in the volume — so it naturally handles occlusion and uses all views
   jointly.
3. **"s" = social:** sDANNCE extends this to **multiple interacting animals** (per-animal COMs +
   identity handling).

Because it learns from data, it needs **ground-truth 3D poses** to train on.

---

## 9. Getting ground truth: Label3D

**Label3D** (a MATLAB GUI) is how humans create training labels. You click a keypoint in **≥2
camera views**; using the calibration, it **triangulates** to a 3D point and reprojects it into
the other views so you can check/correct it. You repeat for each keypoint, across many frames.

- Recommended: **~250–400 labeled frames** per animal, chosen **diverse and non-contiguous** (so
  the network sees varied poses, not 400 near-identical frames).
- Output: a label file the network trains on.

> Live bonus: when a clicked point reprojects correctly into the other views, that's your
> **calibration validating itself in real time**.

---

## 10. Training & prediction

On a GPU (Linux): **train the COM network** → **train sDANNCE** (often *finetuning* a pretrained
model, which needs far less data) → **predict** on new video. Output is per-frame 3D coordinates
for every keypoint — the animal's movement reconstructed in 3D, ready for behavioral analysis.

---

## 11. What *we* actually did (and what it taught us)

We built an **OpenCV/Python calibration pipeline** (a free replacement for the standard MATLAB
toolbox) for a rig of **4× ArduCam OV9281 global-shutter cameras**, hardware-synchronized by an
**Arduino TTL trigger**, targeting **sDANNCE** for single-mouse 3D pose.

Concretely, the pipeline:
1. **Checks synchronization** — all four videos must have matching frame counts (a dropped frame
   shifts everything after it).
2. **Intrinsic calibration** per camera from board footage.
3. **Extrinsic calibration** from a shared board → all cameras in one world frame.
4. **Exports** camera parameters in the exact DANNCE format and **QCs** them (reprojection +
   triangulated-square-size + camera layout).

**Hard-won practical lessons (great talk material — "the things nobody tells you"):**
- **Synchronization is non-negotiable.** TTL triggering keeps frame *N* the same instant in every
  camera; we added a check for it.
- **Intrinsic board coverage decides quality.** A board kept central gave a *plausible-looking but
  wrong* calibration (focal length and lens center drifted); filling the frame fixed it.
- **Bigger board beat the smaller one** — coarse corners are detected reliably even at the frame
  edges; fine corners are missed when far/oblique.
- **Weak data needs constraints.** We enforced square pixels (`fx = fy`), zeroed negligible
  tangential distortion, and pinned the principal point to center — physically reasonable
  assumptions that stabilize an under-constrained fit.
- **USB camera order can silently swap between sessions** (re-enumeration). We detected and
  remapped it. *Lesson: record intrinsics and extrinsics in the same session and don't replug.*
- **A static board is immune to sync error** — the safe choice for extrinsics.
- **Exposure/gain must be fixed** (manual), or an auto-loop changes brightness with scene content
  and ruins consistency.

We then **labeled a mouse in Label3D** and **staged everything for GPU training** (COM → sDANNCE
→ prediction).

---

## 12. Talk takeaways (one-liners)

- **Calibration turns pixels into geometry.** It's what makes multiple cameras one 3D instrument.
- **Intrinsics = the lens** (focal length, distortion; independent of position). **Extrinsics =
  the pose** (where the camera is in the world). You need both.
- **Triangulation = intersecting rays.** Calibrated cameras turn matching pixels into 3D points.
- **sDANNCE reasons in 3D** (top-down COM + volumetric fusion), so it beats triangulating
  independently-detected 2D points, especially under occlusion.
- **Garbage in, garbage out:** good 3D needs good *synchronization*, good *coverage* during
  calibration, and *enough* labeled training frames.

---

*Glossary:* **Intrinsics** — internal optics (K + distortion). **Extrinsics** — camera pose
(R, t) in the world. **Reprojection error** — pixel gap when a 3D point is projected back; the
quality metric. **COM** — center of mass, the animal's rough 3D location. **Triangulation** —
recovering a 3D point by intersecting rays from ≥2 calibrated cameras. **TTL** — the electrical
trigger pulse that synchronizes the cameras.

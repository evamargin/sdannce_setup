"""Generate a mouse skeleton .mat for Label3D / sDANNCE.

sDANNCE/Label3D need a skeleton file with at least:
  joint_names : cell array of keypoint names
  joints_idx  : M x 2 connectivity (1-BASED indices into joint_names) defining
                which keypoints are joined by a limb (for drawing + priors)
We also write `color` (M x 3) for nicer limb colors in the GUI.

This ships a sensible ~20-keypoint single-mouse skeleton. EDIT JOINTS / EDGES
below to match the keypoints you actually want to label -- that choice is yours
and must stay fixed once you start labeling.

Usage:
    python make_mouse_skeleton.py [--out skeletons/mouse.mat]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat

# --- keypoints (order defines the index used everywhere downstream) ---------
JOINTS = [
    "Snout",                                  # 1
    "EarL", "EarR",                           # 2, 3
    "Neck",                                   # 4
    "SpineM", "SpineL",                       # 5, 6
    "TailBase", "TailTip",                    # 7, 8
    "ShoulderL", "ElbowL", "PawFL",           # 9, 10, 11
    "ShoulderR", "ElbowR", "PawFR",           # 12, 13, 14
    "HipL", "KneeL", "PawHL",                 # 15, 16, 17
    "HipR", "KneeR", "PawHR",                 # 18, 19, 20
]

# --- limbs as pairs of 1-BASED joint indices --------------------------------
EDGES = [
    (1, 4), (2, 4), (3, 4), (1, 2), (1, 3),   # head
    (4, 5), (5, 6), (6, 7), (7, 8),           # spine + tail
    (4, 9), (9, 10), (10, 11),                # left foreleg
    (4, 12), (12, 13), (13, 14),              # right foreleg
    (6, 15), (15, 16), (16, 17),              # left hindleg
    (6, 18), (18, 19), (19, 20),              # right hindleg
]


def build_skeleton(joints, edges) -> dict:
    names = np.empty((len(joints), 1), dtype=object)
    for i, n in enumerate(joints):
        names[i, 0] = n
    joints_idx = np.array(edges, dtype=np.float64)   # 1-based, M x 2
    # distinct-ish colors per limb (0-1 RGB)
    rng = np.linspace(0, 1, len(edges))
    color = np.stack([rng, 1 - rng, 0.5 * np.ones_like(rng)], axis=1)
    return {"joint_names": names, "joints_idx": joints_idx, "color": color}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="skeletons/mouse.mat")
    args = ap.parse_args()

    if max(max(e) for e in EDGES) > len(JOINTS) or min(min(e) for e in EDGES) < 1:
        raise ValueError("EDGES reference a joint index outside JOINTS range.")

    skel = build_skeleton(JOINTS, EDGES)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out), skel)
    print(f"Wrote {out}")
    print(f"  {len(JOINTS)} keypoints, {len(EDGES)} limbs")
    print(f"  keypoints: {', '.join(JOINTS)}")
    print("Edit JOINTS/EDGES in this script to change the skeleton, then re-run.")


if __name__ == "__main__":
    main()

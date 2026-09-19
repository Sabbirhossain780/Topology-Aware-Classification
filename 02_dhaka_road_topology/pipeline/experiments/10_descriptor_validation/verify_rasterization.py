#!/usr/bin/env python
"""Experiment 10b -- diagnose WHY rotation invariance failed, and test the fix.

verify_logic.py found that both grid_score and |c_2| vary substantially as a
synthetic grid is rotated (spreads 0.225 and 0.359), even though both are
rotation-invariant in exact arithmetic. Two candidate causes:

  H1  RASTERIZATION. PIL draws 1px Bresenham lines with no anti-aliasing. An
      axis-aligned or 45-degree line rasterizes exactly; a 15 or 30 degree line
      stair-steps, and the staircase injects spurious high-frequency energy
      spread across orientations. Note the pattern in the failure: rotations
      0 and 45 scored HIGHEST (0.945, 0.981) and the oblique ones LOWEST
      (0.755-0.800) -- exactly the angles where a raster is exact vs stepped.

  H2  The metric maths is genuinely rotation-variant.

If H1 is the cause, anti-aliased rendering should shrink the spread, and
computing the angular histogram directly from VECTOR geometry (no raster at
all) should remove it almost entirely.

Usage:
    python experiments/10_descriptor_validation/verify_rasterization.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "09_patch_mixture_pilot"))

import numpy as np
from PIL import Image, ImageDraw

from run import PATCH_PX, angular_histogram, grid_score_from_hist, power_spectrum
from verify_logic import angular_signature

ROTS = [0, 7, 15, 22, 30, 37, 45, 52, 60, 67, 75, 82]


def _segments_grid(rot_deg: float, spacing_m: float = 100.0, size: int = PATCH_PX):
    """Return the grid's line segments as ((x1,y1),(x2,y2)) in pixel space."""
    sp = spacing_m / 3.125
    segs = []
    c = np.array([size / 2.0, size / 2.0])
    L = size * 1.6
    for off in (0.0, 90.0):
        a = np.radians(rot_deg + off)
        d = np.array([np.cos(a), np.sin(a)])
        n = np.array([-np.sin(a), np.cos(a)])
        for k in range(-int(size * 1.6 / sp), int(size * 1.6 / sp) + 1):
            base = c + k * sp * n
            segs.append((tuple(base - L * d), tuple(base + L * d)))
    return segs


def render(segs, size: int = PATCH_PX, supersample: int = 1, width: int = 1):
    """Rasterize segments, optionally with supersampling (poor man's AA)."""
    s = size * supersample
    img = Image.new("L", (s, s), 0)
    draw = ImageDraw.Draw(img)
    for p1, p2 in segs:
        draw.line([(p1[0] * supersample, p1[1] * supersample),
                   (p2[0] * supersample, p2[1] * supersample)],
                  fill=255, width=width * supersample)
    if supersample > 1:
        img = img.resize((size, size), Image.BOX)
    return np.asarray(img, dtype=float) / 255.0


def vector_angular_histogram(segs, n_bins: int = 180) -> np.ndarray:
    """Length-weighted bearing histogram straight from geometry -- NO raster.

    Rotation-invariant by construction up to bin quantisation: rotating every
    segment circularly shifts this histogram and nothing else.
    """
    h = np.zeros(n_bins)
    for (x1, y1), (x2, y2) in segs:
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length <= 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        h[int(ang * n_bins / 180.0) % n_bins] += length
    s = h.sum()
    return h / s if s > 0 else h


def spread(vals):
    return float(max(vals) - min(vals))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    methods = {}

    # --- A: current renderer (1px Bresenham, no AA) -----------------------
    # --- B: 4x supersampled                        -----------------------
    # --- C: 4x supersampled, 2px lines             -----------------------
    for name, kw in [("A_raster_1px_noAA", dict(supersample=1, width=1)),
                     ("B_raster_4x_AA", dict(supersample=4, width=1)),
                     ("C_raster_4x_AA_2px", dict(supersample=4, width=2))]:
        gs, c1, c2 = [], [], []
        for r in ROTS:
            img = render(_segments_grid(r), **kw)
            h = angular_histogram(power_spectrum(img))
            sig = angular_signature(h)
            gs.append(grid_score_from_hist(h)[0])
            c1.append(sig[0]); c2.append(sig[1])
        methods[name] = {"grid_score": gs, "c1": c1, "c2": c2}

    # --- D: vector bearings, no raster at all -----------------------------
    gs, c1, c2 = [], [], []
    for r in ROTS:
        h = vector_angular_histogram(_segments_grid(r))
        sig = angular_signature(h)
        gs.append(grid_score_from_hist(h)[0])
        c1.append(sig[0]); c2.append(sig[1])
    methods["D_vector_no_raster"] = {"grid_score": gs, "c1": c1, "c2": c2}

    print("=" * 84)
    print("ROTATION INVARIANCE BY RENDERING METHOD -- synthetic grid, 12 rotations")
    print("=" * 84)
    print(f"{'method':<22}{'grid_score mean':>17}{'spread':>9}"
          f"{'|c2| mean':>12}{'spread':>9}{'|c1| mean':>12}")
    print("-" * 84)
    for name, m in methods.items():
        print(f"{name:<22}{np.mean(m['grid_score']):>17.4f}{spread(m['grid_score']):>9.4f}"
              f"{np.mean(m['c2']):>12.4f}{spread(m['c2']):>9.4f}{np.mean(m['c1']):>12.4f}")

    print("\nPer-rotation grid_score (watch 0 and 45 vs the oblique angles):")
    print(f"{'rot':>5}" + "".join(f"{n.split('_')[0]:>10}" for n in methods))
    print("-" * 84)
    for i, r in enumerate(ROTS):
        print(f"{r:>5}" + "".join(f"{methods[n]['grid_score'][i]:>10.4f}" for n in methods))

    a, d = methods["A_raster_1px_noAA"], methods["D_vector_no_raster"]
    print("\n" + "-" * 84)
    print(f"H1 (rasterization) predicts: spread shrinks as rendering improves.")
    print(f"  A no-AA        grid_score spread = {spread(a['grid_score']):.4f}")
    print(f"  B 4x AA        grid_score spread = {spread(methods['B_raster_4x_AA']['grid_score']):.4f}")
    print(f"  C 4x AA 2px    grid_score spread = {spread(methods['C_raster_4x_AA_2px']['grid_score']):.4f}")
    print(f"  D vector       grid_score spread = {spread(d['grid_score']):.4f}")
    verdict = spread(d["grid_score"]) < 0.05 and spread(d["grid_score"]) < spread(a["grid_score"])
    print(f"\n  -> H1 {'CONFIRMED' if verdict else 'NOT confirmed'}: the variance was "
          f"a rendering artifact, not the metric.")
    print(f"  -> vector method |c2| spread = {spread(d['c2']):.5f} "
          f"(vs {spread(a['c2']):.5f} rasterized)")

    with open(os.path.join(here, "rasterization.json"), "w") as f:
        json.dump({"rotations": ROTS,
                   "methods": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()}
                               for k, v in methods.items()},
                   "h1_confirmed": bool(verdict)}, f, indent=2)
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())

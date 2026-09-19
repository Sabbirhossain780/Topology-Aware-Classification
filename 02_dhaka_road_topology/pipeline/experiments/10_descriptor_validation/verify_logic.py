#!/usr/bin/env python
"""Experiment 10 (verification) -- test every claim made about rotation-invariant
angular descriptors BEFORE building an experiment on top of them.

Each claim is checked against synthetic patterns whose morphology is known by
construction, so "correct" means measurable, not arguable.

CLAIMS UNDER TEST
  C1  Experiment 09's grid_score is already rotation-invariant.
  C2  |c_k| (Fourier magnitudes of the angular histogram) are rotation-invariant.
  C3  On the folded [0,180) axis, a grid shows in |c_2|.
  C4  |c_1| vs |c_2| separates grid from single-corridor.
      (asserted earlier that |c_2| alone means "grid" -- tested here properly)
  C5  grid_score cannot distinguish radial from organic (both score low).
  C6  The |c_k| signature CAN separate grid / corridor / three-way / organic.
  C7  Does |c_k| separate radial from organic, or is a spatial measure needed?
  C8  UFFM's bearing fingerprint + linear-axis Wasserstein is rotation-VARIANT,
      i.e. two identical grids at different rotations look further apart than a
      grid and an organic network.

Usage:
    python experiments/10_descriptor_validation/verify_logic.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "09_patch_mixture_pilot"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import wasserstein_distance

from run import (PATCH_PX, angular_histogram, grid_score_from_hist,
                 power_spectrum, synthetic_random)

RNG = np.random.default_rng(7)


# ------------------------------------------------- synthetic pattern library

def _canvas():
    img = Image.new("L", (PATCH_PX, PATCH_PX), 0)
    return img, ImageDraw.Draw(img)


def _family(draw, angle_deg: float, spacing_px: float):
    """One family of parallel lines at `angle_deg`, spaced `spacing_px` apart."""
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    n = np.array([-np.sin(a), np.cos(a)])
    c = np.array([PATCH_PX / 2.0, PATCH_PX / 2.0])
    L = PATCH_PX * 1.6
    kmax = int(PATCH_PX * 1.6 / spacing_px)
    for k in range(-kmax, kmax + 1):
        base = c + k * spacing_px * n
        draw.line([tuple(base - L * d), tuple(base + L * d)], fill=255, width=1)


def _finish(img):
    return np.asarray(img, dtype=float) / 255.0


def synth_grid(rot_deg=0.0, spacing_m=100.0):
    img, draw = _canvas()
    sp = spacing_m / 3.125
    _family(draw, rot_deg, sp)
    _family(draw, rot_deg + 90.0, sp)
    return _finish(img)


def synth_corridor(rot_deg=0.0, spacing_m=100.0):
    """Parallel roads only -- strongly oriented but NOT a grid."""
    img, draw = _canvas()
    _family(draw, rot_deg, spacing_m / 3.125)
    return _finish(img)


def synth_threeway(rot_deg=0.0, spacing_m=120.0):
    img, draw = _canvas()
    sp = spacing_m / 3.125
    for off in (0.0, 60.0, 120.0):
        _family(draw, rot_deg + off, sp)
    return _finish(img)


def synth_radial(n_spokes=16, offset=(0.0, 0.0), n_rings=0):
    """Spokes from a hub. offset moves the hub off-centre, as in a real patch
    that contains only part of a radial system."""
    img, draw = _canvas()
    c = np.array([PATCH_PX / 2.0 + offset[0], PATCH_PX / 2.0 + offset[1]])
    for i in range(n_spokes):
        a = 2 * np.pi * i / n_spokes
        p2 = c + PATCH_PX * 1.6 * np.array([np.cos(a), np.sin(a)])
        draw.line([tuple(c), tuple(p2)], fill=255, width=1)
    for r in range(1, n_rings + 1):
        rad = r * PATCH_PX / (2.0 * (n_rings + 1))
        draw.ellipse([c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad], outline=255)
    return _finish(img)


# ------------------------------------------------------------- the descriptor

def angular_signature(h: np.ndarray, kmax: int = 8) -> np.ndarray:
    """|c_k| / |c_0| of the 180-bin circular angular histogram.

    A rotation of the network circularly shifts h, which multiplies c_k by a
    pure phase -- so the magnitudes are rotation-invariant BY CONSTRUCTION,
    not by searching over orientations.
    """
    c = np.fft.rfft(h)
    c0 = np.abs(c[0])
    return np.abs(c[1:kmax + 1]) / (c0 + 1e-12)


def describe(img: np.ndarray) -> dict:
    h = angular_histogram(power_spectrum(img))
    sig = angular_signature(h)
    score, orient = grid_score_from_hist(h)
    return {"grid_score": score, "orientation": orient, "sig": sig, "hist": h}


# ------------------------------------------------- UFFM-style bearing compare

def bearing_hist_from_img_lines(angle_list, n_bins=36) -> np.ndarray:
    """Bearing histogram folded to [0,180), matching uffm.bearing_fingerprint."""
    a = np.asarray(angle_list, dtype=float) % 180.0
    hist, _ = np.histogram(a, bins=n_bins, range=(0, 180))
    s = hist.sum()
    return hist / s if s > 0 else np.ones(n_bins) / n_bins


def uffm_distance(b1, b2, n_bins=36) -> float:
    """Exactly uffm.build_distance_matrix's bearing term: LINEAR axis."""
    x = np.linspace(0, 1, n_bins)
    return float(wasserstein_distance(x, x, b1, b2))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    print("=" * 78)
    print("DESCRIPTOR LOGIC VERIFICATION -- synthetic ground truth")
    print("=" * 78)

    # ---- C1 / C2 / C3: rotation invariance -------------------------------
    rots = [0, 15, 30, 45, 60, 75]
    gs, sigs = [], []
    for r in rots:
        d = describe(synth_grid(r))
        gs.append(d["grid_score"])
        sigs.append(d["sig"])
    sigs = np.array(sigs)

    print("\n[C1/C2/C3] GRID ROTATED 0-75 deg -- do the measures stay put?")
    print(f"{'rot':>5}{'grid_score':>12}{'|c1|':>9}{'|c2|':>9}{'|c3|':>9}{'|c4|':>9}")
    print("-" * 78)
    for r, g, s in zip(rots, gs, sigs):
        print(f"{r:>5}{g:>12.4f}{s[0]:>9.4f}{s[1]:>9.4f}{s[2]:>9.4f}{s[3]:>9.4f}")
    print(f"\n  grid_score spread : {max(gs) - min(gs):.5f}  "
          f"(mean {np.mean(gs):.4f})  -> C1 {'HOLDS' if max(gs)-min(gs) < 0.05 else 'FAILS'}")
    c2_spread = sigs[:, 1].max() - sigs[:, 1].min()
    print(f"  |c2| spread       : {c2_spread:.5f}  "
          f"(mean {sigs[:,1].mean():.4f})  -> C2 {'HOLDS' if c2_spread < 0.05 else 'FAILS'}")
    dom = int(np.argmax(sigs.mean(axis=0))) + 1
    print(f"  dominant harmonic : k={dom}  -> C3 {'HOLDS' if dom == 2 else 'FAILS'}"
          f" (claim was k=2)")
    out["C1_grid_score_spread"] = float(max(gs) - min(gs))
    out["C2_c2_spread"] = float(c2_spread)
    out["C3_dominant_k"] = dom

    # ---- C4 / C5 / C6 / C7: does the signature separate morphologies? -----
    patterns = {
        "grid_0deg":      synth_grid(0),
        "grid_37deg":     synth_grid(37),
        "corridor_0deg":  synth_corridor(0),
        "corridor_37deg": synth_corridor(37),
        "threeway":       synth_threeway(11),
        "radial_centred": synth_radial(16, (0, 0)),
        "radial_offset":  synth_radial(16, (45, 30)),
        "organic_a":      synthetic_random(40, 1),
        "organic_b":      synthetic_random(80, 2),
    }
    desc = {k: describe(v) for k, v in patterns.items()}

    print("\n[C4/C5/C6/C7] SIGNATURE ACROSS MORPHOLOGIES")
    print(f"{'pattern':<17}{'grid_score':>11}{'|c1|':>8}{'|c2|':>8}{'|c3|':>8}"
          f"{'|c4|':>8}{'|c6|':>8}")
    print("-" * 78)
    for k, d in desc.items():
        s = d["sig"]
        print(f"{k:<17}{d['grid_score']:>11.4f}{s[0]:>8.4f}{s[1]:>8.4f}{s[2]:>8.4f}"
              f"{s[3]:>8.4f}{s[5]:>8.4f}")

    g, co = desc["grid_0deg"]["sig"], desc["corridor_0deg"]["sig"]
    print(f"\n  C4  grid   |c1|={g[0]:.3f} |c2|={g[1]:.3f}")
    print(f"      corridor |c1|={co[0]:.3f} |c2|={co[1]:.3f}")
    c4 = co[0] - g[0] > 0.15
    print(f"      -> |c1| separates grid from corridor? {'YES' if c4 else 'NO'}"
          f"   C4 {'HOLDS' if c4 else 'FAILS'}")
    print(f"      (earlier claim '|c2| alone = grid' is {'WRONG' if co[1] > 0.3 else 'ok'}"
          f": corridor |c2|={co[1]:.3f})")

    rad = np.mean([desc["radial_centred"]["grid_score"], desc["radial_offset"]["grid_score"]])
    org = np.mean([desc["organic_a"]["grid_score"], desc["organic_b"]["grid_score"]])
    gsc = np.mean([desc["grid_0deg"]["grid_score"], desc["grid_37deg"]["grid_score"]])
    print(f"\n  C5  grid_score: radial={rad:.3f}  organic={org:.3f}  grid={gsc:.3f}")
    c5 = abs(rad - org) < 0.15
    print(f"      -> scalar conflates radial with organic? {'YES' if c5 else 'NO'}"
          f"   C5 {'HOLDS' if c5 else 'FAILS'}")

    print(f"\n  C7  radial vs organic on the SIGNATURE:")
    for k in ("radial_centred", "radial_offset", "organic_a", "organic_b"):
        s = desc[k]["sig"]
        print(f"      {k:<16} " + " ".join(f"{v:.3f}" for v in s[:6]))
    r_off = desc["radial_offset"]["sig"][:6]
    o_mean = np.mean([desc["organic_a"]["sig"][:6], desc["organic_b"]["sig"][:6]], axis=0)
    sep = float(np.abs(r_off - o_mean).max())
    print(f"      max per-harmonic gap, radial_offset vs organic = {sep:.3f}")
    print(f"      -> C7: {'signature DOES separate' if sep > 0.2 else 'signature does NOT separate -- a SPATIAL measure is needed'}")
    out["C5_radial_vs_organic_gridscore_gap"] = float(abs(rad - org))
    out["C7_max_harmonic_gap"] = sep

    # ---- C8: is UFFM's bearing distance rotation-variant? ----------------
    def grid_bearings(rot):
        return [rot % 180, (rot + 90) % 180] * 50

    def organic_bearings(seed):
        return list(np.random.default_rng(seed).uniform(0, 180, 100))

    b_g0 = bearing_hist_from_img_lines(grid_bearings(0))
    b_g45 = bearing_hist_from_img_lines(grid_bearings(45))
    b_org = bearing_hist_from_img_lines(organic_bearings(3))

    d_same_shape = uffm_distance(b_g0, b_g45)      # identical morphology, rotated
    d_diff_shape = uffm_distance(b_g0, b_org)      # different morphology
    print("\n[C8] UFFM BEARING DISTANCE (uffm.py:161, linear-axis Wasserstein)")
    print(f"  d(grid@0deg , grid@45deg) = {d_same_shape:.4f}   <- SAME morphology")
    print(f"  d(grid@0deg , organic)    = {d_diff_shape:.4f}   <- DIFFERENT morphology")
    c8 = d_same_shape >= d_diff_shape
    print(f"  -> rotated twin looks {'FARTHER' if c8 else 'closer'} than a genuine "
          f"mismatch.  C8 {'HOLDS' if c8 else 'FAILS'}")
    # circular-aware control: same data, correct metric
    n = len(b_g0)
    circ = lambda p, q: float(np.abs(np.cumsum(p - q))[:-1].sum()) / n
    print(f"  (for contrast, a circular-aware distance gives "
          f"same-shape {circ(b_g0, b_g45):.4f} vs different-shape {circ(b_g0, b_org):.4f})")
    out["C8_d_rotated_twin"] = d_same_shape
    out["C8_d_true_mismatch"] = d_diff_shape

    # ---- figure ----------------------------------------------------------
    make_figure(here, patterns, desc)
    with open(os.path.join(here, "verification.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, np.generic) else float(v))
                   for k, v in out.items()}, f, indent=2)

    print("\n" + "=" * 78)
    print("See descriptor_validation.png -- renders, spectra and signatures together.")
    print("=" * 78)
    return 0


def make_figure(here, patterns, desc):
    n = len(patterns)
    fig, axes = plt.subplots(3, n, figsize=(2.3 * n, 7.6))
    for i, (name, img) in enumerate(patterns.items()):
        axes[0, i].imshow(1.0 - img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(name.replace("_", "\n"), fontsize=9)
        axes[0, i].axis("off")

        P = power_spectrum(img)
        cy, cx = np.array(P.shape) // 2
        crop = P[cy - 40:cy + 40, cx - 40:cx + 40]
        axes[1, i].imshow(np.log1p(crop / (crop.max() + 1e-12)), cmap="inferno")
        axes[1, i].set_title(f"gs={desc[name]['grid_score']:.3f}", fontsize=9)
        axes[1, i].axis("off")

        s = desc[name]["sig"][:6]
        axes[2, i].bar(range(1, 7), s, color="tab:purple")
        axes[2, i].set_ylim(0, 1.0)
        axes[2, i].set_xlabel("k", fontsize=8)
        axes[2, i].tick_params(labelsize=7)
        if i == 0:
            axes[2, i].set_ylabel("|c_k|", fontsize=9)
    axes[0, 0].text(-0.15, 0.5, "pattern", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11, weight="bold")
    axes[1, 0].text(-0.15, 0.5, "spectrum", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11, weight="bold")
    fig.suptitle("Rotation-invariant angular signature vs the scalar grid_score\n"
                 "(synthetic patterns, morphology known by construction)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(here, "descriptor_validation.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

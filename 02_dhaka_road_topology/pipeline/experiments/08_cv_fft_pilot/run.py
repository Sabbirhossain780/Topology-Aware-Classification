#!/usr/bin/env python
"""Experiment 08 (pilot) — can 2D spectral analysis separate grid from
organic street patterns, where scalar graph metrics could not?

Rationale: a human separates "planned grid" from "organic lanes" instantly
by looking, but 23 scalar graph features could not. The information lives
in the 2D spatial arrangement -- specifically in *periodicity*, which
scalar metrics destroy. A grid repeats at a fixed spacing and orientation;
an organic network does not.

Anti-confound design (the size trap from Experiment 06 must not return):
  - Fixed geographic scale for every zone (same metres-per-pixel), so
    street spacing is comparable rather than normalised away.
  - DC component zeroed before analysis -- DC *is* "amount of ink", the
    exact quantity we must not measure.
  - Low frequencies below a small radius also zeroed (overall blob shape).
  - Power spectrum normalised to sum to 1 before computing any statistic,
    so total energy (i.e. how much road there is) cancels out.

Pilot set: 4 zones whose ground truth is already known from Experiment 07's
manual map inspection -- 2 unmistakable planned grids, 2 unmistakable
organic. If this cannot separate those, the approach is dead.

Usage:
    python experiments/08_cv_fft_pilot/run.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ZONES_DIR = os.path.join("..", "..", "..", "..", "pipeline_outputs", "v2km", "data", "zones")

# Ground truth from Experiment 07's manual OpenStreetMap inspection.
PILOT = {
    "Zone_DH168": "grid",       # numbered roads, RAJUK-style sector
    "Zone_DH167": "grid",       # numbered-road grid, Dhanmondi/Mohammadpur
    "Zone_DH130": "organic",    # curving canal-side lanes, Kalabagan
    "Zone_DH116": "organic",    # fine-grained organic lanes, Mirpur Road
}

SIZE_PX = 512
EXTENT_M = 3600.0     # 2km core + 800m buffer either side
HIGHPASS_RADIUS = 3   # bins around DC to zero out


def render_zone(zone_dir: str, size_px: int = SIZE_PX, extent_m: float = EXTENT_M) -> np.ndarray:
    """Draw a zone's street network as a binary raster at fixed geographic scale."""
    nodes_df = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links_df = pd.read_csv(os.path.join(zone_dir, "links.csv"))

    cx, cy = nodes_df["x_utm"].mean(), nodes_df["y_utm"].mean()
    half = extent_m / 2.0
    scale = size_px / extent_m

    img = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(img)
    pos = dict(zip(nodes_df["node_id"], zip(nodes_df["x_utm"], nodes_df["y_utm"])))

    for _, r in links_df.iterrows():
        u, v = r["from_node"], r["to_node"]
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        p1 = ((x1 - cx + half) * scale, size_px - (y1 - cy + half) * scale)
        p2 = ((x2 - cx + half) * scale, size_px - (y2 - cy + half) * scale)
        draw.line([p1, p2], fill=255, width=1)

    return np.asarray(img, dtype=float) / 255.0


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """Windowed 2D power spectrum, DC/low-frequency removed, normalised to sum 1."""
    hann = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(img * hann))
    P = np.abs(F) ** 2

    cy, cx = np.array(P.shape) // 2
    rr, cc = np.ogrid[: P.shape[0], : P.shape[1]]
    radius = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
    P[radius < HIGHPASS_RADIUS] = 0.0   # kill "amount of ink" + blob shape

    total = P.sum()
    return P / total if total > 0 else P


def regularity_metrics(P: np.ndarray) -> dict:
    """Statistics of a normalised power spectrum. All are energy-independent."""
    nz = P[P > 0]
    spectral_entropy = float(-np.sum(nz * np.log2(nz)) / np.log2(nz.size))

    flat = np.sort(P.ravel())[::-1]
    top_01pct = float(flat[: max(1, int(0.001 * flat.size))].sum())
    top_1pct = float(flat[: max(1, int(0.01 * flat.size))].sum())

    # Angular concentration: grids put energy on a few orientations.
    cy, cx = np.array(P.shape) // 2
    rr, cc = np.ogrid[: P.shape[0], : P.shape[1]]
    ang = (np.degrees(np.arctan2(rr - cy, cc - cx)) % 180).astype(int)
    ang_hist = np.array([P[ang == a].sum() for a in range(180)])
    ang_hist = ang_hist / ang_hist.sum() if ang_hist.sum() > 0 else ang_hist
    anz = ang_hist[ang_hist > 0]
    angular_entropy = float(-np.sum(anz * np.log2(anz)) / np.log2(180))

    return {
        "spectral_entropy": spectral_entropy,
        "top_0.1pct_energy": top_01pct,
        "top_1pct_energy": top_1pct,
        "angular_entropy": angular_entropy,
    }


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    zones_root = os.path.abspath(os.path.join(here, ZONES_DIR))

    results = {}
    fig, axes = plt.subplots(2, len(PILOT), figsize=(4 * len(PILOT), 8))

    for i, (zone_id, truth) in enumerate(PILOT.items()):
        zone_dir = os.path.join(zones_root, zone_id)
        if not os.path.isdir(zone_dir):
            print(f"MISSING: {zone_dir}")
            return 1

        img = render_zone(zone_dir)
        P = power_spectrum(img)
        m = regularity_metrics(P)
        m["ground_truth"] = truth
        m["ink_fraction"] = float(img.mean())   # sanity: how much road, should NOT drive results
        results[zone_id] = m

        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"{zone_id}\n(known: {truth})", fontsize=10)
        axes[0, i].axis("off")

        cy, cx = np.array(P.shape) // 2
        crop = P[cy - 64:cy + 64, cx - 64:cx + 64]
        axes[1, i].imshow(np.log1p(crop / (crop.max() + 1e-12)), cmap="inferno")
        axes[1, i].set_title(f"spectrum\nentropy={m['spectral_entropy']:.4f}", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(here, "pilot_renders_and_spectra.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(here, "result.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 74)
    print("CV / FFT PILOT — can spectral regularity separate grid from organic?")
    print("=" * 74)
    hdr = f"{'zone':<14}{'truth':<10}{'spec_ent':>10}{'ang_ent':>10}{'top0.1%':>10}{'ink':>9}"
    print(hdr)
    print("-" * 74)
    for z, m in results.items():
        print(f"{z:<14}{m['ground_truth']:<10}{m['spectral_entropy']:>10.4f}"
              f"{m['angular_entropy']:>10.4f}{m['top_0.1pct_energy']:>10.4f}{m['ink_fraction']:>9.4f}")

    grids = [m for m in results.values() if m["ground_truth"] == "grid"]
    orgs = [m for m in results.values() if m["ground_truth"] == "organic"]
    print("-" * 74)
    for key in ("spectral_entropy", "angular_entropy", "top_0.1pct_energy"):
        g = np.mean([m[key] for m in grids])
        o = np.mean([m[key] for m in orgs])
        g_rng = (min(m[key] for m in grids), max(m[key] for m in grids))
        o_rng = (min(m[key] for m in orgs), max(m[key] for m in orgs))
        separated = g_rng[1] < o_rng[0] or o_rng[1] < g_rng[0]
        print(f"{key:<22} grid={g:.4f} {g_rng}  organic={o:.4f} {o_rng}  "
              f"{'SEPARATES' if separated else 'overlaps'}")
    print("=" * 74)
    print("Renders + spectra saved to pilot_renders_and_spectra.png — LOOK AT THIS before")
    print("trusting any number above (a broken renderer makes all metrics meaningless).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

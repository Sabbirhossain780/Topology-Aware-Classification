#!/usr/bin/env python
"""Experiment 10c -- how much of Experiment 09's result was the rasterization artifact?

verify_rasterization.py proved the renderer injects a rotation-dependent error of
up to 0.23 in grid_score. Experiment 09 reported a between-zone sd of only 0.045
and a within-zone sd of 0.135 -- both SMALLER than the artifact. So the artifact
is not a rounding detail; it is potentially the same size as the reported signal.

This recomputes Experiment 09's exact 100 patches from VECTOR geometry (no raster,
no FFT) and compares. Segments are subdivided into short steps and each step's
length is accumulated into the bearing histogram of whichever patch it falls in,
so a road crossing a patch boundary contributes to each patch in proportion.

Usage:
    python experiments/10_descriptor_validation/verify_exp09_impact.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "09_patch_mixture_pilot"))

import numpy as np
import pandas as pd

from run import (CORE_M, N_PATCH, PATCH_M, PILOT, ZONES_DIR, grid_score_from_hist)
from verify_logic import angular_signature

STEP_M = 5.0      # segment subdivision for patch assignment
N_BINS = 180


def zone_patch_histograms(zone_dir: str):
    """Length-weighted bearing histogram for every patch of a zone, from vectors."""
    nodes = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links = pd.read_csv(os.path.join(zone_dir, "links.csv"))
    cx, cy = nodes["x_utm"].mean(), nodes["y_utm"].mean()

    pos = dict(zip(nodes["node_id"], zip(nodes["x_utm"], nodes["y_utm"])))
    half = CORE_M / 2.0
    hists = np.zeros((N_PATCH, N_PATCH, N_BINS))
    total_len = np.zeros((N_PATCH, N_PATCH))

    for _, r in links.iterrows():
        u, v = r["from_node"], r["to_node"]
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        bearing = np.degrees(np.arctan2(dy, dx)) % 180.0
        b = int(bearing * N_BINS / 180.0) % N_BINS

        n_steps = max(1, int(np.ceil(length / STEP_M)))
        seg_len = length / n_steps
        ts = (np.arange(n_steps) + 0.5) / n_steps
        xs = x1 + ts * dx
        ys = y1 + ts * dy

        # patch index: col from x (left->right), row from y (top->bottom, to
        # match Experiment 09's raster orientation where y was flipped)
        cols = np.floor((xs - cx + half) / PATCH_M).astype(int)
        rows = np.floor((half - (ys - cy)) / PATCH_M).astype(int)
        ok = (cols >= 0) & (cols < N_PATCH) & (rows >= 0) & (rows < N_PATCH)
        for rr, cc in zip(rows[ok], cols[ok]):
            hists[rr, cc, b] += seg_len
            total_len[rr, cc] += seg_len

    return hists, total_len


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    zones_root = os.path.abspath(os.path.join(here, "..", "09_patch_mixture_pilot", ZONES_DIR))
    old = pd.read_csv(os.path.join(here, "..", "09_patch_mixture_pilot", "patch_metrics.csv"))

    rows = []
    for zone_id in PILOT:
        hists, total_len = zone_patch_histograms(os.path.join(zones_root, zone_id))
        for r in range(N_PATCH):
            for c in range(N_PATCH):
                h = hists[r, c]
                s = h.sum()
                if s <= 0:
                    continue
                h = h / s
                score, orient = grid_score_from_hist(h)
                sig = angular_signature(h)
                rows.append({
                    "zone_id": zone_id, "row": r, "col": c,
                    "patch_id": f"{zone_id}_r{r}c{c}",
                    "grid_score_vec": score, "orientation_vec": orient,
                    "road_len_m": float(total_len[r, c]),
                    **{f"c{k+1}": float(sig[k]) for k in range(6)},
                })

    new = pd.DataFrame(rows)
    merged = new.merge(old[["patch_id", "grid_score", "ink_fraction", "valid"]],
                       on="patch_id", how="inner").rename(
                           columns={"grid_score": "grid_score_raster"})
    merged.to_csv(os.path.join(here, "exp09_vector_recompute.csv"), index=False)
    v = merged[merged["valid"]]

    print("=" * 80)
    print("EXPERIMENT 09 RECOMPUTED FROM VECTOR GEOMETRY -- how much changes?")
    print("=" * 80)
    r = float(np.corrcoef(v["grid_score_raster"], v["grid_score_vec"])[0, 1])
    mad = float(np.abs(v["grid_score_raster"] - v["grid_score_vec"]).mean())
    print(f"\nn = {len(v)} patches")
    print(f"  corr(raster, vector)      = {r:+.3f}")
    print(f"  mean |raster - vector|    = {mad:.3f}")
    print(f"  raster mean {v['grid_score_raster'].mean():.3f}  "
          f"vector mean {v['grid_score_vec'].mean():.3f}")

    print(f"\n{'zone':<14}{'raster med':>12}{'vector med':>12}{'raster sd':>11}"
          f"{'vector sd':>11}")
    print("-" * 80)
    for z in PILOT:
        s = v[v["zone_id"] == z]
        print(f"{z:<14}{s['grid_score_raster'].median():>12.3f}"
              f"{s['grid_score_vec'].median():>12.3f}"
              f"{s['grid_score_raster'].std():>11.3f}{s['grid_score_vec'].std():>11.3f}")

    print("\nTHE HEADLINE NUMBER -- within-zone vs between-zone variation:")
    for col, label in [("grid_score_raster", "raster (Exp 09 as published)"),
                       ("grid_score_vec", "vector (corrected)")]:
        w = v.groupby("zone_id")[col].std().mean()
        b = v.groupby("zone_id")[col].mean().std()
        print(f"  {label:<30} within={w:.3f}  between={b:.3f}  ratio={w/(b+1e-12):.1f}x")

    print("\nWHAT THE VECTOR METHOD ADDS -- angular signature (grid vs corridor):")
    print(f"{'zone':<14}{'|c1| mean':>11}{'|c2| mean':>11}{'gridlike':>10}{'corridor':>10}")
    print("-" * 80)
    for z in PILOT:
        s = v[v["zone_id"] == z]
        gridlike = int(((s["c2"] > 0.35) & (s["c1"] < 0.25)).sum())
        corridor = int((s["c1"] > 0.35).sum())
        print(f"{z:<14}{s['c1'].mean():>11.3f}{s['c2'].mean():>11.3f}"
              f"{gridlike:>10}{corridor:>10}")

    out = {
        "n": int(len(v)),
        "corr_raster_vs_vector": r,
        "mean_abs_diff": mad,
        "raster": {"within": float(v.groupby('zone_id')['grid_score_raster'].std().mean()),
                   "between": float(v.groupby('zone_id')['grid_score_raster'].mean().std())},
        "vector": {"within": float(v.groupby('zone_id')['grid_score_vec'].std().mean()),
                   "between": float(v.groupby('zone_id')['grid_score_vec'].mean().std())},
    }
    with open(os.path.join(here, "exp09_impact.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

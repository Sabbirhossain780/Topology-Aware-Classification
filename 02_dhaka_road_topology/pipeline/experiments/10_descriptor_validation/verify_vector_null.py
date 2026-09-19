#!/usr/bin/env python
"""Experiment 10d -- does the VECTOR grid_score mean anything, or is it inflated?

verify_exp09_impact.py found the vector method gives a mean score of 0.497 vs the
rasterized 0.366. Before treating that as "more signal", two things must be ruled out:

  RISK 1 -- SMALL-SAMPLE INFLATION. The vector histogram puts each segment's whole
    length into ONE of 180 bins. With few segments the histogram is spiky by chance,
    and grid_score maximises over 90 candidate orientations, so it can score high on
    pure noise. The rasterized version smeared energy and did not have this problem
    in the same way. A patch with 20 segments and one with 400 do NOT have the same
    chance baseline -- so a single global threshold would be wrong.

  RISK 2 -- THE RANKING MIGHT NO LONGER MATCH THE EYE. Experiment 09's visual
    validation was done on the RASTER scores. corr(raster, vector) is only +0.398,
    so that validation does not transfer. It has to be redone.

Null model: keep each patch's real segment lengths, assign bearings uniformly at
random, recompute. This holds sample size and length distribution fixed and destroys
only orientation structure -- so it gives each patch its OWN chance baseline.

IMPORTANT -- the unit of randomisation. Roads are subdivided into 5m steps only to
decide WHICH patch each bit of road falls in. The null must NOT randomise steps
independently: all steps of one road share one bearing, so a step-level null would
have hundreds of independent draws where reality has tens, making the null far too
tight and the significance count badly inflated. Bearings are therefore randomised
once per (road segment, patch) pair, carrying that segment's in-patch length.

Usage:
    python experiments/10_descriptor_validation/verify_vector_null.py
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
import pandas as pd

from run import (CORE_M, MIN_INK, N_PATCH, PATCH_M, PATCH_PX, PILOT, ZONES_DIR,
                 grid_score_from_hist, render_zone_full, FULL_PX, M_PER_PX)
from verify_logic import angular_signature

N_BINS = 180
STEP_M = 5.0
N_NULL = 300
RNG = np.random.default_rng(31337)


def patch_segments(zone_dir: str):
    """Per-patch lists of (bearing_deg, length_m), split at patch boundaries."""
    nodes = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links = pd.read_csv(os.path.join(zone_dir, "links.csv"))
    cx, cy = nodes["x_utm"].mean(), nodes["y_utm"].mean()
    pos = dict(zip(nodes["node_id"], zip(nodes["x_utm"], nodes["y_utm"])))
    half = CORE_M / 2.0

    buckets = {(r, c): [] for r in range(N_PATCH) for c in range(N_PATCH)}
    for _, row in links.iterrows():
        u, v = row["from_node"], row["to_node"]
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        bearing = np.degrees(np.arctan2(dy, dx)) % 180.0
        n_steps = max(1, int(np.ceil(length / STEP_M)))
        seg_len = length / n_steps
        ts = (np.arange(n_steps) + 0.5) / n_steps
        cols = np.floor((x1 + ts * dx - cx + half) / PATCH_M).astype(int)
        rows = np.floor((half - (y1 + ts * dy - cy)) / PATCH_M).astype(int)
        ok = (cols >= 0) & (cols < N_PATCH) & (rows >= 0) & (rows < N_PATCH)
        if not ok.any():
            continue
        # One entry per (segment, patch): the segment's length INSIDE that patch,
        # carrying its single real bearing. Steps are not independent draws.
        keys, counts = np.unique(np.stack([rows[ok], cols[ok]], axis=1),
                                 axis=0, return_counts=True)
        for (rr, cc), n in zip(keys, counts):
            buckets[(int(rr), int(cc))].append((bearing, seg_len * int(n)))
    return buckets


def hist_from(bearings, lengths) -> np.ndarray:
    idx = (np.asarray(bearings) * N_BINS / 180.0).astype(int) % N_BINS
    h = np.bincount(idx, weights=lengths, minlength=N_BINS)
    s = h.sum()
    return h / s if s > 0 else h


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    zones_root = os.path.abspath(os.path.join(here, "..", "09_patch_mixture_pilot", ZONES_DIR))

    rows, crops = [], {}
    lo = (FULL_PX - int(round(CORE_M / M_PER_PX))) // 2

    for zone_id in PILOT:
        zone_dir = os.path.join(zones_root, zone_id)
        buckets = patch_segments(zone_dir)
        full = render_zone_full(zone_dir)          # display only
        core = full[lo:lo + N_PATCH * PATCH_PX, lo:lo + N_PATCH * PATCH_PX]

        for (r, c), segs in buckets.items():
            if not segs:
                continue
            bearings = np.array([b for b, _ in segs])
            lengths = np.array([l for _, l in segs])
            h = hist_from(bearings, lengths)
            score, _ = grid_score_from_hist(h)
            sig = angular_signature(h)

            # per-patch null: same lengths, orientation structure destroyed
            nulls = np.empty(N_NULL)
            for i in range(N_NULL):
                rb = RNG.uniform(0, 180, size=lengths.size)
                nulls[i] = grid_score_from_hist(hist_from(rb, lengths))[0]

            patch_img = core[r * PATCH_PX:(r + 1) * PATCH_PX,
                             c * PATCH_PX:(c + 1) * PATCH_PX]
            pid = f"{zone_id}_r{r}c{c}"
            crops[pid] = patch_img
            rows.append({
                "patch_id": pid, "zone_id": zone_id, "row": r, "col": c,
                "grid_score": score,
                "null_mean": float(nulls.mean()), "null_p95": float(np.percentile(nulls, 95)),
                "excess": float(score - nulls.mean()),
                "z": float((score - nulls.mean()) / (nulls.std() + 1e-12)),
                "p": float(((nulls >= score).sum() + 1) / (N_NULL + 1)),
                "n_units": int(lengths.size), "road_len_m": float(lengths.sum()),
                "ink_fraction": float(patch_img.mean()),
                **{f"c{k+1}": float(sig[k]) for k in range(6)},
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(here, "vector_null.csv"), index=False)

    print("=" * 80)
    print("VECTOR grid_score AGAINST A SAMPLE-SIZE-AWARE NULL")
    print("=" * 80)
    print(f"\nn = {len(df)} patches")
    print(f"  raw score      mean {df['grid_score'].mean():.3f}  "
          f"range {df['grid_score'].min():.3f}-{df['grid_score'].max():.3f}")
    print(f"  its own null   mean {df['null_mean'].mean():.3f}  "
          f"range {df['null_mean'].min():.3f}-{df['null_mean'].max():.3f}")
    print(f"  EXCESS         mean {df['excess'].mean():.3f}  "
          f"range {df['excess'].min():.3f}-{df['excess'].max():.3f}")
    print(f"\n  patches with p < 0.05 : {(df['p'] < 0.05).sum()} / {len(df)}")
    print(f"  patches with z > 2    : {(df['z'] > 2).sum()} / {len(df)}")

    r_n = float(np.corrcoef(df["grid_score"], df["n_units"])[0, 1])
    r_nl = float(np.corrcoef(df["null_mean"], df["n_units"])[0, 1])
    print(f"\nRISK 1 CHECK -- small-sample inflation:")
    print(f"  corr(raw score, n_units) = {r_n:+.3f}")
    print(f"  corr(NULL mean, n_units) = {r_nl:+.3f}   "
          f"<- if strongly negative, the null IS sample-size dependent,")
    print(f"     which means a single global threshold on the raw score would be wrong.")
    r_e = float(np.corrcoef(df["excess"], df["n_units"])[0, 1])
    print(f"  corr(EXCESS, n_units)    = {r_e:+.3f}   "
          f"{'<- excess is the safe quantity' if abs(r_e) < 0.3 else '<- STILL confounded'}")

    print(f"\nPER-ZONE (excess over each patch's own null):")
    print(f"{'zone':<14}{'raw med':>10}{'null med':>10}{'excess med':>12}{'p<.05':>8}")
    print("-" * 80)
    for z in PILOT:
        s = df[df["zone_id"] == z]
        print(f"{z:<14}{s['grid_score'].median():>10.3f}{s['null_mean'].median():>10.3f}"
              f"{s['excess'].median():>12.3f}{int((s['p']<0.05).sum()):>8}")

    for col, lab in [("grid_score", "raw"), ("excess", "excess")]:
        w = df.groupby("zone_id")[col].std().mean()
        b = df.groupby("zone_id")[col].mean().std()
        print(f"\n  within/between on {lab:<7}: within={w:.3f} between={b:.3f} "
              f"ratio={w/(b+1e-12):.1f}x")

    make_ranking(here, df, crops)
    with open(os.path.join(here, "vector_null_summary.json"), "w") as f:
        json.dump({"n": int(len(df)), "mean_raw": float(df["grid_score"].mean()),
                   "mean_null": float(df["null_mean"].mean()),
                   "mean_excess": float(df["excess"].mean()),
                   "n_p05": int((df["p"] < 0.05).sum()),
                   "corr_score_n": r_n, "corr_null_n": r_nl, "corr_excess_n": r_e}, f, indent=2)
    print("\n" + "=" * 80)
    print("RISK 2: check vector_ranking_check.png -- the Exp 09 visual validation was")
    print("done on RASTER scores and does NOT transfer (corr was only +0.40).")
    print("=" * 80)
    return 0


def make_ranking(here, df, crops, k=8):
    d = df.sort_values("excess", ascending=False)
    top = d.head(k).to_dict("records")
    bot = d.tail(k)[::-1].to_dict("records")
    fig, axes = plt.subplots(2, k, figsize=(2.1 * k, 5.2))
    for col, m in enumerate(top):
        axes[0, col].imshow(1.0 - crops[m["patch_id"]], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{m['grid_score']:.2f} (exc {m['excess']:+.2f})\n"
                               f"{m['zone_id'][-5:]}", fontsize=8)
        axes[0, col].axis("off")
    for col, m in enumerate(bot):
        axes[1, col].imshow(1.0 - crops[m["patch_id"]], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{m['grid_score']:.2f} (exc {m['excess']:+.2f})\n"
                               f"{m['zone_id'][-5:]}", fontsize=8)
        axes[1, col].axis("off")
    axes[0, 0].text(-0.12, 0.5, f"TOP {k}", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, weight="bold")
    axes[1, 0].text(-0.12, 0.5, f"BOTTOM {k}", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, weight="bold")
    fig.suptitle("VECTOR method, ranked by excess over each patch's own null\n"
                 "re-validation: the raster-based check from Experiment 09 does not transfer",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(here, "vector_ranking_check.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

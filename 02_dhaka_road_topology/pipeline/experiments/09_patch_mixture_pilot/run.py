#!/usr/bin/env python
"""Experiment 09 (pilot) -- can a zone be characterised by its grid-vs-organic
*mixture ratio*, measured at patch scale?

Why this exists: Experiment 08 asked "is this zone a grid or organic?" and got a
null result -- but the figure showed the question itself was ill-posed. At 2km,
every zone is visibly a mixture of planned grid blocks and organic fabric, so a
single zone-level morphology label has nothing to latch onto. The measurable
replacement: chop the zone into patches small enough that one patch plausibly
has ONE morphology, score each patch, and describe the zone by the proportion
of its patches that are grid-like.

Patch size: 400m. Chosen because Dhaka's block scale is 100-300m (the team's own
methodology report), so a 400m window holds a few blocks -- enough for a repeat
to exist if the fabric is regular, small enough not to straddle two different
neighbourhood fabrics.

The measure (grid_score): a grid is not just "oriented", it is oriented in TWO
perpendicular directions at once. A single arterial road is strongly oriented
but is not a grid. So:

    grid_score = 2 * max over a in [0,90) of  min( E(a), E(a+90) )

where E(a) is the fraction of spectral power within +/-10 degrees of orientation
a. Taking the *min* of the orthogonal pair is what rejects the single-road case.

Anti-confound design (Experiment 06's size trap must not return):
  - Fixed geographic scale for every patch (3.125 m/px), never normalised per patch.
  - DC + low frequencies zeroed: DC is "amount of ink", the thing we must not measure.
  - Power spectrum normalised to sum 1, so total road quantity cancels.
  - grid_score is a *fraction* of angular energy -- dimensionless, density-free.
  - Reported alongside ink_fraction so any surviving correlation with patch
    density is visible rather than hidden.

Calibration (the part Experiment 08 lacked): the same metric is run on two
synthetic controls rendered identically -- a perfect 100m-spacing grid and a
Poisson-random network. These bracket the score scale, so a real patch's number
means something instead of floating free.

Per-patch null: the angular histogram is shuffled N times and the metric
recomputed. This absorbs the upward bias from maximising over 90 candidate
orientations, which would otherwise push even isotropic patches above the
analytic 0.233 baseline.

Usage:
    python experiments/09_patch_mixture_pilot/run.py
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

# Same 4 zones as Experiment 08, so the two pilots are directly comparable.
# NOTE: the "prior" labels come from Experiment 07's map inspection, and
# Experiment 08 showed they are unreliable AT ZONE LEVEL. They are carried here
# only as a loose expectation -- this experiment does not score itself against them.
PILOT = {
    "Zone_DH168": "grid-leaning",
    "Zone_DH167": "grid-leaning",
    "Zone_DH130": "organic-leaning",
    "Zone_DH116": "organic-leaning",
}

M_PER_PX = 3.125
EXTENT_M = 3600.0                            # 2km core + 800m buffer either side
FULL_PX = int(round(EXTENT_M / M_PER_PX))    # 1152
CORE_M = 2000.0                              # analyse the zone core only
PATCH_M = 400.0
PATCH_PX = int(round(PATCH_M / M_PER_PX))    # 128
N_PATCH = int(round(CORE_M / PATCH_M))       # 5 -> 25 patches per zone

HIGHPASS_RADIUS = 3       # bins around DC to zero (amount of ink + blob shape)
BAND_HALFWIDTH = 10       # degrees either side of an orientation
MIN_INK = 0.010           # below this a patch is too empty to have a morphology
N_NULL = 200              # angular shuffles per patch
RNG_SEED = 20260919


# ----------------------------------------------------------------- rendering

def render_zone_full(zone_dir: str) -> np.ndarray:
    """Whole zone as a binary raster at fixed geographic scale (3.125 m/px)."""
    nodes_df = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links_df = pd.read_csv(os.path.join(zone_dir, "links.csv"))

    cx, cy = nodes_df["x_utm"].mean(), nodes_df["y_utm"].mean()
    half = EXTENT_M / 2.0
    scale = FULL_PX / EXTENT_M

    img = Image.new("L", (FULL_PX, FULL_PX), 0)
    draw = ImageDraw.Draw(img)
    pos = dict(zip(nodes_df["node_id"], zip(nodes_df["x_utm"], nodes_df["y_utm"])))

    for _, r in links_df.iterrows():
        u, v = r["from_node"], r["to_node"]
        if u not in pos or v not in pos:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        p1 = ((x1 - cx + half) * scale, FULL_PX - (y1 - cy + half) * scale)
        p2 = ((x2 - cx + half) * scale, FULL_PX - (y2 - cy + half) * scale)
        draw.line([p1, p2], fill=255, width=1)

    return np.asarray(img, dtype=float) / 255.0


def synthetic_grid(spacing_m: float = 100.0) -> np.ndarray:
    """Control: a perfect orthogonal grid at Dhaka block spacing."""
    img = Image.new("L", (PATCH_PX, PATCH_PX), 0)
    draw = ImageDraw.Draw(img)
    step = spacing_m / M_PER_PX
    k = 0
    while k * step < PATCH_PX:
        p = k * step
        draw.line([(p, 0), (p, PATCH_PX)], fill=255, width=1)
        draw.line([(0, p), (PATCH_PX, p)], fill=255, width=1)
        k += 1
    return np.asarray(img, dtype=float) / 255.0


def synthetic_random(n_nodes: int = 40, seed: int = 0) -> np.ndarray:
    """Control: a Poisson-random network, each node joined to its 3 nearest."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, PATCH_PX, size=(n_nodes, 2))
    img = Image.new("L", (PATCH_PX, PATCH_PX), 0)
    draw = ImageDraw.Draw(img)
    for i in range(n_nodes):
        d = np.hypot(*(pts - pts[i]).T)
        for j in np.argsort(d)[1:4]:
            draw.line([tuple(pts[i]), tuple(pts[j])], fill=255, width=1)
    return np.asarray(img, dtype=float) / 255.0


# ------------------------------------------------------------------- metrics

_ANG_MAP = None


def _angle_map(shape) -> np.ndarray:
    """Orientation (0-179 deg) of every spectral bin. Cached -- shape is constant."""
    global _ANG_MAP
    if _ANG_MAP is None or _ANG_MAP.shape != shape:
        cy, cx = np.array(shape) // 2
        rr, cc = np.ogrid[: shape[0], : shape[1]]
        _ANG_MAP = (np.degrees(np.arctan2(rr - cy, cc - cx)) % 180).astype(np.int32)
    return _ANG_MAP


def power_spectrum(img: np.ndarray) -> np.ndarray:
    """Windowed 2D power spectrum, DC/low-frequency removed, normalised to sum 1."""
    hann = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
    P = np.abs(np.fft.fftshift(np.fft.fft2(img * hann))) ** 2

    cy, cx = np.array(P.shape) // 2
    rr, cc = np.ogrid[: P.shape[0], : P.shape[1]]
    P[np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2) < HIGHPASS_RADIUS] = 0.0

    total = P.sum()
    return P / total if total > 0 else P


def angular_histogram(P: np.ndarray) -> np.ndarray:
    ang = _angle_map(P.shape)
    h = np.bincount(ang.ravel(), weights=P.ravel(), minlength=180)
    s = h.sum()
    return h / s if s > 0 else h


def _band(h: np.ndarray, centre: int) -> float:
    idx = np.arange(centre - BAND_HALFWIDTH, centre + BAND_HALFWIDTH + 1) % 180
    return float(h[idx].sum())


def grid_score_from_hist(h: np.ndarray) -> tuple[float, int]:
    """2 * max_a min(E(a), E(a+90)).  Returns (score, best orientation)."""
    best, best_a = -1.0, 0
    for a in range(90):
        m = min(_band(h, a), _band(h, a + 90))
        if m > best:
            best, best_a = m, a
    return 2.0 * best, best_a


def patch_metrics(img: np.ndarray, rng: np.random.Generator) -> dict:
    P = power_spectrum(img)
    h = angular_histogram(P)
    score, orient = grid_score_from_hist(h)

    # Null: same angular energy, orientation structure destroyed. Absorbs the
    # optimistic bias of maximising over 90 candidate orientations.
    nulls = np.empty(N_NULL)
    for i in range(N_NULL):
        nulls[i] = grid_score_from_hist(rng.permutation(h))[0]

    nz = h[h > 0]
    return {
        "grid_score": score,
        "orientation_deg": int(orient),
        "null_mean": float(nulls.mean()),
        "null_p95": float(np.percentile(nulls, 95)),
        "excess": float(score - nulls.mean()),
        "z_vs_null": float((score - nulls.mean()) / (nulls.std() + 1e-12)),
        "angular_entropy": float(-np.sum(nz * np.log2(nz)) / np.log2(180)),
        "ink_fraction": float(img.mean()),
    }


# ---------------------------------------------------------------------- main

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    zones_root = os.path.abspath(os.path.join(here, ZONES_DIR))
    rng = np.random.default_rng(RNG_SEED)

    # --- calibrate the scale on synthetic controls ------------------------
    controls = {
        "synthetic_grid_100m": patch_metrics(synthetic_grid(100.0), rng),
        "synthetic_grid_200m": patch_metrics(synthetic_grid(200.0), rng),
        "synthetic_random_a": patch_metrics(synthetic_random(40, 1), rng),
        "synthetic_random_b": patch_metrics(synthetic_random(80, 2), rng),
    }

    # --- score every patch of every pilot zone ----------------------------
    lo = (FULL_PX - int(round(CORE_M / M_PER_PX))) // 2
    rows, renders = [], {}

    for zone_id, prior in PILOT.items():
        zone_dir = os.path.join(zones_root, zone_id)
        if not os.path.isdir(zone_dir):
            print(f"MISSING: {zone_dir}")
            return 1

        full = render_zone_full(zone_dir)
        core = full[lo:lo + N_PATCH * PATCH_PX, lo:lo + N_PATCH * PATCH_PX]
        renders[zone_id] = (core, [])

        for r in range(N_PATCH):
            for c in range(N_PATCH):
                patch = core[r * PATCH_PX:(r + 1) * PATCH_PX,
                             c * PATCH_PX:(c + 1) * PATCH_PX]
                m = patch_metrics(patch, rng)
                m.update(zone_id=zone_id, zone_prior=prior, row=r, col=c,
                         patch_id=f"{zone_id}_r{r}c{c}",
                         valid=bool(m["ink_fraction"] >= MIN_INK))
                rows.append(m)
                renders[zone_id][1].append((patch, m))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(here, "patch_metrics.csv"), index=False)

    # --- figures ----------------------------------------------------------
    make_contact_sheet(here, renders)
    make_overlay(here, renders)
    make_distributions(here, df, controls)
    make_ranking_check(here, renders)

    # --- report -----------------------------------------------------------
    print("=" * 78)
    print("PATCH MIXTURE PILOT -- is grid-ness measurable at 400m, and does it vary?")
    print("=" * 78)
    print(f"{N_PATCH}x{N_PATCH} patches of {PATCH_M:.0f}m over each zone's central "
          f"{CORE_M:.0f}m, at {M_PER_PX} m/px.\n")

    print("CALIBRATION -- same metric on synthetic controls:")
    print(f"{'control':<22}{'grid_score':>12}{'null_mean':>12}{'z':>9}{'ink':>9}")
    print("-" * 78)
    for k, m in controls.items():
        print(f"{k:<22}{m['grid_score']:>12.4f}{m['null_mean']:>12.4f}"
              f"{m['z_vs_null']:>9.1f}{m['ink_fraction']:>9.4f}")

    valid = df[df["valid"]]
    ctl_rand = np.mean([controls[k]["grid_score"] for k in controls if "random" in k])
    ctl_grid = np.mean([controls[k]["grid_score"] for k in controls if "grid" in k])
    print(f"\nScale anchored: random ~= {ctl_rand:.3f}, perfect grid ~= {ctl_grid:.3f}")
    print(f"Real patches span {valid['grid_score'].min():.3f} - "
          f"{valid['grid_score'].max():.3f} (median {valid['grid_score'].median():.3f})")

    print(f"\nPER-ZONE ({len(df) - len(valid)} of {len(df)} patches dropped as too empty, "
          f"ink < {MIN_INK}):")
    print(f"{'zone':<14}{'prior':<18}{'n':>4}{'median':>9}{'mean':>9}{'max':>9}"
          f"{'sd':>8}{'ink':>8}")
    print("-" * 78)
    for z in PILOT:
        v = valid[valid["zone_id"] == z]
        print(f"{z:<14}{PILOT[z]:<18}{len(v):>4}{v['grid_score'].median():>9.3f}"
              f"{v['grid_score'].mean():>9.3f}{v['grid_score'].max():>9.3f}"
              f"{v['grid_score'].std():>8.3f}{v['ink_fraction'].mean():>8.4f}")

    # Does the score just track how much road is in the patch? (Experiment 06's ghost)
    r_ink = float(np.corrcoef(valid["grid_score"], valid["ink_fraction"])[0, 1])
    print(f"\nCONFOUND CHECK  corr(grid_score, ink_fraction) = {r_ink:+.3f}"
          f"   {'<-- WATCH THIS' if abs(r_ink) > 0.4 else '(weak, good)'}")

    # Within-zone spread vs between-zone spread: is "mixture" the right word?
    within = valid.groupby("zone_id")["grid_score"].std().mean()
    between = valid.groupby("zone_id")["grid_score"].mean().std()
    print(f"Mean WITHIN-zone sd = {within:.3f}   BETWEEN-zone sd of means = {between:.3f}")
    print(f"  -> ratio {within / (between + 1e-12):.1f}x. If >> 1, zones are internally")
    print("     heterogeneous (mixtures) and a single zone-level label cannot be right.")

    summary = {
        "params": {"patch_m": PATCH_M, "m_per_px": M_PER_PX, "core_m": CORE_M,
                   "band_halfwidth_deg": BAND_HALFWIDTH, "min_ink": MIN_INK,
                   "n_null": N_NULL, "seed": RNG_SEED},
        "controls": controls,
        "control_anchors": {"random": ctl_rand, "perfect_grid": ctl_grid},
        "n_patches": int(len(df)), "n_valid": int(len(valid)),
        "corr_score_vs_ink": r_ink,
        "within_zone_sd_mean": float(within),
        "between_zone_sd_of_means": float(between),
        "per_zone": {z: {
            "prior": PILOT[z],
            "n_valid": int((valid["zone_id"] == z).sum()),
            "median": float(valid[valid["zone_id"] == z]["grid_score"].median()),
            "mean": float(valid[valid["zone_id"] == z]["grid_score"].mean()),
            "max": float(valid[valid["zone_id"] == z]["grid_score"].max()),
            "sd": float(valid[valid["zone_id"] == z]["grid_score"].std()),
        } for z in PILOT},
    }
    with open(os.path.join(here, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 78)
    print("Figures written. LOOK AT patch_contact_sheet.png FIRST -- if the patches it")
    print("ranks highest are not visibly the grid-like ones, every number above is void.")
    return 0


def make_contact_sheet(here, renders):
    """Patches sorted by grid_score, descending. The metric's honesty check."""
    fig, axes = plt.subplots(4 * N_PATCH, N_PATCH,
                             figsize=(2.0 * N_PATCH, 2.15 * 4 * N_PATCH))
    for zi, (zone_id, (_core, patches)) in enumerate(renders.items()):
        order = sorted(patches, key=lambda pm: -pm[1]["grid_score"])
        for k, (patch, m) in enumerate(order):
            ax = axes[zi * N_PATCH + k // N_PATCH, k % N_PATCH]
            ax.imshow(1.0 - patch, cmap="gray", vmin=0, vmax=1)
            ok = m["ink_fraction"] >= MIN_INK
            ax.set_title(f"{m['grid_score']:.3f}" + ("" if ok else "  (empty)"),
                         fontsize=9, color="black" if ok else "red")
            ax.axis("off")
            if k == 0:
                ax.text(-0.08, 0.5, zone_id, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(here, "patch_contact_sheet.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)


def make_ranking_check(here, renders, k: int = 8):
    """The validation that matters: pooled across all zones, do the top-scored
    patches look grid-like and the bottom-scored ones look organic? If a human
    cannot see the difference between the two rows, the metric is measuring noise.
    """
    allp = [(p, m) for _, ps in renders.values() for p, m in ps
            if m["ink_fraction"] >= MIN_INK]
    allp.sort(key=lambda pm: -pm[1]["grid_score"])
    top, bottom = allp[:k], allp[-k:][::-1]

    fig, axes = plt.subplots(2, k, figsize=(2.1 * k, 5.0))
    for col, (patch, m) in enumerate(top):
        axes[0, col].imshow(1.0 - patch, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{m['grid_score']:.3f}\n{m['zone_id'][-5:]}", fontsize=9)
        axes[0, col].axis("off")
    for col, (patch, m) in enumerate(bottom):
        axes[1, col].imshow(1.0 - patch, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{m['grid_score']:.3f}\n{m['zone_id'][-5:]}", fontsize=9)
        axes[1, col].axis("off")
    axes[0, 0].text(-0.12, 0.5, f"TOP {k}", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=13, weight="bold")
    axes[1, 0].text(-0.12, 0.5, f"BOTTOM {k}", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=13, weight="bold")
    fig.suptitle("Does the score rank patches the way a human would?\n"
                 "highest-scoring 400m patches (top) vs lowest (bottom), all zones pooled",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(here, "ranking_check.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_overlay(here, renders):
    """Where the grid-like patches sit inside each zone."""
    fig, axes = plt.subplots(1, len(renders), figsize=(5.2 * len(renders), 5.8))
    cmap = plt.get_cmap("RdYlBu_r")
    scores = [m["grid_score"] for _, ps in renders.values() for _, m in ps]
    vmin, vmax = min(scores), max(scores)

    for i, (zone_id, (core, patches)) in enumerate(renders.items()):
        ax = axes[i]
        ax.imshow(1.0 - core, cmap="gray", vmin=0, vmax=1)
        for patch, m in patches:
            x0, y0 = m["col"] * PATCH_PX, m["row"] * PATCH_PX
            ok = m["ink_fraction"] >= MIN_INK
            col = cmap((m["grid_score"] - vmin) / (vmax - vmin + 1e-12)) if ok else "0.7"
            ax.add_patch(plt.Rectangle((x0, y0), PATCH_PX, PATCH_PX,
                                       facecolor=col, alpha=0.38,
                                       edgecolor="k", linewidth=0.6))
            ax.text(x0 + PATCH_PX / 2, y0 + PATCH_PX / 2,
                    f"{m['grid_score']:.2f}" if ok else "-",
                    ha="center", va="center", fontsize=8, weight="bold")
        ax.set_title(f"{zone_id}\nprior: {PILOT[zone_id]}", fontsize=11)
        ax.axis("off")
    fig.suptitle("400m patch grid-score, warm = more grid-like "
                 "(grey = too empty to score)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(here, "zone_mixture_overlay.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def make_distributions(here, df, controls):
    """Per-zone score distributions against the synthetic control anchors."""
    valid = df[df["valid"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rng = np.random.default_rng(0)

    for i, z in enumerate(PILOT):
        v = valid[valid["zone_id"] == z]["grid_score"].values
        ax.scatter(np.full_like(v, i) + rng.uniform(-0.13, 0.13, v.size), v,
                   s=42, alpha=0.75, edgecolor="k", linewidth=0.4, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color="k", linewidth=2.2, zorder=4)

    for k, m in controls.items():
        colour = "tab:blue" if "random" in k else "tab:red"
        ax.axhline(m["grid_score"], color=colour, linestyle="--",
                   linewidth=1.3, alpha=0.8)
        ax.text(len(PILOT) - 0.45, m["grid_score"], f" {k}", fontsize=8,
                va="bottom", color=colour)

    ax.set_xticks(range(len(PILOT)))
    ax.set_xticklabels([f"{z}\n{PILOT[z]}" for z in PILOT], fontsize=9)
    ax.set_ylabel("patch grid_score  (2 x min orthogonal-pair angular energy)")
    ax.set_title("Every 400m patch, against synthetic controls\n"
                 "black bar = zone median", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(here, "score_distributions.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

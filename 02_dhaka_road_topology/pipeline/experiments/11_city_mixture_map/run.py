#!/usr/bin/env python
"""Experiment 11 -- city-wide morphology map: does grid/organic MIXTURE RATIO vary
across Dhaka, even though morphology TYPE does not?

Experiments 08-09 established that a 2km zone has no single morphology, and
Experiment 10 established the correct way to measure one: length-weighted bearing
histograms straight from the graph's vector geometry. That is exact,
rotation-invariant (a grid scores 1.0000 at every rotation) and needs no rendering,
which makes a city-wide sliding-window run affordable.

This is the properly-powered version of Experiment 09. n=4 zones could not test
whether mixture ratio varies between zones; 224 zones can.

DESIGN DECISIONS AND WHY
  Vector geometry, not raster -- Experiment 10 showed the renderer injected a
    rotation-dependent error of up to 0.23, larger than the between-zone signal.

  Sliding window (400m window, 100m stride), not a fixed lattice -- a fixed lattice
    dices a grid area broader than one patch and penalises it at the seams. With
    vector scoring, overlapping windows cost almost nothing.

  True zone-grid centroids from metadata.csv, not node centroids -- node centroids
    are pulled toward wherever roads happen to be, so zone cores would overlap and
    leave gaps. Using the real grid centres makes the 2km cores tile the city exactly.

  Window centres confined to the 2km CORE, while segments are drawn from the full
    buffered zone -- this is the pipeline's existing two-tier design (Casali &
    Heinimann 2019; He et al. 2023), applied one scale down. A window at the core
    edge legitimately sees 200m into the buffer instead of being clipped.

  Integral image over per-cell bearing histograms -- makes each window O(180)
    instead of O(segments), which is what makes ~90,000 windows tractable.

  Null by lookup, VALIDATED on a subsample -- a per-window permutation null at this
    scale is too slow. The null depends on how many independent bearings a window
    contains, so it is precomputed as a function of that and then checked against
    exact segment-level nulls on randomly chosen windows. If that check fails, the
    excess values are not trustworthy and the script says so.

  OSM completeness carried through as a first-class flag -- the proposal already
    flags zones below 0.5 completeness (predominantly Old Dhaka's informal lanes).
    At window scale this matters MORE, not less: sparse mapping reads as "no grid
    structure". Flagged windows are scored but marked, never silently mixed in.

Usage:
    python experiments/11_city_mixture_map/run.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
ZONES_DIR = os.path.join(ROOT, "pipeline_outputs", "v2km", "data", "zones")
META_CSV = os.path.join(ROOT, "pipeline_outputs", "v2km", "outputs", "csv", "metadata.csv")

CELL_M = 100.0                       # spatial grid / sliding stride
WIN_CELLS = 4                        # 4 x 100m = 400m window
HALF_EXTENT_M = 1800.0               # 1km core half-width + 800m buffer
N_CELL = int(2 * HALF_EXTENT_M / CELL_M)          # 36
CORE_HALF_M = 1000.0
STEP_M = 10.0                        # segment subdivision for cell assignment
N_BINS = 180
BAND = 10                            # +/- degrees for an orientation band

MIN_ROAD_M = 500.0                   # below this a 400m window is too empty to score
COMPLETENESS_MIN = 0.5               # the proposal's own flag threshold
N_NULL_SIM = 300
N_VALIDATE = 300                     # windows given an exact segment-level null
RNG = np.random.default_rng(11_2026)


# --------------------------------------------------------------- core scoring

def grid_scores_batch(H: np.ndarray) -> np.ndarray:
    """2 * max_a min(E(a), E(a+90)) for a batch of normalised histograms (N, 180)."""
    k = 2 * BAND + 1
    ext = np.concatenate([H, H[:, : k - 1]], axis=1)
    cs = np.zeros((H.shape[0], ext.shape[1] + 1))
    cs[:, 1:] = np.cumsum(ext, axis=1)
    S = cs[:, k : k + N_BINS] - cs[:, 0:N_BINS]      # S[:,j] = sum of ext[j:j+k]
    band = np.roll(S, BAND, axis=1)                  # band[:,a] centred on a
    mins = np.minimum(band[:, :90], band[:, 90:180])
    return 2.0 * mins.max(axis=1)


def signature_batch(H: np.ndarray, kmax: int = 3) -> np.ndarray:
    """|c_k| / |c_0| -- rotation-invariant by construction (Experiment 10)."""
    c = np.fft.rfft(H, axis=1)
    return np.abs(c[:, 1 : kmax + 1]) / (np.abs(c[:, :1]) + 1e-12)


# ------------------------------------------------------------------ null model

def build_null_lookup(length_pool: np.ndarray):
    """null_mean / null_p95 as a function of the number of independent bearings."""
    ns = np.unique(np.round(np.logspace(np.log10(3), np.log10(1500), 26)).astype(int))
    means, p95s = [], []
    for n in ns:
        H = np.zeros((N_NULL_SIM, N_BINS))
        for i in range(N_NULL_SIM):
            b = (RNG.uniform(0, 180, n) * N_BINS / 180.0).astype(int) % N_BINS
            w = RNG.choice(length_pool, size=n)
            H[i] = np.bincount(b, weights=w, minlength=N_BINS)
        H /= H.sum(axis=1, keepdims=True) + 1e-12
        s = grid_scores_batch(H)
        means.append(s.mean())
        p95s.append(np.percentile(s, 95))
    return ns, np.array(means), np.array(p95s)


def lookup(ns, vals, n_query):
    return np.interp(np.clip(n_query, ns[0], ns[-1]), ns, vals)


# ------------------------------------------------------------------ zone pass

def zone_cell_data(zone_dir: str, cx: float, cy: float):
    """Per-cell bearing histograms, road length, and segment-incidence counts."""
    nodes = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links = pd.read_csv(os.path.join(zone_dir, "links.csv"))
    if nodes.empty or links.empty:
        return None

    pos_x = pd.Series(nodes["x_utm"].values, index=nodes["node_id"].values)
    pos_y = pd.Series(nodes["y_utm"].values, index=nodes["node_id"].values)
    m = links["from_node"].isin(pos_x.index) & links["to_node"].isin(pos_x.index)
    if m.sum() == 0:
        return None
    fn = links.loc[m, "from_node"].values
    tn = links.loc[m, "to_node"].values
    x1, y1 = pos_x[fn].values, pos_y[fn].values
    x2, y2 = pos_x[tn].values, pos_y[tn].values

    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    keep = length > 0
    x1, y1, dx, dy, length = x1[keep], y1[keep], dx[keep], dy[keep], length[keep]
    if length.size == 0:
        return None

    bearing_bin = ((np.degrees(np.arctan2(dy, dx)) % 180.0)
                   * N_BINS / 180.0).astype(int) % N_BINS

    n_steps = np.maximum(1, np.ceil(length / STEP_M).astype(int))
    seg_id = np.repeat(np.arange(length.size), n_steps)
    step_len = np.repeat(length / n_steps, n_steps)
    bb = np.repeat(bearing_bin, n_steps)
    t = (np.concatenate([np.arange(k) for k in n_steps]) + 0.5) / np.repeat(n_steps, n_steps)
    px = np.repeat(x1, n_steps) + t * np.repeat(dx, n_steps)
    py = np.repeat(y1, n_steps) + t * np.repeat(dy, n_steps)

    col = np.floor((px - cx + HALF_EXTENT_M) / CELL_M).astype(int)
    row = np.floor((HALF_EXTENT_M - (py - cy)) / CELL_M).astype(int)
    ok = (col >= 0) & (col < N_CELL) & (row >= 0) & (row < N_CELL)
    if not ok.any():
        return None
    col, row, bb, step_len, seg_id = col[ok], row[ok], bb[ok], step_len[ok], seg_id[ok]

    flat = (row * N_CELL + col) * N_BINS + bb
    H_cells = np.bincount(flat, weights=step_len,
                          minlength=N_CELL * N_CELL * N_BINS).reshape(N_CELL, N_CELL, N_BINS)

    # distinct (segment, cell) incidences -> proxy for independent bearings
    inc = np.unique(np.stack([seg_id, row, col], axis=1), axis=0)
    C_cells = np.bincount(inc[:, 1] * N_CELL + inc[:, 2],
                          minlength=N_CELL * N_CELL).reshape(N_CELL, N_CELL).astype(float)

    return H_cells, C_cells, (px, py, bb, step_len, seg_id, length)


def integral(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:])
    out[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return out


def window_indices():
    """Cell offsets whose window CENTRE falls inside the 2km core."""
    lo = int((HALF_EXTENT_M - CORE_HALF_M) / CELL_M) - WIN_CELLS // 2   # 6
    hi = lo + int(2 * CORE_HALF_M / CELL_M)                             # 26
    idx = np.arange(lo, hi)
    rr = np.repeat(idx, idx.size)
    cc = np.tile(idx, idx.size)
    return rr, cc


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()

    meta = pd.read_csv(META_CSV)
    zone_ids = sorted(os.listdir(ZONES_DIR))
    meta = meta[meta["zone_id"].isin(zone_ids)].set_index("zone_id")
    print(f"zones with metadata: {len(meta)} / {len(zone_ids)} on disk")

    rr, cc = window_indices()
    n_win_per_zone = rr.size
    print(f"windows per zone: {n_win_per_zone}  ({int(np.sqrt(n_win_per_zone))}^2 "
          f"at {CELL_M:.0f}m stride, {WIN_CELLS*CELL_M:.0f}m window)")

    # length pool for the null, from a sample of zones
    pool = []
    for z in list(meta.index)[::20]:
        p = os.path.join(ZONES_DIR, z, "links.csv")
        if os.path.exists(p):
            pool.append(pd.read_csv(p, usecols=["length_m"])["length_m"].values)
    length_pool = np.concatenate(pool)
    length_pool = length_pool[length_pool > 0]
    print(f"null length pool: {length_pool.size} segments, "
          f"median {np.median(length_pool):.1f}m")

    print("building null lookup ...")
    ns, null_means, null_p95s = build_null_lookup(length_pool)

    validate_zones = set(RNG.choice(list(meta.index),
                                    size=min(60, len(meta)), replace=False))
    rows, val_rows = [], []

    for zi, zone_id in enumerate(meta.index):
        if zi % 40 == 0:
            print(f"  [{zi:>3}/{len(meta)}] {zone_id}  ({time.time()-t0:.0f}s)")
        zd = os.path.join(ZONES_DIR, zone_id)
        cx = float(meta.loc[zone_id, "centroid_x_utm"])
        cy = float(meta.loc[zone_id, "centroid_y_utm"])
        data = zone_cell_data(zd, cx, cy)
        if data is None:
            continue
        H_cells, C_cells, raw = data

        IH, IC = integral(H_cells), integral(C_cells)
        W = (IH[rr + WIN_CELLS, cc + WIN_CELLS] - IH[rr, cc + WIN_CELLS]
             - IH[rr + WIN_CELLS, cc] + IH[rr, cc])
        Cn = (IC[rr + WIN_CELLS, cc + WIN_CELLS] - IC[rr, cc + WIN_CELLS]
              - IC[rr + WIN_CELLS, cc] + IC[rr, cc])

        road_len = W.sum(axis=1)
        good = road_len > 0
        if not good.any():
            continue
        Hn = np.zeros_like(W)
        Hn[good] = W[good] / road_len[good, None]

        scores = np.zeros(W.shape[0])
        scores[good] = grid_scores_batch(Hn[good])
        sig = np.zeros((W.shape[0], 3))
        sig[good] = signature_batch(Hn[good])

        nm = lookup(ns, null_means, Cn)
        np95 = lookup(ns, null_p95s, Cn)

        wx = cx - HALF_EXTENT_M + (cc + WIN_CELLS / 2.0) * CELL_M
        wy = cy + HALF_EXTENT_M - (rr + WIN_CELLS / 2.0) * CELL_M
        comp = float(meta.loc[zone_id, "osm_completeness"])

        for i in range(W.shape[0]):
            rows.append({
                "zone_id": zone_id, "wx": wx[i], "wy": wy[i],
                "grid_score": scores[i], "null_mean": nm[i], "null_p95": np95[i],
                "excess": scores[i] - nm[i], "above_p95": bool(scores[i] > np95[i]),
                "c1": sig[i, 0], "c2": sig[i, 1], "c3": sig[i, 2],
                "road_len_m": road_len[i], "n_seg": Cn[i],
                "osm_completeness": comp,
                "sparse": bool(road_len[i] < MIN_ROAD_M),
                "low_completeness": bool(comp < COMPLETENESS_MIN),
            })

        # ---- exact segment-level null on a few windows, to validate the lookup
        if zone_id in validate_zones:
            px, py, bb, step_len, seg_id, seg_len_all = raw
            for i in RNG.choice(np.flatnonzero(good), size=min(5, int(good.sum())),
                                replace=False):
                x0, y0 = wx[i] - WIN_CELLS * CELL_M / 2, wy[i] - WIN_CELLS * CELL_M / 2
                sel = ((px >= x0) & (px < x0 + WIN_CELLS * CELL_M) &
                       (py >= y0) & (py < y0 + WIN_CELLS * CELL_M))
                if sel.sum() < 3:
                    continue
                sids = seg_id[sel]
                uniq, inv = np.unique(sids, return_inverse=True)
                seg_w = np.bincount(inv, weights=step_len[sel])
                seg_b = bb[sel][np.unique(inv, return_index=True)[1]]
                if uniq.size < 3:
                    continue
                Hn_sim = np.zeros((N_NULL_SIM, N_BINS))
                for s in range(N_NULL_SIM):
                    rb = (RNG.uniform(0, 180, uniq.size) * N_BINS / 180.0).astype(int) % N_BINS
                    Hn_sim[s] = np.bincount(rb, weights=seg_w, minlength=N_BINS)
                Hn_sim /= Hn_sim.sum(axis=1, keepdims=True) + 1e-12
                true_null = grid_scores_batch(Hn_sim).mean()
                val_rows.append({"zone_id": zone_id, "n_true_seg": int(uniq.size),
                                 "n_est": float(Cn[i]), "true_null": float(true_null),
                                 "lookup_null": float(nm[i])})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(here, "windows.csv"), index=False)
    val = pd.DataFrame(val_rows)
    val.to_csv(os.path.join(here, "null_validation.csv"), index=False)

    print(f"\nscored {len(df)} windows across {df['zone_id'].nunique()} zones "
          f"in {time.time()-t0:.0f}s")
    report(here, df, val, ns, null_means)
    return 0


def report(here, df, val, ns, null_means):
    print("=" * 82)
    print("EXPERIMENT 11 -- CITY-WIDE MORPHOLOGY MAP")
    print("=" * 82)

    # ---- null validation FIRST: without it the excess values mean nothing ----
    print("\n[1] NULL LOOKUP VALIDATION (exact segment-level null on sampled windows)")
    if len(val) < 10:
        print("  INSUFFICIENT validation sample -- excess values NOT trustworthy.")
        ok_null = False
    else:
        err = val["lookup_null"] - val["true_null"]
        r = float(np.corrcoef(val["lookup_null"], val["true_null"])[0, 1])
        bias = float(err.mean())
        print(f"  n = {len(val)} windows")
        print(f"  corr(lookup, true) = {r:+.3f}")
        print(f"  mean error         = {bias:+.4f}   (positive = lookup too strict)")
        print(f"  max |error|        = {float(err.abs().max()):.4f}")
        print(f"  corr(n_est, n_true_seg) = "
              f"{float(np.corrcoef(val['n_est'], val['n_true_seg'])[0,1]):+.3f} "
              f"(n_est over-counts segments spanning cells)")
        ok_null = abs(bias) < 0.03 and float(err.abs().max()) < 0.10
        print(f"  -> lookup {'USABLE' if ok_null else 'NOT RELIABLE -- report raw scores only'}")

    valid = df[~df["sparse"]]
    flagged = valid[valid["low_completeness"]]
    clean = valid[~valid["low_completeness"]]
    print(f"\n[2] COVERAGE")
    print(f"  total windows        {len(df)}")
    print(f"  too sparse to score  {int(df['sparse'].sum())} "
          f"({100*df['sparse'].mean():.1f}%)")
    print(f"  low OSM completeness {len(flagged)} "
          f"({100*len(flagged)/max(len(valid),1):.1f}% of scorable)")
    print(f"  clean windows        {len(clean)}")

    print(f"\n[3] SCORE DISTRIBUTION (clean windows only)")
    print(f"  grid_score  mean {clean['grid_score'].mean():.3f}  "
          f"median {clean['grid_score'].median():.3f}  "
          f"range {clean['grid_score'].min():.3f}-{clean['grid_score'].max():.3f}")
    print(f"  null        mean {clean['null_mean'].mean():.3f}")
    print(f"  excess      mean {clean['excess'].mean():.3f}")
    print(f"  above null p95   {int(clean['above_p95'].sum())} / {len(clean)} "
          f"({100*clean['above_p95'].mean():.1f}%)")

    # ---- THE QUESTION: does mixture ratio vary between zones? ---------------
    g = clean.groupby("zone_id")
    zone = pd.DataFrame({
        "n": g.size(),
        "mixture_ratio": g["above_p95"].mean(),
        "median_excess": g["excess"].median(),
        "mean_score": g["grid_score"].mean(),
    })
    zone = zone[zone["n"] >= 50]
    zone.to_csv(os.path.join(here, "zone_mixture.csv"))

    within = g["grid_score"].std().mean()
    between = g["grid_score"].mean().std()
    print(f"\n[4] THE QUESTION -- does MIXTURE RATIO vary between zones?")
    print(f"  zones with >=50 clean windows: {len(zone)}")
    print(f"  mixture_ratio (fraction of windows above null p95):")
    print(f"    mean {zone['mixture_ratio'].mean():.3f}   sd {zone['mixture_ratio'].std():.3f}")
    print(f"    range {zone['mixture_ratio'].min():.3f} - {zone['mixture_ratio'].max():.3f}")
    print(f"    IQR   {zone['mixture_ratio'].quantile(.25):.3f} - "
          f"{zone['mixture_ratio'].quantile(.75):.3f}")
    print(f"\n  within-zone sd (mean) = {within:.3f}")
    print(f"  between-zone sd       = {between:.3f}")
    print(f"  ratio                 = {within/(between+1e-12):.1f}x")
    print("    (Experiment 09 found 3.0x, Experiment 10's corrected recompute 4.4x,")
    print("     both on n=4 zones. This is the n=224 answer.)")

    print(f"\n  MOST grid-like zones:")
    for z, r_ in zone.nlargest(5, "mixture_ratio").iterrows():
        print(f"    {z:<14} mixture_ratio={r_['mixture_ratio']:.3f}  "
              f"median_excess={r_['median_excess']:+.3f}")
    print(f"  LEAST grid-like zones:")
    for z, r_ in zone.nsmallest(5, "mixture_ratio").iterrows():
        print(f"    {z:<14} mixture_ratio={r_['mixture_ratio']:.3f}  "
              f"median_excess={r_['median_excess']:+.3f}")

    print(f"\n[5] CONFOUND CHECKS")
    for col, lab in [("road_len_m", "road length"), ("n_seg", "segment count"),
                     ("osm_completeness", "OSM completeness")]:
        r_ = float(np.corrcoef(clean["grid_score"], clean[col])[0, 1])
        print(f"  corr(grid_score, {lab:<17}) = {r_:+.3f}"
              f"{'   <-- WATCH' if abs(r_) > 0.4 else ''}")
    rz = float(np.corrcoef(zone["mixture_ratio"],
                           clean.groupby('zone_id')['osm_completeness'].first()
                           .reindex(zone.index))[0, 1])
    print(f"  corr(zone mixture_ratio, zone completeness) = {rz:+.3f}"
          f"{'   <-- mixture ratio may be tracking MAPPING, not form' if abs(rz) > 0.4 else ''}")

    print(f"\n[6] MORPHOLOGY CLASSES (Experiment 10: grid = low |c1| AND high |c2|)")
    corridor = (clean["c1"] > 0.35)
    gridlike = (clean["c2"] > 0.35) & (clean["c1"] < 0.25)
    other = ~(corridor | gridlike)
    for lab, m in [("grid-like", gridlike), ("corridor", corridor), ("organic/other", other)]:
        print(f"  {lab:<14} {int(m.sum()):>7} ({100*m.mean():>5.1f}%)")

    figures(here, df, clean, zone, val, ns, null_means)

    summary = {
        "n_windows": int(len(df)), "n_clean": int(len(clean)),
        "n_zones": int(clean["zone_id"].nunique()),
        "null_lookup_ok": bool(ok_null),
        "within_zone_sd": float(within), "between_zone_sd": float(between),
        "ratio": float(within / (between + 1e-12)),
        "mixture_ratio": {"mean": float(zone["mixture_ratio"].mean()),
                          "sd": float(zone["mixture_ratio"].std()),
                          "min": float(zone["mixture_ratio"].min()),
                          "max": float(zone["mixture_ratio"].max())},
        "corr_mixture_vs_completeness": rz,
        "classes": {"grid_like": int(gridlike.sum()), "corridor": int(corridor.sum()),
                    "organic_other": int(other.sum())},
    }
    with open(os.path.join(here, "result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 82)
    print("LOOK AT city_mixture_map.png before trusting any number above.")
    print("=" * 82)


def figures(here, df, clean, zone, val, ns, null_means):
    # --- the map ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    sp = df[df["sparse"]]
    axes[0].scatter(sp["wx"] / 1000, sp["wy"] / 1000, s=3, c="0.85", marker="s",
                    label="too sparse")
    fl = df[(~df["sparse"]) & (df["low_completeness"])]
    axes[0].scatter(fl["wx"] / 1000, fl["wy"] / 1000, s=3, c="0.55", marker="s",
                    label="low OSM completeness")
    s = axes[0].scatter(clean["wx"] / 1000, clean["wy"] / 1000, s=3,
                        c=clean["excess"], cmap="RdYlBu_r", marker="s",
                        vmin=clean["excess"].quantile(.02),
                        vmax=clean["excess"].quantile(.98))
    plt.colorbar(s, ax=axes[0], label="excess over null (grid-ness)", shrink=.8)
    axes[0].set_title(f"Dhaka, 400m windows at 100m stride\n{len(clean):,} scored "
                      f"windows, warm = more grid-like", fontsize=12)
    axes[0].set_xlabel("UTM easting (km)"); axes[0].set_ylabel("UTM northing (km)")
    axes[0].legend(loc="upper right", fontsize=8, markerscale=3)
    axes[0].set_aspect("equal")

    zc = clean.groupby("zone_id")[["wx", "wy"]].mean().reindex(zone.index)
    s2 = axes[1].scatter(zc["wx"] / 1000, zc["wy"] / 1000, s=180,
                         c=zone["mixture_ratio"], cmap="RdYlBu_r", marker="s",
                         edgecolor="k", linewidth=.4)
    plt.colorbar(s2, ax=axes[1], label="mixture ratio (share of grid-like windows)",
                 shrink=.8)
    axes[1].set_title(f"Zone mixture ratio, {len(zone)} zones\n"
                      f"range {zone['mixture_ratio'].min():.2f}-"
                      f"{zone['mixture_ratio'].max():.2f}", fontsize=12)
    axes[1].set_xlabel("UTM easting (km)")
    axes[1].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(here, "city_mixture_map.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)

    # --- diagnostics -----------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].hist(clean["grid_score"], bins=60, color="tab:blue", alpha=.8, label="observed")
    ax[0].hist(clean["null_mean"], bins=60, color="tab:red", alpha=.6, label="its null")
    ax[0].set_xlabel("grid_score"); ax[0].set_ylabel("windows")
    ax[0].set_title("Score vs null, clean windows"); ax[0].legend(fontsize=8)

    if len(val) >= 10:
        ax[1].scatter(val["true_null"], val["lookup_null"], s=22, alpha=.7,
                      edgecolor="k", linewidth=.3)
        lim = [min(val["true_null"].min(), val["lookup_null"].min()) - .01,
               max(val["true_null"].max(), val["lookup_null"].max()) + .01]
        ax[1].plot(lim, lim, "k--", linewidth=1)
        ax[1].set_xlabel("exact segment-level null"); ax[1].set_ylabel("lookup null")
        ax[1].set_title(f"Null validation, n={len(val)}")

    ax[2].hist(zone["mixture_ratio"], bins=30, color="tab:purple", alpha=.85)
    ax[2].set_xlabel("zone mixture ratio"); ax[2].set_ylabel("zones")
    ax[2].set_title("Does mixture ratio vary between zones?")
    plt.tight_layout()
    plt.savefig(os.path.join(here, "diagnostics.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

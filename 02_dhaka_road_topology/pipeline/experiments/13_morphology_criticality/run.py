#!/usr/bin/env python
"""Experiment 13 -- context-dependent criticality: does a critical node mean
something different in grid fabric than in organic fabric?

This is the proposal's Principle 2 -- "node importance must be evaluated relative to
local topological regime" -- which is the mechanism that makes the framework
topology-AWARE rather than a centrality ranking with extra steps. It has never been
testable, because it needs a structural label per location and the zone-level
typology never produced a usable one (Experiments 08-12). Experiment 11 produced one
that survives its own confound check, so Principle 2 can finally be tested.

The abstract's motivating claim is concrete and checkable: "A signal timing
optimization that reduces delay by 18% in Dhanmondi's planned grid may simultaneously
induce 27% spillover congestion in Old Dhaka's organic alleyways." The structural
half of that -- that criticality behaves differently in the two fabrics -- is what
this measures.

JOIN STRATEGY
  Betweenness exists only for the 1km run (v3, 338,429 node rows); the morphology
  field came from the 2km run (v2km). That mismatch does not matter: the morphology
  layer is a GEOGRAPHIC field of 400m windows and the node table carries lon/lat, so
  the two join on SPACE, not on zone id. No re-run required.

THREE DATA HAZARDS, HANDLED EXPLICITLY
  1. Node duplication. 338,429 rows cover only 57,999 distinct nodes (5.8x), because
     every node appears in its own zone plus its buffered neighbours. Each node is
     assigned to the zone whose centroid is NEAREST, approximating core membership,
     consistent with the pipeline's two-tier design (core for labelling, buffer for
     computation).
  2. Articulation-point ambiguity. Articulation status is computed per buffered
     subgraph, and 14.4% of nodes DISAGREE between the zones that contain them. The
     nearest-zone value is used, with "any zone" and "all zones" reported as
     sensitivity bounds.
  3. Betweenness is zone-local. BC was computed on each zone's buffered subgraph, so
     raw BC magnitudes are NOT comparable across zones. Only within-zone percentile
     rank is used. Articulation status is local topology and is comparable; tier is
     assigned within zone and is therefore already context-relative.

THE CONFOUND THAT MUST NOT RETURN
  Experiments 06, 11 and 12 all found density masquerading as structure. If
  articulation rate simply tracks how sparse an area is, then "fragility" would just
  be "sparse" relabelled. Every relationship below is therefore reported both raw and
  after removing local road density.

Usage:
    python experiments/13_morphology_criticality/run.py
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
from pyproj import Transformer
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
BC_CSV = os.path.join(ROOT, "pipeline_outputs", "v3", "gat", "csv", "all_nodes_bc.csv")
V3_META = os.path.join(ROOT, "pipeline_outputs", "v3", "outputs", "csv", "metadata.csv")
WINDOWS = os.path.join(HERE, "..", "11_city_mixture_map", "windows.csv")
UTM = "EPSG:32646"          # UTM 46N -- Dhaka
MAX_SNAP_M = 250.0          # a node further than this from any window is unlabelled


def main() -> int:
    # ---------------------------------------------------------------- load
    bc = pd.read_csv(BC_CSV)
    meta = pd.read_csv(V3_META)
    win = pd.read_csv(WINDOWS)
    print(f"nodes rows {len(bc):,}  distinct {bc['node_id'].nunique():,}  "
          f"zones {len(meta)}  windows {len(win):,}")

    tr = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    bc["x"], bc["y"] = tr.transform(bc["lon"].values, bc["lat"].values)

    # Sanity: does our projection agree with the pipeline's own UTM coordinates?
    # NOTE: do NOT check against a zone's nodes.csv. `tiling.py:112` writes
    #   "lon": round(d.get("lon", d.get("x", 0)), 6)
    # and on a PROJECTED OSMnx graph d["x"] is the UTM easting, so those lon/lat
    # columns actually hold UTM in both v3 and v2km. Latent pipeline bug, logged
    # separately; it has never affected an analysis because everything reads x_utm.
    # metadata.csv carries genuine centroid_lon/centroid_lat, so check against that.
    mcx, mcy = tr.transform(meta["centroid_lon"].values, meta["centroid_lat"].values)
    err = float(np.abs(mcx - meta["centroid_x_utm"].values).max())
    print(f"projection check vs metadata centroids: max error {err:.2f} m "
          f"-> {'OK' if err < 1.0 else 'MISMATCH, ABORT'}")
    if err >= 1.0:
        return 1

    # ------------------------------------------- hazard 1 & 2: deduplicate
    art_any = bc.groupby("node_id")["is_art_pt"].max()
    art_all = bc.groupby("node_id")["is_art_pt"].min()

    zc = meta.set_index("zone_id")[["centroid_x_utm", "centroid_y_utm"]]
    bc = bc.join(zc, on="zone_id")
    bc["d_to_zone"] = np.hypot(bc["x"] - bc["centroid_x_utm"], bc["y"] - bc["centroid_y_utm"])
    bc["bc_pct"] = bc.groupby("zone_id")["bc"].rank(pct=True)   # zone-local BC only
    nodes = bc.loc[bc.groupby("node_id")["d_to_zone"].idxmin()].copy()
    nodes["art_any"] = nodes["node_id"].map(art_any)
    nodes["art_all"] = nodes["node_id"].map(art_all)
    print(f"deduplicated to {len(nodes):,} nodes (nearest-zone-centroid assignment)")
    print(f"  articulation rate  nearest-zone {nodes['is_art_pt'].mean():.4f}   "
          f"any-zone {nodes['art_any'].mean():.4f}   all-zones {nodes['art_all'].mean():.4f}")

    # ------------------------------------------------ spatial join to morphology
    w = win[~win["sparse"]].reset_index(drop=True)
    tree = cKDTree(w[["wx", "wy"]].values)
    dist, idx = tree.query(nodes[["x", "y"]].values, k=1)
    nodes["snap_m"] = dist
    ok = dist <= MAX_SNAP_M
    for col in ("excess", "grid_score", "c1", "c2", "c3", "road_len_m"):
        nodes.loc[:, f"loc_{col}"] = np.where(ok, w[col].values[idx], np.nan)
    nodes["labelled"] = ok
    print(f"  morphology assigned to {ok.sum():,} / {len(nodes):,} nodes "
          f"({100*ok.mean():.1f}%); median snap {np.median(dist):.0f} m")

    n = nodes[nodes["labelled"]].copy()
    n.to_csv(os.path.join(HERE, "nodes_morphology.csv"), index=False)

    print("\n" + "=" * 84)
    print("EXPERIMENT 13 -- DOES CRITICALITY DEPEND ON LOCAL MORPHOLOGY?")
    print("=" * 84)

    # ---- [1] articulation rate vs local grid-ness ------------------------
    print("\n[1] FRAGILITY -- are organic areas more articulation-dependent?")
    n["mbin"] = pd.qcut(n["loc_excess"], 6, labels=False, duplicates="drop")
    tab = n.groupby("mbin").agg(grid_ness=("loc_excess", "mean"),
                                art_rate=("is_art_pt", "mean"),
                                road_len=("loc_road_len_m", "mean"),
                                deg=("degree", "mean"), nodes=("node_id", "size"))
    print(f"  {'bin':<5}{'grid-ness':>12}{'artic. rate':>14}{'mean degree':>13}"
          f"{'road m/win':>12}{'n':>9}")
    print("-" * 84)
    for i, r in tab.iterrows():
        print(f"  {int(i):<5}{r['grid_ness']:>12.3f}{r['art_rate']:>14.4f}"
              f"{r['deg']:>13.2f}{r['road_len']:>12.0f}{int(r['nodes']):>9,}")
    r_raw = stats.pearsonr(n["loc_excess"], n["is_art_pt"])
    print(f"\n  corr(local grid-ness, is_art_pt) = {r_raw[0]:+.4f}  (p={r_raw[1]:.2e})")

    # the confound: is this just density?
    r_dens = stats.pearsonr(n["loc_road_len_m"], n["is_art_pt"])
    print(f"  corr(local road density, is_art_pt) = {r_dens[0]:+.4f}")
    X = n[["loc_road_len_m"]].values
    resid_grid = n["loc_excess"].values - LinearRegression().fit(X, n["loc_excess"]).predict(X)
    resid_art = n["is_art_pt"].values - LinearRegression().fit(X, n["is_art_pt"]).predict(X)
    r_part = stats.pearsonr(resid_grid, resid_art)
    print(f"  PARTIAL corr, density removed     = {r_part[0]:+.4f}  (p={r_part[1]:.2e})")
    print(f"  -> {'survives' if abs(r_part[0]) > 0.5*abs(r_raw[0]) else 'LARGELY THE DENSITY CONFOUND'}")

    # sensitivity to the articulation ambiguity
    for lab, col in [("any-zone", "art_any"), ("all-zones", "art_all")]:
        rr = stats.pearsonr(n["loc_excess"], n[col])
        print(f"  sensitivity ({lab:<9}): corr = {rr[0]:+.4f}")

    # ---- [2] do the MOST critical nodes sit in different fabric? ---------
    print("\n[2] CONTEXT -- where do the top-criticality nodes actually sit?")
    top = n[n["bc_pct"] >= 0.99]
    rest = n[n["bc_pct"] < 0.99]
    print(f"  top 1% BC (within zone): n={len(top):,}   mean local grid-ness "
          f"{top['loc_excess'].mean():+.3f}")
    print(f"  all others            : n={len(rest):,}   mean local grid-ness "
          f"{rest['loc_excess'].mean():+.3f}")
    u = stats.mannwhitneyu(top["loc_excess"].dropna(), rest["loc_excess"].dropna())
    print(f"  Mann-Whitney p = {u.pvalue:.2e}")
    print(f"\n  Of the top-1%-BC nodes, what share are ALSO articulation points?")
    print(f"  (a critical node that is also a cut vertex has NO local alternative route)")
    print(f"\n  {'fabric':<12}{'top-1% BC':>12}{'all nodes':>12}{'ratio':>8}"
          f"{'Fisher p':>12}{'n top':>8}")
    print("-" * 84)
    interaction = {}
    for lo, hi, lab in [(-9, -0.05, "organic"), (-0.05, 0.15, "mixed"), (0.15, 9, "grid-like")]:
        band = n[(n["loc_excess"] >= lo) & (n["loc_excess"] < hi)]
        s = band[band["bc_pct"] >= 0.99]
        o = band[band["bc_pct"] < 0.99]
        if len(s) < 30:
            continue
        table = [[int(s["is_art_pt"].sum()), int(len(s) - s["is_art_pt"].sum())],
                 [int(o["is_art_pt"].sum()), int(len(o) - o["is_art_pt"].sum())]]
        odds, p = stats.fisher_exact(table)
        ratio = s["is_art_pt"].mean() / (o["is_art_pt"].mean() + 1e-12)
        interaction[lab] = {"top": float(s["is_art_pt"].mean()),
                            "rest": float(o["is_art_pt"].mean()),
                            "ratio": float(ratio), "p": float(p), "n_top": int(len(s))}
        print(f"  {lab:<12}{s['is_art_pt'].mean():>12.3f}{o['is_art_pt'].mean():>12.3f}"
              f"{ratio:>8.2f}{p:>12.2e}{len(s):>8,}")
    print("\n  A ratio ABOVE 1 means: in this fabric, being critical also means being a")
    print("  cut vertex. BELOW 1 means critical nodes there have alternatives nearby.")
    print("  If the direction FLIPS between fabrics, Principle 2 holds -- the same")
    print("  betweenness rank means a structurally different thing in each regime.")
    print("\n  A critical node that is ALSO a cut vertex has no local alternative route:")
    print("  optimisation cannot help it, only redundancy can. That is the policy-relevant")
    print("  distinction Principle 2 asks for.")

    # ---- [3] window-level fragility layer --------------------------------
    print("\n[3] FRAGILITY LAYER (per 400m window)")
    # Each window is 400m across but the stride is 100m, so windows OVERLAP. Pooling
    # by nearest-window would give each window only its ~100m catchment (~1 node) --
    # it must instead pool every node that falls INSIDE the 400m window.
    node_tree = cKDTree(n[["x", "y"]].values)
    members = node_tree.query_ball_point(w[["wx", "wy"]].values, r=200.0)
    art = n["is_art_pt"].values
    bcp = n["bc_pct"].values
    cnt = np.array([len(m) for m in members])
    keep = cnt >= 10
    fw = w.loc[keep].copy()
    fw["n_nodes"] = cnt[keep]
    fw["art_rate"] = np.array([art[m].mean() if len(m) >= 10 else np.nan
                               for m in members])[keep]
    fw["mean_bcpct"] = np.array([bcp[m].mean() if len(m) >= 10 else np.nan
                                 for m in members])[keep]
    print(f"  pooled by 200m radius around each window centre "
          f"(median {int(np.median(cnt))} nodes/window)")
    # fragility = articulation-dependent AND not grid-like, both as percentiles
    fw["fragility"] = (stats.rankdata(fw["art_rate"]) / len(fw)
                       + stats.rankdata(-fw["excess"]) / len(fw)) / 2.0
    fw.to_csv(os.path.join(HERE, "window_fragility.csv"), index=False)
    print(f"  {len(fw):,} windows with >=10 nodes")
    print(f"  articulation rate: mean {fw['art_rate'].mean():.3f}  "
          f"range {fw['art_rate'].min():.3f}-{fw['art_rate'].max():.3f}")
    r_fd = stats.pearsonr(fw["fragility"], fw["road_len_m"])
    print(f"  corr(fragility, local road density) = {r_fd[0]:+.3f}"
          f"{'   <-- fragility is substantially density' if abs(r_fd[0]) > 0.6 else ''}")
    hi = fw.nlargest(12, "fragility")
    print(f"  most fragile windows cluster in zones: "
          f"{', '.join(hi['zone_id'].value_counts().head(5).index)}")

    figures(n, tab, fw)
    out = {
        "n_nodes": int(len(nodes)), "n_labelled": int(len(n)),
        "art_rate": {"nearest_zone": float(nodes["is_art_pt"].mean()),
                     "any_zone": float(nodes["art_any"].mean()),
                     "all_zones": float(nodes["art_all"].mean())},
        "corr_gridness_artpt_raw": float(r_raw[0]),
        "corr_density_artpt": float(r_dens[0]),
        "partial_corr_density_removed": float(r_part[0]),
        "top1pct_gridness": float(top["loc_excess"].mean()),
        "rest_gridness": float(rest["loc_excess"].mean()),
        "mannwhitney_p": float(u.pvalue),
        "n_fragility_windows": int(len(fw)),
        "corr_fragility_density": float(r_fd[0]),
        "artic_rate_by_gridness_bin": {str(int(i)): float(r["art_rate"]) for i, r in tab.iterrows()},
        "artic_rate_ratio_organic_vs_grid": float(tab["art_rate"].iloc[0] / tab["art_rate"].iloc[-1]),
        "principle2_interaction": interaction,
    }
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 84)
    print("See morphology_criticality.png")
    print("=" * 84)
    return 0


def figures(n, tab, fw):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))

    ax[0].plot(tab["grid_ness"], tab["art_rate"], "o-", color="tab:red", linewidth=2)
    ax[0].set_xlabel("local grid-ness (excess over null)")
    ax[0].set_ylabel("articulation-point rate")
    ax[0].set_title("Are organic areas more dependent\non single cut vertices?", fontsize=11)
    ax[0].grid(alpha=.3)
    a2 = ax[0].twinx()
    a2.plot(tab["grid_ness"], tab["road_len"], "s--", color="0.6", linewidth=1.2)
    a2.set_ylabel("local road density (m/window)", color="0.5", fontsize=9)
    a2.tick_params(labelcolor="0.5")

    s = ax[1].scatter(fw["wx"] / 1000, fw["wy"] / 1000, s=5, c=fw["fragility"],
                      cmap="inferno_r", marker="s")
    plt.colorbar(s, ax=ax[1], label="fragility (organic AND cut-vertex dependent)", shrink=.85)
    ax[1].set_title(f"Fragility layer, {len(fw):,} windows\n"
                    "dark = one blockage from isolation", fontsize=11)
    ax[1].set_xlabel("UTM easting (km)"); ax[1].set_ylabel("UTM northing (km)")
    ax[1].set_aspect("equal")

    for lab, lo, hi, c in [("organic", -9, -0.05, "tab:blue"),
                           ("mixed", -0.05, 0.15, "tab:orange"),
                           ("grid-like", 0.15, 9, "tab:green")]:
        s_ = n[(n["loc_excess"] >= lo) & (n["loc_excess"] < hi)]
        if len(s_) > 100:
            byt = s_.groupby("tier")["is_art_pt"].mean()
            ax[2].plot(byt.index, byt.values, "o-", color=c, label=lab, linewidth=2)
    ax[2].set_xlabel("betweenness tier (1 = most critical)")
    ax[2].set_ylabel("articulation-point rate")
    ax[2].set_title("Does a 'critical node' mean the same\nthing in each fabric?", fontsize=11)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "morphology_criticality.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

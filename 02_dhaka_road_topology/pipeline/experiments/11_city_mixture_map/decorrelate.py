#!/usr/bin/env python
"""Experiment 11b -- is the mixture ratio measuring urban FORM or OSM MAPPING?

run.py found corr(zone mixture_ratio, osm_completeness) = +0.724. That is the
confound the proposal itself warns about ("apparent 'maze' clustering may partly
reflect mapping sparsity rather than true topological fragmentation"), and it is
large enough to explain the result on its own. Unresolved, the city map is
uninterpretable and the equity framing would be actively misleading.

Two rival explanations, which have DIFFERENT testable signatures:

  ARTIFACT -- under-mapped areas are missing their small lanes. What survives in OSM
    is the arterial skeleton, which is straighter and more orthogonal, so sparse
    zones score... actually LOWER here, meaning the mechanism would be that missing
    lanes leave too few segments to establish any orthogonal order. If this is the
    cause, the correlation should be driven by the poorly-mapped tail and should
    FLATTEN among well-mapped zones.

  GENUINE CO-OCCURRENCE -- RAJUK-planned areas (Dhanmondi, Gulshan, Uttara) are both
    better mapped AND actually gridded; informal areas are both under-mapped AND
    actually organic. Then the trend should CONTINUE among well-mapped zones, because
    completeness is proxying for planned-ness, not causing the score.

These cannot be fully separated with OSM alone -- that is an honest limitation -- but
the shape of the relationship discriminates between them, and the decorrelated
residual shows what survives either way (the same treatment Experiment 06 applied to
the zone-size confound).

Usage:
    python experiments/11_city_mixture_map/decorrelate.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
META_CSV = os.path.join(ROOT, "pipeline_outputs", "v2km", "outputs", "csv", "metadata.csv")


def main() -> int:
    zone = pd.read_csv(os.path.join(HERE, "zone_mixture.csv")).set_index("zone_id")
    win = pd.read_csv(os.path.join(HERE, "windows.csv"))
    meta = pd.read_csv(META_CSV).set_index("zone_id")
    zone["completeness"] = meta["osm_completeness"].reindex(zone.index)
    zone["mean_road_len"] = (win[~win["sparse"] & ~win["low_completeness"]]
                             .groupby("zone_id")["road_len_m"].mean().reindex(zone.index))
    zone = zone.dropna(subset=["completeness", "mixture_ratio"])

    print("=" * 80)
    print("IS THE MIXTURE RATIO MEASURING FORM, OR MAPPING?")
    print("=" * 80)
    r_all = stats.pearsonr(zone["completeness"], zone["mixture_ratio"])
    print(f"\n[0] n = {len(zone)} zones")
    print(f"    corr(completeness, mixture_ratio) = {r_all[0]:+.3f}  (p={r_all[1]:.2e})")
    print(f"    completeness range {zone['completeness'].min():.2f} - "
          f"{zone['completeness'].max():.2f}, median {zone['completeness'].median():.2f}")

    # ---- [1] shape of the relationship: artifact tail or continuing trend? ----
    print("\n[1] SHAPE -- binned means of mixture_ratio by completeness decile")
    zone["dec"] = pd.qcut(zone["completeness"], 10, labels=False, duplicates="drop")
    b = zone.groupby("dec").agg(comp=("completeness", "mean"),
                                mix=("mixture_ratio", "mean"),
                                n=("mixture_ratio", "size"))
    print(f"    {'decile':<8}{'completeness':>14}{'mixture_ratio':>16}{'n':>6}")
    for d, row in b.iterrows():
        print(f"    {int(d):<8}{row['comp']:>14.3f}{row['mix']:>16.3f}{int(row['n']):>6}")

    for thresh in (0.6, 0.8, 1.0):
        sub = zone[zone["completeness"] >= thresh]
        if len(sub) > 20:
            rr = stats.pearsonr(sub["completeness"], sub["mixture_ratio"])
            print(f"\n    completeness >= {thresh:.1f}:  n={len(sub):>3}  "
                  f"corr={rr[0]:+.3f}  p={rr[1]:.3f}")
    top = zone[zone["completeness"] >= zone["completeness"].median()]
    r_top = stats.pearsonr(top["completeness"], top["mixture_ratio"])
    print(f"\n    -> among the BETTER-MAPPED half, corr = {r_top[0]:+.3f} (p={r_top[1]:.3f})")
    if abs(r_top[0]) < 0.25:
        print("       Trend FLATTENS -> the correlation is driven by the under-mapped")
        print("       tail, i.e. substantially an ARTIFACT of mapping sparsity.")
    else:
        print("       Trend CONTINUES among well-mapped zones -> completeness is partly")
        print("       proxying for planned-ness (GENUINE CO-OCCURRENCE), not purely")
        print("       an artifact. Cannot be fully separated with OSM data alone.")

    # ---- [2] decorrelate, as Experiment 06 did for zone size ----------------
    X = zone[["completeness"]].values
    y = zone["mixture_ratio"].values
    reg = LinearRegression().fit(X, y)
    zone["residual"] = y - reg.predict(X)
    r2 = float(reg.score(X, y))
    print(f"\n[2] DECORRELATION (mixture_ratio ~ completeness)")
    print(f"    R^2 explained by completeness alone = {r2:.3f}")
    print(f"    -> {100*r2:.0f}% of between-zone mixture variation is completeness")
    print(f"    raw       sd = {zone['mixture_ratio'].std():.3f}  "
          f"range {zone['mixture_ratio'].min():.3f}-{zone['mixture_ratio'].max():.3f}")
    print(f"    residual  sd = {zone['residual'].std():.3f}  "
          f"range {zone['residual'].min():.3f}-{zone['residual'].max():.3f}")
    print(f"    surviving spread = {100*zone['residual'].std()/zone['mixture_ratio'].std():.0f}% "
          f"of the original")

    rho = stats.spearmanr(zone["mixture_ratio"], zone["residual"])
    print(f"    rank correlation raw vs residual = {rho.statistic:+.3f} "
          f"({'ordering largely preserved' if rho.statistic > 0.7 else 'ORDERING CHANGES'})")

    print(f"\n    MOST grid-like AFTER correcting for mapping:")
    for z, r_ in zone.nlargest(6, "residual").iterrows():
        print(f"      {z:<14} residual={r_['residual']:+.3f}  raw={r_['mixture_ratio']:.3f}"
              f"  completeness={r_['completeness']:.2f}")
    print(f"    LEAST grid-like AFTER correcting for mapping:")
    for z, r_ in zone.nsmallest(6, "residual").iterrows():
        print(f"      {z:<14} residual={r_['residual']:+.3f}  raw={r_['mixture_ratio']:.3f}"
              f"  completeness={r_['completeness']:.2f}")

    # ---- [3] is the residual spatially coherent, or noise? ------------------
    zc = win[~win["sparse"]].groupby("zone_id")[["wx", "wy"]].mean().reindex(zone.index)
    zone["wx"], zone["wy"] = zc["wx"], zc["wy"]
    moran = spatial_autocorr(zone["wx"].values, zone["wy"].values, zone["residual"].values)
    moran_raw = spatial_autocorr(zone["wx"].values, zone["wy"].values,
                                 zone["mixture_ratio"].values)
    print(f"\n[3] SPATIAL COHERENCE (neighbour correlation, 2.5km radius)")
    print(f"    raw mixture_ratio : {moran_raw:+.3f}")
    print(f"    residual          : {moran:+.3f}")
    print("    A residual that is pure noise would sit near 0. Spatial structure means")
    print("    something real survives the mapping correction.")

    zone.to_csv(os.path.join(HERE, "zone_mixture_decorrelated.csv"))
    figures(zone, b)
    with open(os.path.join(HERE, "decorrelation.json"), "w") as f:
        json.dump({"n_zones": int(len(zone)), "corr_all": float(r_all[0]),
                   "corr_better_mapped_half": float(r_top[0]),
                   "r2_completeness": r2,
                   "raw_sd": float(zone["mixture_ratio"].std()),
                   "residual_sd": float(zone["residual"].std()),
                   "rank_corr_raw_vs_residual": float(rho.statistic),
                   "spatial_autocorr_raw": float(moran_raw),
                   "spatial_autocorr_residual": float(moran)}, f, indent=2)
    print("\n" + "=" * 80)
    return 0


def spatial_autocorr(x, y, v, radius=2500.0):
    """Mean correlation between each zone's value and its neighbours' mean."""
    v = np.asarray(v, dtype=float)
    neigh = np.full(v.size, np.nan)
    for i in range(v.size):
        d = np.hypot(x - x[i], y - y[i])
        m = (d > 0) & (d <= radius)
        if m.sum() >= 2:
            neigh[i] = v[m].mean()
    ok = ~np.isnan(neigh)
    return float(np.corrcoef(v[ok], neigh[ok])[0, 1]) if ok.sum() > 10 else np.nan


def figures(zone, binned):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    ax[0].scatter(zone["completeness"], zone["mixture_ratio"], s=30, alpha=.65,
                  edgecolor="k", linewidth=.3)
    ax[0].plot(binned["comp"], binned["mix"], "r-o", linewidth=2, markersize=5,
               label="decile means")
    ax[0].set_xlabel("OSM completeness"); ax[0].set_ylabel("zone mixture ratio")
    ax[0].set_title("The confound: does grid-ness track mapping?", fontsize=11)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    s = ax[1].scatter(zone["wx"] / 1000, zone["wy"] / 1000, s=150,
                      c=zone["residual"], cmap="RdYlBu_r", marker="s",
                      edgecolor="k", linewidth=.4)
    plt.colorbar(s, ax=ax[1], label="mixture ratio, mapping removed", shrink=.85)
    ax[1].set_title("Residual grid-ness after removing\nOSM completeness", fontsize=11)
    ax[1].set_xlabel("UTM easting (km)"); ax[1].set_ylabel("UTM northing (km)")
    ax[1].set_aspect("equal")

    ax[2].hist(zone["mixture_ratio"] - zone["mixture_ratio"].mean(), bins=28,
               alpha=.6, label=f"raw (sd {zone['mixture_ratio'].std():.3f})")
    ax[2].hist(zone["residual"], bins=28, alpha=.6,
               label=f"residual (sd {zone['residual'].std():.3f})")
    ax[2].set_xlabel("deviation from mean"); ax[2].set_ylabel("zones")
    ax[2].set_title("How much variation survives?", fontsize=11)
    ax[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "completeness_confound.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

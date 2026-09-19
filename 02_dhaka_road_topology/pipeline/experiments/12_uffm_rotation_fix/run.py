#!/usr/bin/env python
"""Experiment 12 -- was UFFM's null result caused by a rotation-variance bug?

UFFM (Urban Form Fingerprint Matching) is documented in this project as a failure:
k=2, silhouette 0.4685, but its crosstab against the scalar-feature clusters doesn't
line up (40/144 and 582/61 at 1km), so it "didn't independently confirm" anything.

Experiment 10 (C8) identified a specific, mechanical cause. `uffm.py:37` builds a
street-bearing histogram folded to [0,180 deg). `uffm.py:161` then compares two of
them with `wasserstein_distance` over `np.linspace(0, 1, bins)` -- a LINEAR axis for
a CIRCULAR variable. Two consequences:

  1. Bin 0 and bin 35 are treated as maximally far apart when they are 5 degrees apart.
  2. The distance is rotation-VARIANT: the same street pattern rotated 45 degrees
     reads as a different urban form.

That is fatal for a morphology measure. "Is this a grid?" must not depend on which
way north points -- a rotated grid is still a grid.

WHY ONLY THE BEARING TERM IS CHANGED
  angle_fingerprint  -- intersection angles between adjoining streets, via arccos of
    unit vectors. These are RELATIVE angles, so already rotation-invariant, and 0 deg
    (collinear same direction) and 180 deg (straight through-road) are genuinely
    different situations, so the linear axis is correct. Left untouched.
  length_fingerprint -- segment lengths are a genuinely linear variable. Left untouched.
  bearing_fingerprint -- absolute compass bearings. The one defective term.

This is therefore a single-variable experiment: change the bearing distance, hold
everything else fixed, and see whether the documented failure moves.

THE FIX: represent each bearing histogram by the magnitudes |c_k| of its circular
Fourier coefficients (Experiment 10). A rotation circularly shifts the histogram,
which changes only the PHASE, so |c_k| is rotation-invariant by construction. On a
36-bin [0,180 deg) axis, |c_2| carries grid-ness and |c_1| a single dominant axis.

Usage:
    python experiments/12_uffm_rotation_fix/run.py
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
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSV = os.path.join(ROOT, "pipeline_outputs", "v2km", "outputs", "csv")

N_BEAR, N_ANG, N_LEN = 36, 36, 40
W_BEAR, W_ANG, W_LEN = 0.40, 0.35, 0.25
K_RANGE = range(2, 9)
KMAX_HARMONIC = 6
SEED = 42


# ------------------------------------------------------------------- distances

def linear_w(p, q, n):
    """Exactly uffm.build_distance_matrix's term: Wasserstein on a LINEAR axis."""
    x = np.linspace(0, 1, n)
    return float(wasserstein_distance(x, x, p, q))


def circular_w(p, q):
    """Wasserstein on a CIRCLE. Optimal transport on a ring shifts by the median
    of the CDF difference; without that shift the metric charges for wrap-around."""
    d = np.cumsum(p - q)
    c = np.median(d)
    return float(np.abs(d - c).mean())


def harmonics(h, kmax=KMAX_HARMONIC):
    """|c_k| / |c_0| -- rotation-invariant by construction."""
    c = np.fft.rfft(h)
    return np.abs(c[1 : kmax + 1]) / (np.abs(c[0]) + 1e-12)


def harmonic_d(p, q):
    return float(np.linalg.norm(harmonics(p) - harmonics(q)))


# ------------------------------------------------------------------- clustering

def spectral(dist, zone_ids):
    nz = dist[dist > 0]
    sigma = float(np.median(nz))
    sim = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(sim, 1.0)
    out = []
    for k in K_RANGE:
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=SEED, n_init=10, assign_labels="kmeans")
        lab = sc.fit_predict(sim)
        out.append({"k": k, "sil": float(silhouette_score(dist, lab, metric="precomputed")),
                    "labels": lab})
    best = max(out, key=lambda r: r["sil"])
    return best, pd.DataFrame([{"k": r["k"], "silhouette": round(r["sil"], 4)} for r in out])


def _pairwise(vals_fn, n):
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = vals_fn(i, j)
    return D


def _unit_median(D):
    """Rescale a distance matrix to median 1 over its off-diagonal entries.

    Necessary because the three bearing metrics have different natural scales
    (linear ~0.073, harmonic ~0.339 between zones). With fixed weights
    0.40/0.35/0.25, an unscaled harmonic term would simply SWAMP the angle and
    length terms, and the comparison would measure that rather than the metric.
    Applied identically to every term in every mode, so the test stays fair.
    """
    off = D[~np.eye(D.shape[0], dtype=bool)]
    med = np.median(off[off > 0])
    return D / (med + 1e-12)


def build_matrix(B, A, L, bearing_mode, normalize=True):
    n = B.shape[0]
    fn = {"linear": lambda i, j: linear_w(B[i], B[j], N_BEAR),
          "circular": lambda i, j: circular_w(B[i], B[j]),
          "harmonic": lambda i, j: harmonic_d(B[i], B[j])}[bearing_mode]
    Db = _pairwise(fn, n)
    Da = _pairwise(lambda i, j: linear_w(A[i], A[j], N_ANG), n)
    Dl = _pairwise(lambda i, j: linear_w(L[i], L[j], N_LEN), n)
    if normalize:
        Db, Da, Dl = _unit_median(Db), _unit_median(Da), _unit_median(Dl)
    return W_BEAR * Db + W_ANG * Da + W_LEN * Dl


def main() -> int:
    fp = pd.read_csv(os.path.join(CSV, "uffm_fingerprints.csv"))
    zone_ids = fp["zone_id"].tolist()
    B = fp[[f"b_{i}" for i in range(N_BEAR)]].values.astype(float)
    A = fp[[f"a_{i}" for i in range(N_ANG)]].values.astype(float)
    L = fp[[f"l_{i}" for i in range(N_LEN)]].values.astype(float)
    B = B / (B.sum(axis=1, keepdims=True) + 1e-12)
    print(f"loaded {len(zone_ids)} zone fingerprints\n")

    # ---- [1] rotation test on REAL fingerprints ---------------------------
    print("=" * 82)
    print("[1] ROTATION TEST -- rotate a real zone, does its fingerprint move?")
    print("=" * 82)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(zone_ids), size=min(40, len(zone_ids)), replace=False)
    shifts = np.arange(0, N_BEAR)
    curves = {m: np.zeros((len(sample), len(shifts))) for m in ("linear", "circular", "harmonic")}
    for si, zi in enumerate(sample):
        h = B[zi]
        for k, s in enumerate(shifts):
            hs = np.roll(h, s)                     # rotate the zone by s * 5 degrees
            curves["linear"][si, k] = linear_w(h, hs, N_BEAR)
            curves["circular"][si, k] = circular_w(h, hs)
            curves["harmonic"][si, k] = harmonic_d(h, hs)

    # how far apart are genuinely DIFFERENT zones, for scale?
    pairs = rng.choice(len(zone_ids), size=(400, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    between = {m: np.array([{"linear": linear_w(B[i], B[j], N_BEAR),
                             "circular": circular_w(B[i], B[j]),
                             "harmonic": harmonic_d(B[i], B[j])}[m]
                            for i, j in pairs]) for m in curves}

    print(f"{'metric':<12}{'d(zone, rotated self)':>24}{'d(zone, other zone)':>22}{'ratio':>10}")
    print("-" * 82)
    rot_summary = {}
    for m in ("linear", "circular", "harmonic"):
        self_rot = curves[m][:, 1:].mean()          # exclude shift 0
        other = between[m].mean()
        ratio = self_rot / (other + 1e-12)
        rot_summary[m] = {"self_rotated": float(self_rot), "between_zones": float(other),
                          "ratio": float(ratio)}
        print(f"{m:<12}{self_rot:>24.4f}{other:>22.4f}{ratio:>10.2f}")
    print("\n  A rotation-invariant metric scores 0 in the first column.")
    print(f"  linear   : rotating a zone moves it {rot_summary['linear']['ratio']*100:.0f}% as far")
    print(f"             as a genuinely different zone -- the bug, on real data.")
    print(f"  harmonic : {rot_summary['harmonic']['ratio']*100:.1f}% -- invariant as designed.")

    # ---- [2] rebuild the distance matrix, one term changed ----------------
    print("\n" + "=" * 82)
    print("[2] RE-CLUSTER -- change ONLY the bearing distance")
    print("=" * 82)
    scalar = pd.read_csv(os.path.join(CSV, "cluster_assignments.csv"))
    scalar_map = dict(zip(scalar["zone_id"], scalar["cluster_id"]))

    # Unnormalised linear = the published pipeline exactly, as a reproduction control.
    D_raw = build_matrix(B, A, L, "linear", normalize=False)
    best_raw, _ = spectral(D_raw, zone_ids)
    print(f"  CONTROL (unnormalised linear, i.e. uffm.py as published): "
          f"k={best_raw['k']} sil={best_raw['sil']:.4f}")
    print(f"  published uffm_clustering_scores.csv says k=2 sil=0.4685 -> "
          f"{'REPRODUCED' if abs(best_raw['sil'] - 0.4685) < 0.002 else 'MISMATCH'}\n")

    results = {}
    for mode in ("linear", "circular", "harmonic"):
        D = build_matrix(B, A, L, mode)
        best, summary = spectral(D, zone_ids)
        lab = best["labels"]
        sv = np.array([scalar_map.get(z, -1) for z in zone_ids])
        ok = sv >= 0
        ct = pd.crosstab(pd.Series(lab[ok], name="uffm"), pd.Series(sv[ok], name="scalar"))
        # agreement = best achievable match rate between the two labelings
        agree = ct.max(axis=0).sum() / ct.values.sum()
        ari = adjusted_rand(lab[ok], sv[ok])
        results[mode] = {"best_k": best["k"], "sil": best["sil"], "labels": lab,
                         "crosstab": ct, "agreement": float(agree), "ari": float(ari),
                         "summary": summary, "D": D}
        tag = " (reproduces the published UFFM run)" if mode == "linear" else ""
        print(f"\n  {mode.upper()} bearing term{tag}")
        print(f"    best k = {best['k']}   silhouette = {best['sil']:.4f}")
        print(f"    vs scalar clusters: agreement = {agree:.3f}   ARI = {ari:+.4f}")
        print(f"    crosstab:\n{ct.to_string()}")

    # ---- [3] independent check against Experiment 11 ----------------------
    print("\n" + "=" * 82)
    print("[3] INDEPENDENT CHECK -- do UFFM clusters track Experiment 11's mixture ratio?")
    print("=" * 82)
    ex11_path = os.path.join(HERE, "..", "11_city_mixture_map", "zone_final.csv")
    ex11 = pd.read_csv(ex11_path).set_index("zone_id") if os.path.exists(ex11_path) else None
    mix_summary = {}
    if ex11 is None:
        print("  Experiment 11 output not found -- skipped.")
    else:
        print("  UFFM and the patch metric share no code and no inputs beyond the graph,")
        print("  so agreement here is genuine cross-method confirmation.\n")
        print(f"  {'bearing term':<14}{'eta^2':>9}{'Kruskal p':>12}   cluster means (residual grid-ness)")
        print("-" * 82)
        for mode in ("linear", "circular", "harmonic"):
            lab = results[mode]["labels"]
            vals, groups = [], []
            for z, l in zip(zone_ids, lab):
                if z in ex11.index:
                    vals.append(ex11.loc[z, "resid"]); groups.append(l)
            vals, groups = np.array(vals), np.array(groups)
            gs = [vals[groups == g] for g in np.unique(groups) if (groups == g).sum() >= 3]
            if len(gs) < 2:
                continue
            h, p = stats.kruskal(*gs)
            grand = vals.mean()
            ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
            eta2 = ss_b / (((vals - grand) ** 2).sum() + 1e-12)
            means = "  ".join(f"{g.mean():+.3f}(n={len(g)})" for g in gs)
            mix_summary[mode] = {"eta2": float(eta2), "p": float(p)}
            print(f"  {mode:<14}{eta2:>9.3f}{p:>12.2e}   {means}")
        print("\n  eta^2 = share of Experiment 11's zone grid-ness explained by the UFFM label.")

        # ---- [4] the decisive check: is UFFM splitting on DENSITY? ----------
        print("\n" + "=" * 82)
        print("[4] WHAT IS UFFM ACTUALLY SPLITTING ON?")
        print("=" * 82)
        print("  Experiment 06 showed the scalar clusters are substantially a size/density")
        print("  artifact, so 'agrees with the scalar clusters' is NOT a success criterion.")
        print("  Compare how much of each UFFM split is explained by density vs by")
        print("  independently-measured morphology (Experiment 11's decorrelated residual).\n")
        meta = pd.read_csv(os.path.join(CSV, "metadata.csv")).set_index("zone_id")
        print(f"  {'bearing term':<14}{'eta^2 density':>16}{'eta^2 morphology':>19}"
              f"{'ratio':>10}")
        print("-" * 82)
        for mode in ("linear", "circular", "harmonic"):
            lab = results[mode]["labels"]
            dens, morph, grp = [], [], []
            for z, l in zip(zone_ids, lab):
                if z in ex11.index and z in meta.index:
                    dens.append(np.log(max(meta.loc[z, "osm_completeness"], 1e-3)))
                    morph.append(ex11.loc[z, "resid"])
                    grp.append(l)
            dens, morph, grp = np.array(dens), np.array(morph), np.array(grp)

            def eta2(v):
                gs = [v[grp == g] for g in np.unique(grp) if (grp == g).sum() >= 3]
                grand = v.mean()
                ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
                return ssb / (((v - grand) ** 2).sum() + 1e-12)

            ed, em = eta2(dens), eta2(morph)
            results[mode]["eta2_density"] = float(ed)
            results[mode]["eta2_morph"] = float(em)
            print(f"  {mode:<14}{ed:>16.3f}{em:>19.3f}{ed/(em+1e-12):>10.1f}x")
        print("\n  If density >> morphology for every variant, UFFM is a density measure")
        print("  wearing a geometry costume, and the rotation bug is not what broke it.")

        # ---- [5] which of the three terms carries the density? -------------
        print("\n" + "=" * 82)
        print("[5] TERM ABLATION -- where does the density signal enter?")
        print("=" * 82)
        print("  All three fingerprints are normalised to sum 1, so density has to be")
        print("  entering through the SHAPE of a distribution. Cluster on each term alone.\n")
        n = B.shape[0]
        terms = {
            "bearing (harmonic)": _unit_median(_pairwise(
                lambda i, j: harmonic_d(B[i], B[j]), n)),
            "angle": _unit_median(_pairwise(
                lambda i, j: linear_w(A[i], A[j], N_ANG), n)),
            "length": _unit_median(_pairwise(
                lambda i, j: linear_w(L[i], L[j], N_LEN), n)),
        }
        print(f"  {'term alone':<22}{'k':>4}{'sil':>9}{'eta2 density':>15}"
              f"{'eta2 morphology':>18}")
        print("-" * 82)
        ablation = {}
        for name, D in terms.items():
            best_t, _ = spectral(D, zone_ids)
            lab = best_t["labels"]
            dens, morph, grp = [], [], []
            for z, l in zip(zone_ids, lab):
                if z in ex11.index and z in meta.index:
                    dens.append(np.log(max(meta.loc[z, "osm_completeness"], 1e-3)))
                    morph.append(ex11.loc[z, "resid"])
                    grp.append(l)
            dens, morph, grp = np.array(dens), np.array(morph), np.array(grp)

            def eta2(v):
                gs = [v[grp == g] for g in np.unique(grp) if (grp == g).sum() >= 3]
                grand = v.mean()
                ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
                return ssb / (((v - grand) ** 2).sum() + 1e-12)

            ed, em = eta2(dens), eta2(morph)
            ablation[name] = {"k": int(best_t["k"]), "sil": best_t["sil"],
                              "eta2_density": float(ed), "eta2_morph": float(em)}
            print(f"  {name:<22}{best_t['k']:>4}{best_t['sil']:>9.4f}{ed:>15.3f}{em:>18.3f}")
        results["_ablation"] = ablation

    figures(curves, shifts, between, results, rot_summary)
    with open(os.path.join(HERE, "result.json"), "w") as f:
        json.dump({"rotation": rot_summary,
                   "ablation": results.get("_ablation", {}),
                   "clustering": {m: {"best_k": int(r["best_k"]), "sil": r["sil"],
                                      "agreement": r["agreement"], "ari": r["ari"],
                                      "eta2_density": r.get("eta2_density"),
                                      "eta2_morph": r.get("eta2_morph")}
                                  for m, r in results.items() if m != "_ablation"},
                   "vs_experiment11": mix_summary}, f, indent=2)
    for m, r in ((k, v) for k, v in results.items() if k != "_ablation"):
        pd.DataFrame({"zone_id": zone_ids, "cluster": r["labels"]}).to_csv(
            os.path.join(HERE, f"clusters_{m}.csv"), index=False)
        r["summary"].to_csv(os.path.join(HERE, f"scores_{m}.csv"), index=False)

    print("\n" + "=" * 82)
    print("See uffm_rotation_fix.png")
    print("=" * 82)
    return 0


def adjusted_rand(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def figures(curves, shifts, between, results, rot_summary):
    results = {k: v for k, v in results.items() if k != "_ablation"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    deg = shifts * (180.0 / len(shifts))
    colors = {"linear": "tab:red", "circular": "tab:orange", "harmonic": "tab:green"}
    for m in curves:
        mu = curves[m].mean(axis=0)
        sd = curves[m].std(axis=0)
        ax[0].plot(deg, mu, color=colors[m], label=f"{m} bearing distance", linewidth=2)
        ax[0].fill_between(deg, mu - sd, mu + sd, color=colors[m], alpha=.15)
        ax[0].axhline(between[m].mean(), color=colors[m], linestyle=":", linewidth=1.2)
    ax[0].set_xlabel("rotation applied to the zone (degrees)")
    ax[0].set_ylabel("distance from the zone to itself, rotated")
    ax[0].set_title("Rotate a real zone: does it become\na 'different' urban form?", fontsize=11)
    ax[0].legend(fontsize=8)
    ax[0].text(0.02, 0.96, "dotted = mean distance between DIFFERENT zones",
               transform=ax[0].transAxes, fontsize=7.5, va="top")
    ax[0].grid(alpha=.3)

    modes = list(results)
    ax[1].bar(range(len(modes)), [results[m]["sil"] for m in modes],
              color=[colors[m] for m in modes])
    for i, m in enumerate(modes):
        ax[1].text(i, results[m]["sil"] + .005, f"k={results[m]['best_k']}",
                   ha="center", fontsize=9)
    ax[1].set_xticks(range(len(modes))); ax[1].set_xticklabels(modes, fontsize=9)
    ax[1].set_ylabel("best silhouette"); ax[1].set_title("Clustering quality", fontsize=11)

    ax[2].bar(range(len(modes)), [results[m]["ari"] for m in modes],
              color=[colors[m] for m in modes])
    ax[2].axhline(0, color="k", linewidth=.8)
    ax[2].set_xticks(range(len(modes))); ax[2].set_xticklabels(modes, fontsize=9)
    ax[2].set_ylabel("Adjusted Rand vs scalar clusters")
    ax[2].set_title("Does UFFM now agree with the\nscalar-feature clustering?", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "uffm_rotation_fix.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

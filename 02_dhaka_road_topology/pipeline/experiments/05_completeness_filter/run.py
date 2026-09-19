#!/usr/bin/env python
"""Experiment 05: exclude low-OSM-completeness zones (mostly informal/
under-mapped areas) before clustering, on the best-performing config so far
(2km scalar-only, Experiment 03).

Usage:
    python experiments/05_completeness_filter/run.py --config configs/v2km.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from dhaka_topology.clustering import compare_algorithms, permutation_test, run_dbscan, run_gmm, run_kmeans, run_pca
from dhaka_topology.config import Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--completeness-threshold", type=float, default=0.5)
    parser.add_argument("--n-permutations", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = Config.from_yaml(args.config)
    norm_df = pd.read_csv(os.path.join(config.out_csv, "features_normalized.csv"))
    meta_df = pd.read_csv(os.path.join(config.out_csv, "metadata.csv"))

    low_completeness = meta_df.loc[meta_df["osm_completeness"] < args.completeness_threshold, "zone_id"]
    filtered_df = norm_df[~norm_df["zone_id"].isin(low_completeness)].reset_index(drop=True)

    out_dir = os.path.dirname(__file__)
    filtered_df.to_csv(os.path.join(out_dir, "filtered_features.csv"), index=False)

    pca_result = run_pca(filtered_df, config)
    km = run_kmeans(pca_result["X_pca"], config)
    db = run_dbscan(pca_result["X_pca"], config)
    gmm = run_gmm(pca_result["X_pca"], config)

    class _LocalConfig:
        out_csv = out_dir

    comparison = compare_algorithms(km, db, gmm, _LocalConfig())
    sig = permutation_test(filtered_df, config, n_permutations=args.n_permutations)

    result = {
        "n_zones_before": len(norm_df),
        "n_zones_excluded": int(len(low_completeness)),
        "n_zones_after": len(filtered_df),
        "kmeans": {"best_k": km["best_k"], "best_sil": km["best_sil"]},
        "dbscan": {"n_clusters": db["n_clusters"], "sil": db["sil"]},
        "gmm": {"best_sil": gmm["best_sil"]},
        "winner": comparison["best_algo"],
        "winner_sil": comparison["comparison"][comparison["best_algo"]][0],
        "significance": {k: v for k, v in sig.items() if k != "null_sils"},
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("OSM-COMPLETENESS-FILTERED CLUSTERING RESULT")
    print("=" * 60)
    print(f"Zones: {result['n_zones_before']} -> {result['n_zones_after']} "
          f"({result['n_zones_excluded']} excluded, completeness < {args.completeness_threshold})")
    print(f"KMeans : k={km['best_k']}  silhouette={km['best_sil']:.4f}")
    print(f"DBSCAN : n_clusters={db['n_clusters']}  silhouette={db['sil']:.4f}")
    print(f"GMM    : silhouette={gmm['best_sil']:.4f}")
    print(f"Winner : {comparison['best_algo']} (silhouette={result['winner_sil']:.4f})")
    print(f"\nSignificance: p={sig['p_value']:.4f}  null mean={sig['null_mean']:.4f}  "
          f"null 95th pct={sig['null_p95']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

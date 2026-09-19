#!/usr/bin/env python
"""Experiment 02 (plan's "Experiment 1"): re-cluster using only the 5
bootstrap-stable features (CV <= 0.35), dropping the 18 unstable ones.

Usage:
    python experiments/02_stable_features_only/run.py --config configs/v3.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from dhaka_topology.clustering import (
    compare_algorithms, filter_stable_features, permutation_test,
    run_dbscan, run_gmm, run_kmeans, run_pca,
)
from dhaka_topology.config import Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cv-threshold", type=float, default=0.35)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger(__name__)

    config = Config.from_yaml(args.config)
    norm_df = pd.read_csv(os.path.join(config.out_csv, "features_normalized.csv"))
    stability_df = pd.read_csv(os.path.join(config.out_csv, "feature_stability.csv"))

    stable_df = filter_stable_features(norm_df, stability_df, cv_threshold=args.cv_threshold)

    pca_result = run_pca(stable_df, config)
    km = run_kmeans(pca_result["X_pca"], config)
    db = run_dbscan(pca_result["X_pca"], config)
    gmm = run_gmm(pca_result["X_pca"], config)

    # compare_algorithms writes clustering_scores.csv to config.out_csv - redirect to
    # this experiment's own folder instead, so it doesn't clobber the baseline.
    out_dir = os.path.dirname(__file__)

    class _LocalConfig:
        out_csv = out_dir

    comparison = compare_algorithms(km, db, gmm, _LocalConfig())

    sig = permutation_test(stable_df, config, n_permutations=args.n_permutations)

    result = {
        "n_features_used": len(pca_result["feat_cols"]),
        "features_used": pca_result["feat_cols"],
        "pca_n_components": pca_result["n_comp"],
        "kmeans": {"best_k": km["best_k"], "best_sil": km["best_sil"]},
        "dbscan": {"eps": db["eps"], "n_clusters": db["n_clusters"], "sil": db["sil"]},
        "gmm": {"best_sil": gmm["best_sil"]},
        "winner": comparison["best_algo"],
        "winner_sil": comparison["comparison"][comparison["best_algo"]][0],
        "significance": {k: v for k, v in sig.items() if k != "null_sils"},
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print("STABLE-FEATURES-ONLY CLUSTERING RESULT")
    print("=" * 60)
    print(f"Features used ({result['n_features_used']}): {result['features_used']}")
    print(f"PCA components: {result['pca_n_components']}")
    print(f"KMeans : k={km['best_k']}  silhouette={km['best_sil']:.4f}")
    print(f"DBSCAN : n_clusters={db['n_clusters']}  silhouette={db['sil']:.4f}")
    print(f"GMM    : silhouette={gmm['best_sil']:.4f}")
    print(f"Winner : {comparison['best_algo']} (silhouette={result['winner_sil']:.4f})")
    print(f"\nSignificance vs baseline (24 features, sil=0.408):")
    print(f"  This result's p-value: {sig['p_value']:.4f}")
    print(f"  Null mean: {sig['null_mean']:.4f}  Null 95th pct: {sig['null_p95']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

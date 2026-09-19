#!/usr/bin/env python
"""Experiment 06: regress each feature against zone size (log node_count)
and cluster on the residuals, to test whether any real (non-size-driven)
structure survives after removing the confound found while preparing the
satellite validation step.

Usage:
    python experiments/06_decorrelate_from_size/run.py --config configs/v2km.yaml
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
    assign_clusters, compare_algorithms, permutation_test, residualize_features,
    run_dbscan, run_gmm, run_kmeans, run_pca,
)
from dhaka_topology.config import Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True, help="e.g. '1km' or '2km', used in output filenames")
    parser.add_argument("--n-permutations", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = Config.from_yaml(args.config)
    norm_df = pd.read_csv(os.path.join(config.out_csv, "features_normalized.csv"))
    raw_df = pd.read_csv(config.features_csv)

    resid_df, r2_per_feature = residualize_features(norm_df, raw_df)

    out_dir = os.path.dirname(__file__)
    resid_df.to_csv(os.path.join(out_dir, f"residual_features_{args.label}.csv"), index=False)

    pca_result = run_pca(resid_df, config)
    km = run_kmeans(pca_result["X_pca"], config)
    db = run_dbscan(pca_result["X_pca"], config)
    gmm = run_gmm(pca_result["X_pca"], config)

    class _LocalConfig:
        out_csv = out_dir
        clusters_csv = os.path.join(out_dir, f"cluster_assignments_{args.label}.csv")
        metadata_csv = os.path.join(out_dir, "__no_metadata_to_merge__.csv")  # skip merge safely

    comparison = compare_algorithms(km, db, gmm, _LocalConfig())
    cluster_df = assign_clusters(comparison["best_labels"], resid_df, raw_df, _LocalConfig())
    sig = permutation_test(resid_df, config, n_permutations=args.n_permutations)

    result = {
        "label": args.label,
        "r2_against_size": r2_per_feature,
        "pca_n_components": pca_result["n_comp"],
        "kmeans": {"best_k": km["best_k"], "best_sil": km["best_sil"]},
        "dbscan": {"n_clusters": db["n_clusters"], "sil": db["sil"]},
        "gmm": {"best_sil": gmm["best_sil"]},
        "winner": comparison["best_algo"],
        "winner_sil": comparison["comparison"][comparison["best_algo"]][0],
        "significance": {k: v for k, v in sig.items() if k != "null_sils"},
    }
    with open(os.path.join(out_dir, f"result_{args.label}.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SIZE-DECORRELATED CLUSTERING RESULT ({args.label})")
    print("=" * 60)
    print("R^2 against log(node_count) per feature (higher = more of a size proxy):")
    for feat, r2 in sorted(r2_per_feature.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {feat:<28} R^2={r2:.4f}")
    print(f"\nKMeans : k={km['best_k']}  silhouette={km['best_sil']:.4f}")
    print(f"DBSCAN : n_clusters={db['n_clusters']}  silhouette={db['sil']:.4f}")
    print(f"GMM    : silhouette={gmm['best_sil']:.4f}")
    print(f"Winner : {comparison['best_algo']} (silhouette={result['winner_sil']:.4f})")
    print(f"\nSignificance: p={sig['p_value']:.4f}  null mean={sig['null_mean']:.4f}  "
          f"null 95th pct={sig['null_p95']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

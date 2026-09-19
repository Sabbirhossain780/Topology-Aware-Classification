#!/usr/bin/env python
"""Experiment 1 (plan's "Experiment 4"): is the baseline clustering result
statistically distinguishable from noise?

Usage:
    python experiments/01_significance_test/run.py --config configs/v3.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd

from dhaka_topology.clustering import permutation_test
from dhaka_topology.config import Config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--out-dir", default=None,
                         help="Where to write result.json/null_distribution.csv "
                              "(default: this script's own directory)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = Config.from_yaml(args.config)
    norm_df = pd.read_csv(os.path.join(config.out_csv, "features_normalized.csv"))

    result = permutation_test(norm_df, config, n_permutations=args.n_permutations)

    out_dir = args.out_dir or os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)
    result_to_save = {k: v for k, v in result.items() if k != "null_sils"}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result_to_save, f, indent=2)

    pd.DataFrame({"null_silhouette": result["null_sils"]}).to_csv(
        os.path.join(out_dir, "null_distribution.csv"), index=False)

    print("\n" + "=" * 60)
    print("SIGNIFICANCE TEST RESULT")
    print("=" * 60)
    print(f"Real best silhouette   : {result['real_best_sil']:.4f}  (k={result['real_best_k']})")
    print(f"Null distribution mean : {result['null_mean']:.4f} +/- {result['null_std']:.4f}")
    print(f"Null 95th percentile   : {result['null_p95']:.4f}")
    print(f"p-value                : {result['p_value']:.4f}")
    print(f"n permutations         : {result['n_permutations']}")
    verdict = "SIGNIFICANT (p < 0.05)" if result["p_value"] < 0.05 else "NOT significant (p >= 0.05)"
    print(f"\nVerdict: {verdict}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

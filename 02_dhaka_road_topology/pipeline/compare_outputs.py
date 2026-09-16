#!/usr/bin/env python
"""Compare new pipeline outputs against the original notebook-based outputs.

Usage:
    python compare_outputs.py \
        --old "../../backup_original_topology/02_dhaka_road_topology/outputs/v3" \
        --new "../../pipeline_outputs/v3/outputs" \
        --old-gat "../../backup_original_topology/02_dhaka_road_topology/gat" \
        --new-gat "../../pipeline_outputs/v3/gat"

For each matching CSV, reports whether values are numerically identical
(within a small float tolerance) — this is the ground-truth check that the
refactor changed nothing about the actual analysis.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

TOLERANCE = 1e-6

# Columns that are expected to differ even when the underlying data matches
# (labels the user assigns by hand, algorithm-internal object ids, or
# wall-clock timings that are never reproducible run-to-run).
IGNORE_COLS = {"cluster_label", "uffm_cluster_label", "compute_time_s"}


def compare_csv(old_path: str, new_path: str) -> dict:
    if not os.path.exists(old_path):
        return {"status": "old_missing"}
    if not os.path.exists(new_path):
        return {"status": "new_missing"}

    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    if "zone_id" in old_df.columns and "zone_id" in new_df.columns:
        old_df = old_df.sort_values("zone_id").reset_index(drop=True)
        new_df = new_df.sort_values("zone_id").reset_index(drop=True)

    if list(old_df.columns) != list(new_df.columns):
        return {"status": "column_mismatch",
                "old_cols": list(old_df.columns), "new_cols": list(new_df.columns)}

    if len(old_df) != len(new_df):
        return {"status": "row_count_mismatch", "old_rows": len(old_df), "new_rows": len(new_df)}

    mismatches = {}
    for col in old_df.columns:
        if col in IGNORE_COLS:
            continue
        old_col, new_col = old_df[col], new_df[col]
        if pd.api.types.is_numeric_dtype(old_col) and pd.api.types.is_numeric_dtype(new_col):
            diff = np.abs(old_col.fillna(0).values - new_col.fillna(0).values)
            n_bad = int((diff > TOLERANCE).sum())
            if n_bad:
                mismatches[col] = {"n_mismatched": n_bad, "max_diff": float(diff.max())}
        else:
            n_bad = int((old_col.astype(str) != new_col.astype(str)).sum())
            if n_bad:
                mismatches[col] = {"n_mismatched": n_bad}

    return {"status": "match" if not mismatches else "value_mismatch", "mismatches": mismatches,
            "n_rows": len(old_df)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Old outputs/csv directory")
    parser.add_argument("--new", required=True, help="New outputs/csv directory")
    parser.add_argument("--old-gat", default=None, help="Old gat/csv directory")
    parser.add_argument("--new-gat", default=None, help="New gat/csv directory")
    args = parser.parse_args()

    targets = [
        "features.csv", "features_normalized.csv", "cluster_assignments.csv",
        "clustering_scores.csv", "metadata.csv",
        "uffm_fingerprints.csv", "uffm_cluster_assignments.csv", "uffm_clustering_scores.csv",
    ]

    old_csv = os.path.join(args.old, "csv") if os.path.isdir(os.path.join(args.old, "csv")) else args.old
    new_csv = os.path.join(args.new, "csv") if os.path.isdir(os.path.join(args.new, "csv")) else args.new

    all_ok = True
    print("=" * 70)
    print("TOPOLOGY OUTPUT COMPARISON")
    print("=" * 70)

    for fname in targets:
        result = compare_csv(os.path.join(old_csv, fname), os.path.join(new_csv, fname))
        status = result["status"]
        mark = "OK" if status == "match" else "FAIL"
        if status != "match":
            all_ok = False
        print(f"\n[{mark}] {fname}: {status}")
        if status == "value_mismatch":
            for col, info in result["mismatches"].items():
                print(f"    {col}: {info}")
        elif status == "column_mismatch":
            print(f"    old cols: {result['old_cols']}")
            print(f"    new cols: {result['new_cols']}")
        elif status == "row_count_mismatch":
            print(f"    old rows: {result['old_rows']}  new rows: {result['new_rows']}")
        elif status == "match":
            print(f"    {result['n_rows']} rows, all numeric columns within {TOLERANCE}")

    if args.old_gat and args.new_gat:
        print("\n" + "=" * 70)
        print("GAT (BETWEENNESS) COMPARISON")
        print("=" * 70)
        for fname in ["all_zones_summary.csv", "all_nodes_bc.csv"]:
            result = compare_csv(
                os.path.join(args.old_gat, "csv", fname),
                os.path.join(args.new_gat, "csv", fname),
            )
            status = result["status"]
            mark = "OK" if status == "match" else "FAIL"
            if status != "match":
                all_ok = False
            print(f"\n[{mark}] {fname}: {status}")
            if status == "value_mismatch":
                for col, info in result["mismatches"].items():
                    print(f"    {col}: {info}")

    print("\n" + "=" * 70)
    print("RESULT:", "ALL MATCH" if all_ok else "MISMATCHES FOUND — see above")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

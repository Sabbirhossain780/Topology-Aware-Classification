"""Experiment 04 — fuse scalar topology features with UFFM geometric
fingerprints into one representation.

Rather than merging all 112 raw UFFM histogram bins (36 bearing + 36 angle +
40 length) into the feature space -- which would dwarf the 23 scalar
features with redundant, highly-correlated dimensions -- this derives a
small set of summary statistics per fingerprint: entropy (how spread out
the distribution is), peak location, and the grid/T-junction diagnostic
fractions already used in uffm.py's own diagnostic report (angle mass near
90 degrees vs 180 degrees). These carry the shape information that scalar
features miss without reintroducing the redundancy UFFM's own separate
clustering already showed doesn't merge cleanly with the scalar clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def derive_uffm_summary_features(fingerprints_df: pd.DataFrame, config) -> pd.DataFrame:
    """One row per zone: entropy + shape summaries of the 3 UFFM fingerprints."""
    bearing_cols = [f"b_{i}" for i in range(config.bearing_bins)]
    angle_cols = [f"a_{i}" for i in range(config.angle_bins)]
    length_cols = [f"l_{i}" for i in range(config.length_bins)]

    B = fingerprints_df[bearing_cols].values.astype(float)
    A = fingerprints_df[angle_cols].values.astype(float)
    L = fingerprints_df[length_cols].values.astype(float)

    x_bearing_deg = np.linspace(0, 180, config.bearing_bins)
    x_angle_deg = np.linspace(0, 180, config.angle_bins)
    x_length_m = np.linspace(0, config.length_max_m, config.length_bins)

    bin_90 = int(round(90 / 180 * (config.angle_bins - 1)))
    bin_180 = config.angle_bins - 1

    rows = []
    for i in range(len(fingerprints_df)):
        b, a, l = B[i], A[i], L[i]

        p90 = float(a[max(0, bin_90 - 1):bin_90 + 2].sum())
        p180 = float(a[max(0, bin_180 - 2):].sum())

        l_cumsum = np.cumsum(l)
        median_bin = min(int(np.searchsorted(l_cumsum, 0.5)), config.length_bins - 1)
        median_length_m = x_length_m[median_bin]

        rows.append({
            "zone_id": fingerprints_df.iloc[i]["zone_id"],
            "uffm_bearing_entropy": _entropy(b),
            "uffm_angle_entropy": _entropy(a),
            "uffm_length_entropy": _entropy(l),
            "uffm_dominant_bearing_deg": float(x_bearing_deg[np.argmax(b)]),
            "uffm_angle_p90": p90,
            "uffm_angle_p180": p180,
            "uffm_median_length_m": median_length_m,
        })

    return pd.DataFrame(rows)


def build_fused_features(norm_df: pd.DataFrame, fingerprints_df: pd.DataFrame, config) -> pd.DataFrame:
    """Merge normalized scalar features with min-max normalized UFFM summaries."""
    uffm_summary = derive_uffm_summary_features(fingerprints_df, config)

    summary_cols = [c for c in uffm_summary.columns if c != "zone_id"]
    for col in summary_cols:
        mn, mx = uffm_summary[col].min(), uffm_summary[col].max()
        uffm_summary[col] = (uffm_summary[col] - mn) / (mx - mn) if mx > mn else 0.0

    fused = norm_df.merge(uffm_summary, on="zone_id", how="inner")
    return fused

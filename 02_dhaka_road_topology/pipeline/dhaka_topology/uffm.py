"""UFFM — Urban Form Fingerprint Matching (formerly uffm_algorithm_v3.ipynb).

Instead of scalar feature vectors, each zone gets three distribution
fingerprints (street bearing, intersection angle, segment length) compared
via weighted Wasserstein distance, then grouped with Spectral Clustering.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from .io_utils import list_zone_ids

log = logging.getLogger(__name__)


def _safe_hist(values, n_bins: int, val_range: tuple[float, float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.ones(n_bins, dtype=float) / n_bins
    counts, _ = np.histogram(values, bins=n_bins, range=val_range)
    total = counts.sum()
    if total == 0:
        return np.ones(n_bins, dtype=float) / n_bins
    return counts.astype(float) / total


def bearing_fingerprint(nodes_df: pd.DataFrame, links_df: pd.DataFrame, n_bins: int) -> np.ndarray:
    """Histogram of street bearing angles folded to [0, 180) (undirected)."""
    if "x_utm" not in nodes_df.columns or nodes_df.empty or links_df.empty:
        return np.ones(n_bins, dtype=float) / n_bins

    pos_x = pd.Series(nodes_df["x_utm"].values, index=nodes_df["node_id"].values)
    pos_y = pd.Series(nodes_df["y_utm"].values, index=nodes_df["node_id"].values)

    mask = (links_df["from_node"].isin(pos_x.index) & links_df["to_node"].isin(pos_x.index))
    if mask.sum() < 3:
        return np.ones(n_bins, dtype=float) / n_bins

    fn = links_df.loc[mask, "from_node"].values
    tn = links_df.loc[mask, "to_node"].values
    dx = pos_x[tn].values - pos_x[fn].values
    dy = pos_y[tn].values - pos_y[fn].values

    nz = (dx != 0) | (dy != 0)
    dx, dy = dx[nz], dy[nz]
    if len(dx) < 3:
        return np.ones(n_bins, dtype=float) / n_bins

    angles = np.degrees(np.arctan2(dy, dx)) % 360
    angles[angles >= 180] -= 180
    return _safe_hist(angles, n_bins, (0, 180))


def angle_fingerprint(nodes_df: pd.DataFrame, links_df: pd.DataFrame, n_bins: int) -> np.ndarray:
    """Angle between every pair of connecting streets at each node with >=2 connections."""
    if "x_utm" not in nodes_df.columns or nodes_df.empty or links_df.empty:
        return np.ones(n_bins, dtype=float) / n_bins

    pos = dict(zip(nodes_df["node_id"].values,
                    zip(nodes_df["x_utm"].values, nodes_df["y_utm"].values)))

    adj = defaultdict(list)
    valid_mask = (links_df["from_node"].isin(pos) & links_df["to_node"].isin(pos))
    for _, r in links_df[valid_mask].iterrows():
        u, v = r["from_node"], r["to_node"]
        adj[u].append(v)
        adj[v].append(u)

    intersection_angles = []
    for node, neighbours in adj.items():
        if len(neighbours) < 2:
            continue
        px, py = pos[node]
        vecs = []
        for nb in neighbours:
            if nb not in pos:
                continue
            dx = pos[nb][0] - px
            dy = pos[nb][1] - py
            length = np.hypot(dx, dy)
            if length > 0:
                vecs.append((dx / length, dy / length))
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dot = np.clip(vecs[i][0] * vecs[j][0] + vecs[i][1] * vecs[j][1], -1.0, 1.0)
                intersection_angles.append(np.degrees(np.arccos(dot)))

    return _safe_hist(intersection_angles, n_bins, (0, 180))


def length_fingerprint(links_df: pd.DataFrame, n_bins: int, max_len: float) -> np.ndarray:
    """Histogram of street segment lengths, capped at max_len metres."""
    if "length_m" in links_df.columns:
        lengths = links_df["length_m"].dropna().values.astype(float)
    elif "length_km" in links_df.columns:
        lengths = links_df["length_km"].dropna().values.astype(float) * 1000.0
    else:
        return np.ones(n_bins, dtype=float) / n_bins

    lengths = np.clip(lengths, 0, max_len)
    return _safe_hist(lengths, n_bins, (0, max_len))


def extract_fingerprints(config) -> pd.DataFrame:
    zone_ids = list_zone_ids(config.data_zones)
    bearing_cols = [f"b_{i}" for i in range(config.bearing_bins)]
    angle_cols = [f"a_{i}" for i in range(config.angle_bins)]
    length_cols = [f"l_{i}" for i in range(config.length_bins)]

    records, failed = [], []
    for zone_id in tqdm(zone_ids, desc="UFFM fingerprints"):
        try:
            zone_dir = os.path.join(config.data_zones, zone_id)
            nodes_df = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
            links_df = pd.read_csv(os.path.join(zone_dir, "links.csv"))

            b_fp = bearing_fingerprint(nodes_df, links_df, config.bearing_bins)
            a_fp = angle_fingerprint(nodes_df, links_df, config.angle_bins)
            l_fp = length_fingerprint(links_df, config.length_bins, config.length_max_m)

            row = {"zone_id": zone_id}
            for col, val in zip(bearing_cols, b_fp): row[col] = round(float(val), 6)
            for col, val in zip(angle_cols, a_fp): row[col] = round(float(val), 6)
            for col, val in zip(length_cols, l_fp): row[col] = round(float(val), 6)
            records.append(row)
        except Exception as e:
            failed.append({"zone_id": zone_id, "error": str(e)})

    fingerprints_df = pd.DataFrame(records)
    os.makedirs(config.out_csv, exist_ok=True)
    fingerprints_df.to_csv(os.path.join(config.out_csv, "uffm_fingerprints.csv"), index=False)

    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(config.out_csv, "uffm_failed_zones.csv"), index=False)

    return fingerprints_df


def build_distance_matrix(fingerprints_df: pd.DataFrame, config) -> pd.DataFrame:
    bearing_cols = [f"b_{i}" for i in range(config.bearing_bins)]
    angle_cols = [f"a_{i}" for i in range(config.angle_bins)]
    length_cols = [f"l_{i}" for i in range(config.length_bins)]

    zone_ids = fingerprints_df["zone_id"].tolist()
    N = len(zone_ids)

    B = fingerprints_df[bearing_cols].values.astype(float)
    A = fingerprints_df[angle_cols].values.astype(float)
    L = fingerprints_df[length_cols].values.astype(float)

    x_bearing = np.linspace(0, 1, config.bearing_bins)
    x_angle = np.linspace(0, 1, config.angle_bins)
    x_length = np.linspace(0, 1, config.length_bins)

    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="Wasserstein distances"):
        for j in range(i + 1, N):
            d = (
                config.w_bearing * wasserstein_distance(x_bearing, x_bearing, B[i], B[j]) +
                config.w_angle * wasserstein_distance(x_angle, x_angle, A[i], A[j]) +
                config.w_length * wasserstein_distance(x_length, x_length, L[i], L[j])
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    dist_df = pd.DataFrame(dist_matrix, index=zone_ids, columns=zone_ids)
    os.makedirs(config.out_csv, exist_ok=True)
    dist_df.to_csv(os.path.join(config.out_csv, "uffm_distance_matrix.csv"))
    return dist_df


def run_spectral_clustering(dist_df: pd.DataFrame, config) -> dict:
    dist_matrix = dist_df.values.astype(np.float32)
    zone_ids = list(dist_df.index)

    nz_vals = dist_matrix[dist_matrix > 0]
    sigma = float(np.median(nz_vals))
    similarity = np.exp(-(dist_matrix ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(similarity, 1.0)

    results = []
    for k in tqdm(config.k_range, desc="Spectral Clustering"):
        try:
            sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                     random_state=config.random_state, n_init=10,
                                     assign_labels="kmeans")
            labels = sc.fit_predict(similarity)
            sil = silhouette_score(dist_matrix, labels, metric="precomputed") if k > 1 else -1.0
            results.append({"k": k, "silhouette": round(float(sil), 4), "labels": labels})
        except Exception as e:
            log.warning("Spectral clustering k=%d failed: %s", k, e)

    if not results:
        raise RuntimeError("All Spectral Clustering attempts failed")

    sc_summary = pd.DataFrame([{"k": r["k"], "silhouette": r["silhouette"]} for r in results])
    os.makedirs(config.out_csv, exist_ok=True)
    sc_summary.to_csv(os.path.join(config.out_csv, "uffm_clustering_scores.csv"), index=False)

    best = max(results, key=lambda r: r["silhouette"])

    uffm_cluster_df = pd.DataFrame({
        "zone_id": zone_ids,
        "uffm_cluster_id": best["labels"].tolist(),
        "uffm_cluster_label": [f"UF_Cluster_{l}" for l in best["labels"]],
    })
    uffm_cluster_df.to_csv(os.path.join(config.out_csv, "uffm_cluster_assignments.csv"), index=False)

    return {"best_k": int(best["k"]), "best_sil": float(best["silhouette"]),
            "labels": best["labels"], "cluster_df": uffm_cluster_df, "sc_summary": sc_summary}


def run_uffm_stage(config) -> dict:
    """Orchestrates fingerprint extraction -> distance matrix -> spectral clustering."""
    fingerprints_df = extract_fingerprints(config)
    dist_df = build_distance_matrix(fingerprints_df, config)
    clustering = run_spectral_clustering(dist_df, config)
    return {"fingerprints_df": fingerprints_df, "dist_df": dist_df, **clustering}


def crosstab_vs_v3(config) -> pd.DataFrame | None:
    """Cross-tabulate UFFM clusters against the scalar-feature (v3) clusters."""
    uffm_path = os.path.join(config.out_csv, "uffm_cluster_assignments.csv")
    if not (os.path.exists(config.clusters_csv) and os.path.exists(uffm_path)):
        return None
    v3_df = pd.read_csv(config.clusters_csv)
    uffm_df = pd.read_csv(uffm_path)
    merged = v3_df.merge(uffm_df, on="zone_id", how="inner")
    ct = pd.crosstab(merged["cluster_label"], merged["uffm_cluster_label"])
    ct.to_csv(os.path.join(config.out_csv, "uffm_v3_crosstab.csv"))
    return ct

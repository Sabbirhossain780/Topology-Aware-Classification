"""Stage 4 — PCA + clustering + diagnostic discovery.

Runs K-Means, DBSCAN (auto-tuned eps), and GMM; picks the winner by
silhouette score; saves neutral Cluster_N labels plus a diagnostic report.
Formulas copied verbatim from dhaka_topology_v3.ipynb Stage 4.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from .features import SIZE_ONLY_FEATURES

log = logging.getLogger(__name__)

KEY_FEATURES = [
    ("node_count", "Network density (raw nodes - SIZE indicator, use cautiously)"),
    ("orientation_entropy", "Street orientation entropy: LOW=grid, HIGH=organic maze"),
    ("betweenness_gini", "Betweenness inequality: LOW=uniform, HIGH=few bottlenecks"),
    ("global_efficiency", "Path efficiency: HIGH=well-connected, LOW=fragmented"),
    ("hub_dominance_ratio", "Hub dominance: HIGH=star/radial, LOW=uniform grid"),
    ("wcc_fragmentation", "Fragmentation: HIGH=many components, LOW=connected"),
    ("gwcc_pct", "Largest WCC %: HIGH=connected, LOW=broken network"),
    ("avg_degree", "Avg connections per node: HIGH=dense mesh"),
    ("density", "Edge density: HIGH=tightly connected"),
    ("transitivity", "Triangle formation: HIGH=local clustering"),
]


def run_pca(norm_df: pd.DataFrame, config):
    feat_cols = [c for c in norm_df.columns if c != "zone_id"]
    X = norm_df[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(random_state=config.random_state)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    n_comp = int(np.searchsorted(cumvar, config.pca_variance)) + 1
    n_comp = max(n_comp, 2)
    n_comp = min(n_comp, len(feat_cols))

    pca = PCA(n_components=n_comp, random_state=config.random_state)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "X_pca": X_pca, "n_comp": n_comp, "cumvar": cumvar,
        "feat_cols": feat_cols, "zone_ids": norm_df["zone_id"].tolist(),
    }


def run_kmeans(X_pca: np.ndarray, config) -> dict:
    results = []
    for k in tqdm(config.k_range, desc="K-Means"):
        km = KMeans(n_clusters=k, random_state=config.random_state, n_init=20, max_iter=500)
        lbl = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca, lbl) if k > 1 else -1
        ch = calinski_harabasz_score(X_pca, lbl) if k > 1 else 0
        results.append({"k": k, "silhouette": sil, "ch": ch, "wcss": km.inertia_, "labels": lbl})

    df = pd.DataFrame(results)
    best = df.loc[df["silhouette"].idxmax()]
    return {"results": results, "best_k": int(best["k"]),
            "best_labels": best["labels"], "best_sil": best["silhouette"]}


def run_dbscan(X_pca: np.ndarray, config) -> dict:
    nbrs = NearestNeighbors(n_neighbors=config.dbscan_min_samples).fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)
    kth_dists = np.sort(distances[:, -1])[::-1]
    d2 = np.diff(np.diff(kth_dists))
    knee_idx = np.argmax(np.abs(d2)) + 2
    eps = float(kth_dists[knee_idx])

    db = DBSCAN(eps=eps, min_samples=config.dbscan_min_samples)
    labels = db.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_pca, labels) if n_clusters > 1 else -1
    return {"eps": eps, "labels": labels, "sil": sil, "n_clusters": n_clusters}


def run_gmm(X_pca: np.ndarray, config) -> dict:
    results = []
    for k in tqdm(config.k_range, desc="GMM"):
        gmm = GaussianMixture(n_components=k, random_state=config.random_state, n_init=5)
        lbl = gmm.fit_predict(X_pca)
        bic = gmm.bic(X_pca)
        sil = silhouette_score(X_pca, lbl) if k > 1 else -1
        results.append({"k": k, "bic": bic, "silhouette": sil, "labels": lbl})

    df = pd.DataFrame(results)
    best = df.loc[df["bic"].idxmin()]
    return {"results": results, "best_labels": best["labels"], "best_sil": best["silhouette"]}


def compare_algorithms(km: dict, db: dict, gmm: dict, config) -> dict:
    comparison = {
        "KMeans": (km["best_sil"], km["best_labels"]),
        "DBSCAN": (db["sil"], db["labels"]),
        "GMM": (gmm["best_sil"], gmm["best_labels"]),
    }
    best_algo = max(comparison, key=lambda a: comparison[a][0])
    best_labels = comparison[best_algo][1]

    os.makedirs(config.out_csv, exist_ok=True)
    pd.DataFrame([
        {"algorithm": a, "silhouette": comparison[a][0],
         "n_clusters": len(set(comparison[a][1])) - (1 if -1 in comparison[a][1] else 0)}
        for a in comparison
    ]).to_csv(os.path.join(config.out_csv, "clustering_scores.csv"), index=False)

    return {"best_algo": best_algo, "best_labels": best_labels, "comparison": comparison}


def assign_clusters(best_labels: np.ndarray, norm_df: pd.DataFrame, raw_df: pd.DataFrame, config) -> pd.DataFrame:
    """Assign neutral Cluster_N labels, compute confidence scores, save + print diagnostics."""
    feat_cols = [c for c in norm_df.columns if c != "zone_id"]
    zone_ids = norm_df["zone_id"].tolist()

    unique_ids = sorted(np.unique(best_labels))
    final_map = {cid: ("Noise" if cid == -1 else f"Cluster_{cid}") for cid in unique_ids}

    X_for_conf = MinMaxScaler().fit_transform(norm_df[feat_cols].fillna(0).values)
    centroids = {cid: X_for_conf[best_labels == cid].mean(axis=0)
                 for cid in unique_ids if cid != -1}
    dists = []
    for i, cid in enumerate(best_labels):
        dists.append(1.0 if cid == -1 else float(np.linalg.norm(X_for_conf[i] - centroids[cid])))
    max_d = max(dists) if dists else 1.0

    cluster_df = pd.DataFrame({
        "zone_id": zone_ids,
        "cluster_id": best_labels,
        "cluster_label": [final_map[l] for l in best_labels],
        "confidence_score": [round(1 - d / max_d, 4) for d in dists],
    })
    os.makedirs(config.out_csv, exist_ok=True)
    cluster_df.to_csv(config.clusters_csv, index=False)

    if os.path.exists(config.metadata_csv):
        meta_df = pd.read_csv(config.metadata_csv)
        for c in ["cluster_id", "cluster_label", "confidence_score"]:
            if c in meta_df.columns:
                meta_df.drop(columns=[c], inplace=True)
        meta_df = meta_df.merge(
            cluster_df[["zone_id", "cluster_id", "cluster_label", "confidence_score"]],
            on="zone_id", how="left")
        meta_df.to_csv(config.metadata_csv, index=False)

    _print_diagnostic_report(cluster_df, raw_df, feat_cols)
    return cluster_df


def _print_diagnostic_report(cluster_df: pd.DataFrame, raw_df: pd.DataFrame, feat_cols: list[str]) -> None:
    clust_feats = raw_df[[c for c in feat_cols if c in raw_df.columns]].copy()
    clust_feats["cluster_label"] = cluster_df["cluster_label"].values
    profiles = clust_feats.groupby("cluster_label").mean(numeric_only=True)

    log.info("=" * 70)
    log.info("CLUSTER DIAGNOSTIC REPORT")
    log.info("=" * 70)

    for cid_name in sorted(profiles.index):
        row = profiles.loc[cid_name]
        n_zones = (cluster_df["cluster_label"] == cid_name).sum()
        mean_conf = cluster_df[cluster_df["cluster_label"] == cid_name]["confidence_score"].mean()
        log.info("-- %s (%d zones, mean confidence=%.2f) --", cid_name, n_zones, mean_conf)
        for feat, desc in KEY_FEATURES:
            if feat not in row.index:
                continue
            val = row[feat]
            all_vals = profiles[feat].sort_values()
            rank_pos = list(all_vals.index).index(cid_name)
            n_cl = len(all_vals)
            if n_cl > 1:
                if rank_pos == n_cl - 1: rank_str = "HIGHEST"
                elif rank_pos == 0: rank_str = "LOWEST"
                elif rank_pos >= n_cl * 0.66: rank_str = "high"
                elif rank_pos <= n_cl * 0.33: rank_str = "low"
                else: rank_str = "mid"
            else:
                rank_str = ""
            log.info("  %-28s %8.4f  %-8s  %s", feat, val, rank_str, desc)

        top_zones = (cluster_df[cluster_df["cluster_label"] == cid_name]
                     .nlargest(5, "confidence_score")["zone_id"].tolist())
        log.info("  Top confidence zones: %s", top_zones)


def apply_user_labels(user_labels: dict[str, str], config) -> pd.DataFrame:
    """Rename Cluster_N -> human-readable names (equivalent to notebook Cell 27)."""
    cluster_df = pd.read_csv(config.clusters_csv)
    cluster_df["cluster_label"] = cluster_df["cluster_label"].map(
        lambda x: user_labels.get(x, x))
    cluster_df.to_csv(config.clusters_csv, index=False)

    if os.path.exists(config.metadata_csv):
        meta_df = pd.read_csv(config.metadata_csv)
        if "cluster_label" in meta_df.columns:
            meta_df["cluster_label"] = meta_df["cluster_label"].map(
                lambda x: user_labels.get(str(x), x) if pd.notna(x) else x)
            meta_df.to_csv(config.metadata_csv, index=False)
    return cluster_df


def run_clustering_stage(config) -> dict:
    """Orchestrates PCA -> KMeans/DBSCAN/GMM -> comparison -> cluster assignment."""
    norm_df = pd.read_csv(os.path.join(config.out_csv, "features_normalized.csv"))
    raw_df = pd.read_csv(config.features_csv)

    pca_result = run_pca(norm_df, config)
    X_pca = pca_result["X_pca"]

    km = run_kmeans(X_pca, config)
    db = run_dbscan(X_pca, config)
    gmm = run_gmm(X_pca, config)
    comparison = compare_algorithms(km, db, gmm, config)

    cluster_df = assign_clusters(comparison["best_labels"], norm_df, raw_df, config)

    return {
        "pca": pca_result, "kmeans": km, "dbscan": db, "gmm": gmm,
        "comparison": comparison, "cluster_df": cluster_df,
    }

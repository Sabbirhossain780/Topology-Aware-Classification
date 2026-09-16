"""Stage 3 — Topological feature engineering.

26 size-independent features per zone. Formulas are copied verbatim from
dhaka_topology_v3.ipynb (the v2-fixes version): corrected Gini coefficient,
orientation_entropy replacing saturated reciprocity, hub_dominance_ratio,
wcc_fragmentation, and size-normalized path length.
"""
from __future__ import annotations

import logging
import os
import random

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import list_zone_ids, load_zone_graph

log = logging.getLogger(__name__)

SIZE_ONLY_FEATURES = ["node_count", "edge_count", "path_length_p90"]


def get_lcc(G: nx.DiGraph) -> nx.DiGraph:
    """Return subgraph of the Largest Weakly Connected Component."""
    wccs = list(nx.weakly_connected_components(G))
    if len(wccs) == 1:
        return G
    return G.subgraph(max(wccs, key=len)).copy()


def gini_coefficient(values) -> float:
    """Standard Gini via pairwise absolute differences. Always in [0, 1]."""
    v = np.abs(np.array(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    v_sorted = np.sort(v)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v_sorted) - (n + 1) * v_sorted.sum()) /
                 (n * v_sorted.sum()))


def orientation_entropy(nodes_df: pd.DataFrame, links_df: pd.DataFrame, n_bins: int = 36) -> float:
    """Shannon entropy of street bearing angles, folded to [0, 180).

    Low entropy = planned grid (few dominant bearings).
    High entropy = organic maze (angles spread across the range).
    """
    pos = {r["node_id"]: (r["x_utm"], r["y_utm"])
           for _, r in nodes_df.iterrows() if "x_utm" in nodes_df.columns}
    if not pos:
        return 0.0

    bearings = []
    for _, r in links_df.iterrows():
        u, v = r["from_node"], r["to_node"]
        if u in pos and v in pos:
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            if dx == 0 and dy == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            if angle >= 180:
                angle -= 180
            bearings.append(angle)

    if len(bearings) < 3:
        return 0.0

    counts, _ = np.histogram(bearings, bins=n_bins // 2, range=(0, 180))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return round(float(entropy), 4)


def compute_features(G_multi: nx.MultiDiGraph, zone_id: str, config,
                      nodes_df: pd.DataFrame | None = None,
                      links_df: pd.DataFrame | None = None) -> dict:
    """Compute the 26 size-independent topological features for one zone."""
    feats = {"zone_id": zone_id}

    DG = nx.DiGraph()
    for u, v, data in G_multi.edges(data=True):
        length = data.get("length", 1.0)
        if DG.has_edge(u, v):
            if length > DG[u][v]["weight"]:
                DG[u][v]["weight"] = length
        else:
            DG.add_edge(u, v, weight=length)
    for node, d in G_multi.nodes(data=True):
        if node not in DG:
            DG.add_node(node)

    n = DG.number_of_nodes()
    m = DG.number_of_edges()
    UG = DG.to_undirected()

    feats["node_count"] = n
    feats["edge_count"] = m

    feats["density"] = round(m / max(n * (n - 1), 1), 6)
    feats["transitivity"] = round(nx.transitivity(UG), 4)
    feats["avg_clustering_coef"] = round(nx.average_clustering(UG), 4)
    feats["avg_degree"] = round(np.mean([DG.in_degree(v) + DG.out_degree(v)
                                          for v in DG.nodes()]), 4)

    sccs = list(nx.strongly_connected_components(DG))
    wccs = list(nx.weakly_connected_components(DG))
    feats["n_scc"] = len(sccs)
    feats["n_wcc"] = len(wccs)
    feats["gscc_pct"] = round(len(max(sccs, key=len)) / n * 100, 2) if n > 0 else 0.0
    feats["gwcc_pct"] = round(len(max(wccs, key=len)) / n * 100, 2) if n > 0 else 0.0
    feats["wcc_fragmentation"] = round(feats["n_wcc"] / max(n, 1), 4)

    if nodes_df is not None and links_df is not None:
        feats["orientation_entropy"] = orientation_entropy(
            nodes_df, links_df, n_bins=config.orientation_bins)
    else:
        feats["orientation_entropy"] = 0.0

    LCC = get_lcc(DG)
    lcc_nodes = list(LCC.nodes())
    lcc_n = len(lcc_nodes)

    if lcc_n >= 3:
        sample_nodes = random.sample(lcc_nodes, min(config.diameter_sample_n, lcc_n))
        path_lengths = [
            l for src in sample_nodes
            for l in nx.single_source_shortest_path_length(LCC, src).values()
            if l > 0
        ]
        raw_p90 = float(np.percentile(path_lengths, 90)) if path_lengths else 0.0
    else:
        raw_p90 = 0.0
    feats["path_length_p90"] = round(raw_p90, 4)
    feats["path_length_p90_norm"] = round(raw_p90 / max(np.sqrt(n), 1), 4)

    zero_centrality = [
        "mean_in_degree_centrality", "mean_out_degree_centrality",
        "mean_closeness_centrality", "mean_betweenness_centrality",
        "betweenness_gini", "mean_katz_centrality", "mean_pagerank",
        "mean_local_clustering", "mean_strength_norm",
        "global_efficiency", "local_efficiency", "hub_dominance_ratio",
    ]
    if lcc_n < 3:
        for key in zero_centrality:
            feats[key] = 0.0
        return feats

    in_deg = nx.in_degree_centrality(LCC)
    out_deg = nx.out_degree_centrality(LCC)
    feats["mean_in_degree_centrality"] = round(np.mean(list(in_deg.values())), 6)
    feats["mean_out_degree_centrality"] = round(np.mean(list(out_deg.values())), 6)

    raw_in_degrees = np.array([d for _, d in LCC.in_degree()])
    mean_in = raw_in_degrees.mean()
    feats["hub_dominance_ratio"] = round(
        float(raw_in_degrees.max() / max(mean_in, 0.001)), 4)

    feats["mean_closeness_centrality"] = round(
        np.mean(list(nx.closeness_centrality(LCC).values())), 6)

    bw = nx.betweenness_centrality(LCC, k=min(config.betweenness_k, lcc_n), normalized=True)
    bw_v = np.array(list(bw.values()))
    feats["mean_betweenness_centrality"] = round(float(np.mean(bw_v)), 6)
    feats["betweenness_gini"] = round(gini_coefficient(bw_v), 4)

    try:
        katz = nx.katz_centrality(LCC, alpha=config.katz_alpha, max_iter=1000, normalized=True)
        feats["mean_katz_centrality"] = round(np.mean(list(katz.values())), 6)
    except Exception:
        feats["mean_katz_centrality"] = feats["mean_in_degree_centrality"]

    try:
        pr = nx.pagerank(LCC, alpha=config.pagerank_alpha, max_iter=200)
        feats["mean_pagerank"] = round(np.mean(list(pr.values())), 6)
    except Exception:
        feats["mean_pagerank"] = 1.0 / lcc_n

    LCC_U = LCC.to_undirected()
    feats["mean_local_clustering"] = round(np.mean(list(nx.clustering(LCC_U).values())), 4)

    edge_weights = [d.get("weight", 1.0) for _, _, d in LCC.edges(data=True)]
    mean_w = np.mean(edge_weights) if edge_weights else 1.0
    strengths = [sum(d.get("weight", 1.0) for _, _, d in LCC.edges(v, data=True))
                 for v in LCC.nodes()]
    raw_mean_strength = np.mean(strengths) if strengths else 0.0
    feats["mean_strength_norm"] = round(raw_mean_strength / max(mean_w * lcc_n, 1), 4)

    try:
        feats["global_efficiency"] = round(nx.global_efficiency(LCC_U), 6)
    except Exception:
        feats["global_efficiency"] = 0.0
    try:
        feats["local_efficiency"] = round(nx.local_efficiency(LCC_U), 6)
    except Exception:
        feats["local_efficiency"] = 0.0

    return feats


def run_feature_engineering(config) -> pd.DataFrame:
    """Compute features for every zone, save features.csv, return the DataFrame."""
    random.seed(config.random_state)
    np.random.seed(config.random_state)

    zone_ids = list_zone_ids(config.data_zones)
    log.info("Computing features for %d zones", len(zone_ids))

    all_features, failed_zones = [], []
    for zone_id in tqdm(zone_ids, desc="Feature engineering"):
        try:
            subG, nodes_df, links_df = load_zone_graph(config.data_zones, zone_id)
            feats = compute_features(subG, zone_id, config, nodes_df, links_df)
            all_features.append(feats)
        except Exception as e:
            failed_zones.append({"zone_id": zone_id, "error": str(e)})

    features_df = pd.DataFrame(all_features)
    os.makedirs(config.out_csv, exist_ok=True)
    features_df.to_csv(config.features_csv, index=False)

    if failed_zones:
        pd.DataFrame(failed_zones).to_csv(
            os.path.join(config.out_csv, "failed_zones.csv"), index=False)
        log.warning("Failed zones: %d", len(failed_zones))

    log.info("Feature engineering complete: %d ok, %d failed", len(all_features), len(failed_zones))
    return features_df


def normalize_features(features_df: pd.DataFrame, config) -> pd.DataFrame:
    """Min-max normalize clustering features, excluding raw size columns."""
    all_feat_cols = [c for c in features_df.columns if c != "zone_id"]
    cluster_feat_cols = [c for c in all_feat_cols if c not in SIZE_ONLY_FEATURES]

    norm_df = features_df[["zone_id"] + cluster_feat_cols].copy()
    for col in cluster_feat_cols:
        mn, mx = features_df[col].min(), features_df[col].max()
        norm_df[col] = (features_df[col] - mn) / (mx - mn) if mx > mn else 0.0

    os.makedirs(config.out_csv, exist_ok=True)
    norm_path = os.path.join(config.out_csv, "features_normalized.csv")
    norm_df.to_csv(norm_path, index=False)
    return norm_df


def bootstrap_stability(features_df: pd.DataFrame, config) -> pd.DataFrame | None:
    """Resample 80% of nodes per zone BOOTSTRAP_N times; report feature CV."""
    random.seed(config.random_state)
    cluster_feat_cols = [c for c in features_df.columns
                          if c not in ["zone_id"] + SIZE_ONLY_FEATURES]
    sample_zones = random.sample(list(features_df["zone_id"]),
                                  min(30, len(features_df)))
    bootstrap_records = []

    for zone_id in tqdm(sample_zones, desc="Bootstrap stability"):
        try:
            subG, ndf, ldf = load_zone_graph(config.data_zones, zone_id)
            nodes = list(subG.nodes())
            for _ in range(config.bootstrap_n):
                sampled = random.choices(nodes, k=max(config.min_nodes, int(0.8 * len(nodes))))
                sub = subG.subgraph(set(sampled)).copy()
                try:
                    r = compute_features(sub, "bs", config, ndf, ldf)
                    bootstrap_records.append({k: v for k, v in r.items() if k != "zone_id"})
                except Exception:
                    pass
        except Exception:
            pass

    if not bootstrap_records:
        return None

    bs_df = pd.DataFrame(bootstrap_records)
    cv_summary = pd.DataFrame({
        "feature": cluster_feat_cols,
        "mean_cv": [
            round(bs_df[c].std() / max(abs(bs_df[c].mean()), 1e-9), 4)
            if c in bs_df.columns else 0.0
            for c in cluster_feat_cols
        ],
    })
    os.makedirs(config.out_csv, exist_ok=True)
    cv_summary.to_csv(os.path.join(config.out_csv, "feature_stability.csv"), index=False)
    return cv_summary

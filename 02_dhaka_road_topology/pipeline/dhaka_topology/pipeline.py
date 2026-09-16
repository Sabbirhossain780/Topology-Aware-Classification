"""Orchestrates the full pipeline: acquisition -> tiling -> features ->
clustering -> betweenness (GAT) -> UFFM -> visualizations.

Each stage can be run independently (skip=True re-loads existing CSVs
instead of recomputing), mirroring how the notebooks let you re-run a
single cell using cached files from disk.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from . import acquisition, betweenness, clustering, features, tiling, uffm, visualization
from .config import Config

log = logging.getLogger(__name__)

STAGES = ["acquire", "tile", "features", "cluster", "betweenness", "uffm", "plots"]


def run_stage(stage: str, config: Config, force: bool = False) -> dict:
    config.ensure_dirs()
    result: dict = {}

    if stage == "acquire":
        G = acquisition.acquire_graph(config, force_redownload=force)
        result["graph"] = G

    elif stage == "tile":
        import osmnx as ox
        G = ox.load_graphml(config.graph_cache)
        metadata_df = tiling.run_tiling_stage(G, config)
        result["metadata_df"] = metadata_df

    elif stage == "features":
        features_df = features.run_feature_engineering(config)
        norm_df = features.normalize_features(features_df, config)
        features.bootstrap_stability(features_df, config)
        result["features_df"] = features_df
        result["norm_df"] = norm_df

    elif stage == "cluster":
        result = clustering.run_clustering_stage(config)

    elif stage == "betweenness":
        if config.gat_enabled:
            result = betweenness.run_betweenness_stage(config)

    elif stage == "uffm":
        if config.uffm_enabled:
            result = uffm.run_uffm_stage(config)
            result["crosstab"] = uffm.crosstab_vs_v3(config)

    elif stage == "plots":
        result = _run_plots(config)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return result


def _run_plots(config: Config) -> dict:
    import geopandas as gpd
    out = {}

    if os.path.exists(config.grid_cache) and os.path.exists(config.metadata_csv):
        try:
            import osmnx as ox
            G = ox.load_graphml(config.graph_cache)
            valid_grid = gpd.read_file(config.grid_cache)
            metadata_df = pd.read_csv(config.metadata_csv)
            out["road_network"] = visualization.plot_road_network(G, config)
            out["grid_tiling"] = visualization.plot_grid_tiling(G, valid_grid, metadata_df, config)
        except Exception as e:
            log.warning("Network/grid plots skipped: %s", e)

    if os.path.exists(config.features_csv):
        features_df = pd.read_csv(config.features_csv)
        out["feature_distributions"] = visualization.plot_feature_distributions(features_df, config)

        if os.path.exists(config.clusters_csv):
            cluster_df = pd.read_csv(config.clusters_csv)
            out["feature_heatmap"] = visualization.plot_feature_heatmap(features_df, cluster_df, config)
            out["cluster_choropleth"] = visualization.plot_cluster_choropleth(config)
            out["cluster_sample_subnetworks"] = visualization.plot_cluster_sample_subnetworks(config)
            try:
                out["interactive_map"] = visualization.plot_interactive_map(config)
            except ImportError:
                log.warning("folium not installed - skipping interactive map")

    gat_summary_path = os.path.join(config.gat_csv, "all_zones_summary.csv")
    if os.path.exists(gat_summary_path):
        summary_df = pd.read_csv(gat_summary_path)
        out["city_distributions"] = visualization.plot_city_distributions(summary_df, config)
        out["city_master_nodes"] = visualization.plot_city_master_nodes(summary_df, config)

    return out


def run_full_pipeline(config: Config, stages: list[str] | None = None, force: bool = False) -> dict:
    stages = stages or STAGES
    results = {}
    for stage in stages:
        log.info("=" * 60)
        log.info("STAGE: %s", stage)
        log.info("=" * 60)
        results[stage] = run_stage(stage, config, force=force)
    return results

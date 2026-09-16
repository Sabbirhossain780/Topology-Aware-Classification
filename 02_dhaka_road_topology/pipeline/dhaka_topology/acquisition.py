"""Stage 1 — OSM data acquisition.

Downloads Dhaka's road network from OpenStreetMap, projects to UTM,
validates connectivity, and caches to GraphML. Identical logic to
dhaka_topology_v3.ipynb Stage 1, extracted into a reusable function.
"""
from __future__ import annotations

import inspect
import logging
import os
import time

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)


def _download_osm_graph(bbox: dict, network_type: str = "all"):
    """Download OSM graph — auto-detects osmnx v1.x vs v2.x API."""
    import osmnx as ox

    params = list(inspect.signature(ox.graph_from_bbox).parameters.keys())
    log.info("osmnx %s | network_type=%s", ox.__version__, network_type)
    if params[0] == "bbox":  # v2.x
        return ox.graph_from_bbox(
            bbox=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
            network_type=network_type,
        )
    else:  # v1.x
        return ox.graph_from_bbox(
            bbox["north"], bbox["south"], bbox["east"], bbox["west"],
            network_type=network_type, retain_all=True, simplify=True,
        )


def acquire_graph(config, force_redownload: bool = False) -> nx.MultiDiGraph:
    """Load the cached graph, or download+cache it if missing/forced.

    Also runs the same validation checks and writes network_stats.csv.
    """
    import osmnx as ox

    graph_cache = config.graph_cache

    if os.path.exists(graph_cache) and not force_redownload:
        log.info("Cache found - loading from %s", graph_cache)
        t0 = time.time()
        G = ox.load_graphml(graph_cache)
        log.info("Loaded in %.1fs", time.time() - t0)
    else:
        log.info("Downloading OSM (network_type=%s)", config.network_type)
        t0 = time.time()
        G = _download_osm_graph(config.bbox, network_type=config.network_type)
        log.info("Download complete in %.1fs", time.time() - t0)

        G.remove_edges_from(nx.selfloop_edges(G))
        G = ox.project_graph(G, to_crs=config.crs_metric)
        os.makedirs(os.path.dirname(graph_cache), exist_ok=True)
        ox.save_graphml(G, graph_cache)
        log.info("Saved -> %s", graph_cache)

    _validate_graph(G, config)
    return G


def _validate_graph(G: nx.MultiDiGraph, config) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    wccs = list(nx.weakly_connected_components(G))
    sccs = list(nx.strongly_connected_components(G))
    degrees = [d for _, d in G.degree()]

    stats = {
        "Total Nodes": f"{n_nodes:,}",
        "Total Edges": f"{n_edges:,}",
        "Weakly Connected Components": len(wccs),
        "Largest WCC (%)": f"{len(max(wccs, key=len)) / n_nodes * 100:.1f}%",
        "Strongly Connected Components": len(sccs),
        "Largest SCC (%)": f"{len(max(sccs, key=len)) / n_nodes * 100:.1f}%",
        "Avg Degree": f"{sum(degrees) / len(degrees):.2f}",
        "Edges per Node": f"{n_edges / n_nodes:.2f}",
    }
    log.info("Network validation: %s", stats)

    assert n_nodes > 10_000, f"Only {n_nodes} nodes - check bbox or OSM connection"
    assert len(max(wccs, key=len)) / n_nodes > 0.80, "Largest WCC < 80% - fragmented download"

    os.makedirs(config.data_raw, exist_ok=True)
    pd.DataFrame([stats]).to_csv(os.path.join(config.data_raw, "network_stats.csv"), index=False)
    return stats

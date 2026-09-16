"""Betweenness-hierarchy stage (formerly gat_pipeline.ipynb).

For every zone: build an undirected weighted graph, compute exact
(Brandes) betweenness centrality, classify nodes into 4 tiers, flag
articulation points and the "master" (max-BC) node.
"""
from __future__ import annotations

import logging
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_utils import list_zone_ids

log = logging.getLogger(__name__)


def utm46n_to_wgs84(easting, northing):
    """UTM Zone 46N (EPSG:32646) -> WGS84. Accurate to <1m for Dhaka."""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2
    k0 = 0.9996
    E0 = 500000.0
    lon0 = np.radians(93.0)
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    x1 = np.array(easting, dtype=float) - E0
    M = np.array(northing, dtype=float) / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu))
    N1 = a / np.sqrt(1 - e2 * np.sin(phi1) ** 2)
    T1v = np.tan(phi1) ** 2
    C1 = e2 / (1 - e2) * np.cos(phi1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * np.sin(phi1) ** 2) ** 1.5
    D = x1 / (N1 * k0)
    lat_r = phi1 - (N1 * np.tan(phi1) / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1v + 10 * C1 - 4 * C1 ** 2 - 9 * e2 / (1 - e2)) * D ** 4 / 24
        + (61 + 90 * T1v + 298 * C1 + 45 * T1v ** 2 - 252 * e2 / (1 - e2) - 3 * C1 ** 2) * D ** 6 / 720)
    lon_r = lon0 + ((D
                     - (1 + 2 * T1v + C1) * D ** 3 / 6
                     + (5 - 2 * C1 + 28 * T1v - 3 * C1 ** 2 + 8 * e2 / (1 - e2) + 24 * T1v ** 2) * D ** 5 / 120)
                    / np.cos(phi1))
    return np.degrees(lon_r), np.degrees(lat_r)


def build_graph(nodes_df: pd.DataFrame, links_df: pd.DataFrame) -> nx.Graph:
    """Build undirected graph from zone CSVs. Returns largest connected component."""
    G = nx.Graph()
    for _, r in nodes_df.iterrows():
        G.add_node(int(r["node_id"]), lon=float(r["lon_wgs"]), lat=float(r["lat_wgs"]))
    for _, r in links_df.iterrows():
        u, v = int(r["from_node"]), int(r["to_node"])
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, weight=float(r["length_m"]), highway=str(r["highway_type"]))
    if G.number_of_nodes() == 0:
        return G
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def compute_bc_and_tiers(G: nx.Graph):
    """Exact betweenness (Brandes, weighted). Returns (bc_dict, tier_dict)."""
    bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    arr = np.array(list(bc.values()))
    t1c = np.percentile(arr, 99)
    t2c = np.percentile(arr, 95)
    t3c = np.percentile(arr, 75)
    tier = {n: (1 if v >= t1c else 2 if v >= t2c else 3 if v >= t3c else 4)
            for n, v in bc.items()}
    return bc, tier


def load_zone_for_bc(data_zones_dir: str, zone_id: str):
    """Load a zone, converting UTM to WGS84 lon/lat if needed."""
    zone_dir = os.path.join(data_zones_dir, zone_id)
    nodes_df = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links_df = pd.read_csv(os.path.join(zone_dir, "links.csv"))

    if nodes_df["lon"].mean() > 1000:  # UTM stored in the lon column
        lons, lats = utm46n_to_wgs84(nodes_df["x_utm"].values, nodes_df["y_utm"].values)
    else:
        lons = nodes_df["lon"].values
        lats = nodes_df["lat"].values

    nodes_df = nodes_df.copy()
    nodes_df["lon_wgs"] = np.round(lons, 7)
    nodes_df["lat_wgs"] = np.round(lats, 7)

    G = build_graph(nodes_df, links_df)
    return G, nodes_df, links_df


def run_betweenness_stage(config) -> dict:
    """Process every zone: betweenness, tiers, articulation points, master node.

    Writes gat/csv/Zone_*/nodes_bc.csv, gat/csv/all_zones_summary.csv,
    gat/csv/all_nodes_bc.csv. Plotting is left to visualization.py.
    """
    zone_ids = list_zone_ids(config.data_zones)
    if config.gat_max_zones:
        zone_ids = zone_ids[: config.gat_max_zones]

    os.makedirs(config.gat_csv, exist_ok=True)
    os.makedirs(config.gat_plots, exist_ok=True)

    summary_rows, all_node_rows, failed = [], [], []
    t_global = time.time()

    for zone_id in tqdm(zone_ids, desc="Betweenness"):
        t0 = time.time()
        try:
            G, nodes_df, links_df = load_zone_for_bc(config.data_zones, zone_id)

            if G.number_of_nodes() < 10:
                failed.append((zone_id, f"LCC only {G.number_of_nodes()} nodes"))
                continue

            n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
            bc, tier = compute_bc_and_tiers(G)
            bc_arr = np.array(list(bc.values()))

            master = max(bc, key=bc.get)
            master_bc = bc[master]
            art_pts = list(nx.articulation_points(G))
            art_set = set(art_pts)

            c_lon = nodes_df["lon_wgs"].mean()
            c_lat = nodes_df["lat_wgs"].mean()

            zone_out = os.path.join(config.gat_csv, zone_id)
            os.makedirs(zone_out, exist_ok=True)

            node_rows = [{
                "zone_id": zone_id, "node_id": nid,
                "lon": G.nodes[nid]["lon"], "lat": G.nodes[nid]["lat"],
                "bc": round(bc[nid], 8), "tier": tier[nid],
                "degree": G.degree(nid),
                "is_master": int(nid == master),
                "is_art_pt": int(nid in art_set),
            } for nid in G.nodes()]
            pd.DataFrame(node_rows).to_csv(os.path.join(zone_out, "nodes_bc.csv"), index=False)
            all_node_rows.extend(node_rows)

            summary_rows.append({
                "zone_id": zone_id,
                "centre_lat": round(c_lat, 5), "centre_lon": round(c_lon, 5),
                "n_nodes": n_nodes, "n_edges": n_edges,
                "avg_degree": round(2 * n_edges / n_nodes, 3),
                "max_bc": round(master_bc, 6), "mean_bc": round(bc_arr.mean(), 6),
                "p95_bc": round(np.percentile(bc_arr, 95), 6),
                "p99_bc": round(np.percentile(bc_arr, 99), 6),
                "tier1_count": sum(1 for v in tier.values() if v == 1),
                "tier2_count": sum(1 for v in tier.values() if v == 2),
                "art_point_count": len(art_pts),
                "master_node_id": master,
                "master_lon": round(G.nodes[master]["lon"], 6),
                "master_lat": round(G.nodes[master]["lat"], 6),
                "master_degree": G.degree(master),
                "compute_time_s": round(time.time() - t0, 2),
            })

            if config.gat_do_plots:
                from .visualization import plot_zone_hierarchy
                plot_path = os.path.join(config.gat_plots, f"{zone_id}_hierarchy.png")
                plot_zone_hierarchy(zone_id, G, bc, tier, plot_path, dpi=config.gat_plot_dpi)

        except Exception as ex:
            failed.append((zone_id, str(ex)))
            log.error("Zone %s failed: %s", zone_id, ex)

    summary_df = pd.DataFrame(summary_rows).sort_values("max_bc", ascending=False)
    summary_df.to_csv(os.path.join(config.gat_csv, "all_zones_summary.csv"), index=False)

    all_nodes_df = pd.DataFrame(all_node_rows)
    all_nodes_df.to_csv(os.path.join(config.gat_csv, "all_nodes_bc.csv"), index=False)

    log.info("Betweenness stage done: %d zones in %.1f min, %d failed",
              len(summary_df), (time.time() - t_global) / 60, len(failed))
    return {"summary_df": summary_df, "all_nodes_df": all_nodes_df, "failed": failed}

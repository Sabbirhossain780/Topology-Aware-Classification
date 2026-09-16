"""Shared I/O helpers used across pipeline stages."""
from __future__ import annotations

import os
from typing import Iterable, Tuple

import networkx as nx
import pandas as pd


def list_zone_ids(data_zones_dir: str) -> list[str]:
    """Sorted list of Zone_* subfolder names under data/zones."""
    return sorted(
        d for d in os.listdir(data_zones_dir)
        if os.path.isdir(os.path.join(data_zones_dir, d)) and d.startswith("Zone_")
    )


def load_zone_graph(data_zones_dir: str, zone_id: str) -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame]:
    """Reconstruct a MultiDiGraph from a zone's nodes.csv/links.csv.

    Matches dhaka_topology_v3.ipynb's load_zone_graph exactly (directed,
    multigraph, used for feature engineering).
    """
    zone_dir = os.path.join(data_zones_dir, zone_id)
    nodes_df = pd.read_csv(os.path.join(zone_dir, "nodes.csv"))
    links_df = pd.read_csv(os.path.join(zone_dir, "links.csv"))

    G = nx.MultiDiGraph()
    for _, r in nodes_df.iterrows():
        G.add_node(r["node_id"], x=r.get("x_utm", 0), y=r.get("y_utm", 0))
    for _, r in links_df.iterrows():
        G.add_edge(
            r["from_node"], r["to_node"],
            length=r.get("length_m", 1.0),
            highway=r.get("highway_type", "unknown"),
        )
    return G, nodes_df, links_df


def zone_csv_paths(data_zones_dir: str, zone_id: str) -> Tuple[str, str]:
    zone_dir = os.path.join(data_zones_dir, zone_id)
    return os.path.join(zone_dir, "nodes.csv"), os.path.join(zone_dir, "links.csv")

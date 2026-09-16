"""Unit tests for the betweenness/tier classification logic."""
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhaka_topology.betweenness import compute_bc_and_tiers


def test_star_graph_center_has_max_betweenness():
    # Star: center node 0 connects to 1..5. All shortest paths pass through it.
    G = nx.star_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    bc, tier = compute_bc_and_tiers(G)
    assert max(bc, key=bc.get) == 0
    assert tier[0] == 1  # center should land in the top tier


def test_tiers_partition_all_nodes():
    G = nx.path_graph(10)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    bc, tier = compute_bc_and_tiers(G)
    assert set(tier.keys()) == set(G.nodes())
    assert set(tier.values()) <= {1, 2, 3, 4}

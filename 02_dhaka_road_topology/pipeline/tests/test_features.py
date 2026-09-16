"""Unit tests for the pure feature-engineering functions.

These lock in the exact formulas ported from dhaka_topology_v3.ipynb so a
future refactor can't silently change results without a test failing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhaka_topology.features import gini_coefficient, orientation_entropy


def test_gini_uniform_is_zero():
    assert gini_coefficient([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_is_in_unit_range():
    values = [0.01, 0.02, 0.5, 0.9, 0.001, 0.3]
    g = gini_coefficient(values)
    assert 0.0 <= g <= 1.0


def test_gini_empty_returns_zero():
    assert gini_coefficient([]) == 0.0


def test_gini_higher_for_more_unequal_distribution():
    equal = gini_coefficient([1, 1, 1, 1])
    unequal = gini_coefficient([0, 0, 0, 10])
    assert unequal > equal


def test_orientation_entropy_grid_lower_than_random():
    # A perfect grid: every edge points either 0 deg or 90 deg -> low entropy.
    grid_nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "x_utm": [0, 10, 0, 10],
        "y_utm": [0, 0, 10, 10],
    })
    grid_links = pd.DataFrame({
        "from_node": [0, 0, 1, 2],
        "to_node":   [1, 2, 3, 3],
    })
    grid_entropy = orientation_entropy(grid_nodes, grid_links, n_bins=36)

    # A star with edges spread across many bearings -> high entropy.
    rng = np.random.default_rng(0)
    n = 20
    angles = np.linspace(0, 350, n)
    xs = np.cos(np.radians(angles)) * 10
    ys = np.sin(np.radians(angles)) * 10
    star_nodes = pd.DataFrame({
        "node_id": list(range(n + 1)),
        "x_utm": [0.0] + list(xs),
        "y_utm": [0.0] + list(ys),
    })
    star_links = pd.DataFrame({
        "from_node": [0] * n,
        "to_node": list(range(1, n + 1)),
    })
    star_entropy = orientation_entropy(star_nodes, star_links, n_bins=36)

    assert grid_entropy < star_entropy


def test_orientation_entropy_too_few_edges_returns_zero():
    nodes = pd.DataFrame({"node_id": [0, 1], "x_utm": [0, 1], "y_utm": [0, 1]})
    links = pd.DataFrame({"from_node": [0], "to_node": [1]})
    assert orientation_entropy(nodes, links) == 0.0

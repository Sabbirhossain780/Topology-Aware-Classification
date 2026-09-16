"""Unit tests for UFFM fingerprint functions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhaka_topology.uffm import bearing_fingerprint, length_fingerprint, _safe_hist


def test_safe_hist_sums_to_one():
    h = _safe_hist([1, 2, 3, 4, 5, 6, 7, 8], n_bins=4, val_range=(0, 10))
    assert h.sum() == pytest.approx(1.0)


def test_safe_hist_uniform_fallback_for_sparse_input():
    h = _safe_hist([1, 2], n_bins=5, val_range=(0, 10))
    assert h == pytest.approx(np.ones(5) / 5)


def test_bearing_fingerprint_sums_to_one():
    nodes = pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "x_utm": [0, 10, 0, 10],
        "y_utm": [0, 0, 10, 10],
    })
    links = pd.DataFrame({
        "from_node": [0, 0, 1, 2],
        "to_node": [1, 2, 3, 3],
    })
    fp = bearing_fingerprint(nodes, links, n_bins=36)
    assert fp.sum() == pytest.approx(1.0)
    assert len(fp) == 36


def test_length_fingerprint_caps_at_max_len():
    links = pd.DataFrame({"length_m": [10, 50, 600, 1000]})
    fp = length_fingerprint(links, n_bins=10, max_len=500)
    assert fp.sum() == pytest.approx(1.0)
    # The two long segments should land in the final bin (clipped to 500).
    assert fp[-1] > 0

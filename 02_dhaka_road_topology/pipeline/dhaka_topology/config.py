"""Central configuration for the topology pipeline.

Every notebook variant (v1, v2, v3, v4_2km, thana) differed only in a handful
of these values — grid cell size, bbox, zone unit. Here that becomes one
dataclass loaded from a small YAML file instead of a forked notebook.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    name: str = "v3"

    # ── Study area ───────────────────────────────────────────────────────
    bbox_north: float = 23.95
    bbox_south: float = 23.65
    bbox_east: float = 90.48
    bbox_west: float = 90.25
    network_type: str = "all"          # 'all' = footpaths+lanes+service roads

    # ── Coordinate systems ───────────────────────────────────────────────
    crs_metric: str = "EPSG:32646"     # UTM 46N
    crs_geo: str = "EPSG:4326"

    # ── Zone unit ────────────────────────────────────────────────────────
    # 'grid'  -> square-cell tiling (v1/v2/v3/v4_2km)
    # 'thana' -> administrative-boundary tiling (thana variant)
    zone_unit: str = "grid"
    cell_size_m: float = 1000.0
    buffer_m: float = 800.0
    min_nodes: int = 15
    thana_boundary_path: Optional[str] = None  # required when zone_unit == 'thana'

    # ── Feature engineering ──────────────────────────────────────────────
    betweenness_k: int = 50
    pagerank_alpha: float = 0.85
    katz_alpha: float = 0.1
    diameter_sample_n: int = 300
    bootstrap_n: int = 20
    cv_threshold: float = 0.35
    orientation_bins: int = 36

    # ── Clustering ───────────────────────────────────────────────────────
    k_min: int = 2
    k_max: int = 8                     # inclusive; mirrors range(2, 9)
    pca_variance: float = 0.95
    random_state: int = 42
    dbscan_min_samples: int = 5

    # ── UFFM (Urban Form Fingerprint Matching) ──────────────────────────
    uffm_enabled: bool = True
    bearing_bins: int = 36
    angle_bins: int = 36
    length_bins: int = 40
    length_max_m: float = 500.0
    w_bearing: float = 0.40
    w_angle: float = 0.35
    w_length: float = 0.25

    # ── GAT / betweenness-hierarchy stage ───────────────────────────────
    gat_enabled: bool = True
    gat_plot_dpi: int = 150
    gat_do_plots: bool = True
    gat_max_zones: Optional[int] = None  # cap for a quick test run

    # ── Directories (relative to base_dir unless absolute) ──────────────
    base_dir: str = "."

    CLUSTER_PALETTE = [
        "#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
        "#9B59B6", "#1ABC9C", "#E67E22", "#2C3E50",
    ]

    # -- derived path helpers -------------------------------------------
    @property
    def bbox(self) -> dict:
        return dict(north=self.bbox_north, south=self.bbox_south,
                     east=self.bbox_east, west=self.bbox_west)

    @property
    def k_range(self):
        return range(self.k_min, self.k_max + 1)

    def _p(self, *parts) -> str:
        return os.path.join(self.base_dir, *parts)

    @property
    def data_raw(self) -> str: return self._p("data", "raw")
    @property
    def data_processed(self) -> str: return self._p("data", "processed")
    @property
    def data_zones(self) -> str: return self._p("data", "zones")

    @property
    def out_csv(self) -> str: return self._p("outputs", "csv")
    @property
    def out_maps(self) -> str: return self._p("outputs", "maps")
    @property
    def out_plots(self) -> str: return self._p("outputs", "plots")

    @property
    def gat_dir(self) -> str: return self._p("gat")
    @property
    def gat_csv(self) -> str: return self._p("gat", "csv")
    @property
    def gat_plots(self) -> str: return self._p("gat", "plots")

    @property
    def graph_cache(self) -> str: return os.path.join(self.data_raw, "dhaka_graph.graphml")
    @property
    def grid_cache(self) -> str: return os.path.join(self.data_processed, "grid_zones.gpkg")
    @property
    def features_csv(self) -> str: return os.path.join(self.out_csv, "features.csv")
    @property
    def metadata_csv(self) -> str: return os.path.join(self.out_csv, "metadata.csv")
    @property
    def clusters_csv(self) -> str: return os.path.join(self.out_csv, "cluster_assignments.csv")

    def ensure_dirs(self) -> None:
        for d in (self.data_raw, self.data_processed, self.data_zones,
                  self.out_csv, self.out_maps, self.out_plots):
            os.makedirs(d, exist_ok=True)
        if self.gat_enabled:
            os.makedirs(self.gat_csv, exist_ok=True)
            os.makedirs(self.gat_plots, exist_ok=True)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**known)

    def to_yaml(self, path: str) -> None:
        d = asdict(self)
        d.pop("CLUSTER_PALETTE", None)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)

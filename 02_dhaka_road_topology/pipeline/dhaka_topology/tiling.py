"""Stage 2 — Buffered grid tiling & subgraph extraction.

Same logic as dhaka_topology_v3.ipynb Stage 2: lay a square grid over the
metric-projected graph, buffer each cell to avoid boundary dead-ends,
extract per-zone subgraphs, and write nodes.csv/links.csv per zone.

`zone_unit='thana'` swaps the grid for arbitrary polygons read from a
boundary file (dhaka_topology_thana.ipynb's approach) — same extraction
logic, different polygon source.
"""
from __future__ import annotations

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box, Point
from tqdm import tqdm

log = logging.getLogger(__name__)

EXPECTED_NODE_DENSITY = 40.0  # intersections/km2 - OSM completeness baseline


def build_grid(G, config) -> gpd.GeoDataFrame:
    """Lay a cell_size_m x cell_size_m grid over the graph's bounding box."""
    import osmnx as ox

    nodes_gdf, _ = ox.graph_to_gdfs(G)
    nodes_gdf = nodes_gdf.to_crs(config.crs_metric)
    xmin, ymin, xmax, ymax = nodes_gdf.total_bounds

    xmin -= config.cell_size_m / 2
    ymin -= config.cell_size_m / 2

    xs = np.arange(xmin, xmax + config.cell_size_m, config.cell_size_m)
    ys = np.arange(ymin, ymax + config.cell_size_m, config.cell_size_m)

    cells = []
    idx = 0
    for i, x0 in enumerate(xs[:-1]):
        for j, y0 in enumerate(ys[:-1]):
            core = box(x0, y0, x0 + config.cell_size_m, y0 + config.cell_size_m)
            buffered = core.buffer(config.buffer_m)
            cells.append({
                "zone_id": f"Zone_DH{idx:03d}",
                "col_idx": i, "row_idx": j,
                "geometry": core,
                "geometry_buf": buffered,
                "centroid_x": core.centroid.x,
                "centroid_y": core.centroid.y,
                "area_km2": core.area / 1e6,
            })
            idx += 1

    return gpd.GeoDataFrame(cells, geometry="geometry", crs=config.crs_metric)


def build_thana_zones(config) -> gpd.GeoDataFrame:
    """Load administrative (thana) boundary polygons as the zone grid.

    Requires config.thana_boundary_path pointing to a vector file readable
    by geopandas, with a name/id column that becomes zone_id.
    """
    if not config.thana_boundary_path:
        raise ValueError("zone_unit='thana' requires thana_boundary_path in config")
    gdf = gpd.read_file(config.thana_boundary_path).to_crs(config.crs_metric)

    id_col = next((c for c in gdf.columns if c.lower() in
                   ("name", "thana", "thana_name", "zone_id")), gdf.columns[0])

    out = gdf.rename(columns={id_col: "zone_id"}).copy()
    out["zone_id"] = [f"Zone_TH{i:03d}_{str(v)[:20]}" for i, v in enumerate(out["zone_id"])]
    out["geometry_buf"] = out.geometry.buffer(config.buffer_m)
    out["centroid_x"] = out.geometry.centroid.x
    out["centroid_y"] = out.geometry.centroid.y
    out["area_km2"] = out.geometry.area / 1e6
    out["col_idx"] = 0
    out["row_idx"] = range(len(out))
    return out


def extract_zones(G, grid_gdf: gpd.GeoDataFrame, config) -> pd.DataFrame:
    """Extract per-zone subgraphs, write nodes.csv/links.csv, return metadata."""
    transformer = Transformer.from_crs(config.crs_metric, config.crs_geo, always_xy=True)

    metadata_rows = []
    valid_zone_ids = []

    for _, row in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="Extracting subgraphs"):
        zone_id = row["zone_id"]
        buf_poly = row["geometry_buf"]

        node_ids = [n for n, d in G.nodes(data=True)
                    if buf_poly.contains(Point(d["x"], d["y"]))]

        if len(node_ids) < config.min_nodes:
            continue

        subG = G.subgraph(node_ids).copy()

        zone_dir = os.path.join(config.data_zones, zone_id)
        os.makedirs(zone_dir, exist_ok=True)

        node_rows = [{
            "node_id": n,
            "x_utm": round(d.get("x", 0), 3),
            "y_utm": round(d.get("y", 0), 3),
            "lon": round(d.get("lon", d.get("x", 0)), 6),
            "lat": round(d.get("lat", d.get("y", 0)), 6),
        } for n, d in subG.nodes(data=True)]
        pd.DataFrame(node_rows).to_csv(os.path.join(zone_dir, "nodes.csv"), index=False)

        link_rows = [{
            "link_id": li,
            "from_node": u, "to_node": v,
            "length_m": round(d.get("length", 1.0), 2),
            "length_km": round(d.get("length", 1.0) / 1000, 4),
            "oneway": d.get("oneway", False),
            "highway_type": str(d.get("highway", "unknown")),
        } for li, (u, v, d) in enumerate(subG.edges(data=True))]
        pd.DataFrame(link_rows).to_csv(os.path.join(zone_dir, "links.csv"), index=False)

        lon, lat = transformer.transform(row["centroid_x"], row["centroid_y"])
        cell_km2 = (config.cell_size_m / 1000) ** 2
        osm_completeness = round(
            (subG.number_of_nodes() / cell_km2) / EXPECTED_NODE_DENSITY, 3)

        metadata_rows.append({
            "zone_id": zone_id,
            "col_idx": int(row["col_idx"]),
            "row_idx": int(row["row_idx"]),
            "centroid_lon": round(lon, 6),
            "centroid_lat": round(lat, 6),
            "centroid_x_utm": round(row["centroid_x"], 2),
            "centroid_y_utm": round(row["centroid_y"], 2),
            "core_area_km2": round(row["area_km2"], 4),
            "n_nodes_buffered": subG.number_of_nodes(),
            "n_edges_buffered": subG.number_of_edges(),
            "osm_completeness": osm_completeness,
            "cluster_label": None,
            "cluster_id": None,
        })
        valid_zone_ids.append(zone_id)

    metadata_df = pd.DataFrame(metadata_rows)
    os.makedirs(config.out_csv, exist_ok=True)
    metadata_df.to_csv(config.metadata_csv, index=False)

    valid_grid = grid_gdf[grid_gdf["zone_id"].isin(valid_zone_ids)].drop(columns=["geometry_buf"])
    os.makedirs(config.data_processed, exist_ok=True)
    valid_grid.to_file(config.grid_cache, driver="GPKG")

    log.info("Zone extraction complete: %d/%d valid zones", len(valid_zone_ids), len(grid_gdf))
    return metadata_df


def run_tiling_stage(G, config) -> pd.DataFrame:
    if config.zone_unit == "thana":
        grid_gdf = build_thana_zones(config)
    else:
        grid_gdf = build_grid(G, config)
    return extract_zones(G, grid_gdf, config)

"""All plotting functions, extracted from the notebooks' Stage 5 / GAT / UFFM cells.

Each function takes already-computed data and a config, and saves exactly
the same PNG/HTML filenames the notebooks produced.
"""
from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection

from .io_utils import load_zone_graph


# ── Stage 1/2: network + grid ──────────────────────────────────────────────

def plot_road_network(G, config):
    import osmnx as ox
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    edges_gdf = ox.graph_to_gdfs(G, nodes=False).to_crs(config.crs_geo)
    edges_gdf.plot(ax=ax, color="#7f8fc9", linewidth=0.25, alpha=0.6)

    ax.set_title("Dhaka Road Network (OSM)", fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()
    out = os.path.join(config.out_plots, "dhaka_road_network.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out


def plot_grid_tiling(G, valid_grid, metadata_df, config):
    import osmnx as ox
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    for ax in axes: ax.set_facecolor("#1a1a2e")

    edges_gdf = ox.graph_to_gdfs(G, nodes=False).to_crs(config.crs_geo)
    ax = axes[0]
    edges_gdf.plot(ax=ax, color="#7f8fc9", linewidth=0.25, alpha=0.5)
    valid_grid.to_crs(config.crs_geo).boundary.plot(ax=ax, color="#F39C12", linewidth=0.5, alpha=0.6)
    ax.set_title("Road Network + Analysis Grid", fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")

    ax = axes[1]
    merged_viz = valid_grid.to_crs(config.crs_geo).merge(
        metadata_df[["zone_id", "osm_completeness"]], on="zone_id")
    merged_viz.plot(column="osm_completeness", ax=ax, cmap="RdYlGn", legend=True,
                     edgecolor="white", linewidth=0.2, alpha=0.85,
                     legend_kwds={"label": "OSM Completeness", "shrink": 0.7})
    ax.set_title("OSM Data Completeness", fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Longitude", color="white"); ax.set_ylabel("Latitude", color="white")
    ax.tick_params(colors="white")

    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()
    out = os.path.join(config.out_plots, "grid_tiling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out


# ── Stage 3: feature distributions ─────────────────────────────────────────

def plot_feature_distributions(features_df: pd.DataFrame, config):
    desired = ["orientation_entropy", "gwcc_pct", "betweenness_gini",
               "global_efficiency", "mean_closeness_centrality",
               "hub_dominance_ratio", "wcc_fragmentation", "density",
               "path_length_p90_norm", "avg_clustering_coef"]
    key_features = [f for f in desired if f in features_df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for ax, feat in zip(axes, key_features):
        vals = features_df[feat].dropna()
        ax.hist(vals, bins=30, color="#3498DB", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(vals.mean(), color="#E74C3C", linestyle="--", linewidth=1.5,
                    label=f"Mean: {vals.mean():.3f}")
        ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for ax in axes[len(key_features):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(config.out_plots, "feature_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Stage 4: PCA / clustering diagnostics ──────────────────────────────────

def plot_pca_explained(pca_result: dict, config):
    cumvar = pca_result["cumvar"]
    n_comp = pca_result["n_comp"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(range(1, len(cumvar) + 1), np.diff(np.concatenate([[0], cumvar])) * 100,
                color="#3498DB", alpha=0.8)
    axes[0].axvline(n_comp, color="#E74C3C", linestyle="--", label=f"Cutoff PC{n_comp}")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance (%)")
    axes[0].legend()

    axes[1].plot(range(1, len(cumvar) + 1), cumvar * 100, "o-", color="#E74C3C", lw=2)
    axes[1].axvline(n_comp, color="#E74C3C", linestyle="--")
    axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(config.out_plots, "pca_explained.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_elbow_silhouette(km_results: dict, config):
    df = pd.DataFrame(km_results["results"])
    ks = df["k"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col, ylabel, color in zip(
        axes, ["wcss", "silhouette", "ch"],
        ["WCSS (Inertia)", "Silhouette Score", "Calinski-Harabasz"],
        ["#E74C3C", "#2ECC71", "#3498DB"],
    ):
        ax.plot(ks, df[col].tolist(), "o-", color=color, lw=2)
        ax.axvline(km_results["best_k"], color="grey", linestyle="--")
        ax.set_xlabel("k"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.out_plots, "elbow_silhouette.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pca_clusters(pca_result: dict, km_results: dict, comparison: dict, config):
    X_pca = pca_result["X_pca"]
    best_labels = comparison["best_labels"]
    palette = config.CLUSTER_PALETTE

    unique = sorted(np.unique(best_labels))
    colors = {cid: palette[i % len(palette)] for i, cid in enumerate(unique) if cid != -1}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, lbl_arr, title in zip(
        axes, [km_results["best_labels"], best_labels],
        [f"K-Means (k={km_results['best_k']})", f"{comparison['best_algo']} (winner)"],
    ):
        for cid in np.unique(lbl_arr):
            if cid == -1: continue
            mask = lbl_arr == cid
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors.get(cid, "#95A5A6"),
                        label=f"Cluster_{cid}", alpha=0.75, s=50, edgecolors="white", linewidths=0.4)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title)
        ax.legend(title="Cluster"); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.out_plots, "pca_clusters.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_heatmap(features_df: pd.DataFrame, cluster_df: pd.DataFrame, config):
    from .features import SIZE_ONLY_FEATURES
    feat_cols = [c for c in features_df.columns if c not in ["zone_id"] + SIZE_ONLY_FEATURES]

    df_heat = features_df[feat_cols].copy()
    df_heat["cluster_label"] = cluster_df.set_index("zone_id").loc[
        features_df["zone_id"], "cluster_label"].values

    grp_mean = df_heat.groupby("cluster_label")[feat_cols].mean()
    grp_norm = (grp_mean - grp_mean.min()) / (grp_mean.max() - grp_mean.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(16, len(feat_cols) * 0.65), 5))
    sns.heatmap(grp_norm, cmap="RdYlGn", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Normalized Mean Feature Value"})
    ax.set_title("Feature Profiles per Topology Cluster", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    out = os.path.join(config.out_plots, "feature_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Stage 5: choropleth / subnetworks / interactive map ────────────────────

def plot_cluster_choropleth(config):
    import geopandas as gpd

    grid_gdf = gpd.read_file(config.grid_cache)
    meta_df = pd.read_csv(config.metadata_csv)
    cluster_df = pd.read_csv(config.clusters_csv)

    meta_cols = ["zone_id"] + [c for c in
        ["osm_completeness", "centroid_lat", "centroid_lon", "n_nodes_buffered", "n_edges_buffered"]
        if c in meta_df.columns]

    merged = (grid_gdf.merge(meta_df[meta_cols], on="zone_id", how="left")
              .merge(cluster_df[["zone_id", "cluster_label", "confidence_score"]], on="zone_id", how="left")
              .to_crs(config.crs_geo))
    merged["cluster_label"] = merged["cluster_label"].fillna("Unknown")

    unique_labels = sorted(merged["cluster_label"].unique().tolist())
    colors = {lbl: config.CLUSTER_PALETTE[i % len(config.CLUSTER_PALETTE)]
              for i, lbl in enumerate(unique_labels)}
    colors["Unknown"] = "#95A5A6"

    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax in axes: ax.set_facecolor("#f5f5f5")

    ax = axes[0]
    for label in unique_labels:
        sub = merged[merged["cluster_label"] == label]
        if not sub.empty:
            sub.plot(ax=ax, color=colors[label], alpha=0.75, edgecolor="white", linewidth=0.3)
    patches = [mpatches.Patch(color=colors[l], label=l) for l in unique_labels]
    ax.legend(handles=patches, title="Cluster", loc="lower right")
    ax.set_title("Dhaka Intra-Urban Topology Clusters", fontsize=13, fontweight="bold")

    ax = axes[1]
    if "osm_completeness" in merged.columns:
        merged.plot(column="osm_completeness", ax=ax, cmap="RdYlGn", legend=True,
                     edgecolor="white", linewidth=0.2, alpha=0.85,
                     legend_kwds={"label": "OSM Completeness", "shrink": 0.7})
    ax.set_title("OSM Data Completeness", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(config.out_plots, "cluster_choropleth.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_sample_subnetworks(config):
    cluster_df = pd.read_csv(config.clusters_csv)
    unique_labels = sorted(cluster_df["cluster_label"].unique().tolist())
    colors = {lbl: config.CLUSTER_PALETTE[i % len(config.CLUSTER_PALETTE)]
              for i, lbl in enumerate(unique_labels)}
    n = len(unique_labels)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1: axes = [axes]
    fig.patch.set_facecolor("#1a1a2e")

    for ax, label in zip(axes, unique_labels):
        ax.set_facecolor("#1a1a2e"); ax.axis("off")
        group = cluster_df[cluster_df["cluster_label"] == label]
        rep = group.loc[group["confidence_score"].idxmax(), "zone_id"]
        try:
            subG, nodes_df, _ = load_zone_graph(config.data_zones, rep)
            pos = {row["node_id"]: (row["x_utm"], row["y_utm"])
                   for _, row in nodes_df.iterrows()}
            color = colors.get(label, "#aaa")
            nx.draw_networkx_edges(subG, pos, ax=ax, edge_color=color, alpha=0.7, width=0.8, arrows=False)
            nx.draw_networkx_nodes(subG, pos, ax=ax, node_color=color, node_size=10, alpha=0.9)
            ax.set_title(f"{label}\n({rep})", fontsize=10, fontweight="bold", color="white")
        except Exception as e:
            ax.text(0.5, 0.5, f"{label}\nError: {e}", ha="center", va="center",
                    color="white", transform=ax.transAxes, fontsize=8, wrap=True)

    plt.tight_layout()
    out = os.path.join(config.out_plots, "cluster_sample_subnetworks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    return out


def plot_interactive_map(config):
    import folium
    import geopandas as gpd
    from folium.plugins import Fullscreen

    cluster_df = pd.read_csv(config.clusters_csv)
    meta_df = pd.read_csv(config.metadata_csv)
    grid_gdf = gpd.read_file(config.grid_cache)

    meta_cols = ["zone_id"] + [c for c in
        ["osm_completeness", "n_nodes_buffered", "n_edges_buffered"] if c in meta_df.columns]
    merged = (grid_gdf.merge(meta_df[meta_cols], on="zone_id", how="left")
              .merge(cluster_df[["zone_id", "cluster_label", "cluster_id", "confidence_score"]],
                     on="zone_id", how="left").to_crs(config.crs_geo))
    for col in ["osm_completeness", "confidence_score", "n_nodes_buffered", "n_edges_buffered"]:
        if col not in merged.columns: merged[col] = 0
        merged[col] = merged[col].fillna(0)
    merged["cluster_label"] = merged["cluster_label"].fillna("Unknown")

    unique_labels = sorted(merged["cluster_label"].unique().tolist())
    colors = {lbl: config.CLUSTER_PALETTE[i % len(config.CLUSTER_PALETTE)]
              for i, lbl in enumerate(unique_labels)}
    colors["Unknown"] = "#95A5A6"

    center_lat = merged.geometry.centroid.y.mean()
    center_lon = merged.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB dark_matter")
    Fullscreen().add_to(m)

    layer_groups = {lbl: folium.FeatureGroup(name=lbl) for lbl in unique_labels}
    for _, row in merged.iterrows():
        label = str(row.get("cluster_label", "Unknown"))
        color = colors.get(label, "#95A5A6")
        popup_html = (
            f"<div style='font-family:monospace;font-size:12px;min-width:200px'>"
            f"<b>{row.get('zone_id','N/A')}</b><hr style='margin:4px 0'>"
            f"<b>Cluster:</b> {label}<br>"
            f"<b>Confidence:</b> {float(row.get('confidence_score',0)):.2f}<br>"
            f"<b>Nodes:</b> {int(row.get('n_nodes_buffered',0))}<br>"
            f"<b>Edges:</b> {int(row.get('n_edges_buffered',0))}<br>"
            f"<b>OSM Completeness:</b> {float(row.get('osm_completeness',0)):.2f}"
            f"</div>"
        )
        try:
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=color: {"fillColor": c, "color": "white", "weight": 0.5, "fillOpacity": 0.65},
                highlight_function=lambda _: {"weight": 2, "fillOpacity": 0.9},
                tooltip=folium.Tooltip(f"{row.get('zone_id','N/A')} - {label}"),
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(layer_groups.get(label, folium.FeatureGroup(name=label)))
        except Exception:
            pass

    for grp in layer_groups.values(): grp.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(config.out_maps, exist_ok=True)
    html_path = os.path.join(config.out_maps, "dhaka_cluster_map.html")
    m.save(html_path)
    return html_path


# ── GAT hierarchy plot ─────────────────────────────────────────────────────

_BG = "#05080E"; _WHITE = "#EEF2FF"
_T1 = "#FF3B00"; _T2 = "#FF9900"; _T3 = "#00CCFF"; _T4 = "#0D1E3C"; _GRN = "#00E676"


def plot_zone_hierarchy(zone_id: str, G: nx.Graph, bc: dict, tier: dict, out_path: str, dpi: int = 150):
    """Save 4-tier betweenness hierarchy map for one zone (matches gat_pipeline.ipynb)."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=_BG)
    ax.set_facecolor(_BG)

    lons = np.array([G.nodes[n]["lon"] for n in G.nodes()])
    lats = np.array([G.nodes[n]["lat"] for n in G.nodes()])

    segs = {1: [], 2: [], 3: [], 4: []}
    for u, v in G.edges():
        t = min(tier.get(u, 4), tier.get(v, 4))
        segs[t].append([(G.nodes[u]["lon"], G.nodes[u]["lat"]), (G.nodes[v]["lon"], G.nodes[v]["lat"])])
    for t, (c, lw, al) in zip([4, 3, 2, 1], [(_T4, .25, .07), (_T3, .5, .18), (_T2, .9, .45), (_T1, 1.6, .80)]):
        if segs[t]:
            ax.add_collection(LineCollection(segs[t], colors=c, linewidths=lw, alpha=al, zorder=2))

    for t_n, col, sz, al in [(4, _T4, 6, 0.40), (3, _T3, 14, 0.60), (2, _T2, 30, 0.82)]:
        m = np.array([tier.get(n, 4) == t_n for n in G.nodes()])
        if m.any():
            ax.scatter(lons[m], lats[m], s=sz, c=col, alpha=al, linewidths=0, zorder=4 + t_n)

    m1 = np.array([tier.get(n, 4) == 1 for n in G.nodes()])
    if m1.any():
        for sz, al in [(600, .04), (240, .12), (75, .35), (24, .80)]:
            ax.scatter(lons[m1], lats[m1], s=sz, c=_T1, alpha=al, linewidths=0, zorder=8)

    master = max(bc, key=bc.get)
    mlon, mlat, mbc = G.nodes[master]["lon"], G.nodes[master]["lat"], bc[master]
    for sz, al in [(2000, .04), (800, .12), (240, .30), (70, .80)]:
        ax.scatter(mlon, mlat, s=sz, c=_GRN, alpha=al, linewidths=0, zorder=9)
    ax.scatter(mlon, mlat, s=280, marker="*", c="#FFEE00", alpha=0.95,
               edgecolors=_GRN, linewidths=1.2, zorder=10)

    pad = max(lons.max() - lons.min(), lats.max() - lats.min())
    ax.annotate(f"BC={mbc:.5f}", xy=(mlon, mlat), xytext=(mlon + pad * 0.05, mlat + pad * 0.05),
                fontsize=8, color="#FFEE00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_GRN, lw=0.9),
                bbox=dict(boxstyle="round,pad=0.25", fc=_BG, ec=_GRN, alpha=0.95, lw=1.0), zorder=11)

    p = pad * 0.04
    ax.set_xlim(lons.min() - p, lons.max() + p)
    ax.set_ylim(lats.min() - p, lats.max() + p)
    ax.tick_params(colors="#2A4060", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#0A1830")

    n1 = sum(1 for v in tier.values() if v == 1)
    n2 = sum(1 for v in tier.values() if v == 2)
    art = len(list(nx.articulation_points(G)))
    ax.set_title(f"{zone_id} - Node Importance Hierarchy\n"
                 f"{G.number_of_nodes():,} intersections * {G.number_of_edges():,} roads * "
                 f"MaxBC={mbc:.4f} * T1={n1} * T2={n2} * ArtPts={art}",
                 fontsize=9.5, color=_WHITE, fontweight="bold", pad=8)

    ax.legend(handles=[
        plt.scatter([], [], s=20, c=_GRN, marker="*", label=f"Master BC={mbc:.4f}"),
        plt.scatter([], [], s=20, c=_T1, label=f"Tier 1 (n={n1})"),
        plt.scatter([], [], s=14, c=_T2, label=f"Tier 2 (n={n2})"),
        plt.scatter([], [], s=9, c=_T3, label="Tier 3"),
        plt.scatter([], [], s=5, c=_T4, label="Tier 4"),
    ], loc="lower left", fontsize=7, facecolor=_BG, edgecolor="#0A1830", labelcolor=_WHITE, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)


def plot_city_distributions(summary_df: pd.DataFrame, config):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0D1117")
    PANEL = "#161B22"; WHITE = "#EEF2FF"

    ax = axes[0]; ax.set_facecolor(PANEL)
    ax.hist(summary_df["max_bc"], bins=30, color=_T1, alpha=0.85, edgecolor="none")
    ax.set_title("Max Betweenness per Zone", color=WHITE)

    ax = axes[1]; ax.set_facecolor(PANEL)
    ax.hist(summary_df["art_point_count"], bins=30, color=_T2, alpha=0.85, edgecolor="none")
    ax.set_title("Articulation Points per Zone", color=WHITE)

    ax = axes[2]; ax.set_facecolor(PANEL)
    ax.scatter(summary_df["art_point_count"], summary_df["max_bc"],
               c=summary_df["max_bc"], cmap="hot", s=40, alpha=0.80, linewidths=0)
    ax.set_title("BC vs Articulation Points", color=WHITE)

    plt.tight_layout()
    out = os.path.join(config.gat_dir, "city_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    return out


def plot_city_master_nodes(summary_df: pd.DataFrame, config):
    fig, ax = plt.subplots(figsize=(14, 16), facecolor="#05080E")
    ax.set_facecolor("#05080E")

    sizes = 20 + 600 * (summary_df["max_bc"] / summary_df["max_bc"].max())
    sc = ax.scatter(summary_df["master_lon"], summary_df["master_lat"], s=sizes,
                     c=summary_df["max_bc"], cmap="hot", alpha=0.85, linewidths=0, zorder=3)

    top10 = summary_df.head(10)
    for _, row in top10.iterrows():
        ax.annotate(f"{row['zone_id']}\nBC={row['max_bc']:.4f}",
                    xy=(row["master_lon"], row["master_lat"]),
                    xytext=(row["master_lon"] + 0.003, row["master_lat"] + 0.002),
                    fontsize=6.5, color="#FFEE00", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#FF9900", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.2", fc="#05080E", ec="#FF9900", alpha=0.90, lw=0.8), zorder=5)

    plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title(f"Dhaka - Master Node per Zone\n{len(summary_df)} zones", fontsize=12, color="#EEF2FF", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(config.gat_dir, "city_master_nodes.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#05080E")
    plt.close(fig)
    return out

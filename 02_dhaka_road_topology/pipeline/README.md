# Dhaka Road Topology Pipeline

> **Status: work in progress.** The pipeline runs end-to-end and its output
> is validated bit-for-bit against the original notebooks (see
> [Validation](#validation--the-port-changed-nothing) below). The
> *underlying clustering result itself* is weak — see
> [Known Limitations](#known-limitations) before citing any cluster labels
> as a real finding.

Part of an MSc thesis (CSE, BUET) on topology-aware transport policy
targeting for Dhaka. Decomposes Dhaka's road network into 1km zones,
extracts 26 topology features per zone, clusters zones by structural type,
and identifies critical nodes (high betweenness centrality / articulation
points) as candidate policy-intervention points.

This repo contains the **pipeline code** and a **snapshot of its output**
from a full run over 827 zones. It does not contain the paper drafts,
presentations, or raw/intermediate data (see [What's not here](#whats-not-here)).

## Why this exists

The original analysis was six Jupyter notebooks that each forked off the
last one — `dhaka_topology_v1.ipynb` → `v2.ipynb` → `v3.ipynb` (same
pipeline, copy-pasted forward every time a bug was fixed or the bounding
box changed), plus `v4_2km.ipynb` and `thana.ipynb` (new notebooks just to
test a different grid size or zone unit), plus `gat_pipeline.ipynb` and
`uffm_algorithm_v3.ipynb` bolted on top. Grid size, bounding box, and zone
unit were never parameters — they were reasons to fork a notebook.

This package turns those forks into one config-driven pipeline with six
independent, testable stages.

## What's here

```
dhaka_topology/            the pipeline package
  config.py                  Config dataclass — bbox, grid size, zone unit,
                              every threshold — loaded from configs/*.yaml
  io_utils.py                 shared zone-loading helpers
  acquisition.py               Stage 1 — OSM download, cache, validate
  tiling.py                     Stage 2 — grid or thana zone extraction
  features.py                   Stage 3 — 26 topology features per zone
  clustering.py                  Stage 4 — PCA + KMeans/DBSCAN/GMM + report
  betweenness.py                  GAT stage — Brandes betweenness, 4-tier
                                   hierarchy, articulation points
  uffm.py                          UFFM stage — Wasserstein-distance
                                   geometric fingerprints + spectral clustering
  visualization.py                 every plot the pipeline produces
  pipeline.py                      stage orchestration

configs/                   one YAML per variant (v1, v3, v2km, thana)
tests/                     12 unit tests on the pure functions
compare_outputs.py         diffs this pipeline's output against the
                            original notebooks' saved results
results/                   a full run's output (see below) — this is the
                            actual analysis result, not just example output
```

### `results/` — what a full run over 827 zones produced

```
results/outputs/csv/
  features.csv                 26 features x 827 zones
  features_normalized.csv      min-max normalized, size features excluded
  feature_stability.csv        bootstrap CV per feature (see Limitations)
  cluster_assignments.csv      zone -> cluster label + confidence score
  clustering_scores.csv        KMeans/DBSCAN/GMM comparison
  uffm_fingerprints.csv        bearing/angle/length distribution fingerprints
  uffm_cluster_assignments.csv  spectral clustering on Wasserstein distances
  uffm_clustering_scores.csv    silhouette per k, k=2..8
  uffm_v3_crosstab.csv          scalar-feature clusters vs UFFM clusters
  metadata.csv                  per-zone centroid, node/edge counts, cluster

results/outputs/plots/     9 PNGs — road network, grid, PCA, elbow,
                            feature heatmap, choropleth, sample subnetworks
results/outputs/maps/      interactive Folium map (dhaka_cluster_map.html)

results/gat/csv/
  all_zones_summary.csv        822 zones — max/mean/p95/p99 betweenness,
                                tier counts, articulation point count,
                                master (highest-BC) node per zone
  all_nodes_bc.csv              every node in every zone: BC, tier (1-4),
                                degree, is_master, is_articulation_point
  Zone_DH*/nodes_bc.csv          per-zone breakdown (822 files)

results/gat/plots/         822 per-zone hierarchy maps (one PNG per zone)
results/gat/city_distributions.png   city-wide BC/articulation histograms
results/gat/city_master_nodes.png    every zone's master node, one map
```

## What came out

**Clustering (scalar features, Stage 4):** KMeans won with **k=2**,
silhouette **0.408** — a 184/643 zone split. DBSCAN found essentially no
structure (1 cluster, silhouette -1.0, i.e. failed). GMM's best silhouette
was **0.046** — noise. The v3 notebook's own documentation expected "k=3 or
k=4" once footpaths/lanes were included in the network; getting k=2 back,
with two of three algorithms failing to find any real separation, is a weak
result.

**UFFM (geometric fingerprints, Wasserstein distance + spectral
clustering):** best k=2, silhouette **0.437** — marginally better than the
scalar approach, but the cross-tabulation against the scalar clusters
(`uffm_v3_crosstab.csv`) doesn't line up cleanly: scalar Cluster_0 (184
zones) splits 40/144 across the two UFFM clusters, and scalar Cluster_1
(643 zones) splits 582/61. Two different feature representations of the
same zones don't agree on where the boundary is — a sign the "2 topology
types" story isn't a stable property of the data, it's close to the
boundary both methods happen to draw.

**Betweenness/GAT:** this part is more solid — betweenness centrality,
tier assignment, and articulation-point detection are exact, deterministic
graph computations (Brandes' algorithm), not a fitted model. `all_zones_summary.csv`
and the master-node map are a reasonable output on their own, independent
of whether the clustering holds up.

## Known limitations

- **Bootstrap feature stability is bad.** `feature_stability.csv` reports
  coefficient of variation per feature under 80%-node resampling. Several
  features the clustering leans on have CV well above the notebook's own
  0.35 flag threshold: `transitivity` (1.35), `avg_clustering_coef` (1.32),
  `n_scc`/`n_wcc` (~1.2), `density` (0.87). A feature with CV > 1 means its
  bootstrap standard deviation exceeds its mean — the feature is not a
  stable property of a zone, it's noisy at the sample sizes involved.
- **Only one of three clustering algorithms found real structure.** DBSCAN
  and GMM effectively failed (silhouette -1.0 and 0.046). KMeans's 0.408 is
  moderate at best — not the kind of clean separation you'd want before
  naming clusters after urban-form archetypes (RAJUK grid, Moghul organic,
  etc., as the notebook's own Stage 4 comments suggested).
- **UFFM doesn't independently confirm the scalar clustering.** If two
  different feature representations agreed on the same 2-way split, that
  would be real evidence. They don't — see the crosstab above.
- **k=2 is a coarse result for a "topology-aware policy targeting"
  framework.** A binary split across 827 zones doesn't give a policymaker
  much to target differently.

None of this is a code bug — it's ported faithfully from the original
notebooks (see Validation below) and reflects genuine weak signal in the
26-feature representation at the 1km-grid scale. Candidates worth trying
before trusting cluster labels as a finding: richer features (the UFFM
fingerprints are a step in that direction but don't resolve it either),
a coarser or admin-boundary zone unit (`thana` config, unvalidated — see
TODO), or treating this as a two-way (dense-connected vs.
fragmented-organic) distinction rather than searching for more types that
the data doesn't support.

## Validation — the port changed nothing

Every formula here is copied verbatim from the notebooks (each module has
a comment pointing at its source cell). To catch any silent change during
the port, the pipeline was run against the same 827 zones the original
`dhaka_topology_v3.ipynb` already extracted, and every output was diffed
column-by-column against the notebooks' saved CSVs.

| Stage | Output | Result |
|---|---|---|
| Features | `features.csv` (26 x 827) | **exact match** (±1e-6) |
| Features | `features_normalized.csv` | **exact match** |
| Clustering | `cluster_assignments.csv`, `clustering_scores.csv` | **exact match** |
| Betweenness/GAT | `all_zones_summary.csv`, `all_nodes_bc.csv` (822 zones) | **exact match** (excl. wall-clock timing) |
| UFFM | `uffm_fingerprints.csv`, `uffm_cluster_assignments.csv`, `uffm_clustering_scores.csv` | **exact match** |

`compare_outputs.py` is the tool that did this diff. It's included so the
same check can be re-run if the pipeline changes.

**Not yet validated:** the `tile` stage (Stage 2 — grid construction from a
freshly downloaded OSM graph) and the `thana` zone unit. Both are direct
ports but haven't been diffed the way the stages above were, because that
requires a multi-hour OSM download to test against.

## Install & run

```bash
pip install -r requirements.txt

# Full pipeline, canonical 1km / 827-zone variant
python run_pipeline.py --config configs/v3.yaml --stage all

# Just re-run feature engineering + clustering (zones already extracted)
python run_pipeline.py --config configs/v3.yaml --stage features cluster

# 2km grid experiment
python run_pipeline.py --config configs/v2km.yaml --stage all
```

Stages run in order: `acquire` -> `tile` -> `features` -> `cluster` ->
`betweenness` -> `uffm` -> `plots`. Each can be run alone; it reloads
whatever earlier-stage CSVs it needs from disk.

```bash
pytest tests/
```
12 tests on the pure functions: Gini coefficient, orientation entropy,
UFFM fingerprint histograms, betweenness/tier classification.

## What's not here

Paper drafts, the LaTeX thesis document, and presentation slides are kept
outside this repo (local only) — this repo is code + results + docs. Raw
OSM data, per-zone intermediate CSVs, and the original notebooks (kept as
a local backup for reference) are gitignored: they're large, regeneratable
by running the pipeline, and not needed to understand or reproduce the
analysis from what's tracked here.

## TODO

- [ ] Address the clustering weakness above — try a richer or different
      feature set before treating k=2 as a real finding
- [ ] Validate the `tile` stage against a fresh OSM download
- [ ] Validate the `thana` zone-unit path (no boundary file tested yet)
- [ ] Add a smoke config (`gat_max_zones` set low) for fast sanity checks

# Dhaka Road Topology Pipeline

> **Status: work in progress.** Feature engineering, clustering, betweenness
> (GAT), and UFFM are built and validated. Zone tiling and plotting are
> ported but not yet re-validated end-to-end against a fresh OSM download.
> See [Status](#status--whats-validated) below before relying on this for
> new results.

Part of an MSc thesis (CSE, BUET) on topology-aware transport policy
targeting for Dhaka. This package replaces six overlapping Jupyter
notebooks with one config-driven pipeline.

## Why this exists

The original analysis was six notebooks that each forked off the last one:

- `dhaka_topology_v1.ipynb` → `v2.ipynb` → `v3.ipynb` — the same pipeline,
  copy-pasted forward each time a bug was fixed or the bbox changed
- `dhaka_topology_v4_2km.ipynb` — a whole new notebook just to test a 2km
  grid instead of 1km
- `dhaka_topology_thana.ipynb` — another new notebook to swap the grid for
  administrative boundaries
- `gat_pipeline.ipynb`, `uffm_algorithm_v3.ipynb` — two more notebooks
  bolted on top of v3's output

Grid size, bounding box, and zone-unit choice were never parameters — they
were reasons to fork a notebook. That made the analysis hard to reproduce,
hard to diff, and hard to hand off.

## What we did

Extracted every notebook's logic into a Python package with one stage per
pipeline step, all driven by a single YAML config per variant:

```
dhaka_topology/
  config.py          Config dataclass — bbox, grid size, zone unit, all
                      thresholds — loaded from configs/*.yaml
  io_utils.py         shared zone-loading helpers
  acquisition.py       Stage 1 — OSM download, cache, validate
  tiling.py            Stage 2 — grid or thana zone extraction
  features.py          Stage 3 — 26 topology features (density, orientation
                        entropy, betweenness gini, hub dominance, ...)
  clustering.py         Stage 4 — PCA + KMeans/DBSCAN/GMM + diagnostic report
  betweenness.py         GAT stage — Brandes betweenness, 4-tier hierarchy,
                          articulation points (was gat_pipeline.ipynb)
  uffm.py                UFFM stage — Wasserstein-distance geometric
                          fingerprints + spectral clustering
                          (was uffm_algorithm_v3.ipynb)
  visualization.py       every plot the notebooks produced
  pipeline.py            stage orchestration
```

What used to require forking a notebook is now a config value:

| Config | Replaces | What differs |
|---|---|---|
| `configs/v1.yaml` | `dhaka_topology_v1.ipynb` | Narrower bbox, drive-only network |
| `configs/v3.yaml` | `dhaka_topology_v3.ipynb` (canonical) | Full bbox, all-roads network, 827 zones |
| `configs/v2km.yaml` | `dhaka_topology_v4_2km.ipynb` | 2km grid cell instead of 1km |
| `configs/thana.yaml` | `dhaka_topology_thana.ipynb` | Admin boundaries instead of a grid |

`gat_pipeline.ipynb` and `uffm_algorithm_v3.ipynb` aren't separate configs —
they're stages (`betweenness`, `uffm`) toggled per-config via
`gat_enabled` / `uffm_enabled`, since they always ran against v3's output.

## Status / what's validated

The formulas are copied verbatim from the notebooks (see comments in each
module pointing back to the source cell). To make sure the port didn't
silently change any math, the pipeline was run against the same 827 zones
already extracted by the original `dhaka_topology_v3.ipynb`, and every
output was diffed column-by-column against the notebooks' saved results.

| Stage | Output | Result |
|---|---|---|
| Features | `features.csv` (26 features × 827 zones) | **exact match** (±1e-6) |
| Features | `features_normalized.csv` | **exact match** |
| Clustering | `cluster_assignments.csv` | **exact match** |
| Clustering | `clustering_scores.csv` | **exact match** |
| Betweenness/GAT | `all_zones_summary.csv` | **exact match** (excl. wall-clock timing column) |
| Betweenness/GAT | `all_nodes_bc.csv` (per-node BC, tier, articulation flags) | **exact match** |
| UFFM | `uffm_fingerprints.csv` | **exact match** |
| UFFM | `uffm_cluster_assignments.csv` | **exact match** |
| UFFM | `uffm_clustering_scores.csv` | **exact match** |

**Not yet validated:** the `tile` stage (Stage 2 — grid construction and
zone extraction from a freshly downloaded OSM graph) and the `thana` zone
unit. Both are direct ports of the notebook logic but haven't been run
end-to-end and diffed the way the stages above were, because that requires
a multi-hour OSM download to test against. Treat `tile` as unverified until
it's been run and compared.

Reproducing this validation requires the original zone data (`data/v3/`)
and notebook outputs, which are not part of this repository (kept locally
only, since they're hundreds of megabytes of generated data). If you have
them, `compare_outputs.py` does the diff:

```bash
python compare_outputs.py \
    --old  path/to/original/outputs/v3 \
    --new  path/to/pipeline/outputs/v3 \
    --old-gat path/to/original/gat \
    --new-gat path/to/pipeline/gat
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline, canonical 1km / 827-zone variant
python run_pipeline.py --config configs/v3.yaml --stage all

# Just re-run feature engineering + clustering (zones already extracted)
python run_pipeline.py --config configs/v3.yaml --stage features cluster

# 2km grid experiment
python run_pipeline.py --config configs/v2km.yaml --stage all

# Force re-download of the OSM graph
python run_pipeline.py --config configs/v3.yaml --stage acquire --force
```

Stages run in order: `acquire` → `tile` → `features` → `cluster` →
`betweenness` → `uffm` → `plots`. Each can be run alone and will reload
whatever earlier-stage CSVs it needs from disk — the same "re-run one cell"
workflow the notebooks had, minus the notebook.

## Tests

```bash
pytest tests/
```

Covers the pure, most failure-prone functions: Gini coefficient,
orientation entropy, UFFM fingerprint histograms, and betweenness/tier
classification. 12 tests, all passing.

## TODO

- [ ] Validate the `tile` stage against a fresh OSM download
- [ ] Validate the `thana` zone-unit path (no boundary file has been tested against it yet)
- [ ] Wire up a `--dry-run` / cost estimate before triggering an OSM download
- [ ] Add a smoke config with `gat_max_zones` set low, for fast CI-style checks
- [ ] Decide on a permanent home for large intermediate data (currently gitignored, lives outside the repo)

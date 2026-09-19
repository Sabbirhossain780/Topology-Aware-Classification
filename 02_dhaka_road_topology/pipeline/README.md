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

The port's initial clustering result was weak (see "Original weak result"
below) — but a systematic experiment sequence since then (full log in
[`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md)) found and fixed
the actual problem, and arrived at a validated, defensible finding:

**Dhaka's road network does not separate into discrete morphological
archetypes (Maze/Grid/Radial/Periphery) at 1-2km grid scale** — this
confirms the team's own prior null-result finding. **But it does separate,
robustly, into a connectivity-level gradient**: dense/continuous urban
fabric (whether organically grown or formally planned) vs.
sparse/fragmented/peripheral development. Unlike every earlier attempt,
this result is:
- **Statistically significant** against a permutation null (p=0.005,
  ~4-5x above the 95th-percentile of what random data produces)
- **Not a zone-size artifact** — most "normalized" centrality features
  turned out to be 60-85% explained by raw zone size alone
  (`mean_katz_centrality` R²=0.85, `global_efficiency` R²=0.83); the
  finding above is what survives after regressing that out
- **Visually confirmed on the ground** — 7/7 sampled zones matched their
  predicted profile when checked against OpenStreetMap (dense continuous
  fabric for the "connected" cluster including both organic and
  RAJUK-style planned-grid examples; sparse/rural/institutional/
  urbanizing-fringe fabric for the other)

See `experiments/EXPERIMENTS.md` Experiments 01, 06, and 07 for the full
methodology. The original weak numbers below are kept for context on how
this result was reached, not as the final word.

### Original weak result (superseded, kept for context)

KMeans won with k=2, silhouette 0.408 (1km) / 0.513 (2km, before the size
confound was found and corrected). DBSCAN found no structure at any stage
of this project. GMM's silhouette was 0.046-0.242 depending on
configuration. UFFM (geometric fingerprints) didn't independently confirm
the scalar-feature split. All of this turned out to be substantially a
zone-size confound, not a measure of the feature set's real
discriminative power — see Experiment 06.

**Betweenness/GAT:** this part was solid throughout — betweenness
centrality, tier assignment, and articulation-point detection are exact,
deterministic graph computations (Brandes' algorithm), not a fitted model.
`all_zones_summary.csv` and the master-node map are a reasonable output on
their own, independent of the clustering work above.

## Known limitations

- **This is a connectivity-level finding, not a morphology-type
  finding.** The validated cluster split doesn't distinguish "Grid" from
  "Maze" — both appear inside the same "connected" cluster in the
  Experiment 07 spot-check. It distinguishes dense/continuous networks
  from sparse/fragmented ones, which is a real and useful distinction for
  policy targeting, but narrower than the original four-archetype goal.
  **Experiment 08 shows visually why**: rendering whole zones
  ([`experiments/08_cv_fft_pilot/pilot_renders_and_spectra.png`](experiments/08_cv_fft_pilot/pilot_renders_and_spectra.png))
  reveals that no 2km zone *has* a single morphology — every one is a
  mixture of grid patches and organic fabric, and their 2D spectra are
  near-identical as a result. A zone-level morphology label is not a
  well-defined thing to ask for at this scale. **Experiments 09-10 then
  measured this** at 400m patch scale: a rotation-invariant per-patch
  grid-score, validated by eye
  ([`vector_ranking_check.png`](experiments/10_descriptor_validation/vector_ranking_check.png)),
  shows **within-zone morphological variation is ~4x the between-zone
  variation**. So the zone-level label isn't badly measured, it's
  poorly defined — and the **mixture ratio** is the measurable replacement.
  (Experiment 09's own numbers were later found to carry a rasterization
  artifact and are superseded by Experiment 10's vector recompute; the
  conclusion held and strengthened.)
- **The satellite validation (Experiment 07) was a small, single-reviewer
  sample** (7 zones) — not the original proposal's planned n=30
  expert-morphological-assessment panel. The 7/7 agreement is a genuine
  signal, not a substitute for that more rigorous check.
- **The size-confound correction (Experiment 06) is not perfectly clean.**
  A ~3x mean node-count gap remains between the two clusters after
  regressing out `log(node_count)` (down from ~15x before) — better, but
  worth keeping in mind rather than treating as fully solved.
- **Bootstrap feature stability is still uneven.** `feature_stability.csv`
  flags several features (mostly the same ones later found to be
  size-confounded) with CV well above 0.35 under 80%-node resampling.

None of the above is a code bug — the pipeline is validated bit-for-bit
against the original notebooks (see Validation below). The weak initial
result and the confound that explained it were both genuine properties of
the feature set and the data, tracked down through the experiment sequence
in `experiments/EXPERIMENTS.md` rather than assumed or dismissed.

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
- [x] Validate the `tile` stage — done via the 2km grid run (Experiment 03), 224/234 zones
      extracted correctly from a freshly-tiled (not re-downloaded) graph
- [ ] `thana` zone-unit path is explicitly out of scope per user direction, not just
      unvalidated (see `experiments/EXPERIMENTS.md` Aim section)
- [x] **Patch-scale mixture-ratio pilot** — done as Experiment 09. A 400m patch grid-score
      (orthogonal-pair spectral energy) is calibrated, density-decoupled and visually
      validated; within-zone variation is 3.0x between-zone variation on n=4 zones.
- [ ] **Scale the patch metric to all 224 zones** (~5,600 patches, runs in minutes) — the live
      research direction. The n=4 pilot cannot test whether *mixture ratio* varies between
      zones even though morphology *type* does not; the full run can. See
      `experiments/EXPERIMENTS.md` Experiment 09.
- [ ] Try a sliding patch window instead of a fixed lattice — a grid straddling two patch
      boundaries is currently penalised (Experiment 09 caveat)
- [ ] **Fix UFFM's fingerprint comparison** — `uffm.py:161` uses linear-axis Wasserstein on a
      circular bearing variable, making it rotation-VARIANT: two identical grids rotated 45°
      apart score further apart than a grid and an organic network (Experiment 10, C8). This
      is a candidate root cause for UFFM's documented null result, and the fix is small.
- [ ] **Add a spatial convergence measure for "radial"** — Experiment 10 (C7) showed radial has
      no stable angular signature, so it cannot come from the |c_k| descriptor at all.
- [ ] Add a smoke config (`gat_max_zones` set low) for fast sanity checks
- [ ] Run the Experiment 07 satellite check at proper scale (n=30, ideally a second reviewer)
      to actually meet the original proposal's validation target rather than approximate it
- [ ] Tighten the size-confound correction in Experiment 06 — a ~3x zone-size gap remains
      between clusters after decorrelation; worth investigating whether a stronger
      (e.g. non-linear) size correction removes it further without destroying the real signal

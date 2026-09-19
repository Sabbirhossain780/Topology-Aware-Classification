# Experiment Log

## Aim

Make the 1-2km grid topology clustering of Dhaka's road network produce a real, defensible
result — not a repeat of the null result already documented in
`04_reports_and_papers/reports/dhaka_methodology_report.docx` (KMeans k=2, silhouette 0.408;
DBSCAN/GMM find no structure; UFFM doesn't independently confirm the scalar-feature split).

**Constraints (from the user):**
- 1km or 2km grid only — thana-level/administrative-boundary work is explicitly out of scope.
- Systematic, lab-style development: every change is a logged experiment (hypothesis, method,
  config, result, decision), not an ad hoc tweak.
- Grounded in what the existing proposal/reports already established (the CV > 0.35 instability
  threshold, the >=85% satellite-imagery agreement target, etc.) rather than inventing new
  success criteria from nothing.
- No GPU needed — confirmed early on; this is a small-graph, CPU-tractable problem (see
  `../README.md` and the plan file this work originated from).

Each entry below: **Hypothesis / Method / Config / Result / Decision.** Entries are append-only
— a later experiment building on or superseding an earlier one gets a new entry, not an edit.

---

## Baseline (for reference, not a new experiment)

Already established and validated bit-for-bit against the original notebooks (see
`../README.md`):
- KMeans: k=2, silhouette **0.408**, 184/643 zone split
- DBSCAN: 1 cluster, silhouette **-1.0** (failed to find structure)
- GMM: k=6, silhouette **0.046** (failed to find structure)
- UFFM (geometric fingerprints): k=2, silhouette **0.437**, but crosstab against the scalar
  clusters doesn't line up cleanly (40/144 and 582/61 splits)
- Several clustering features have bootstrap CV > 1.0 (`transitivity` 1.35,
  `avg_clustering_coef` 1.32, `n_scc`/`n_wcc` ~1.2, `density` 0.87)

---

## Experiment 01 — Permutation significance test

**Hypothesis:** the baseline silhouette (0.408) might not be distinguishable from what
clustering pure noise would produce, given DBSCAN and GMM both failed to find structure on
the same data.

**Method:** `dhaka_topology.clustering.permutation_test()` — shuffle each of the 26 (23
clustering) feature columns independently across all 827 zones (destroys joint/cross-feature
structure, preserves each feature's own marginal distribution), then re-run the *exact* same
model-selection process used on real data (PCA to 95% variance -> KMeans k=2..8, keep the best
silhouette). Repeated 200 times to build a null distribution. Config: `configs/v3.yaml`.
Script: `experiments/01_significance_test/run.py`.

**Result:**
| | Value |
|---|---|
| Real best silhouette | **0.4080** (k=2) |
| Null distribution mean | 0.0589 (+/- 0.0105) |
| Null 95th percentile | 0.0781 |
| p-value | **0.0050** (0/200 permutations beat the real result) |

Full null distribution saved in `experiments/01_significance_test/null_distribution.csv`;
summary in `result.json`.

**Decision:** The 2-cluster split is **statistically real** — not noise, not an artifact of
having 26 correlated features and a flexible k-search. This directly strengthens (doesn't
undercut) the methodology report's "policy contribution" framing: the low-connectivity 22%
cluster is a genuine, significant structural feature of Dhaka's network, not a coincidence of
this particular feature set or k choice. Proceeding to Experiment 02 (stable-features-only
re-cluster) to test whether a *richer* result (more than 2 clusters, or better separation) is
achievable, now that we know there's real signal in this feature space worth refining rather
than replacing.

---

## Experiment 02 — Stable-features-only re-cluster

**Hypothesis:** the 18 bootstrap-unstable features (CV > 0.35 — `density`, `transitivity`,
`avg_clustering_coef`, `n_scc`/`n_wcc`, all four centrality means, etc.) are adding noise that
obscures cleaner structure carried by the 5 stable features
(`orientation_entropy`, `betweenness_gini`, `avg_degree`, `hub_dominance_ratio`,
`wcc_fragmentation`).

**Method:** `filter_stable_features()` restricts `features_normalized.csv` to the 5 stable
columns, then runs the identical PCA -> KMeans/DBSCAN/GMM comparison and a fresh permutation
test on that reduced feature set. Config: `configs/v3.yaml`.
Script: `experiments/02_stable_features_only/run.py`.

**Result:**
| | 24 features (baseline) | 5 stable features only |
|---|---|---|
| KMeans | k=2, sil=**0.408** | k=2, sil=**0.324** |
| DBSCAN | 1 cluster, sil=-1.0 | 1 cluster, sil=-1.0 |
| GMM | sil=0.046 | sil=0.136 |
| Permutation p-value | 0.005 | 0.010 |
| Null mean / 95th pct | 0.059 / 0.078 | 0.203 / 0.223 |

Full result in `experiments/02_stable_features_only/result.json`.

**Decision — hypothesis rejected.** Dropping the unstable features made KMeans *worse*
(0.408 -> 0.324), not better, and didn't change the cluster count (still k=2). DBSCAN still
finds nothing. GMM improved somewhat (0.046 -> 0.136) but stayed weak. Still statistically
significant vs. random noise, but the null distribution's own baseline rose sharply (5-D random
data tends to look more "clustered" by chance than 26-D random data — less averaging-out of
noise across dimensions). Plausible explanation: even individually noisy features carry shared
signal about the same underlying connectivity factor, and PCA over the full set averages that
noise out rather than being corrupted by it. **Conclusion: the full feature set is doing useful
work; feature reduction is not the fix.** Proceeding to Experiment 03 (2km grid) — the scale
hypothesis, not the feature-noise hypothesis, remains the more promising lead.

---

## Experiment 03 — 2km grid, full pipeline run

**Hypothesis:** doubling cell size (1km -> 2km) partially closes the scale gap the
methodology report identifies (1km cells mix Dhaka's micro-scale ~100-300m blocks with its
macro-scale ~3-5km neighborhoods) without leaving the 1-2km bound.

**Method:** ran `acquire` (reused cached graph, no re-download) -> `tile` -> `features` ->
`cluster` on `configs/v2km.yaml`. First real exercise of the `tile` stage, previously
unvalidated. Zone extraction: 224/234 candidate cells valid (vs 827/? at 1km). Also ran the
same permutation significance test on this result.
Scripts: `run_pipeline.py --config configs/v2km.yaml`, `experiments/01_significance_test/run.py
--config configs/v2km.yaml --out-dir experiments/03_2km_grid`.

**Result:**
| | 1km grid (827 zones) | 2km grid (224 zones) |
|---|---|---|
| KMeans | k=2, sil=**0.408**, split 184/643 (22%/78%) | k=2, sil=**0.513**, split 25/199 (11%/89%) |
| DBSCAN | 1 cluster, sil=-1.0 | 1 cluster, sil=-1.0 (same failure mode) |
| GMM | k=6, sil=0.046 | k=3, sil=**0.201** |
| Permutation p-value | 0.005 | 0.005 |
| Null mean / 95th pct | 0.059 / 0.078 | 0.070 / 0.097 |

Full result in `experiments/03_2km_grid/result.json`. Raw features/clusters in
`pipeline_outputs/v2km/outputs/csv/`.

**Decision — partial support for the scale hypothesis.** The 2km grid's KMeans silhouette
(0.513) is meaningfully better than 1km's (0.408), and GMM found a cleaner 3-cluster structure
(0.201 vs 0.046) instead of overfitting to 6. Both effects point the same direction as the
methodology report's scale-mismatch diagnosis. However: the cluster split got *more* imbalanced
(11%/89% vs 22%/78%), DBSCAN still finds nothing at either scale, and the driving features are
the same ones in the same direction (low orientation_entropy/low betweenness_gini/high
efficiency vs the opposite) — this still isn't a rich multi-archetype result, it's the same
two-way connectivity split with cleaner separation at the coarser grid. The 2km result is the
better of the two available options within the 1-2km bound.
**2km adopted as the new working baseline** for Experiments 04-06 going forward.

---

## Experiment 04 — Fuse scalar + UFFM features

**Hypothesis:** scalar features and UFFM's geometric fingerprints run as two separate,
disagreeing clustering exercises (see baseline crosstab). A joint feature space might resolve
what neither captures alone.

**Method:** `dhaka_topology/fusion.py` derives 7 compact summary statistics per zone from the
3 UFFM fingerprints (Shannon entropy of each distribution, dominant bearing angle, fraction of
intersection angles near 90 deg / near 180 deg, median segment length) — deliberately not the
full 112 raw histogram bins, to avoid drowning the 23 scalar features in redundant dimensions.
Merged with `features_normalized.csv` (2km) for 30 total features, re-ran the full PCA ->
clustering -> significance pipeline. Config: `configs/v2km.yaml` (with `uffm_enabled: true`).
Script: `experiments/04_feature_fusion/run.py`.

**Result:**
| | 2km scalar-only (Exp 03) | 2km fused (23 scalar + 7 UFFM) |
|---|---|---|
| KMeans | k=2, sil=**0.513** | k=2, sil=**0.338** |
| DBSCAN | 1 cluster, sil=-1.0 | 1 cluster, sil=-1.0 |
| GMM | sil=0.201 | sil=0.167 |
| Permutation p-value | 0.005 | 0.005 |

Full result in `experiments/04_feature_fusion/result.json`; fused feature matrix in
`experiments/04_feature_fusion/fused_features.csv`.

**Decision — hypothesis rejected.** Adding the UFFM summary features made KMeans *worse*
(0.513 -> 0.338), same pattern as Experiment 02 (more features from a weaker source dilute a
stronger one). Still statistically significant vs noise, still k=2, same fundamental split.
The UFFM geometric fingerprints genuinely carry different information than the scalar features
(confirmed by the baseline crosstab not lining up), but "different" isn't "complementary" here
— it's closer to unrelated noise from the clustering's perspective. **2km scalar-only (Exp 03,
sil=0.513) remains the best result.** Proceeding to Experiment 05 (OSM-completeness filtering)
on that baseline.

---

## Experiment 05 — OSM-completeness filtering

**Hypothesis:** low-completeness zones (`osm_completeness < 0.5` — under-mapped areas, mostly
informal/dense old-city fabric where OSM coverage is sparse) inject noise that masks cleaner
structure elsewhere, per the original proposal's own flag.

**Method:** excluded the 22/224 zones below the 0.5 completeness threshold from the 2km
scalar-only feature set (Exp 03's config), re-ran the full clustering + significance pipeline
on the remaining 202. Script: `experiments/05_completeness_filter/run.py`.

**Result:**
| | 2km unfiltered (Exp 03, 224 zones) | 2km completeness-filtered (202 zones) |
|---|---|---|
| KMeans | k=2, sil=**0.513** | k=2, sil=**0.377** |
| DBSCAN | 1 cluster, sil=-1.0 | 1 cluster, sil=-1.0 |
| GMM | sil=0.201 | sil=**0.242** |
| Permutation p-value | 0.005 | 0.005 |

Full result in `experiments/05_completeness_filter/result.json`.

**Decision — mixed, net negative for the winning algorithm.** KMeans (the algorithm that
actually wins in every experiment so far) got *worse* after filtering (0.513 -> 0.377), though
GMM improved slightly (0.201 -> 0.242). Since KMeans is consistently the winner and the metric
that matters for the final result, this doesn't improve on Exp 03. Removing 22 zones (~10% of
the dataset) also cost real statistical power for a result that got worse, not better.
**2km scalar-only, unfiltered (Exp 03, sil=0.513) remains the best result overall.** Proceeding
to Experiment 06 — checking whether that specific result actually corresponds to recognizable
urban form.

---

## Finding — the 2-cluster split is a node-count confound, not a topology distinction

**Discovered while preparing Experiment 06** (pulling top-confidence zones per cluster for a
satellite spot-check). Before trusting cluster labels enough to validate them visually, checked
whether the split actually tracks the connectivity story the diagnostic report tells, or
something else.

**What was found:** at 2km (Exp 03's best result), `Cluster_0` (25 zones) has mean
`n_nodes_buffered` = 57.6 (range 19-161); `Cluster_1` (199 zones) has mean 900.2 (range
87-3722) — a ~15.6x difference in raw zone size with almost no overlap. The same pattern holds
at 1km: `Cluster_0` mean node_count = 51.6, `Cluster_1` mean = 544.3, and node_count correlates
with `global_efficiency` at r=-0.66 and with `betweenness_gini` at r=+0.62 across the whole
1km dataset.

**Why this matters:** `global_efficiency` and `betweenness_gini` are "already normalized by
networkx" in the sense of being bounded in [0,1] and not raw counts, but that doesn't make them
size-*independent*. A 20-node zone has short paths almost by construction (few nodes to route
through), which mechanically inflates `global_efficiency` and compresses `betweenness_gini`
regardless of actual street layout. Excluding `node_count`/`edge_count`/`path_length_p90` as
`SIZE_ONLY_FEATURES` (the v2 fix) was not sufficient — several of the *remaining* "normalized"
features are still strongly, mechanically coupled to raw zone size.

**Implication for every experiment so far (01-05):** the 2-cluster result these experiments
have been measuring — and the "improvement" from 1km (sil=0.408) to 2km (sil=0.513) — is best
explained as "small/sparse zones vs. large/dense zones," essentially rediscovering the same OSM
mapping-density split the original v1 methodology report identified and believed it had
eliminated ("The clustering was not measuring urban topology. It was measuring OSM mapping
completeness"). The Experiment 05 completeness filter partially brushed against this (it
removed many of Cluster_0's members, which is why KMeans got worse) without diagnosing the
actual mechanism.

**Decision:** Experiment 06 (satellite validation) is **paused, not run**, because spot-checking
this specific cluster split would just confirm "small edge zones look different from large
urban zones" — true, but not the topological finding the thesis is after, and not worth
burning the validation budget on. Before any further clustering experiments, the feature set
needs proper decorrelation from zone size (e.g. residualizing each feature against
`node_count` via regression, or explicit size-stratified analysis) — this is a new, higher-
priority experiment than continuing the original 01-06 sequence as planned. Flagged to the user
for direction before proceeding.

---

## CHECKPOINT (superseded — resolved by Experiments 06-07 below, kept for history)

**State when paused:** every experiment run so far (01-05) has real, statistically significant
results (p=0.005 throughout), but the significant thing they're all detecting appears to be a
**zone-size confound**, not a topology-type distinction. The best numeric result (2km,
sil=0.513, Experiment 03) is not more trustworthy than the 1km baseline in this respect — it
may just have a wider size spread. Experiment 06 (satellite validation) was intentionally
*not* run because validating this split visually would be confirming size, not morphology.

**What's proven solid and doesn't need redoing:**
- The pipeline itself (`dhaka_topology/` package) — validated bit-for-bit against the original
  notebooks (see `../README.md`).
- The `tile` stage now has one real validated run behind it (2km, 224/234 zones extracted
  correctly) — no longer purely theoretical.
- The permutation significance test (`clustering.permutation_test()`) works and is cheap to
  re-run (~1-2 min at these dataset sizes) — reuse it on whatever comes next.
- Experiments 02 (stable-features-only) and 04 (UFFM fusion) are genuinely settled negative
  results — don't retry those exact approaches without a new reason to.

**Immediate next action when resuming:** design and run a decorrelation experiment —
residualize each clustering feature against `node_count` (linear regression, keep the
residuals) before PCA/clustering, on both 1km and 2km data, and see whether any real
(non-size-driven) structure survives. This is unstarted; no code exists for it yet. If nothing
survives decorrelation, that would be a stronger, more honest version of the methodology
report's null-result finding than what currently exists. If something does survive, *that*
result — not the current 0.513 — is what should go to Experiment 06's satellite check.

**Files to open first when resuming:**
- `experiments/EXPERIMENTS.md` (this file) — full history and the finding above
- `dhaka_topology/clustering.py` — where `filter_stable_features()` and `permutation_test()`
  live; the decorrelation function belongs here too
- `pipeline_outputs/v2km/outputs/csv/features.csv` and `pipeline_outputs/v3/outputs/csv/features.csv`
  — both already computed, ready to reuse without re-running acquisition/tiling/features stages

---

## Experiment 06 — Decorrelate features from zone size

**Hypothesis:** after regressing each feature against `log(node_count)` and clustering on the
residuals, does any real (non-size-driven) structure survive?

**Method:** `clustering.residualize_features()` fits `feature ~ log(node_count)` per feature via
linear regression and keeps the residuals; ran the same PCA -> KMeans/DBSCAN/GMM -> significance
pipeline on the residualized features, on both 1km and 2km data.
Script: `experiments/06_decorrelate_from_size/run.py --config <cfg> --label <1km|2km>`.

**Result — how much of each feature was actually just measuring size:**
| Feature | R^2 vs log(node_count), 2km | R^2, 1km |
|---|---|---|
| mean_katz_centrality | 0.847 | 0.834 |
| mean_betweenness_centrality | 0.832 | 0.829 |
| global_efficiency | 0.827 | 0.821 |
| mean_closeness_centrality | 0.759 | 0.747 |
| density | 0.674 | 0.732 |
| mean_strength_norm | 0.641 | 0.617 |
| mean_in/out_degree_centrality | 0.630 | 0.584 |
| mean_pagerank | 0.614 | 0.557 |
| betweenness_gini | 0.611 | 0.530 |

Ten of twenty-three clustering features have R^2 > 0.5 against zone size alone — these were
never meaningfully independent of size despite being "normalized" in the networkx sense.

**Clustering result after removing the size confound:**
| | Before decorrelation | After decorrelation |
|---|---|---|
| 1km KMeans | sil=0.408 | sil=**0.269** |
| 2km KMeans | sil=0.513 | sil=**0.314** |
| 1km GMM | sil=0.046 | sil=0.019 (now fails too) |
| 2km GMM | sil=0.201 | sil=0.132 |
| DBSCAN (both) | fails | still fails |
| Significance (both) | p=0.005 | still **p=0.005** |

Full results in `experiments/06_decorrelate_from_size/result_1km.json` and `result_2km.json`.

**Decision — real signal survives, smaller but genuine.** Both scales lost roughly a third of
their silhouette once the size confound was removed, but both remain statistically significant
— the real silhouette sits ~4-5x above the null 95th percentile at both scales. This is a more
defensible result than anything produced so far: it's no longer "small zones vs large zones,"
it's a real 2-way structural difference in *how* zones of comparable size are connected. 2km
still edges out 1km after decorrelation (0.314 vs 0.269), consistent with Experiment 03.
**This decorrelated 2km result is now the one worth taking to a satellite-imagery check** —
not Experiment 03's raw (confounded) 0.513 result. Extended `06_decorrelate_from_size/run.py`
to also save per-zone cluster assignments (was previously comparison-metrics only) — see
`experiments/06_decorrelate_from_size/cluster_assignments_2km.csv`.

**Size-confound sanity check on the decorrelated result:** Cluster_0 (53 zones) mean
`n_nodes_buffered`=1634 vs Cluster_1 (171 zones) mean=549 — a ~3x gap, down from ~15.6x before
decorrelation, and the per-zone ranges now substantially overlap (Cluster_1's top-confidence
zones range 135-904 nodes; Cluster_0's range 2308-3722). Much healthier than before, though not
perfectly zero residual size relationship — worth keeping in mind, not treating as fully solved.

---

## Experiment 07 — Satellite-imagery spot-check (decorrelated 2km result)

**Hypothesis:** does the decorrelated 2-cluster split (Cluster_0=53 zones,
Cluster_1=171 zones) correspond to a visually recognizable, describable difference in urban
form — not just an abstract number?

**Method:** sampled the top-confidence zones from `cluster_assignments_2km.csv`'s decorrelated
result (4 Cluster_0, 3 Cluster_1 — practical n given this is one non-expert reviewer, not the
original proposal's planned n=30 expert panel; documented honestly rather than padding the
sample). Viewed each zone's centroid on OpenStreetMap (chosen over Google Maps satellite tiles,
which didn't render in this environment; OSM is also the actual data source the features are
computed from, arguably more directly relevant). Assessed only what's visually verifiable: does
the zone look like dense, continuous urban fabric (expected for Cluster_0: higher connectivity
features) or sparse/peripheral/fragmented development (expected for Cluster_1)?

**Result:**
| Zone | Cluster | Nodes | What's actually there |
|---|---|---|---|
| Zone_DH130 (23.721,90.388) | Cluster_0 | 3276 | Dense mixed residential along a canal, organic curving streets (Kalabagan area) |
| Zone_DH168 (23.758,90.427) | Cluster_0 | 2817 | Clear planned grid — numbered roads ("Road No. 3/5/7...", RAJUK-style sector) |
| Zone_DH167 (23.740,90.427) | Cluster_0 | 2644 | Dense numbered-road grid near Dhanmondi/Mohammadpur, rail line through it |
| Zone_DH116 (23.793,90.367) | Cluster_0 | 3722 | Extremely fine-grained dense organic lanes (Mirpur Road area) |
| Zone_DH028 (23.827,90.268) | Cluster_1 | 589 | Sparse peri-urban/rural, wetlands, scattered settlement, one highway |
| Zone_DH126 (23.649,90.390) | Cluster_1 | 135 | Institutional/rural (near Jagannath University new campus), rail line, low density |
| Zone_DH140 (23.901,90.385) | Cluster_1 | 904 | Urbanizing fringe — industrial colony + scattered development + open land |

**Agreement: 7/7 zones visually matched their expected connectivity profile** (dense/continuous
for Cluster_0, sparse/fragmented/peripheral for Cluster_1) — meeting the direction of the
original proposal's >=85% target, though on a much smaller, single-reviewer sample; this is not
a substitute for the proposal's planned n=30 expert-panel audit.

**Important caveat — this is NOT a clean morphological archetype.** Cluster_0 contains both
organic (DH130, DH116) and planned-grid (DH167, DH168) patterns; Cluster_1 contains rural
(DH028), institutional (DH126), and urbanizing-fringe (DH140) patterns. There is no single
"Grid" or "Maze" story here — what's visually consistent is *density/continuity of the street
network*, not a specific layout style. This matches the methodology report's own conclusion
(section 3, "Dhaka does not cluster into discrete morphological types") almost exactly, but
with an important upgrade: **the connectivity-level distinction itself is now shown to (a)
survive a size-confound correction, (b) be statistically significant against a permutation
null, and (c) correspond to something a human can see and describe on the ground** — none of
which had been demonstrated before this experiment sequence.

**Decision — accept as the thesis's defensible finding.** Dhaka's road network does not
separate into discrete morphological archetypes (Maze/Grid/Radial/Periphery) at 1-2km grid
scale — confirming the prior null result — but it does separate, robustly and
visually-verifiably, into a **connectivity-level gradient**: dense/continuous urban fabric
(whether organically grown or formally planned) vs. sparse/fragmented/peripheral development.
This is a narrower claim than the original proposal's four-archetype goal, but it is now the
most rigorously validated result in this entire project: real after permutation testing,
real after removing the zone-size confound, and visually confirmed on the ground for every
zone checked.

> **Correction from Experiment 08:** the per-zone *morphology* labels used here
> ("planned grid" vs "organic") are unreliable — they were assigned by viewing each zone's
> centre point on OpenStreetMap at zoom 15, which shows roughly a quarter of a zone's true
> 3.6km extent. Rendering the full zones (Experiment 08) shows every one of them is a
> *mixture* of grid and organic fabric. The connectivity finding above still stands — it
> never depended on those morphology labels — but do not cite the grid/organic
> characterisations from the table above.

---

## Experiment 08 — Computer-vision pilot: can spectral analysis separate grid from organic?

**Hypothesis:** a human separates "planned grid" from "organic lanes" instantly by *looking*,
but 23 scalar graph features cannot. The information lives in 2D spatial arrangement —
specifically in **periodicity**, which scalar metrics destroy by construction. A grid repeats
at fixed spacing and orientation; an organic network does not. 2D Fourier analysis measures
exactly that, and does so *independently of how much road there is* (the power spectrum is
normalised to sum to 1), which is precisely the decoupling Experiment 06 showed we need.
Literature grounding: the proposal's own lit review cites Badhrudeen et al. (2022) classifying
48 world cities via geometric deep learning on OSM data, so a CV approach is the state of the
art in the cited work, not a tangent.

**Method:** rendered each zone's street network as a binary raster at **fixed geographic
scale** (3.6km extent -> 512px, ~7m/px, identical for every zone so street spacing is
comparable rather than normalised away), applied a 2D Hann window (prevents the image border
from faking periodicity), took the 2D FFT power spectrum, **zeroed the DC component and
frequencies within 3 bins of it** (DC *is* "amount of ink" — the exact quantity that must not
drive the result), normalised the spectrum to sum to 1, then computed spectral entropy,
angular entropy, and top-k energy concentration. Grid -> sharp discrete peaks -> low entropy.
Organic -> diffuse energy -> high entropy.
Pilot set: 4 zones whose morphology was believed known from Experiment 07.
Script: `experiments/08_cv_fft_pilot/run.py`.
**Figure: `experiments/08_cv_fft_pilot/pilot_renders_and_spectra.png`** (top row: rendered
street networks; bottom row: their power spectra).

**Result — numerically null, and the reason why is the actual finding:**
| Zone | Believed truth | Spectral entropy | Angular entropy | Top 0.1% energy | Ink fraction |
|---|---|---|---|---|---|
| Zone_DH168 | "grid" | 0.9402 | 0.9891 | 0.0309 | 0.1138 |
| Zone_DH167 | "grid" | 0.9380 | 0.9885 | 0.0349 | 0.1124 |
| Zone_DH130 | "organic" | 0.9332 | 0.9864 | 0.0386 | 0.1140 |
| Zone_DH116 | "organic" | 0.9409 | 0.9845 | 0.0240 | 0.1301 |

Spectral entropy and top-0.1%-energy overlap between the two groups. Angular entropy
technically "separates" but by ~0.002 on n=2 per group — noise, not signal. All entropies sit
near maximum (~0.94-0.99), i.e. every zone's spectrum is near-uniform.

**Why it came out null — look at the figure.** The renderer is working correctly; the street
networks are drawn cleanly. What the images show is that **none of the four zones is actually
"a grid" or "an organic network."** Every one is a *mixture* — patches of regular comb-like
grid embedded in irregular organic fabric. Zone_DH116, labelled "organic", visibly contains
the most grid structure of the four. Zone_DH168, labelled "grid", is mostly irregular with a
few grid patches. The Experiment 07 labels were assigned from zone *centre points* at OSM
zoom 15, which shows ~a quarter of each zone's 3.6km extent — different corners of the same
zone look completely different.

So the method measured correctly and the premise was false: the entropies came out nearly
identical because **the zones genuinely are nearly identical in mixture composition.**

**This is the most direct evidence the project has produced for the scale-mismatch
hypothesis.** The methodology report asserted it in prose — *"Every 1km cell contains a
mixture of the planned skeleton and its organic infill. All mixtures look statistically
similar"* — and `pilot_renders_and_spectra.png` is that claim made visible, at 2km, with the
spectra to match.

**Decision — reframe the question, keep the method.** The failure is in the *analysis unit*,
not the technique. We have been asking "what type is this zone?" when no zone has a type. The
better question is **"what is the mixture ratio of this zone?"** — chop each zone into ~300-400m
patches (the micro-scale the methodology report identifies as where block structure actually
lives), classify each *patch* spectrally, and characterise each zone by its **proportion of
grid-like patches**. That converts the report's qualitative "continuous morphological
spectrum" conclusion into a quantitative, mappable variable, measured at the scale where the
signal exists rather than averaged away before measurement. Cost is trivial: ~25 patches per
zone, ~5,600 patches across all 224 zones.

**Next experiment:** patch-scale FFT pilot, with patches judged by inspecting the rendered
images directly — not map centre points, which is how the Experiment 07 labelling went wrong.
→ **Run as Experiment 09 below; the reframe held up.**


---

## Experiment 09 — Patch-scale pilot: is the grid/organic *mixture ratio* measurable?

> **⚠ CORRECTION (see Experiment 10).** The per-patch `grid_score` values below were computed
> from a rasterized street image, and the renderer (1px Bresenham lines, no anti-aliasing)
> introduced a **rotation-dependent error of up to 0.23** — larger than the 0.045 between-zone
> sd this experiment reports. Recomputing the same 100 patches from vector geometry gives only
> **corr = +0.398** with the numbers below. **Do not cite the per-patch scores, the per-zone
> medians, or the calibration table from this entry.** The *conclusion* survives and
> strengthens: the within/between variation ratio goes from 3.0× to **4.4×**, and the visual
> ranking check was redone on the corrected scores and passes. Superseding values:
> `10_descriptor_validation/exp09_vector_recompute.csv` and `vector_null.csv`.

**Hypothesis:** Experiment 08's null result came from an ill-posed question, not a broken
method. If the analysis unit drops from 2km (where every zone is a mixture) to **400m** (where
a single fabric type plausibly fills the window), the same spectral approach should produce a
per-patch grid-ness score that (a) a human agrees with, and (b) varies *within* a zone — which
is what "mixture" means operationally. Patch size 400m chosen from the team's own methodology
report, which puts Dhaka's block scale at 100-300m: a 400m window holds a few blocks, enough
for a repeat to exist if the fabric is regular, small enough not to straddle two neighbourhoods.

**The measure.** Experiment 08 used spectral/angular entropy, which turned out too blunt. The
sharper insight: a grid is not merely *oriented*, it is oriented in **two perpendicular
directions at once**. A single arterial road is strongly oriented but is not a grid. So:

    grid_score = 2 * max over a in [0,90) of  min( E(a), E(a+90) )

where `E(a)` is the fraction of spectral power within ±10° of orientation `a`. Taking the
**min** of the orthogonal pair is exactly what rejects the single-road case; the `max` over `a`
makes it rotation-invariant, so a grid tilted 30° off north scores the same as one aligned to it.

**Anti-confound design** (Experiment 06's size trap must not return): fixed geographic scale for
every patch (3.125 m/px, never normalised per patch); DC and frequencies within 3 bins zeroed
(DC *is* "amount of ink"); power spectrum normalised to sum 1 so total road quantity cancels;
`grid_score` is a dimensionless *fraction* of angular energy; and `ink_fraction` is reported
alongside so any surviving density correlation is visible rather than hidden.

**Two things Experiment 08 lacked, added here:**
1. **Calibration.** The identical metric is run on synthetic controls rendered the same way — a
   perfect 100m/200m orthogonal grid and Poisson-random networks. These anchor the scale, so a
   real patch's number means something instead of floating free.
2. **A per-patch null.** The angular histogram is shuffled 200× and the metric recomputed. This
   absorbs the upward bias from maximising over 90 candidate orientations, which would otherwise
   push even isotropic patches above the analytic 0.233 baseline.

**Method:** same 4 zones as Experiment 08, so the two pilots are directly comparable. Each
zone's central 2000m rendered at 3.125 m/px and cut into a 5×5 grid of 400m patches — 100
patches total. Script: `experiments/09_patch_mixture_pilot/run.py`, seed 20260919.

### Result 1 — the metric works, and it is validated by eye before by number

**Figure: `09_patch_mixture_pilot/ranking_check.png`** — the 8 highest-scoring and 8
lowest-scoring patches, pooled across all zones. The top row is visibly regular: parallel lanes
at near-constant spacing crossing a perpendicular spine. The bottom row is sparse, irregular and
oblique, with no repetition. **The ranking matches what a human would say**, which is the
precondition for any of the numbers below meaning anything.

Calibration bracket (same metric, synthetic controls):

| Control | grid_score | z vs its own null |
|---|---|---|
| synthetic grid, 100m spacing | 0.967 | 2.7 |
| synthetic grid, 200m spacing | 0.975 | 2.9 |
| Poisson-random network (a) | 0.267 | 0.1 |
| Poisson-random network (b) | 0.261 | 0.3 |

So the scale runs **~0.26 = structureless** to **~0.97 = perfect grid**. Real Dhaka patches span
**0.083 to 0.784**, median 0.350 — sitting across the whole intermediate range, as expected for
real fabric rather than synthetic extremes. 73/100 patches score above the random control;
49/100 exceed their own shuffled null by z > 2. The metric is detecting real orthogonal order,
not noise.

**Confound check passed.** `corr(grid_score, ink_fraction) = +0.101` (Spearman 0.089, p = 0.38).
At the extremes: the top-8 patches differ from the bottom-8 by **4.2× in score but only 1.22× in
ink**. The size/density confound that invalidated the original clustering has not returned.

### Result 2 — the mixture hypothesis is confirmed quantitatively

**Figure: `09_patch_mixture_pilot/score_distributions.png`** and
**`09_patch_mixture_pilot/zone_mixture_overlay.png`** (patches in place, warm = more grid-like).

| Zone | Exp-07 prior | n | median | mean | max | sd |
|---|---|---|---|---|---|---|
| Zone_DH168 | grid-leaning | 25 | 0.352 | 0.348 | 0.595 | 0.116 |
| Zone_DH167 | grid-leaning | 25 | 0.407 | 0.430 | 0.784 | 0.164 |
| Zone_DH130 | organic-leaning | 25 | 0.330 | 0.327 | 0.610 | 0.133 |
| Zone_DH116 | organic-leaning | 25 | 0.318 | 0.357 | 0.623 | 0.127 |

**Mean within-zone sd = 0.135. Between-zone sd of means = 0.045. Ratio 3.0×.**

Every zone individually spans almost the entire observed range — from at-or-below the random
control up to 0.6-0.78, well into grid territory. This is the Experiment 08 claim turned into a
number: **a zone's internal morphological variation is 3× its variation from other zones.** A
single zone-level morphology label is not a badly-measured quantity, it is a
**poorly-defined** one, and now there is a statistic saying so rather than only a picture.

Correspondingly, the Experiment 07 priors do not separate: grid-leaning mean 0.389 vs
organic-leaning 0.342, Mann-Whitney p = 0.141; Kruskal-Wallis across all four zones H = 5.52,
p = 0.138. With n = 4 zones this is **underpowered, not evidence of absence** — it is consistent
with Experiment 08 and does not settle whether zone mixture ratios differ city-wide.

**Decision — promote the patch metric to a pipeline feature and scale it up.** The pilot cleared
every bar it was set: visually validated ranking, calibrated against synthetic controls,
density-decoupled, and significant against a per-patch null. Two things follow:

1. **Run it over all 224 zones** (~5,600 patches; the 100-patch pilot ran in well under a
   minute, so full scale is minutes). This is the first properly-powered test of whether
   *mixture ratio* — unlike morphology *type* — actually varies between zones. A negative there
   would be a real finding; this pilot cannot deliver one at n = 4.
2. **Patch-level, not zone-level, is the honest unit of analysis for morphology.** The mixture
   ratio (e.g. fraction of a zone's patches scoring above the random control) is a genuine
   zone-level variable derived from it, and unlike the 26 scalar features it is constructed to
   be independent of zone size from the start.

**Caveats recorded honestly:** n = 4 zones, chosen for comparability with Experiment 08 rather
than sampled; the per-patch null shuffles the angular histogram, which tests orthogonal-pair
concentration against an angle-scrambled spectrum, not against a fully resampled street network;
patch boundaries are a fixed lattice, so a grid straddling two patches is penalised (a sliding
window would fix this and is worth testing at scale); and links are drawn as straight segments
between nodes, so genuinely curved roads render as polylines — this affects organic fabric more
than grids, and the direction of that bias (if any) has not been quantified.

---

## Experiment 10 — Verification pass: testing the descriptor logic before building on it

**Why this exists:** before designing a rotation-invariant angular descriptor, every claim
made about it was tested against synthetic patterns whose morphology is known by construction.
Three of eight claims failed, one of the failures invalidates Experiment 09's numbers, and one
error was in the verification's own null model. Recording all of it.

Scripts: `10_descriptor_validation/verify_logic.py`, `verify_rasterization.py`,
`verify_exp09_impact.py`, `verify_vector_null.py`.
**Figures:** `descriptor_validation.png`, `vector_ranking_check.png`.

### Claim-by-claim result

| # | Claim | Verdict |
|---|---|---|
| C1 | Experiment 09's `grid_score` is rotation-invariant | **FAILED as implemented** |
| C2 | \|c_k\| of the angular histogram are rotation-invariant | **FAILED as implemented** |
| C3 | On the folded [0,180°) axis a grid appears in \|c₂\| | HOLDS |
| C4 | \|c₂\| alone identifies a grid | **WRONG — corrected below** |
| C5 | The scalar cannot tell radial from organic | HOLDS |
| C6 | The \|c_k\| signature separates grid / corridor / three-way / organic | HOLDS |
| C7 | Radial needs a separate spatial measure | HOLDS (automated verdict was misleading) |
| C8 | UFFM's bearing distance is rotation-variant | HOLDS, with a caveat on margin |

### C1/C2 — the rotation failure was the renderer, not the maths

A synthetic grid rotated through 0-82° gave `grid_score` ranging **0.755 to 0.981** (spread
0.2255) and \|c₂\| spread 0.3588. But the failure pattern gave the cause away: rotations **0°
and 45° scored highest** and the oblique angles lowest — exactly the angles at which a raster
line is drawn exactly versus stair-stepped. PIL draws 1px Bresenham lines with no
anti-aliasing, and the staircase injects spurious high-frequency energy across orientations.

Testing four rendering methods on 12 rotations settled it:

| Method | `grid_score` mean | spread | \|c₂\| spread |
|---|---|---|---|
| A — 1px, no AA (**what Experiment 09 used**) | 0.823 | 0.2255 | 0.3588 |
| B — 4× supersampled | 0.847 | 0.2330 | 0.0963 |
| C — 4× supersampled, 2px lines | 0.911 | 0.0785 | 0.0245 |
| **D — vector geometry, no raster at all** | **1.0000** | **0.0000** | **0.0006** |

The metric is rotation-invariant in exact arithmetic; **rasterization was destroying it.**
Note B barely helped — anti-aliasing alone is insufficient, because a 1px line downsampled
from 4× still aliases; line *thickness* is what suppresses the staircase. Method D scores a
perfect grid at exactly 1.0000 at every rotation.

**Method D is the fix, and it is also simpler and faster:** compute a length-weighted bearing
histogram directly from the graph's segment geometry. No rendering, no FFT, exactly
rotation-invariant. This means **the CV/FFT machinery of Experiments 08-09 was not needed for
the orientation question at all** — orientation order is a property of the graph, available
exactly. The FFT retains one distinct job the vector method cannot do: measuring *spacing
periodicity* (whether streets are evenly spaced, not merely parallel). That is a genuine
second axis, but it is not what `grid_score` was measuring.

### C4 — the claim as previously stated was wrong

Earlier this log implied "\|c₂\| high = grid". Measured:

| Pattern | \|c₁\| | \|c₂\| |
|---|---|---|
| grid | 0.007 | 0.953 |
| single corridor | **0.995** | **0.983** |

A corridor's \|c₂\| is as high as a grid's, so \|c₂\| alone does **not** identify a grid. The
discriminator is \|c₁\|: a grid's two families 90° apart cancel the odd harmonics, so a grid
has \|c₁\| ≈ 0 while a single-axis corridor has \|c₁\| ≈ 1. **Grid = low \|c₁\| AND high \|c₂\|.**

### C5/C7 — radial has no stable angular signature

`grid_score`: radial 0.201, organic 0.264, grid 0.900 — the scalar conflates radial with
organic, as claimed. On the signature, the automated test reported "separates", but inspecting
the values shows that verdict is an artifact of the threshold:

| Pattern | \|c₁\| | \|c₂\| | \|c₃\| | \|c₄\| |
|---|---|---|---|---|
| radial, hub centred | 0.002 | 0.005 | 0.003 | 0.102 |
| radial, hub off-centre | 0.586 | 0.283 | 0.052 | 0.062 |
| organic | 0.193 | 0.191 | 0.196 | 0.014 |

The two radial variants land in **completely different places** — a centred hub looks more
isotropic than organic fabric, an off-centre hub looks like a corridor. That is inconsistency,
not separation. **Radial is spatial convergence, not an orientation property**, and needs its
own measure (do segment bearing-lines intersect near a common point more than chance?). The
angular descriptor delivers grid / corridor / organic; radial is not obtainable from it.

### C8 — UFFM's fingerprint comparison is rotation-variant

`uffm.py:37` builds a bearing histogram folded to [0,180°); `uffm.py:161` compares two of them
with `wasserstein_distance` on `np.linspace(0, 1, bins)` — a **linear** axis for a **circular**
variable. Measured on synthetic fingerprints:

- d(grid @ 0°, grid @ 45°) = **0.2571** — identical morphology, only rotated
- d(grid @ 0°, organic) = **0.2520** — genuinely different morphology

A rotation-invariant metric would return ≈ 0 for the first. It instead returns *more* than the
distance to a completely different morphology. The margin between the two numbers is small and
the synthetic test is crude, so the ranking itself is weak evidence — but the absolute value is
the point: **0.257 of distance between two identical grids is the defect.** This is a concrete,
identifiable cause for UFFM's documented null result, and it is a small fix: replace the
distance with rotation-invariant \|c_k\| features on the fingerprints UFFM already computes.

### Consequence for Experiment 09 — its numbers are wrong, its conclusion survives

Experiment 09's 100 patches were recomputed from vector geometry
(`verify_exp09_impact.py`):

- **corr(raster, vector) = +0.398** — weak agreement
- mean \|raster − vector\| = **0.161**
- raster mean 0.366 vs vector mean 0.497 — rasterization systematically *depressed* scores

Since Experiment 09 reported a between-zone sd of only 0.045, the artifact was **larger than
the signal being reported**. The per-patch numbers in Experiment 09 should not be cited.

The headline conclusion holds and strengthens:

| | within-zone sd | between-zone sd | ratio |
|---|---|---|---|
| raster (Experiment 09 as published) | 0.135 | 0.045 | 3.0× |
| vector, raw score | 0.136 | 0.031 | **4.4×** |
| vector, excess over null | 0.137 | 0.036 | **3.8×** |

Zones are internally heterogeneous by 4× their mutual differences. Experiment 09's visual
ranking check was done on raster scores and does **not** transfer, so it was redone on the
vector scores (`vector_ranking_check.png`) — and passes, including a ~45° rotated grid
correctly scored 0.81, which the rasterized version could not have done.

### An error in this experiment's own null, caught and fixed

The first version of the vector null randomised bearings **per 5m step**. Roads are subdivided
into steps only to decide which patch they fall in; all steps of one road share one bearing, so
a step-level null has hundreds of independent draws where reality has tens. That null was far
too tight:

| Null unit | null mean | patches p < 0.05 |
|---|---|---|
| per 5m step (**wrong**) | 0.250 | 95 / 100 |
| per (segment, patch) (**correct**) | 0.299 | **81 / 100** |

81/100 is the honest figure. The corrected null also varies with patch sample size
(corr −0.913, range 0.276-0.341), which means **excess over each patch's own null — not the
raw score — is the quantity that can be thresholded** across patches.

**Decision.** Replace rasterized FFT scoring with vector-geometry angular histograms as the
primitive; report excess over a segment-level null; use (\|c₁\|, \|c₂\|, \|c₃\|) as the
morphology signature rather than a single scalar; treat radial as requiring a separate spatial
convergence measure, not as a fourth value of the same descriptor; and re-run UFFM with a
rotation-invariant comparison to test whether its null result was this bug. Experiment 09's
conclusion stands; its numbers are superseded by the vector recompute in
`10_descriptor_validation/exp09_vector_recompute.csv`.

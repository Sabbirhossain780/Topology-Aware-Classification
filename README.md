# Topology-Aware Classification of Dhaka's Road Network

MSc thesis work (CSE, BUET) on classifying Dhaka's road network into
topology types and identifying structurally critical intersections, as a
basis for targeting transport policy interventions by zone rather than
city-wide.

## What's in this repo

```
02_dhaka_road_topology/
  figures/          key result figures (node hierarchy, betweenness explainer)
  pipeline/          the actual project — code, tests, and a full run's output
```

Everything that matters is in **[`02_dhaka_road_topology/pipeline/`](02_dhaka_road_topology/pipeline/README.md)**:
a config-driven pipeline (feature engineering -> clustering -> betweenness
centrality -> geometric fingerprint matching) that replaced six overlapping
notebooks, validated to reproduce their output exactly, plus the results
of running it over all 827 zones.

**Read [`02_dhaka_road_topology/pipeline/README.md`](02_dhaka_road_topology/pipeline/README.md)
for the full picture** — what each stage does, what the actual clustering
results are, and an honest account of where those results are currently
weak.

## Status

Work in progress. The pipeline is validated against the original analysis;
the underlying clustering result is not yet strong enough to treat as a
finding — see the pipeline README's Known Limitations section.

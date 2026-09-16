#!/usr/bin/env python
"""CLI entry point for the Dhaka topology pipeline.

Examples
--------
Run the full v3 pipeline:
    python run_pipeline.py --config configs/v3.yaml --stage all

Run just feature engineering + clustering (re-using existing zone data):
    python run_pipeline.py --config configs/v3.yaml --stage features cluster

Run the 2km grid experiment:
    python run_pipeline.py --config configs/v2km.yaml --stage all

Force re-download of the OSM graph:
    python run_pipeline.py --config configs/v3.yaml --stage acquire --force
"""
from __future__ import annotations

import argparse
import logging
import sys

from dhaka_topology.config import Config
from dhaka_topology.pipeline import STAGES, run_full_pipeline, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Dhaka road topology pipeline")
    parser.add_argument("--config", required=True, help="Path to a config YAML (see configs/)")
    parser.add_argument("--stage", nargs="+", default=["all"],
                         choices=STAGES + ["all"],
                         help="Which stage(s) to run. 'all' runs the full pipeline in order.")
    parser.add_argument("--force", action="store_true",
                         help="Force re-download/recompute even if cached files exist")
    parser.add_argument("--base-dir", default=None,
                         help="Override the config's base_dir (where data/outputs live)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Config.from_yaml(args.config)
    if args.base_dir:
        config.base_dir = args.base_dir

    stages = STAGES if "all" in args.stage else args.stage
    run_full_pipeline(config, stages=stages, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

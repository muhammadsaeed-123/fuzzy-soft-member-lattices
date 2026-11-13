#!/usr/bin/env python3
"""Regenerate the controlled synthetic-benchmark figures from packaged CSV tables.

This lightweight command does not rerun simulation or bootstrapping.  It reads
machine-readable outputs produced by ``run_synthetic_benchmark.py`` and rebuilds
the four publication figures, allowing the reference-mode smoke test to verify
all manuscript artwork without external data or a lengthy refit.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from run_synthetic_benchmark import figures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        type=Path,
        default=root / "outputs" / "synthetic" / "tables",
        help="Directory containing packaged synthetic CSV tables.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "outputs" / "synthetic" / "figures",
        help="Directory for regenerated PDF and PNG figures.",
    )
    args = parser.parse_args()

    comparison = pd.read_csv(args.tables / "method_comparison.csv")
    baseline = pd.read_csv(args.tables / "baseline_path.csv")
    recurrence = pd.read_csv(args.tables / "node_recurrence.csv")
    scaling = pd.read_csv(args.tables / "scaling.csv")

    baseline_path = [ast.literal_eval(value) for value in baseline["node"]]
    node_frequency = {
        ast.literal_eval(row.node): float(row.frequency)
        for row in recurrence.itertuples(index=False)
    }

    args.out.mkdir(parents=True, exist_ok=True)
    figures(args.out, comparison, node_frequency, baseline_path, scaling)
    print(f"Regenerated synthetic figures in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

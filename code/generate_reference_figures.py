#!/usr/bin/env python3
"""Regenerate empirical manuscript figures from packaged machine-readable tables.

This command is useful for checking plotting code without downloading the 96 MB
transaction file.  The full analysis command remains the authoritative way to
recompute the tables and the complete-refit bootstrap histogram from raw data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_utils import ensure_output, parse_tuple, save_figure

plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})


def load_table(tables: Path, name: str) -> pd.DataFrame:
    path = tables / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference table: {path}")
    return pd.read_csv(path)


def coordination_map(tables: Path, out: Path) -> None:
    df = load_table(tables, "coordination_map_audit")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(df))
    ax.scatter(df["Training exact coordination rate"], y, label="Training", marker="o")
    ax.scatter(df["Holdout exact coordination rate"], y, label="Holdout", marker="s")
    ax.set_yticks(y, df["Map"])
    ax.set_xlabel("Three-layer exact coordination rate")
    ax.grid(axis="x", alpha=.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out, "coordination_map_audit")


def closure(tables: Path, out: Path) -> None:
    df = load_table(tables, "closure_audit")
    # Earlier reference tables used ordered pairs including diagonals.  Convert
    # those rates to the mathematically cleaner unordered-distinct convention.
    if "Ambient join nonclosure rate" in df.columns:
        join_rate = df["Ambient join nonclosure rate"].to_numpy(float)
        meet_rate = df["Ambient meet nonclosure rate"].to_numpy(float)
    else:
        n = df["Nodes"].to_numpy(float)
        factor = np.where(n > 1, n / (n - 1), 1.0)
        join_rate = df["Join violation rate"].to_numpy(float) * factor
        meet_rate = df["Meet violation rate"].to_numpy(float) * factor
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = np.arange(len(df)); w = .36
    ax.bar(x-w/2, join_rate, w, label="Ambient join")
    ax.bar(x+w/2, meet_rate, w, label="Ambient meet")
    ax.set_xticks(x, ["Full pullback", "Marginal", "Joint"])
    ax.set_ylabel("Ambient-operation nonclosure rate")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, "closure_audit")


def path_heatmap(tables: Path, out: Path) -> None:
    df = load_table(tables, "optimal_path")
    rows = []
    for _, row in df.iterrows():
        c = parse_tuple(str(row["Commercial"]))
        o = parse_tuple(str(row["Logistics"]))
        m = parse_tuple(str(row["Market"]))
        rows.append([
            (c[0]-1)/3, (c[1]-1)/3, (c[2]-1)/3,
            (o[0]-1)/3, (o[1]-1)/3, o[2]-1,
            (m[0]-1)/4, (m[1]-1)/4, (m[2]-1)/3,
        ])
    matrix = np.asarray(rows).T
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(9), ["Unit value", "Discount preservation", "Order scale", "Actual speed", "Planned speed", "Reliability", "Net revenue", "Order profit", "Profit margin"])
    ax.set_xticks(range(len(df)), range(1, len(df)+1))
    ax.set_xlabel("Ordered-state stage")
    fig.colorbar(image, ax=ax, label="Normalized lattice level")
    fig.tight_layout()
    save_figure(fig, out, "optimal_path_heatmap")


def utility(tables: Path, out: Path) -> None:
    df = load_table(tables, "optimal_path")
    fig, ax = plt.subplots(figsize=(8.1, 4.5))
    stage = df["Stage"]
    ax.plot(stage, df["Quality"], marker="o", label="Fuzzy quality")
    ax.plot(stage, df["Evidence"], marker="s", label="Joint evidence")
    ax.plot(stage, df["Node utility"], marker="^", label="Combined utility")
    ax.set_xlabel("Stage"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    save_figure(fig, out, "path_utility_progression")


def external_bootstrap(tables: Path, out: Path) -> None:
    df = load_table(tables, "external_outcome_bootstrap")
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    y = np.arange(len(df))
    center = df["Point difference"].to_numpy(float)
    lower = np.maximum(0.0, center - df["CI low"].to_numpy(float))
    upper = np.maximum(0.0, df["CI high"].to_numpy(float) - center)
    ax.errorbar(center, y, xerr=np.vstack([lower, upper]), fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y, df["Comparison"])
    ax.set_xlabel("Difference in mean signed order-status score")
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    save_figure(fig, out, "external_outcome_bootstrap")


def rolling(tables: Path, out: Path) -> None:
    df = load_table(tables, "rolling_origin")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(df))
    ax.plot(x, df["Full outcome"], marker="o", label="Full model")
    ax.plot(x, df["Quality-only outcome"], marker="s", label="Quality-only")
    ax.set_xticks(x, df["Test fraction"])
    ax.set_xlabel("Rolling holdout interval within each market")
    ax.set_ylabel("Mean signed order-status score")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    save_figure(fig, out, "rolling_origin")


def operators(tables: Path, out: Path) -> None:
    df = load_table(tables, "operator_sensitivity")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(df["Operator"], df["Node Jaccard with baseline"])
    ax.set_ylabel("Node Jaccard with geometric baseline")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    save_figure(fig, out, "operator_sensitivity")


def full_refit(tables: Path, out: Path, reference_raster: Path | None = None) -> None:
    replicate_path = tables / "full_refit_bootstrap.csv"
    if replicate_path.exists():
        df = pd.read_csv(replicate_path)
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        values = df["Node Jaccard"].dropna()
        ax.hist(values, bins=np.linspace(0,1,21), linewidth=0.4)
        median = float(values.median())
        ax.axvline(median, linestyle="--", label=f"Median = {median:.3f}")
        ax.set_xlabel("Node Jaccard with fitted path")
        ax.set_ylabel("Full-pipeline bootstrap replicates")
        ax.legend(); ax.grid(axis="y", alpha=.2); fig.tight_layout()
        save_figure(fig, out, "full_refit_stability")
        return

    # No replicate-level values are invented.  The packaged summary is enough
    # to generate a different, fully reproducible summary plot.
    summary = load_table(tables, "full_refit_summary").iloc[0]
    labels = ["Exact path\nrecovery", "Median node\nJaccard", "Median edge\nJaccard"]
    values = [
        float(summary["Exact recovery rate"]),
        float(summary["Median node Jaccard"]),
        float(summary["Median edge Jaccard"]),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion / similarity")
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.025, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, out, "full_refit_stability")


def coverage_limitations(tables: Path, out: Path) -> None:
    df = load_table(tables, "validation")
    dataset = load_table(tables, "dataset_summary")
    total = int(float(dataset.loc[dataset["Item"] == "Chronological holdout orders", "Value"].iloc[0]))
    near_col = "Radius-one holdout orders" if "Radius-one holdout orders" in df.columns else "Near-path orders"
    values = 100.0 * df[near_col].to_numpy(float) / total
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(df["Method"], values)
    ax.set_ylabel("Radius-one coverage of complete holdout (%)")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.04, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out, "coverage_limitations")


def generate_all(tables: Path, out: Path) -> None:
    ensure_output(out)
    for draw in (coordination_map, closure, path_heatmap, utility,
                 external_bootstrap, rolling, operators, coverage_limitations,
                 full_refit):
        draw(tables, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--tables", type=Path, default=repo/"outputs"/"reference"/"tables")
    parser.add_argument("--out", type=Path, default=repo/"reference_outputs"/"generated_empirical")
    parser.add_argument("--reference-refit-raster", type=Path, default=repo/"outputs"/"reference"/"manuscript_rasters"/"main_article"/"fig12_refit.png")
    args = parser.parse_args()
    out = ensure_output(args.out)
    generate_all(args.tables, out)
    print(f"Generated reference-table figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

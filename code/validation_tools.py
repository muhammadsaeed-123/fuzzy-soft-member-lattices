"""Validation utilities for the common-protocol analysis."""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from lattice_model import GammaNode, compatible_tuple_series, node_distance


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return Spearman statistic and p-value without constant-input warnings."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def evidence_on_frozen_training_scale(
    evaluation_counts: Mapping[GammaNode, int],
    training_max_count: int,
) -> Callable[[GammaNode], float]:
    """Normalize evaluation support with the training denominator.

    Values are capped at one so the utility remains on its declared [0,1]
    scale even if a holdout node is more frequent than the most frequent
    training node.
    """
    denominator = math.log1p(max(int(training_max_count), 1))

    def score(node: GammaNode) -> float:
        return min(1.0, math.log1p(int(evaluation_counts.get(node, 0))) / denominator)

    return score


def compatible_distance_frame(
    frame: pd.DataFrame,
    paths: Mapping[str, Sequence[GammaNode]],
    maps=None,
) -> pd.DataFrame:
    tuples = compatible_tuple_series(frame, maps)
    rows = []
    for idx, node in tuples.items():
        if node is None:
            continue
        row = {
            "row_index": idx,
            "order_id": frame.at[idx, "Order Id"] if "Order Id" in frame else idx,
            "market": frame.at[idx, "market"],
            "customer_id": frame.at[idx, "customer_id"],
            "order_status": frame.at[idx, "order_status"],
            "signed_status_score": float(frame.at[idx, "status_outcome"]),
        }
        for name, path in paths.items():
            row[f"distance::{name}"] = min(node_distance(node, p) for p in path)
        rows.append(row)
    return pd.DataFrame(rows)


def distance_status_association(distance_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in distance_frame.columns:
        if not column.startswith("distance::"):
            continue
        name = column.split("::", 1)[1]
        x = -distance_frame[column].to_numpy(float)
        y = distance_frame["signed_status_score"].to_numpy(float)
        statistic, pvalue = _safe_spearman(x, y)
        rows.append(
            {
                "Method": name,
                "Compatible holdout orders": len(distance_frame),
                "Spearman correlation of negative distance with signed status": statistic,
                "Two-sided p-value": pvalue,
            }
        )
    return pd.DataFrame(rows)


def market_distance_summary(distance_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize common-cohort path proximity within each market."""
    rows: list[dict] = []
    for market, group in distance_frame.groupby("market", sort=True):
        y = group["signed_status_score"].to_numpy(float)
        for column in [c for c in group.columns if c.startswith("distance::")]:
            method = column.split("::", 1)[1]
            x = -group[column].to_numpy(float)
            statistic, _ = _safe_spearman(x, y)
            radius_one = group.loc[group[column] <= 1]
            rows.append(
                {
                    "Market": market,
                    "Method": method,
                    "Compatible holdout orders": len(group),
                    "Spearman correlation of negative distance with signed status": statistic,
                    "Radius-one orders": len(radius_one),
                    "Radius-one mean signed status score": (
                        float(radius_one["signed_status_score"].mean()) if len(radius_one) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_distance_association(
    distance_frame: pd.DataFrame,
    replicates: int = 2000,
    seed: int = 20260819,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Customer-cluster bootstrap for common-cohort distance associations.

    Customers are resampled with replacement within each market.  Every method
    is evaluated on the same resampled exactly coordinated holdout cases.  The
    function returns percentile-interval summaries and replicate-level values.
    """
    distance_columns = [c for c in distance_frame.columns if c.startswith("distance::")]
    if distance_frame.empty or not distance_columns or replicates <= 0:
        summary_columns = [
            "Method", "Compatible holdout orders", "Point Spearman correlation",
            "Bootstrap CI low", "Bootstrap CI high", "Requested replicates",
            "Valid replicates", "Resampling unit",
        ]
        replicate_columns = ["Replicate", "Method", "Spearman correlation"]
        return pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=replicate_columns)

    rng = np.random.default_rng(seed)
    blocks: dict[str, list[np.ndarray]] = {}
    for market, market_frame in distance_frame.groupby("market", sort=True):
        blocks[str(market)] = [
            group.index.to_numpy()
            for _, group in market_frame.groupby("customer_id", sort=False)
        ]

    replicate_rows: list[dict] = []
    for b in range(replicates):
        sampled_indices: list[object] = []
        for market in sorted(blocks):
            clusters = blocks[market]
            choices = rng.integers(0, len(clusters), size=len(clusters))
            for choice in choices:
                sampled_indices.extend(clusters[int(choice)].tolist())
        sample = distance_frame.loc[sampled_indices]
        y = sample["signed_status_score"].to_numpy(float)
        for column in distance_columns:
            method = column.split("::", 1)[1]
            x = -sample[column].to_numpy(float)
            statistic, _ = _safe_spearman(x, y)
            replicate_rows.append(
                {
                    "Replicate": b + 1,
                    "Method": method,
                    "Spearman correlation": statistic,
                }
            )

    replicate_table = pd.DataFrame(replicate_rows)
    summary_rows: list[dict] = []
    y_point = distance_frame["signed_status_score"].to_numpy(float)
    for column in distance_columns:
        method = column.split("::", 1)[1]
        x_point = -distance_frame[column].to_numpy(float)
        point, _ = _safe_spearman(x_point, y_point)
        values = replicate_table.loc[
            replicate_table["Method"] == method, "Spearman correlation"
        ].dropna()
        low, high = (
            (float(values.quantile(.025)), float(values.quantile(.975)))
            if len(values) else (np.nan, np.nan)
        )
        summary_rows.append(
            {
                "Method": method,
                "Compatible holdout orders": len(distance_frame),
                "Point Spearman correlation": point,
                "Bootstrap CI low": low,
                "Bootstrap CI high": high,
                "Requested replicates": replicates,
                "Valid replicates": len(values),
                "Resampling unit": "customer within market",
            }
        )
    return pd.DataFrame(summary_rows), replicate_table


def radius_coverage_profile(
    distance_frame: pd.DataFrame,
    total_holdout: int,
    radii: Sequence[int] = (0, 1, 2, 3),
) -> pd.DataFrame:
    rows = []
    compatible_n = len(distance_frame)
    for column in distance_frame.columns:
        if not column.startswith("distance::"):
            continue
        name = column.split("::", 1)[1]
        for radius in radii:
            selected = distance_frame.loc[distance_frame[column] <= radius]
            status_counts = selected["order_status"].value_counts().to_dict()
            rows.append(
                {
                    "Method": name,
                    "Radius": int(radius),
                    "Near-path orders": int(len(selected)),
                    "Share of compatible holdout": len(selected) / compatible_n if compatible_n else np.nan,
                    "Share of complete holdout": len(selected) / total_holdout if total_holdout else np.nan,
                    "Mean signed status score": float(selected["signed_status_score"].mean()) if len(selected) else np.nan,
                    "Positive-coded orders": int((selected["signed_status_score"] > 0).sum()),
                    "Neutral-coded orders": int((selected["signed_status_score"] == 0).sum()),
                    "Negative-coded orders": int((selected["signed_status_score"] < 0).sum()),
                    "Status category counts": "; ".join(f"{k}={v}" for k, v in sorted(status_counts.items())),
                }
            )
    return pd.DataFrame(rows)


def fixed_k_profile(
    distance_frame: pd.DataFrame,
    fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Coverage-standardized comparison: each method selects the same k."""
    rows = []
    n = len(distance_frame)
    for fraction in fractions:
        k = max(1, int(math.floor(fraction * n))) if n else 0
        for column in distance_frame.columns:
            if not column.startswith("distance::"):
                continue
            name = column.split("::", 1)[1]
            tie_key = "order_id" if "order_id" in distance_frame else "row_index"
            # The secondary key is fixed before inspecting held-out outcomes.
            chosen = distance_frame.sort_values([column, tie_key], kind="mergesort").head(k) if k else distance_frame.iloc[0:0]
            rows.append(
                {
                    "Method": name,
                    "Target compatible fraction": fraction,
                    "Target k": k,
                    "Selected orders": len(chosen),
                    "Tie-breaking rule": "distance, then ascending order identifier",
                    "Maximum selected distance": float(chosen[column].max()) if len(chosen) else np.nan,
                    "Mean signed status score": float(chosen["signed_status_score"].mean()) if len(chosen) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def status_distribution(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Return original status categories together with the declared signed coding."""
    counts = frame["order_status"].value_counts(dropna=False)
    score_by_status = (
        frame.groupby("order_status", dropna=False)["status_outcome"]
        .first()
        .to_dict()
    )
    statuses = counts.index.astype(str)
    scores = np.asarray([float(score_by_status.get(status, 0.0)) for status in counts.index])
    labels = np.where(scores > 0, "positive", np.where(scores < 0, "negative", "neutral"))
    return pd.DataFrame(
        {
            "Cohort": cohort,
            "Order status": statuses,
            "Signed score": scores,
            "Coding label": labels,
            "Orders": counts.to_numpy(int),
            "Share": counts.to_numpy(float) / len(frame) if len(frame) else np.nan,
        }
    )


def customer_overlap_summary(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    train_ids = set(train["customer_id"])
    holdout_ids = set(holdout["customer_id"])
    overlap = train_ids & holdout_ids
    return pd.DataFrame(
        [
            {
                "Training customers": len(train_ids),
                "Holdout customers": len(holdout_ids),
                "Customers appearing in both": len(overlap),
                "Share of holdout customers previously observed": len(overlap) / len(holdout_ids) if holdout_ids else np.nan,
                "Holdout orders from previously observed customers": int(holdout["customer_id"].isin(overlap).sum()),
                "Holdout orders from unseen customers": int((~holdout["customer_id"].isin(overlap)).sum()),
            }
        ]
    )


def deterministic_dependency_audit(order: pd.DataFrame) -> pd.DataFrame:
    """Numerically verify deterministic construction identities."""
    eps = 1e-12
    reconstructed_sales = order["avg_unit_value"] * order["total_qty"]
    reconstructed_net = order["gross_sales"] - order["discount_amount"]
    reconstructed_margin = order["total_profit"] / order["net_sales"].replace(0, np.nan)
    checks = [
        ("gross_sales = avg_unit_value * total_qty", order["gross_sales"], reconstructed_sales),
        ("net_sales = gross_sales - discount_amount", order["net_sales"], reconstructed_net),
        ("profit_margin = total_profit / net_sales", order["profit_margin"], reconstructed_margin),
    ]
    rows = []
    for label, observed, reconstructed in checks:
        error = (observed - reconstructed).abs()
        rows.append(
            {
                "Identity": label,
                "Maximum absolute discrepancy": float(error.max()),
                "Mean absolute discrepancy": float(error.mean()),
                "Exact within 1e-9": bool((error.fillna(0) <= 1e-9).all()),
            }
        )
    return pd.DataFrame(rows)

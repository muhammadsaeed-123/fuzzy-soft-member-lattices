#!/usr/bin/env python3
"""Controlled benchmark for the mathematical and validation protocol.

No external data are required. The simulation separates order coordinates from
accounting-derived payloads, compares four membership constructions, selects
map tolerance/support/evidence/cost settings inside a nested chronological
split, evaluates all methods on the same holdout cohort and at fixed coverage,
and reports top-k and bootstrap path-family stability.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from order_audit import ambient_closure_audit, audit_summary_table, intrinsic_lattice_audit
from path_algorithms import (
    consensus_path,
    max_mean_path_multi_source,
    path_mean,
    recurrence_tables,
    top_k_mean_paths_dag,
    count_source_target_paths,
)

LocalC = Tuple[int, int]
LocalO = Tuple[int, int]
LocalM = Tuple[int, int]
Gamma = Tuple[LocalC, LocalO, LocalM]

FEATURES = (
    ("c_volume", 4, (0.0, 10.0)),
    ("c_value", 3, (0.0, 100.0)),
    ("o_speed", 4, (0.0, 1.0)),
    ("o_resilience", 2, (0.0, 1.0)),
    ("m_profit", 3, (-40.0, 120.0)),
    ("m_retention", 3, (0.0, 1.0)),
)
DENOMS = np.asarray([3, 2, 3, 1, 2, 2], dtype=float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = np.arange(n)
    drift = 0.12 * (index / max(1, n - 1) - 0.5)
    latent = np.clip(rng.beta(2.2, 2.2, n) + drift, 0, 1)
    volume = np.clip(0.8 + 8.8 * latent + rng.normal(0, 1.15, n), 0, 10)
    value = np.clip(12 + 82 * latent + rng.normal(0, 10, n), 0, 100)
    speed = np.clip(0.08 + 0.88 * latent + rng.normal(0, 0.13, n), 0, 1)
    resilience_prob = sigmoid(-1.4 + 3.0 * latent + rng.normal(0, 0.25, n))
    resilience = rng.binomial(1, resilience_prob).astype(float)
    profit = np.clip(-28 + 132 * latent + rng.normal(0, 23, n), -40, 120)
    retention = np.clip(0.05 + 0.9 * latent + rng.normal(0, 0.16, n), 0, 1)

    # Payload annotations: deliberately excluded from lattice coordinates.
    revenue = np.maximum(0.0, volume * value * (0.88 + 0.08 * rng.random(n)))
    margin = np.divide(profit, np.maximum(revenue, 1.0))
    reliability = (speed >= 0.28 + 0.38 * (1.0 - latent)).astype(int)

    p_positive = sigmoid(-1.0 + 2.6 * latent + 0.45 * resilience + rng.normal(0, 0.25, n))
    p_adverse = sigmoid(-2.0 - 1.9 * latent - 0.25 * resilience + rng.normal(0, 0.20, n))
    draw = rng.random(n)
    signed = np.where(draw < p_adverse, -1, np.where(draw > 1.0 - p_positive, 1, 0))
    return pd.DataFrame(
        {
            "time": index,
            "latent": latent,
            "c_volume": volume,
            "c_value": value,
            "o_speed": speed,
            "o_resilience": resilience,
            "m_profit": profit,
            "m_retention": retention,
            "payload_revenue": revenue,
            "payload_margin": margin,
            "payload_reliability": reliability,
            "signed_outcome": signed.astype(int),
        }
    )


def fit_membership_model(train: pd.DataFrame, scheme: str) -> dict:
    model = {"scheme": scheme, "features": {}}
    for name, _levels, domain in FEATURES:
        x = train[name].to_numpy(float)
        if scheme in {"domain_anchored", "crisp"}:
            lo, hi = domain
            model["features"][name] = {"lo": lo, "hi": hi}
        elif scheme == "robust_minmax":
            lo, hi = np.quantile(x, [0.01, 0.99])
            if hi <= lo:
                hi = np.nextafter(lo, np.inf)
            model["features"][name] = {"lo": float(lo), "hi": float(hi)}
        elif scheme == "rank_only":
            model["features"][name] = {"sorted": np.sort(x).tolist()}
        else:
            raise ValueError(scheme)
    return model


def apply_membership_model(frame: pd.DataFrame, model: Mapping[str, object]) -> pd.DataFrame:
    out = frame.copy()
    scheme = str(model["scheme"])
    params = model["features"]
    for name, levels, _domain in FEATURES:
        p = params[name]
        x = out[name].to_numpy(float)
        if scheme == "rank_only":
            ref = np.asarray(p["sorted"], dtype=float)
            mu = np.searchsorted(ref, x, side="right") / max(1, len(ref))
        else:
            lo, hi = float(p["lo"]), float(p["hi"])
            mu = np.clip((x - lo) / (hi - lo), 0, 1)
        level = np.minimum(levels, np.floor(mu * levels).astype(int) + 1)
        if scheme == "crisp":
            mu = (level - 1) / max(1, levels - 1)
        out[f"{name}_mu"] = mu
        out[f"{name}_level"] = level
    out["C_node"] = list(zip(out.c_volume_level, out.c_value_level))
    out["O_node"] = list(zip(out.o_speed_level, out.o_resilience_level))
    out["M_node"] = list(zip(out.m_profit_level, out.m_retention_level))
    out["Gamma"] = list(zip(out.C_node, out.O_node, out.M_node))
    return out


def leq(a: Gamma, b: Gamma) -> bool:
    return all(x <= y for aa, bb in zip(a, b) for x, y in zip(aa, bb))


def join(a: Gamma, b: Gamma) -> Gamma:
    return tuple(tuple(max(x, y) for x, y in zip(aa, bb)) for aa, bb in zip(a, b))  # type: ignore[return-value]


def meet(a: Gamma, b: Gamma) -> Gamma:
    return tuple(tuple(min(x, y) for x, y in zip(aa, bb)) for aa, bb in zip(a, b))  # type: ignore[return-value]


def rank(g: Gamma) -> int:
    return sum(v - 1 for local in g for v in local)


def q_c(c: LocalC) -> int:
    return (1, 2, 2, 3)[c[0] - 1]


def q_o(o: LocalO) -> int:
    return (1, 2, 2, 3)[o[0] - 1]


def q_m(m: LocalM) -> int:
    return m[0]


def is_compatible(g: Gamma, delta: int) -> bool:
    values = (q_c(g[0]), q_o(g[1]), q_m(g[2]))
    return max(values) - min(values) <= delta


def algebraic_nodes(delta: int) -> List[Gamma]:
    c = product(range(1, 5), range(1, 4))
    o = product(range(1, 5), range(1, 3))
    m = product(range(1, 4), range(1, 4))
    return [g for g in product(list(c), list(o), list(m)) if is_compatible(g, delta)]  # type: ignore[list-item]


def counts(frame: pd.DataFrame, delta: int) -> Counter[Gamma]:
    return Counter(g for g in frame.Gamma if is_compatible(g, delta))


def cover_graph(nodes: Sequence[Gamma]) -> Tuple[Dict[Gamma, List[Gamma]], Dict[Gamma, List[Gamma]]]:
    xs = list(dict.fromkeys(nodes))
    A = np.asarray([[v for local in g for v in local] for g in xs], dtype=np.int16)
    ranks = A.sum(axis=1) - A.shape[1]
    order = np.argsort(ranks, kind="stable")
    succ: Dict[Gamma, List[Gamma]] = {g: [] for g in xs}
    pred: Dict[Gamma, List[Gamma]] = {g: [] for g in xs}
    for ii in order:
        candidate = np.where(np.all(A[ii] <= A, axis=1) & np.any(A[ii] < A, axis=1))[0]
        candidate = candidate[np.argsort(ranks[candidate], kind="stable")]
        minima: List[int] = []
        for jj in candidate:
            if not minima or not np.any(np.all(A[minima] <= A[jj], axis=1)):
                minima.append(int(jj))
        for jj in minima:
            succ[xs[ii]].append(xs[jj])
            pred[xs[jj]].append(xs[ii])
    return succ, pred


def representative_memberships(train: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    for name, levels, _ in FEATURES:
        out[name] = (
            train.groupby(f"{name}_level")[f"{name}_mu"]
            .median()
            .reindex(range(1, levels + 1))
            .ffill()
            .bfill()
            .to_numpy(float)
        )
    return out


def quality(reps: Mapping[str, np.ndarray]) -> Callable[[Gamma], float]:
    def score(g: Gamma) -> float:
        c, o, m = g
        layer_c = np.mean([reps["c_volume"][c[0] - 1], reps["c_value"][c[1] - 1]])
        layer_o = np.mean([reps["o_speed"][o[0] - 1], reps["o_resilience"][o[1] - 1]])
        layer_m = np.mean([reps["m_profit"][m[0] - 1], reps["m_retention"][m[1] - 1]])
        return float(np.mean([layer_c, layer_o, layer_m]))
    return score


def fit_path(train: pd.DataFrame, delta: int, n_min: int, beta: float, rank_cost: float) -> dict:
    support = counts(train, delta)
    nodes = [g for g in algebraic_nodes(delta) if support.get(g, 0) >= n_min]
    if not nodes:
        raise ValueError("No supported node")
    succ, pred = cover_graph(nodes)
    sources = [g for g in nodes if not pred[g]]
    targets = [g for g in nodes if not succ[g]]
    reps = representative_memberships(train)
    q = quality(reps)
    max_count = max(support.values(), default=1)
    evidence = lambda g: math.log1p(support.get(g, 0)) / math.log1p(max_count)
    utility = lambda g: (1 - beta) * q(g) + beta * evidence(g)
    maximum_rank = sum(v - 1 for v in (4, 3, 4, 2, 3, 3))
    edge_cost = lambda a, b: rank_cost * max(0, rank(b) - rank(a) - 1) / maximum_rank
    path, score, _ = max_mean_path_multi_source(
        nodes, succ, pred, utility, sources, targets, transition_cost=edge_cost
    )
    return {
        "nodes": nodes,
        "succ": succ,
        "pred": pred,
        "sources": sources,
        "targets": targets,
        "support": support,
        "quality": q,
        "utility": utility,
        "edge_cost": edge_cost,
        "path": path,
        "score": score,
    }


def distance(g: Gamma, path: Sequence[Gamma]) -> float:
    x = np.asarray([v for local in g for v in local], dtype=float)
    return min(
        float(np.mean(np.abs(x - np.asarray([v for local in p for v in local], dtype=float)) / DENOMS))
        for p in path
    )


def evaluate(frame: pd.DataFrame, path: Sequence[Gamma], delta: int, fraction: float = 0.20) -> dict:
    d = np.asarray([distance(g, path) for g in frame.Gamma], dtype=float)
    y = frame.signed_outcome.to_numpy(float)
    rho = stats.spearmanr(-d, y).statistic
    rho = float(rho) if np.isfinite(rho) else 0.0
    k = max(1, int(math.floor(fraction * len(frame))))
    selected = np.argsort(d, kind="stable")[:k]
    return {
        "common_cohort_spearman": rho,
        "matched_coverage_fraction": k / len(frame),
        "matched_coverage_n": k,
        "matched_coverage_outcome": float(y[selected].mean()),
        "mean_distance": float(d.mean()),
        "compatibility_rate": float(np.mean([is_compatible(g, delta) for g in frame.Gamma])),
        "exact_compatibility_rate": float(np.mean([is_compatible(g, 0) for g in frame.Gamma])),
    }


def nested_select(outer_train0: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    cut = int(math.floor(0.75 * len(outer_train0)))
    train0, valid0 = outer_train0.iloc[:cut], outer_train0.iloc[cut:]
    rows = []
    for scheme in ["domain_anchored", "robust_minmax", "rank_only", "crisp"]:
        membership = fit_membership_model(train0, scheme)
        train = apply_membership_model(train0, membership)
        valid = apply_membership_model(valid0, membership)
        for delta in [0, 1]:
            for n_min in [3, 8]:
                for beta in [0.0, 0.15]:
                    for rank_cost in [0.0, 0.02]:
                        try:
                            model = fit_path(train, delta, n_min, beta, rank_cost)
                        except ValueError:
                            continue
                        ev = evaluate(valid, model["path"], delta)
                        rows.append(
                            {
                                "membership": scheme,
                                "delta": delta,
                                "n_min": n_min,
                                "beta": beta,
                                "rank_gap_cost": rank_cost,
                                "graph_nodes": len(model["nodes"]),
                                "path_nodes": len(model["path"]),
                                "training_mean_utility": model["score"],
                                "inner_common_cohort_spearman": ev["common_cohort_spearman"],
                                "inner_matched_coverage_outcome": ev["matched_coverage_outcome"],
                                "inner_compatibility_rate": ev["compatibility_rate"],
                                "selection_objective": ev["common_cohort_spearman"] + 0.02 * ev["compatibility_rate"],
                            }
                        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(
        ["selection_objective", "inner_compatibility_rate", "graph_nodes", "membership"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]
    config = {
        "membership": str(best.membership),
        "delta": int(best.delta),
        "n_min": int(best.n_min),
        "beta": float(best.beta),
        "rank_gap_cost": float(best.rank_gap_cost),
    }
    return grid, config


def greedy(model: Mapping[str, object]) -> List[Gamma]:
    best_path, best_score = None, -math.inf
    for source in sorted(model["sources"], key=repr):
        path = [source]
        current = source
        while model["succ"].get(current):
            candidates = model["succ"][current]
            current = max(
                candidates,
                key=lambda nxt: (model["utility"](nxt) - model["edge_cost"](current, nxt), repr(nxt)),
            )
            path.append(current)
        score = path_mean(path, model["utility"], model["edge_cost"])
        if score > best_score:
            best_path, best_score = path, score
    if best_path is None:
        raise RuntimeError("No greedy path")
    return best_path


def random_paths(model: Mapping[str, object], n: int, seed: int) -> List[List[Gamma]]:
    rng = np.random.default_rng(seed)
    target_set = set(model["targets"])
    out = []
    for _ in range(n):
        current = model["sources"][int(rng.integers(0, len(model["sources"])))]
        path = [current]
        while current not in target_set and model["succ"].get(current):
            choices = model["succ"][current]
            current = choices[int(rng.integers(0, len(choices)))]
            path.append(current)
        if current in target_set:
            out.append(path)
    return out


def compare_methods(train0: pd.DataFrame, holdout0: pd.DataFrame, config: dict, seed: int):
    rows = []
    fitted = {}
    for scheme in ["domain_anchored", "robust_minmax", "rank_only", "crisp"]:
        membership = fit_membership_model(train0, scheme)
        train = apply_membership_model(train0, membership)
        holdout = apply_membership_model(holdout0, membership)
        model = fit_path(train, config["delta"], config["n_min"], config["beta"], config["rank_gap_cost"])
        rows.append(
            {
                "method": scheme,
                "graph_nodes": len(model["nodes"]),
                "cover_edges": sum(len(x) for x in model["succ"].values()),
                "path_nodes": len(model["path"]),
                "training_mean_utility": model["score"],
                **evaluate(holdout, model["path"], config["delta"]),
            }
        )
        fitted[scheme] = (membership, train, holdout, model)
    membership, train, holdout, model = fitted[config["membership"]]
    greedy_path = greedy(model)
    rows.append(
        {
            "method": "greedy_monotone",
            "graph_nodes": len(model["nodes"]),
            "cover_edges": sum(len(x) for x in model["succ"].values()),
            "path_nodes": len(greedy_path),
            "training_mean_utility": path_mean(greedy_path, model["utility"], model["edge_cost"]),
            **evaluate(holdout, greedy_path, config["delta"]),
        }
    )
    random_family = random_paths(model, 250, seed)
    random_eval = [evaluate(holdout, path, config["delta"]) for path in random_family]
    rows.append(
        {
            "method": "random_monotone_mean",
            "graph_nodes": len(model["nodes"]),
            "cover_edges": sum(len(x) for x in model["succ"].values()),
            "path_nodes": float(np.mean([len(x) for x in random_family])),
            "training_mean_utility": float(
                np.mean([path_mean(x, model["utility"], model["edge_cost"]) for x in random_family])
            ),
            "common_cohort_spearman": float(np.mean([x["common_cohort_spearman"] for x in random_eval])),
            "matched_coverage_fraction": random_eval[0]["matched_coverage_fraction"],
            "matched_coverage_n": random_eval[0]["matched_coverage_n"],
            "matched_coverage_outcome": float(np.mean([x["matched_coverage_outcome"] for x in random_eval])),
            "mean_distance": float(np.mean([x["mean_distance"] for x in random_eval])),
            "compatibility_rate": random_eval[0]["compatibility_rate"],
            "exact_compatibility_rate": random_eval[0]["exact_compatibility_rate"],
        }
    )
    return pd.DataFrame(rows), (membership, train, holdout, model)


def bootstrap_paths(train0: pd.DataFrame, baseline_path: Sequence[Gamma], config: dict, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    rows, paths = [], []
    base_nodes = set(baseline_path)
    base_edges = set(zip(baseline_path[:-1], baseline_path[1:]))
    for replicate in range(1, reps + 1):
        sample = train0.iloc[rng.integers(0, len(train0), size=len(train0))].reset_index(drop=True)
        try:
            membership = fit_membership_model(sample, config["membership"])
            transformed = apply_membership_model(sample, membership)
            model = fit_path(
                transformed,
                config["delta"],
                config["n_min"],
                config["beta"],
                config["rank_gap_cost"],
            )
            path = model["path"]
            paths.append(path)
            nodes, edges = set(path), set(zip(path[:-1], path[1:]))
            rows.append(
                {
                    "replicate": replicate,
                    "path_nodes": len(path),
                    "graph_nodes": len(model["nodes"]),
                    "exact_baseline_path": path == list(baseline_path),
                    "node_jaccard": len(nodes & base_nodes) / len(nodes | base_nodes),
                    "edge_jaccard": len(edges & base_edges) / len(edges | base_edges) if edges | base_edges else 1.0,
                    "training_mean_utility": model["score"],
                    "path": repr(path),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "replicate": replicate,
                    "path_nodes": np.nan,
                    "graph_nodes": 0,
                    "exact_baseline_path": False,
                    "node_jaccard": 0.0,
                    "edge_jaccard": 0.0,
                    "training_mean_utility": np.nan,
                    "path": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows), paths


def structure_examples() -> pd.DataFrame:
    U = frozenset({1, 2, 3})
    B = [frozenset(), frozenset({1}), frozenset({2}), U]
    ambient = ambient_closure_audit(B, lambda a, b: a | b, lambda a, b: a & b)
    intrinsic, _ = intrinsic_lattice_audit(B, lambda a, b: a <= b)
    rows = [audit_summary_table("Boolean counterexample", ambient, intrinsic)]
    exact = algebraic_nodes(0)
    ambient = ambient_closure_audit(exact, join, meet)
    intrinsic, _ = intrinsic_lattice_audit(exact, leq)
    rows.append(audit_summary_table("Exact synthetic pullback", ambient, intrinsic))
    return pd.concat(rows, ignore_index=True)


def scaling(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dimension, levels in [(2, 4), (3, 4), (4, 3), (5, 3), (6, 3)]:
        nodes = list(product(range(1, levels + 1), repeat=dimension))
        succ = {node: [] for node in nodes}
        pred = {node: [] for node in nodes}
        for node in nodes:
            for j in range(dimension):
                if node[j] < levels:
                    nxt = list(node)
                    nxt[j] += 1
                    nxt = tuple(nxt)
                    succ[node].append(nxt)
                    pred[nxt].append(node)
        utility = {node: float(rng.random()) for node in nodes}
        start = time.perf_counter()
        path, score, _ = max_mean_path_multi_source(
            nodes,
            succ,
            pred,
            utility.__getitem__,
            [tuple([1] * dimension)],
            [tuple([levels] * dimension)],
        )
        rows.append(
            {
                "dimension": dimension,
                "levels_per_dimension": levels,
                "nodes": len(nodes),
                "covers": sum(len(x) for x in succ.values()),
                "path_nodes": len(path),
                "mean_utility": score,
                "dp_seconds": time.perf_counter() - start,
            }
        )
    return pd.DataFrame(rows)


def save(fig, directory: Path, name: str) -> None:
    from figure_utils import save_figure
    fig.tight_layout()
    save_figure(fig, directory, name, dpi=600)


def figures(directory: Path, comparison: pd.DataFrame, node_frequency, baseline_path, scaling_table):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = ["Domain\nanchored", "Robust\nmin-max", "Rank\nonly", "Crisp\nordinal", "Greedy", "Random\nmean"]
    label_map = dict(zip(["domain_anchored", "robust_minmax", "rank_only", "crisp", "greedy_monotone", "random_monotone_mean"], labels))
    labels = [label_map.get(x, x.replace("_", "\n")) for x in comparison.method]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(labels, comparison.common_cohort_spearman)
    ax.set_ylabel("Common-cohort Spearman correlation")
    ax.grid(axis="y", alpha=.2)
    save(fig, directory, "synthetic_membership_comparison")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(labels, comparison.matched_coverage_outcome)
    ax.set_ylabel("Mean outcome at matched 20% coverage")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.2)
    save(fig, directory, "synthetic_coverage_outcome")

    fig, ax = plt.subplots(figsize=(7.5, 4.1))
    stages = np.arange(1, len(baseline_path) + 1)
    ax.bar(stages, [node_frequency.get(node, 0.0) for node in baseline_path])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("baseline path stage")
    ax.set_ylabel("bootstrap node recurrence")
    save(fig, directory, "synthetic_path_recurrence")

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.plot(scaling_table.nodes, scaling_table.dp_seconds, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes (log scale)")
    ax.set_ylabel("dynamic-program time in seconds (log scale)")
    save(fig, directory, "synthetic_scaling")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "synthetic")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    table_dir = args.out / "tables"
    figure_dir = args.out / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)

    data = generate_data(args.n, args.seed)
    split = int(math.floor(0.70 * len(data)))
    train0, holdout0 = data.iloc[:split].copy(), data.iloc[split:].copy()
    grid, config = nested_select(train0)
    comparison, fitted = compare_methods(train0, holdout0, config, args.seed + 1)
    membership, train, holdout, model = fitted

    top = top_k_mean_paths_dag(
        model["nodes"], model["succ"], model["pred"], model["utility"],
        model["sources"], model["targets"], model["edge_cost"], k=10
    )
    top_table = pd.DataFrame(
        {
            "rank": range(1, len(top) + 1),
            "path_nodes": [len(path) for path, _ in top],
            "mean_utility": [score for _, score in top],
            "relative_gap_percent": [100 * (top[0][1] - score) / abs(top[0][1]) for _, score in top],
            "path": [repr(path) for path, _ in top],
        }
    )

    bootstrap, paths = bootstrap_paths(train0, model["path"], config, args.bootstrap_reps, args.seed + 2)
    node_frequency, edge_frequency = recurrence_tables(paths)
    consensus, consensus_score = consensus_path(
        model["nodes"],
        model["succ"],
        model["pred"],
        model["sources"],
        model["targets"],
        node_frequency,
        edge_frequency,
    )
    consensus_eval = evaluate(holdout, consensus, config["delta"])

    ambient = ambient_closure_audit(model["nodes"], join, meet)
    intrinsic, witnesses = intrinsic_lattice_audit(model["nodes"], leq)
    fitted_audit = audit_summary_table("Fitted synthetic support order", ambient, intrinsic)
    scaling_table = scaling(args.seed + 3)

    grid.to_csv(table_dir / "nested_selection_grid.csv", index=False)
    pd.DataFrame([config]).to_csv(table_dir / "selected_configuration.csv", index=False)
    comparison.to_csv(table_dir / "method_comparison.csv", index=False)
    top_table.to_csv(table_dir / "top_k_paths.csv", index=False)
    bootstrap.to_csv(table_dir / "bootstrap_paths.csv", index=False)
    fitted_audit.to_csv(table_dir / "fitted_structure_audit.csv", index=False)
    witnesses.to_csv(table_dir / "intrinsic_lattice_witnesses.csv", index=False)
    structure_examples().to_csv(table_dir / "structural_counterexamples.csv", index=False)
    scaling_table.to_csv(table_dir / "scaling.csv", index=False)
    pd.DataFrame(
        [
            {
                "replicates": len(bootstrap),
                "successful_paths": len(paths),
                "exact_path_recoveries": int(bootstrap.exact_baseline_path.sum()),
                "exact_recovery_rate": float(bootstrap.exact_baseline_path.mean()),
                "median_node_jaccard": float(bootstrap.node_jaccard.median()),
                "median_edge_jaccard": float(bootstrap.edge_jaccard.median()),
                "median_path_nodes": float(bootstrap.path_nodes.median()),
                "consensus_path_nodes": len(consensus),
                "consensus_recurrence_score": consensus_score,
                "consensus_common_cohort_spearman": consensus_eval["common_cohort_spearman"],
                "consensus_matched_coverage_outcome": consensus_eval["matched_coverage_outcome"],
            }
        ]
    ).to_csv(table_dir / "stability_summary.csv", index=False)
    pd.DataFrame(
        [{"node": repr(node), "frequency": value, "baseline": node in set(model["path"])} for node, value in node_frequency.items()]
    ).sort_values(["frequency", "node"], ascending=[False, True]).to_csv(table_dir / "node_recurrence.csv", index=False)
    pd.DataFrame(
        [{"edge": repr(edge), "frequency": value, "baseline": edge in set(zip(model["path"][:-1], model["path"][1:]))} for edge, value in edge_frequency.items()]
    ).sort_values(["frequency", "edge"], ascending=[False, True]).to_csv(table_dir / "edge_recurrence.csv", index=False)
    pd.DataFrame(
        {"stage": range(1, len(model["path"]) + 1), "node": [repr(x) for x in model["path"]], "rank": [rank(x) for x in model["path"]], "node_recurrence": [node_frequency.get(x, 0.0) for x in model["path"]]}
    ).to_csv(table_dir / "baseline_path.csv", index=False)
    pd.DataFrame(
        {"stage": range(1, len(consensus) + 1), "node": [repr(x) for x in consensus], "rank": [rank(x) for x in consensus]}
    ).to_csv(table_dir / "consensus_path.csv", index=False)

    summary = {
        "seed": args.seed,
        "observations": len(data),
        "outer_training": len(train0),
        "outer_holdout": len(holdout0),
        "selected_configuration": config,
        "exact_pullback_nodes": len(algebraic_nodes(0)),
        "tolerance_relation_nodes": len(algebraic_nodes(1)),
        "fitted_graph_nodes": len(model["nodes"]),
        "fitted_graph_covers": sum(len(x) for x in model["succ"].values()),
        "baseline_path_nodes": len(model["path"]),
        "source_target_path_count": count_source_target_paths(model["nodes"], model["succ"], model["pred"], model["sources"], model["targets"]),
        "top_path_mean_utility": top[0][1],
        "second_path_relative_gap_percent": float(top_table.iloc[1].relative_gap_percent) if len(top_table) > 1 else None,
        "exact_recovery_rate": float(bootstrap.exact_baseline_path.mean()),
        "median_node_jaccard": float(bootstrap.node_jaccard.median()),
        "median_edge_jaccard": float(bootstrap.edge_jaccard.median()),
        "intrinsic_lattice": bool(intrinsic.is_intrinsic_lattice),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures(figure_dir, comparison, node_frequency, model["path"], scaling_table)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

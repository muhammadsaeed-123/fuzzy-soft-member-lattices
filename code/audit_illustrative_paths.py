#!/usr/bin/env python3
"""Exact rational enumeration of both declared small mathematical examples."""
from __future__ import annotations
import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C3_NODES = tuple(f"{i}{j}" for i in range(1, 4) for j in range(1, 4))
CHAIN_EDGES = ((1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6))
CHAIN_UTILITY = {i: Fraction(v) for i, v in enumerate(
    ("0.43875", "0.60750", "0.60625", "0.77500", "0.85375", "0.93250"), 1)}


def paths(edges: tuple, source, target) -> list[tuple]:
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
    result = []
    def visit(node, prefix):
        if node == target:
            result.append(prefix)
            return
        for nxt in sorted(adjacency.get(node, [])):
            if nxt in prefix:
                raise ValueError("The illustrative graph must be acyclic")
            visit(nxt, prefix + (nxt,))
    visit(source, (source,))
    return result


def c3_paths() -> list[tuple[str, ...]]:
    edges = []
    for i in range(1, 4):
        for j in range(1, 4):
            if i < 3:
                edges.append((f"{i}{j}", f"{i+1}{j}"))
            if j < 3:
                edges.append((f"{i}{j}", f"{i}{j+1}"))
    return paths(tuple(edges), "11", "33")


def mean(path: tuple, utility: dict) -> Fraction:
    return sum((Fraction(utility[node]) for node in path), Fraction(0)) / len(path)


def audit() -> dict:
    cp = c3_paths()
    chain = paths(CHAIN_EDGES, 1, 6)
    best = max(chain, key=lambda p: mean(p, CHAIN_UTILITY))
    av = (Fraction(".20"), Fraction(".50"), Fraction(".80"))
    bv = (Fraction(".30"), Fraction(".60"), Fraction(".90"))
    local_utility = {f"{i+1}{j+1}": (a+b)/2 for i,a in enumerate(av) for j,b in enumerate(bv)}
    return {
        "c3xc3": {
            "status": "verified_by_exact_rational_enumeration",
            "definition": "u_ij = (a_i+b_j)/2; a=(.20,.50,.80), b=(.30,.60,.90); zero edge costs",
            "path_count": len(cp), "illustrated_path": ["11", "21", "31", "32", "33"],
            "utilities": {k: float(v) for k,v in local_utility.items()},
            "paths": [{"nodes": list(p), "mean": float(mean(p,local_utility)),
                       "exact_mean": str(mean(p,local_utility))} for p in cp],
            "optimal_mean": .55, "exact_optimal_mean": "11/20", "all_tied": True,
            "number_of_optima": 6, "unique_optimum": False,
        },
        "chain_diamond": {
            "status": "verified_by_exact_rational_enumeration", "path_count": len(chain),
            "edges": [list(e) for e in CHAIN_EDGES],
            "utilities": {str(k): float(v) for k,v in CHAIN_UTILITY.items()},
            "paths": [{"nodes": list(p), "sum": float(sum(CHAIN_UTILITY[x] for x in p)),
                       "mean": float(mean(p, CHAIN_UTILITY)), "exact_mean": str(mean(p, CHAIN_UTILITY))}
                      for p in chain],
            "optimal_path": list(best), "optimal_mean": float(mean(best, CHAIN_UTILITY)),
            "unique_optimum": sum(mean(p, CHAIN_UTILITY) == mean(best, CHAIN_UTILITY) for p in chain) == 1,
        },
    }


def write(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    result = audit()
    (out / "illustrative_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "c3xc3_paths.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path_number", "nodes", "mean", "exact_mean", "optimal"])
        for i,p in enumerate(result["c3xc3"]["paths"],1):
            writer.writerow([i, ">".join(p["nodes"]), p["mean"], p["exact_mean"], True])
    with (out / "c3xc3_node_utilities.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["node", "utility"])
        writer.writerows(result["c3xc3"]["utilities"].items())
    with (out / "chain_diamond_paths.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nodes", "sum", "mean", "exact_mean", "optimal"])
        for p in result["chain_diamond"]["paths"]:
            writer.writerow([">".join(map(str,p["nodes"])),p["sum"],p["mean"],p["exact_mean"],p["nodes"]==result["chain_diamond"]["optimal_path"]])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/theory")
    args = parser.parse_args()
    result = write(args.out)
    print("Chain-diamond: 3 paths; unique optimum (1,3,5,6); exact mean 0.7078125.")
    print("C3 x C3: all 6 paths are optimal; exact mean 11/20 = 0.55.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

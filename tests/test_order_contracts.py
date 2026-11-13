from __future__ import annotations

from itertools import product

from order_audit import ambient_closure_audit, classification_label, intrinsic_lattice_audit
from path_algorithms import max_mean_path_multi_source, top_k_paths


def test_ambient_nonclosure_does_not_imply_non_lattice():
    U = frozenset({1, 2, 3})
    B = [frozenset(), frozenset({1}), frozenset({2}), U]
    ambient = ambient_closure_audit(B, lambda a, b: a | b, lambda a, b: a & b)
    intrinsic, _ = intrinsic_lattice_audit(B, lambda a, b: a <= b)
    assert not ambient.is_ambient_sublattice
    assert intrinsic.is_intrinsic_lattice
    assert classification_label(ambient, intrinsic) == "intrinsic lattice, not an ambient sublattice"


def test_full_product_is_ambient_sublattice_and_intrinsic_lattice():
    nodes = list(product(range(3), repeat=2))
    leq = lambda a, b: a[0] <= b[0] and a[1] <= b[1]
    ambient = ambient_closure_audit(
        nodes,
        lambda a, b: (max(a[0], b[0]), max(a[1], b[1])),
        lambda a, b: (min(a[0], b[0]), min(a[1], b[1])),
    )
    intrinsic, witnesses = intrinsic_lattice_audit(nodes, leq)
    assert ambient.is_ambient_sublattice
    assert intrinsic.is_intrinsic_lattice
    assert witnesses.empty


def test_multiple_source_recurrence_allows_source_with_incoming_edge():
    nodes = ["a", "b", "c", "d"]
    succ = {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
    pred = {"a": [], "b": ["a"], "c": ["b"], "d": ["c"]}
    u = {"a": 0.1, "b": 0.9, "c": 0.8, "d": 0.7}.__getitem__
    path, score, _ = max_mean_path_multi_source(
        nodes,
        succ,
        pred,
        u,
        sources=["a", "b"],
        targets=["d"],
        transition_cost=lambda _x, _y: 0.05,
    )
    assert path == ["b", "c", "d"]
    assert abs(score - ((0.9 + 0.8 + 0.7 - 0.10) / 3)) < 1e-12


def test_top_k_and_exact_dp_agree():
    nodes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    succ = {(0, 0): [(1, 0), (0, 1)], (1, 0): [(1, 1)], (0, 1): [(1, 1)], (1, 1): []}
    pred = {(0, 0): [], (1, 0): [(0, 0)], (0, 1): [(0, 0)], (1, 1): [(1, 0), (0, 1)]}
    u = lambda x: 0.2 + 0.3 * x[0] + 0.4 * x[1]
    path, score, _ = max_mean_path_multi_source(nodes, succ, pred, u, [(0, 0)], [(1, 1)])
    top = top_k_paths(succ, [(0, 0)], [(1, 1)], u, k=2)
    assert path == top[0][0]
    assert abs(score - top[0][1]) < 1e-12

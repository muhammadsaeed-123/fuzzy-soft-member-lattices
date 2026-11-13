from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

from lattice_model import (
    all_local_nodes,
    alternative_maps,
    build_pullback,
    evidence_function,
    full_product_cover_count,
    induced_cover_graph,
    join,
    max_mean_path,
    meet,
    order_level_aggregate,
    selected_maps,
)


def local_join(a, b):
    return tuple(max(x, y) for x, y in zip(a, b))


def local_meet(a, b):
    return tuple(min(x, y) for x, y in zip(a, b))


def test_full_product_cover_counts_match_manuscript() -> None:
    assert full_product_cover_count((4, 4, 4)) == 144
    assert full_product_cover_count((4, 4, 2)) == 64
    assert full_product_cover_count((5, 5, 4)) == 235


def test_selected_coordination_maps_are_lattice_homomorphisms() -> None:
    local_spaces = all_local_nodes()
    maps = selected_maps()
    for nodes, q in zip(local_spaces, maps):
        for a, b in itertools.product(nodes, repeat=2):
            assert q(local_join(a, b)) == local_join(q(a), q(b))
            assert q(local_meet(a, b)) == local_meet(q(a), q(b))


def test_selected_pullback_cardinality_and_sampled_closure() -> None:
    nodes = build_pullback()
    assert len(nodes) == 3200
    node_set = set(nodes)
    q_c, q_o, q_m = selected_maps()
    sample_indices = [0, 1, 17, 63, 127, 511, 1024, 2047, 3199]
    sample = [nodes[i] for i in sample_indices]
    for g in sample:
        assert q_c(g[0]) == q_o(g[1]) == q_m(g[2])
    for a, b in itertools.product(sample, repeat=2):
        assert join(a, b) in node_set
        assert meet(a, b) in node_set


def test_exact_maximum_mean_path_on_small_diamond() -> None:
    source = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    left = ((2, 1, 1), (1, 1, 1), (1, 1, 1))
    right = ((1, 2, 1), (1, 1, 1), (1, 1, 1))
    target = ((2, 2, 1), (1, 1, 1), (1, 1, 1))
    nodes = [source, left, right, target]
    succ, pred = induced_cover_graph(nodes)
    utility_values = {source: 0.0, left: 0.3, right: 0.9, target: 0.6}
    path, mean, total = max_mean_path(nodes, succ, pred, utility_values.__getitem__, source)
    assert path == [source, right, target]
    assert mean == pytest.approx(0.5)
    assert total == pytest.approx(1.5)


def test_joint_evidence_is_zero_for_unseen_and_one_at_maximum() -> None:
    g1 = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    g2 = ((2, 1, 1), (1, 1, 1), (1, 1, 1))
    unseen = ((3, 1, 1), (1, 1, 1), (1, 1, 1))
    evidence = evidence_function({g1: 3, g2: 27})
    assert evidence(unseen) == 0.0
    assert 0.0 < evidence(g1) < 1.0
    assert evidence(g2) == pytest.approx(1.0)


def test_order_level_aggregation_preserves_the_order_decision_unit() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "minimal_dataco.csv"
    raw = pd.read_csv(fixture)
    orders = order_level_aggregate(raw).set_index("Order Id")
    assert len(orders) == 2
    assert orders.loc[10, "gross_sales"] == pytest.approx(150.0)
    assert orders.loc[10, "total_qty"] == 3
    assert orders.loc[10, "avg_unit_value"] == pytest.approx(50.0)
    assert orders.loc[10, "total_profit"] == pytest.approx(13.5)
    assert orders.loc[10, "reliability"] == 1
    assert orders.loc[10, "status_outcome"] == 1
    assert orders.loc[20, "status_outcome"] == -1


def test_reported_coordination_map_set_has_five_distinct_names() -> None:
    maps = alternative_maps()
    assert list(maps) == [
        "selected", "unit_reliability", "legacy", "actual_service", "profit_service"
    ]


def test_maximum_mean_ties_are_deterministic() -> None:
    source = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    left = ((2, 1, 1), (1, 1, 1), (1, 1, 1))
    right = ((1, 2, 1), (1, 1, 1), (1, 1, 1))
    target = ((2, 2, 1), (1, 1, 1), (1, 1, 1))
    nodes = [source, left, right, target]
    succ, pred = induced_cover_graph(nodes)
    utility = {source: 0.0, left: 0.5, right: 0.5, target: 0.5}.__getitem__
    first = max_mean_path(nodes, succ, pred, utility, source)
    for _ in range(10):
        assert max_mean_path(list(reversed(nodes)), succ, pred, utility, source) == first


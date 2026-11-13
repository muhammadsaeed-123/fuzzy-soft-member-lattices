import math

from validation_tools import evidence_on_frozen_training_scale


def test_frozen_evidence_uses_training_denominator_and_caps_at_one():
    f = evidence_on_frozen_training_scale({"a": 5, "b": 100}, training_max_count=10)
    assert math.isclose(f("a"), math.log1p(5) / math.log1p(10))
    assert f("b") == 1.0
    assert f("missing") == 0.0

import pandas as pd

from validation_tools import cluster_bootstrap_distance_association, market_distance_summary, status_distribution


def test_common_cohort_customer_cluster_bootstrap_is_reproducible():
    rows = []
    for market in ["A", "B"]:
        for customer in range(4):
            for repeat in range(2):
                score = 1.0 if customer < 2 else -1.0
                rows.append(
                    {
                        "market": market,
                        "customer_id": f"{market}-{customer}",
                        "signed_status_score": score,
                        "distance::Method 1": 0 if score > 0 else 2,
                        "distance::Method 2": repeat + customer,
                    }
                )
    frame = pd.DataFrame(rows)
    summary_a, reps_a = cluster_bootstrap_distance_association(frame, replicates=20, seed=17)
    summary_b, reps_b = cluster_bootstrap_distance_association(frame, replicates=20, seed=17)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    pd.testing.assert_frame_equal(reps_a, reps_b)
    assert summary_a.shape[0] == 2
    assert reps_a.shape[0] == 40
    assert set(summary_a["Resampling unit"]) == {"customer within market"}


def test_status_distribution_exposes_declared_signed_coding():
    frame = pd.DataFrame(
        {
            "order_status": ["COMPLETE", "COMPLETE", "CANCELED", "PENDING"],
            "status_outcome": [1, 1, -1, 0],
        }
    )
    table = status_distribution(frame, "toy")
    coding = dict(zip(table["Order status"], table["Signed score"]))
    assert coding == {"COMPLETE": 1.0, "CANCELED": -1.0, "PENDING": 0.0}


def test_market_distance_summary_preserves_method_market_cells():
    frame = pd.DataFrame(
        {
            "market": ["A", "A", "B", "B"],
            "signed_status_score": [1.0, -1.0, 1.0, 0.0],
            "distance::M1": [0, 2, 1, 3],
            "distance::M2": [1, 1, 0, 2],
        }
    )
    table = market_distance_summary(frame)
    assert table.shape[0] == 4
    assert set(table["Market"]) == {"A", "B"}
    assert set(table["Method"]) == {"M1", "M2"}

import numpy as np

from lattice_model import cluster_bootstrap_difference, spearman_bootstrap_joint_support


def _compatible_frame(nodes, outcomes=None):
    if outcomes is None:
        outcomes = [0.0] * len(nodes)
    return pd.DataFrame(
        {
            "C_node": [g[0] for g in nodes],
            "O_node": [g[1] for g in nodes],
            "M_node": [g[2] for g in nodes],
            "market": ["A"] * len(nodes),
            "customer_id": [f"c{i}" for i in range(len(nodes))],
            "status_outcome": outcomes,
        }
    )


def test_cluster_bootstrap_difference_returns_nan_when_one_cohort_is_empty():
    low = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    high = ((4, 4, 4), (4, 4, 2), (4, 4, 4))
    frame = _compatible_frame([low, low], [1.0, -1.0])

    point, interval, replicates = cluster_bootstrap_difference(
        frame,
        path_a=[high],
        path_b=[low],
        reps=20,
        seed=3,
        radius=0,
    )

    assert np.isnan(point)
    assert np.isnan(interval).all()
    assert replicates.size == 0


def test_support_transport_returns_nan_for_constant_training_support():
    node_a = ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    node_b = ((2, 1, 1), (1, 1, 1), (1, 1, 1))
    train = _compatible_frame([node_a, node_b])
    test = _compatible_frame([node_a, node_b])

    rho, interval = spearman_bootstrap_joint_support(
        train,
        test,
        nodes=[node_a, node_b],
        reps=20,
        seed=4,
    )

    assert np.isnan(rho)
    assert np.isnan(interval).all()

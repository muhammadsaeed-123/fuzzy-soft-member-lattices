from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs" / "reference"


def test_headline_reference_values() -> None:
    results = json.loads((REFERENCE / "results.json").read_text(encoding="utf-8"))
    assert results["raw_rows"] == 180_519
    assert results["orders"] == 65_752
    assert results["full_pullback_nodes"] == 3_200
    assert results["joint_nodes"] == 181
    assert results["joint_covers"] == 485
    assert results["path_count"] == 24_870
    assert results["path_nodes"] == 10
    assert results["optimal_mean"] == pytest.approx(0.517913754)
    assert results["radius_one_share_complete_holdout"] == pytest.approx(116 / 13153)


def test_reference_path_and_refit_summary() -> None:
    path = pd.read_csv(REFERENCE / "tables" / "optimal_path.csv")
    assert len(path) == 10
    assert list(path["Rank"]) == [0, 1, 2, 3, 5, 22, 23, 24, 25, 26]
    assert list(path["Joint training support"]) == [17, 31, 24, 7, 8, 3, 9, 8, 5, 3]
    summary = pd.read_csv(REFERENCE / "tables" / "full_refit_summary.csv").iloc[0]
    assert int(summary["Replicates"]) == 500
    assert int(summary["Exact path recoveries"]) == 7
    assert summary["Median node Jaccard"] == pytest.approx(6/17)


def test_reference_figures_exist_and_are_nonempty() -> None:
    theory = ROOT / "outputs" / "theory" / "figures"
    empirical = ROOT / "outputs" / "reference" / "figures"
    expected_theory = {
        "roadmap.pdf", "member_lattice_c3xc3.pdf", "isomorphic_payload.pdf",
        "pullback_chain_diamond.pdf", "ambient_vs_intrinsic.pdf", "exact_paths.pdf",
    }
    expected_empirical = {
        "coordination_map_audit.pdf", "closure_audit.pdf", "optimal_path_heatmap.pdf",
        "path_utility_progression.pdf", "external_outcome_bootstrap.pdf",
        "rolling_origin.pdf", "operator_sensitivity.pdf", "coverage_limitations.pdf",
        "full_refit_stability.pdf",
    }
    for directory, names in ((theory, expected_theory), (empirical, expected_empirical)):
        for name in names:
            path = directory / name
            assert path.exists(), name
            assert path.stat().st_size > 1_000, name

from __future__ import annotations

from pathlib import Path

from generate_reference_figures import (
    closure,
    coordination_map,
    coverage_limitations,
    external_bootstrap,
    full_refit,
    operators,
    path_heatmap,
    rolling,
    utility,
)
from generate_theory_figures import generate

ROOT = Path(__file__).resolve().parents[1]


def test_theory_figure_generators(tmp_path: Path) -> None:
    generate(ROOT/"figure_source", tmp_path, dpi=150)
    expected = {
        "roadmap", "member_lattice_c3xc3", "isomorphic_payload",
        "pullback_chain_diamond", "ambient_vs_intrinsic", "exact_paths",
    }
    assert {p.stem for p in tmp_path.glob("*.pdf")} == expected
    assert {p.stem for p in tmp_path.glob("*.png")} == expected
    assert all(path.stat().st_size > 1_000 for path in tmp_path.glob("*.*"))


def test_reference_table_figure_generators(tmp_path: Path) -> None:
    tables = ROOT / "outputs" / "reference" / "tables"
    coordination_map(tables, tmp_path)
    closure(tables, tmp_path)
    path_heatmap(tables, tmp_path)
    utility(tables, tmp_path)
    external_bootstrap(tables, tmp_path)
    rolling(tables, tmp_path)
    operators(tables, tmp_path)
    coverage_limitations(tables, tmp_path)
    full_refit(tables, tmp_path)
    expected = {
        "coordination_map_audit", "closure_audit", "optimal_path_heatmap",
        "path_utility_progression", "external_outcome_bootstrap", "rolling_origin",
        "operator_sensitivity", "coverage_limitations", "full_refit_stability",
    }
    assert expected == {p.stem for p in tmp_path.glob("*.pdf")}
    assert all((tmp_path / f"{name}.png").exists() for name in expected)

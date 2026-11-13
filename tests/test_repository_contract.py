from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_repository_files_exist() -> None:
    required = [
        "README.md", "QUICKSTART.md", "LICENSE",
        "DATA_LICENSE.md", "OUTPUT_INDEX.md", "FIGURE_INDEX.md",
        "CLAIM_TO_ARTIFACT.md", "create_manifest.py", "verify_manifest.py",
        "manifest.sha256", "run_all.sh", "run_smoke.sh", "run_all.ps1", "run_all.bat",
        "code/lattice_model.py", "code/order_audit.py", "code/path_algorithms.py",
        "code/validation_tools.py", "code/run_analysis.py", "code/run_synthetic_benchmark.py",
        "code/generate_synthetic_figures.py",
        "code/download_data.py", "code/generate_theory_figures.py",
        "code/generate_reference_figures.py", "code/assemble_manuscript_figures.py",
        "code/verify_outputs.py",
        "outputs/synthetic/summary.json", "outputs/synthetic/tables/method_comparison.csv",
    ]
    missing = [name for name in required if not (ROOT / name).exists()]
    assert not missing


def test_expected_results_match_reference_results() -> None:
    expected = json.loads((ROOT / "config" / "expected_headline_results.json").read_text())
    reference = json.loads((ROOT / "outputs" / "reference" / "results.json").read_text())
    for key, value in reference.items():
        assert key in expected
        assert expected[key] == value


def test_figure_map_covers_documents_and_synthetic_resource() -> None:
    mapping = json.loads((ROOT / "config" / "figure_map.json").read_text())
    assert len(mapping["main_article"]) == 11
    assert len(mapping["online_resource_1"]) == 18
    assert set(mapping["synthetic_benchmark"]) == {
        "membership_comparison", "coverage_outcome", "path_recurrence", "scaling"
    }


def test_full_analysis_writes_every_required_empirical_figure() -> None:
    source=(ROOT/"code/run_analysis.py").read_text()
    assert "generate_all(tdir, fdir)" in source
    plotting=(ROOT/"code/generate_reference_figures.py").read_text()
    for stem in ["coordination_map_audit","closure_audit","optimal_path_heatmap","path_utility_progression","external_outcome_bootstrap","rolling_origin","coverage_limitations","operator_sensitivity","full_refit_stability"]:
        assert '"'+stem+'"' in plotting


def test_manifest_verifies() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "verify_manifest.py")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

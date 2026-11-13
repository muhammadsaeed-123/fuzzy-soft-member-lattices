from __future__ import annotations
import hashlib
import json
from pathlib import Path
from audit_illustrative_paths import audit, c3_paths
from assemble_manuscript_figures import assemble

ROOT = Path(__file__).resolve().parents[1]


def test_chain_diamond_exact_example():
    result = audit()["chain_diamond"]
    assert result["path_count"] == 3
    assert result["unique_optimum"]
    assert result["optimal_path"] == [1,3,5,6]
    assert result["optimal_mean"] == 0.7078125
    assert sorted(p["mean"] for p in result["paths"]) == [0.688125,0.6884375,0.7078125]


def test_c3_exact_arithmetic_example():
    result = audit()["c3xc3"]
    assert len(c3_paths()) == 6
    assert all(len(p) == 5 for p in c3_paths())
    assert result["number_of_optima"] == 6
    assert result["optimal_mean"] == .55
    assert not result["unique_optimum"]
    assert {p["exact_mean"] for p in result["paths"]} == {"11/20"}
    assert len(result["utilities"]) == 9


def test_publication_assembly_uses_native_sources(tmp_path):
    result=assemble(out=tmp_path, empirical=ROOT/"outputs/reference/figures",
                    synthetic=ROOT/"outputs/synthetic/figures", theory=ROOT/"outputs/theory/figures")
    assert len(result["files"]) == 87
    assert all(r["generated_from_machine_readable_source"] for r in result["files"])
    for r in result["files"]:
        assert hashlib.sha256((tmp_path/r["relative_destination"]).read_bytes()).hexdigest()==r["sha256"]
    assert len(list((tmp_path/"main_article").glob("*.pdf")))==11
    assert len(list((tmp_path/"online_resource_1").glob("*.eps")))==18


def test_current_numbering_is_contiguous_and_shared_figures_match():
    mapping = json.loads((ROOT/"config/figure_map.json").read_text())
    assert list(mapping["main_article"]) == [f"Fig{i}" for i in range(1,12)]
    assert list(mapping["online_resource_1"]) == [f"FigS{i}" for i in range(1,19)]
    for main,supp in ((2,1),(3,2),(4,3),(5,4),(6,5),(7,11),(8,12),(9,14),(10,15),(11,17)):
        assert mapping["main_article"][f"Fig{main}"] == mapping["online_resource_1"][f"FigS{supp}"]


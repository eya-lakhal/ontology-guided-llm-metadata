"""Tests for the knowledge graph construction and the SPARQL queries."""

from pathlib import Path

import pytest
from rdflib import Graph
from rdflib.namespace import RDF

from src.build_graph import build_graph, main as build_main
from src.namespaces import ABX, ABXD
from src.paths import QUERIES_DIR
from src.query import format_table, load_kg, run_query
from src.rdf_mapping import annotation_to_triples
from src.validate import validate_graph


@pytest.fixture(scope="module")
def kg() -> Graph:
    return load_kg(None)


def q(name: str) -> Path:
    return QUERIES_DIR / name


def test_gold_graph_conforms_to_shacl() -> None:
    g = build_graph()
    assert len(set(g.subjects(RDF.type, None))) == 10
    assert validate_graph(g).conforms


def test_build_graph_cli_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "kg.ttl"
    assert build_main(["--out", str(out)]) == 0
    assert len(Graph().parse(out)) > 100


def test_annotation_to_triples_minimal() -> None:
    triples = annotation_to_triples(
        {"id": "x", "resource_type": "Poster", "title": "T", "topic": ["uti"], "format": "print", "usage_condition": []}
    )
    assert (ABXD.x, RDF.type, ABX.Poster) in triples
    assert (ABXD.x, ABX.coversTopic, ABX.uti) in triples
    assert (ABXD.x, ABX.hasFormat, ABX.print) in triples


def test_at_least_five_queries() -> None:
    assert len(list(QUERIES_DIR.glob("*.rq"))) >= 5


def test_nurse_uti(kg: Graph) -> None:
    _, rows = run_query(kg, q("01_nurse_uti_resources.rq"))
    assert [r[0] for r in rows] == ["abxd:r01", "abxd:r05"]


def test_count_per_topic(kg: Graph) -> None:
    _, rows = run_query(kg, q("02_count_per_topic.rq"))
    counts = {r[0]: int(r[1]) for r in rows}
    assert counts["Antibiotic prescription"] == 4
    assert counts["Antibiotic resistance"] == 4
    assert counts["Urinary tract infection"] == 3
    assert sum(counts.values()) == 18


def test_missing_usage_condition(kg: Graph) -> None:
    _, rows = run_query(kg, q("03_missing_usage_condition.rq"))
    assert [r[0] for r in rows] == ["abxd:r04", "abxd:r07"]


def test_infection_hierarchy(kg: Graph) -> None:
    _, rows = run_query(kg, q("04_infection_resources.rq"))
    assert [r[0] for r in rows] == ["abxd:r01", "abxd:r05", "abxd:r07", "abxd:r08", "abxd:r10"]


def test_audience_fr_labels(kg: Graph) -> None:
    _, rows = run_query(kg, q("05_resources_per_audience_fr.rq"))
    assert rows[0] == ["Infirmiers", "7"]


def test_related_pairs_symmetric(kg: Graph) -> None:
    _, rows = run_query(kg, q("06_related_resources.rq"))
    assert len(rows) == 4  # r01-r05, r02-r08, r03-r06, r08-r10 (each pair once)


def test_recent_french(kg: Graph) -> None:
    _, rows = run_query(kg, q("07_recent_french_for_professionals.rq"))
    years = [int(r[1]) for r in rows]
    assert years and min(years) >= 2023
    assert len(rows) == 5


def test_format_table() -> None:
    assert "| a |" in format_table(["a"], [["1"]], markdown=True)
    assert format_table(["a", "b"], [["1", "22"]]).splitlines()[0].startswith("a")

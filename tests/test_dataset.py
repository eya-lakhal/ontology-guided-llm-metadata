"""Consistency checks between the synthetic dataset and the ontology."""

import json
from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from src.namespaces import ABX
from src.paths import GOLD_DIR, ONTOLOGY_FILE, RAW_DIR

GOLD_FILES = sorted(GOLD_DIR.glob("*.json"))
FIELD_RANGE = {
    "target_audience": "TargetAudience",
    "objective": "Objective",
    "topic": "Topic",
    "usage_condition": "UsageCondition",
}


@pytest.fixture(scope="module")
def onto() -> Graph:
    return Graph().parse(ONTOLOGY_FILE, format="turtle")


def _individuals_of(g: Graph, root: str) -> set[str]:
    classes = set(g.transitive_subjects(RDFS.subClassOf, ABX[root]))
    return {
        str(i).removeprefix(str(ABX))
        for c in classes
        for i in g.subjects(RDF.type, c)
        if (i, RDF.type, OWL.NamedIndividual) in g
    }


def test_ten_documents_with_gold() -> None:
    raw = {p.stem for p in RAW_DIR.glob("*.txt")}
    gold = {p.stem for p in GOLD_FILES}
    assert len(raw) >= 10
    assert raw == gold


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_gold_values_are_in_ontology(onto: Graph, path: Path) -> None:
    gold = json.loads(path.read_text(encoding="utf-8"))
    assert gold["id"] == path.stem
    resource_types = {str(c).removeprefix(str(ABX)) for c in onto.transitive_subjects(RDFS.subClassOf, ABX.Resource)}
    assert gold["resource_type"] in resource_types - {"Resource"}
    for field, root in FIELD_RANGE.items():
        assert set(gold[field]) <= _individuals_of(onto, root), field
    assert gold["format"] in _individuals_of(onto, "Format")
    assert gold["language"] in {"fr", "en"}
    assert gold["target_audience"] and gold["topic"]
    for rel in gold["related_to"]:
        assert (GOLD_DIR / f"{rel}.json").exists()


@pytest.mark.parametrize("path", GOLD_FILES, ids=lambda p: p.stem)
def test_title_appears_in_source(path: Path) -> None:
    gold = json.loads(path.read_text(encoding="utf-8"))
    text = (RAW_DIR / f"{path.stem}.txt").read_text(encoding="utf-8")
    assert gold["title"] in text


def test_ontology_uri_prefix() -> None:
    assert isinstance(ABX.nurse, URIRef)

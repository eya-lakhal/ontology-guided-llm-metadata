"""Structural checks on the OWL ontology."""

from pathlib import Path

import pytest
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ONTOLOGY = Path(__file__).resolve().parents[1] / "ontology" / "abx_resources.ttl"
ABX = Namespace("https://eya-lakhal.github.io/ontology-guided-llm-metadata/abx#")


@pytest.fixture(scope="module")
def graph() -> Graph:
    g = Graph()
    g.parse(ONTOLOGY, format="turtle")
    return g


def _langs(g: Graph, s: URIRef) -> set[str]:
    return {lbl.language for lbl in g.objects(s, RDFS.label)}


@pytest.mark.parametrize(
    "cls",
    ["Resource", "Poster", "PracticalSheet", "Training", "Video", "Tool",
     "TargetAudience", "Nurse", "Physician", "CareAssistant", "Pharmacist", "Resident", "Family",
     "Objective", "Topic", "UrinaryTractInfection", "AntibioticResistance", "HandHygiene",
     "Prescription", "Vaccination", "Format", "UsageCondition"],
)
def test_required_classes_exist(graph: Graph, cls: str) -> None:
    assert (ABX[cls], RDF.type, OWL.Class) in graph


@pytest.mark.parametrize(
    "prop", ["hasTargetAudience", "hasObjective", "coversTopic", "hasFormat", "hasUsageCondition", "relatedTo"]
)
def test_object_properties_have_domain_and_range(graph: Graph, prop: str) -> None:
    p = ABX[prop]
    assert (p, RDF.type, OWL.ObjectProperty) in graph
    assert graph.value(p, RDFS.domain) is not None
    assert graph.value(p, RDFS.range) is not None


@pytest.mark.parametrize("prop", ["title", "publicationYear", "publisher", "language", "url"])
def test_datatype_properties(graph: Graph, prop: str) -> None:
    assert (ABX[prop], RDF.type, OWL.DatatypeProperty) in graph
    assert graph.value(ABX[prop], RDFS.range) is not None


def test_every_entity_has_en_and_fr_labels(graph: Graph) -> None:
    entities = (
        set(graph.subjects(RDF.type, OWL.Class))
        | set(graph.subjects(RDF.type, OWL.ObjectProperty))
        | set(graph.subjects(RDF.type, OWL.DatatypeProperty))
        | set(graph.subjects(RDF.type, OWL.NamedIndividual))
    )
    entities = {e for e in entities if isinstance(e, URIRef) and str(e).startswith(str(ABX))}
    missing = [e for e in entities if not {"en", "fr"} <= _langs(graph, e)]
    assert not missing, f"entities missing EN/FR labels: {missing}"


def test_classes_have_comments(graph: Graph) -> None:
    classes = [c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)]
    assert all(graph.value(c, RDFS.comment) is not None for c in classes)


def test_each_range_class_has_individuals(graph: Graph) -> None:
    """Every controlled dimension must offer at least one individual to pick from."""
    for root in ("TargetAudience", "Topic", "Objective", "Format", "UsageCondition"):
        subclasses = set(graph.transitive_subjects(RDFS.subClassOf, ABX[root]))
        individuals = {i for c in subclasses for i in graph.subjects(RDF.type, c)
                       if (i, RDF.type, OWL.NamedIndividual) in graph}
        assert individuals, root

"""Mapping between JSON annotations and RDF triples.

An *annotation* is a plain dict with the keys used in ``data/gold/*.json``
(``resource_type``, ``title``, ``target_audience``, ``topic``, ``format``, ...),
whose controlled values are local names of ontology individuals (``"nurse"``, ``"uti"``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

from src.namespaces import ABX, ABXD

#: JSON field -> ontology object property (multi-valued fields)
OBJECT_FIELDS: dict[str, URIRef] = {
    "target_audience": ABX.hasTargetAudience,
    "objective": ABX.hasObjective,
    "topic": ABX.coversTopic,
    "usage_condition": ABX.hasUsageCondition,
}
#: single-valued object field
FORMAT_FIELD = "format"
#: JSON field -> (data property, datatype)
DATA_FIELDS: dict[str, tuple[URIRef, URIRef]] = {
    "title": (ABX.title, XSD.string),
    "publisher": (ABX.publisher, XSD.string),
    "publication_year": (ABX.publicationYear, XSD.integer),
    "language": (ABX.language, XSD.string),
    "url": (ABX.url, XSD.anyURI),
}


def bind_prefixes(g: Graph) -> Graph:
    """Bind the project prefixes on ``g`` (for readable Turtle output)."""
    g.bind("abx", ABX)
    g.bind("abxd", ABXD)
    return g


def resource_uri(resource_id: str) -> URIRef:
    """IRI of a resource in the data namespace."""
    return ABXD[resource_id]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def annotation_to_triples(annotation: Mapping[str, Any]) -> list[tuple[URIRef, URIRef, Any]]:
    """Convert one annotation dict into a list of RDF triples."""
    s = resource_uri(annotation["id"])
    triples: list[tuple[URIRef, URIRef, Any]] = []
    if annotation.get("resource_type"):
        triples.append((s, RDF.type, ABX[annotation["resource_type"]]))
    for field, (prop, dtype) in DATA_FIELDS.items():
        value = annotation.get(field)
        if value not in (None, ""):
            triples.append((s, prop, Literal(value, datatype=dtype)))
    for field, prop in OBJECT_FIELDS.items():
        for value in _as_list(annotation.get(field)):
            triples.append((s, prop, ABX[value]))
    for value in _as_list(annotation.get(FORMAT_FIELD)):
        triples.append((s, ABX.hasFormat, ABX[value]))
    for other in _as_list(annotation.get("related_to")):
        triples.append((s, ABX.relatedTo, resource_uri(other)))
    return triples


def annotations_to_graph(annotations: Iterable[Mapping[str, Any]], graph: Graph | None = None) -> Graph:
    """Add the triples of several annotations to ``graph`` (a new graph by default)."""
    g = bind_prefixes(graph if graph is not None else Graph())
    for annotation in annotations:
        for triple in annotation_to_triples(annotation):
            g.add(triple)
    return g

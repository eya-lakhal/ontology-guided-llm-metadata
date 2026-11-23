"""SHACL validation of resource metadata graphs.

Usage (PowerShell)::

    python -m src.validate                       # validates ontology/examples.ttl
    python -m src.validate output/kg.ttl         # validates any Turtle data file
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pyshacl import validate as shacl_validate
from rdflib import Graph

from src.paths import EXAMPLES_FILE, ONTOLOGY_FILE, SHAPES_FILE


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a SHACL validation run."""

    conforms: bool
    report_text: str
    report_graph: Graph


@lru_cache(maxsize=1)
def load_ontology() -> Graph:
    """Load (and cache) the OWL ontology."""
    return Graph().parse(ONTOLOGY_FILE, format="turtle")


@lru_cache(maxsize=1)
def load_shapes() -> Graph:
    """Load (and cache) the SHACL shapes graph."""
    return Graph().parse(SHAPES_FILE, format="turtle")


def validate_graph(data: Graph) -> ValidationResult:
    """Validate ``data`` against the SHACL shapes, using the ontology as background knowledge.

    The ontology is merged into (a copy of) the data graph, so SHACL's built-in
    ``rdfs:subClassOf*`` semantics make ``sh:targetClass abx:Resource`` reach ``abx:Poster``
    instances and let ``sh:class abx:TargetAudience`` accept ``abx:nurse a abx:Nurse``.
    RDFS inference is deliberately *disabled*: with it, ``rdfs:range`` axioms would infer
    that any value of ``abx:coversTopic`` is a ``Topic`` and range errors would go unnoticed.
    """
    conforms, report_graph, report_text = shacl_validate(
        data_graph=data,
        shacl_graph=load_shapes(),
        ont_graph=load_ontology(),
        inference="none",
        abort_on_first=False,
        allow_warnings=True,
    )
    return ValidationResult(bool(conforms), str(report_text), report_graph)


def validate_file(path: Path) -> ValidationResult:
    """Parse a Turtle file and validate it."""
    return validate_graph(Graph().parse(path, format="turtle"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code (0 = conforms)."""
    parser = argparse.ArgumentParser(description="Validate RDF data against ontology/shapes.ttl")
    parser.add_argument("files", nargs="*", type=Path, default=[EXAMPLES_FILE])
    args = parser.parse_args(argv)
    ok = True
    for f in args.files:
        result = validate_file(f)
        print(f"{f}: {'CONFORMS' if result.conforms else 'DOES NOT CONFORM'}")
        if not result.conforms:
            print(result.report_text)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

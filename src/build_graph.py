"""Build the knowledge graph from the gold annotations.

Usage (PowerShell)::

    python -m src.build_graph                    # -> output/kg.ttl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rdflib import Graph

from src.paths import GOLD_DIR, OUTPUT_DIR
from src.rdf_mapping import annotations_to_graph
from src.validate import validate_graph


def load_gold(gold_dir: Path = GOLD_DIR) -> list[dict[str, Any]]:
    """Load all gold annotations, sorted by id."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(gold_dir.glob("*.json"))]


def build_graph(gold_dir: Path = GOLD_DIR) -> Graph:
    """Return the RDF knowledge graph of all gold annotations (data only, no ontology)."""
    return annotations_to_graph(load_gold(gold_dir))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Build output/kg.ttl from data/gold/*.json")
    parser.add_argument("--gold", type=Path, default=GOLD_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "kg.ttl")
    args = parser.parse_args(argv)

    g = build_graph(args.gold)
    result = validate_graph(g)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(args.out, format="turtle")
    print(f"Wrote {len(g)} triples to {args.out} (SHACL: {'conforms' if result.conforms else 'VIOLATIONS'})")
    if not result.conforms:
        print(result.report_text)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

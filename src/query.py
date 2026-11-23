"""Run the SPARQL queries of ``queries/`` against the knowledge graph and print tables.

Usage (PowerShell)::

    python -m src.query                                # all queries on output/kg.ttl
    python -m src.query queries/02_count_per_topic.rq  # a single query
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from src.build_graph import build_graph
from src.paths import OUTPUT_DIR, QUERIES_DIR
from src.rdf_mapping import bind_prefixes
from src.validate import load_ontology

Table = tuple[list[str], list[list[str]]]


def load_kg(kg_path: Path | None = None) -> Graph:
    """Return ontology + data graph. Rebuilds the data from gold if ``kg_path`` does not exist."""
    g = bind_prefixes(Graph())
    g += load_ontology()
    if kg_path is not None and kg_path.exists():
        g.parse(kg_path, format="turtle")
    else:
        g += build_graph()
    return g


def _fmt(term: object, graph: Graph) -> str:
    if term is None:
        return ""
    if isinstance(term, URIRef):
        return graph.namespace_manager.normalizeUri(term)
    if isinstance(term, Literal):
        return str(term)
    return str(term)


def run_query(graph: Graph, query_path: Path) -> Table:
    """Execute a ``.rq`` file and return ``(column names, rows of strings)``."""
    result = graph.query(query_path.read_text(encoding="utf-8"))
    columns = [str(v) for v in result.vars or []]
    rows = [[_fmt(row[i], graph) for i in range(len(columns))] for row in result]
    return columns, rows


def format_table(columns: list[str], rows: list[list[str]], markdown: bool = False) -> str:
    """Render a result table as aligned plain text (or Markdown)."""
    if markdown:
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(columns)]
    line = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in rows]
    return "\n".join([line, sep, *body])


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run SPARQL queries on the knowledge graph")
    parser.add_argument("queries", nargs="*", type=Path, help="query files (default: queries/*.rq)")
    parser.add_argument("--kg", type=Path, default=OUTPUT_DIR / "kg.ttl")
    parser.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    graph = load_kg(args.kg)
    for q in args.queries or sorted(QUERIES_DIR.glob("*.rq")):
        columns, rows = run_query(graph, q)
        print(f"\n=== {q.name} ({len(rows)} rows) ===")
        print(format_table(columns, rows, markdown=args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())

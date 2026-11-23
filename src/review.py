"""Human-in-the-loop validation of LLM proposals (command-line interface).

For every proposed value, the expert sees the value, its ontology label and the source
excerpt (with a faithfulness flag), then chooses **[a]ccept**, **[e]dit** or **[r]eject**.
Edited controlled values must resolve to an ontology identifier. For list fields the expert
can also add values the model missed. The accepted record is validated with SHACL; only a
conforming record is added to the knowledge graph. Every decision is logged.

Usage (PowerShell)::

    python -m src.extract --all                         # produce proposals first
    python -m src.review                                # review output/proposals/guided/*.json
    python -m src.review r01 r05 --reviewer "Dr Martin" --kg output/kg_reviewed.ttl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdflib import Graph

from src.extract import ExtractionResult, proposals_dir
from src.paths import OUTPUT_DIR, RAW_DIR
from src.rdf_mapping import annotation_to_triples, bind_prefixes
from src.validate import validate_graph
from src.vocabulary import CONTROLLED_FIELDS, MULTI_VALUED, Vocabulary, load_vocabulary

FIELD_ORDER = ("resource_type", "title", "publisher", "publication_year", "language",
               "target_audience", "objective", "topic", "format", "usage_condition")
CONTROLLED = {"resource_type", *CONTROLLED_FIELDS}
LOG_FILE = OUTPUT_DIR / "review_log.json"
KG_REVIEWED = OUTPUT_DIR / "kg_reviewed.ttl"

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]


@dataclass
class Decision:
    """One reviewer decision on one proposed (or added) value."""

    doc_id: str
    field: str
    proposed: Any
    excerpt: str
    decision: str  # accept | edit | reject | add
    final: Any
    reviewer: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


@dataclass
class ReviewOutcome:
    """Result of reviewing one document."""

    doc_id: str
    annotation: dict[str, Any]
    decisions: list[Decision]
    shacl_conforms: bool
    shacl_report: str
    added_to_kg: bool = False


class ReviewSession:
    """Interactive review of extraction results; I/O functions are injectable for testing."""

    def __init__(self, vocab: Vocabulary | None = None, reviewer: str = "expert",
                 input_fn: InputFn | None = None, print_fn: PrintFn | None = None) -> None:
        self.vocab = vocab or load_vocabulary()
        self.reviewer = reviewer
        # resolved at call time so that builtins can be patched (tests, notebooks)
        self.ask: InputFn = input_fn or (lambda prompt: input(prompt))
        self.say: PrintFn = print_fn or (lambda msg: print(msg))

    # ------------------------------------------------------------------ helpers
    def _label(self, field_name: str, value: Any) -> str:
        if field_name not in CONTROLLED:
            return ""
        terms = self.vocab.resource_types if field_name == "resource_type" else self.vocab.fields[field_name]
        for t in terms:
            if t.id == value:
                return f" ({t.label_en} / {t.label_fr})"
        return ""

    def _read_value(self, field_name: str) -> Any:
        """Ask for a value until it is valid for the field."""
        while True:
            raw = self.ask(f"    new value for {field_name}: ").strip()
            if field_name in CONTROLLED:
                resolved = self.vocab.resolve(field_name, raw)
                if resolved:
                    return resolved
                self.say(f"    '{raw}' is not in the ontology. Allowed: {', '.join(self.vocab.ids(field_name))}")
            elif field_name == "publication_year":
                if raw.isdigit() and 1990 <= int(raw) <= 2100:
                    return int(raw)
                self.say("    expected a year between 1990 and 2100")
            elif field_name == "language":
                if raw in ("fr", "en"):
                    return raw
                self.say("    expected 'fr' or 'en'")
            elif raw:
                return raw

    def _choose(self) -> str:
        while True:
            choice = self.ask("    [a]ccept / [e]dit / [r]eject ? ").strip().lower()[:1]
            if choice in {"a", "e", "r"}:
                return choice
            self.say("    please type a, e or r")

    def _review_item(self, doc_id: str, field_name: str, item: dict[str, Any], faithful: bool | None,
                     decisions: list[Decision]) -> Any:
        value, excerpt = item.get("value"), item.get("excerpt", "")
        flag = "" if faithful is None else ("  ✓ found in source" if faithful else "  ⚠ NOT found in source")
        self.say(f"\n  {field_name}: {value!r}{self._label(field_name, value)}")
        self.say(f"    excerpt: \"{excerpt}\"{flag}")
        choice = self._choose()
        if choice == "a":
            final, kind = value, "accept"
        elif choice == "e":
            final, kind = self._read_value(field_name), "edit"
        else:
            final, kind = None, "reject"
        decisions.append(Decision(doc_id, field_name, value, excerpt, kind, final, self.reviewer))
        return final

    # ------------------------------------------------------------------ main API
    def review(self, result: ExtractionResult, source_text: str = "") -> ReviewOutcome:
        """Review all values of one extraction result and return the accepted annotation."""
        doc_id = result.doc_id
        proposal = result.proposal or {}
        decisions: list[Decision] = []
        annotation: dict[str, Any] = {"id": doc_id}
        self.say(f"\n================ {doc_id} ({result.model}, guided={result.guided}) ================")
        if source_text:
            self.say(source_text.strip())
        if result.out_of_ontology:
            rejected = ", ".join(f"{o['field']}={o['value']!r}" for o in result.out_of_ontology)
            self.say(f"\n  (automatically rejected, out of ontology: {rejected})")

        for name in FIELD_ORDER:
            value = proposal.get(name)
            if name in MULTI_VALUED:
                kept: list[Any] = []
                for i, item in enumerate(value or []):
                    final = self._review_item(doc_id, name, item, result.faithfulness.get(f"{name}[{i}]"), decisions)
                    if final is not None and final not in kept:
                        kept.append(final)
                while True:
                    extra = self.ask(f"  add a value to {name}? (identifier or label, empty = no) ").strip()
                    if not extra:
                        break
                    resolved = self.vocab.resolve(name, extra)
                    if resolved is None:
                        self.say(f"    not in the ontology. Allowed: {', '.join(self.vocab.ids(name))}")
                        continue
                    if resolved not in kept:
                        kept.append(resolved)
                        decisions.append(Decision(doc_id, name, None, "", "add", resolved, self.reviewer))
                annotation[name] = kept
            elif isinstance(value, dict):
                faithful = None if name == "language" else result.faithfulness.get(name)
                annotation[name] = self._review_item(doc_id, name, value, faithful, decisions)
            else:
                self.say(f"\n  {name}: (no proposal)")
                if self.ask(f"  provide a value for {name}? [y/N] ").strip().lower().startswith("y"):
                    annotation[name] = self._read_value(name)
                    decisions.append(Decision(doc_id, name, None, "", "add", annotation[name], self.reviewer))
                else:
                    annotation[name] = None

        graph = bind_prefixes(Graph())
        for triple in annotation_to_triples(annotation):
            graph.add(triple)
        shacl = validate_graph(graph)
        self.say(f"\n  SHACL: {'conforms' if shacl.conforms else 'VIOLATIONS'}")
        if not shacl.conforms:
            self.say(shacl.report_text)
        return ReviewOutcome(doc_id, annotation, decisions, shacl.conforms, shacl.report_text)


def add_to_kg(outcome: ReviewOutcome, kg_path: Path) -> bool:
    """Add a reviewed record to the KG file if it conforms (previous triples of the record are replaced)."""
    if not outcome.shacl_conforms:
        return False
    kg = bind_prefixes(Graph())
    if kg_path.exists():
        kg.parse(kg_path, format="turtle")
    triples = annotation_to_triples(outcome.annotation)
    subject = triples[0][0]
    kg.remove((subject, None, None))
    for t in triples:
        kg.add(t)
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    kg.serialize(kg_path, format="turtle")
    outcome.added_to_kg = True
    return True


def append_log(outcome: ReviewOutcome, log_path: Path = LOG_FILE) -> None:
    """Append the decisions of one document to the JSON review log."""
    log: list[dict[str, Any]] = []
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
    log.append({
        "doc_id": outcome.doc_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shacl_conforms": outcome.shacl_conforms,
        "added_to_kg": outcome.added_to_kg,
        "annotation": outcome.annotation,
        "decisions": [d.__dict__ for d in outcome.decisions],
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def reviewed_ids(log_path: Path = LOG_FILE) -> set[str]:
    """Documents already present in the log and added to the KG."""
    if not log_path.exists():
        return set()
    return {e["doc_id"] for e in json.loads(log_path.read_text(encoding="utf-8")) if e.get("added_to_kg")}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Review LLM proposals (accept / edit / reject)")
    parser.add_argument("docs", nargs="*", help="document ids (default: all proposals)")
    parser.add_argument("--proposals", type=Path, default=proposals_dir(True))
    parser.add_argument("--kg", type=Path, default=KG_REVIEWED)
    parser.add_argument("--log", type=Path, default=LOG_FILE)
    parser.add_argument("--reviewer", default="expert")
    parser.add_argument("--redo", action="store_true", help="review documents already validated")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    files = [args.proposals / f"{d}.json" for d in args.docs] if args.docs else sorted(args.proposals.glob("*.json"))
    if not files:
        print(f"No proposals in {args.proposals}. Run `python -m src.extract --all` first.")
        return 1
    done = set() if args.redo else reviewed_ids(args.log)
    session = ReviewSession(reviewer=args.reviewer)
    for f in files:
        result = ExtractionResult.from_dict(json.loads(f.read_text(encoding="utf-8")))
        if result.doc_id in done:
            print(f"{result.doc_id}: already reviewed (use --redo)")
            continue
        raw = RAW_DIR / f"{result.doc_id}.txt"
        outcome = session.review(result, raw.read_text(encoding="utf-8") if raw.exists() else "")
        added = add_to_kg(outcome, args.kg)
        append_log(outcome, args.log)
        print(f"  -> {'added to ' + str(args.kg) if added else 'NOT added (SHACL violations)'}; logged in {args.log}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nReview interrupted; decisions of completed documents are saved.")
        sys.exit(130)

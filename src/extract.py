"""Ontology-guided metadata extraction with an LLM.

Pipeline for one document::

    ontology --(rdflib)--> vocabulary --> prompt --> LLM --> JSON
        --> grounding (labels -> ontology ids, out-of-ontology values recorded)
        --> pydantic validation (schema generated from the ontology)
        --> RDF triples --> SHACL validation

Invalid JSON (or, in guided mode, out-of-ontology values) triggers a retry with the
error fed back to the model. Each proposed value keeps the source excerpt justifying it.

Usage (PowerShell)::

    python -m src.extract data/raw/r01.txt                  # guided, provider from .env
    python -m src.extract --all --provider mock             # all documents, offline
    python -m src.extract --all --no-guidance               # ablation: no vocabulary in the prompt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.llm import DOC_END, DOC_START, GUIDED_MARKER, LLMClient, get_client
from src.paths import OUTPUT_DIR, RAW_DIR
from src.rdf_mapping import annotations_to_graph
from src.schema import build_extraction_model
from src.validate import validate_graph
from src.vocabulary import MULTI_VALUED, Vocabulary, load_vocabulary

CONTROLLED = ("resource_type", "target_audience", "objective", "topic", "format", "usage_condition")

JSON_TEMPLATE = """{
  "resource_type": {"value": "...", "excerpt": "..."},
  "title": {"value": "...", "excerpt": "..."},
  "publisher": {"value": "...", "excerpt": "..."} or null,
  "publication_year": {"value": 2024, "excerpt": "..."} or null,
  "language": {"value": "fr|en", "excerpt": "..."},
  "target_audience": [{"value": "...", "excerpt": "..."}],
  "objective": [{"value": "...", "excerpt": "..."}],
  "topic": [{"value": "...", "excerpt": "..."}],
  "format": {"value": "...", "excerpt": "..."} or null,
  "usage_condition": [{"value": "...", "excerpt": "..."}]
}"""

FIELD_DESCRIPTIONS = """- resource_type: the kind of resource (poster, sheet, training, video, tool...)
- title, publisher, publication_year, language (ISO code fr or en)
- target_audience: who the resource is intended for
- objective: what the resource aims to achieve
- topic: the subjects covered
- format: the medium of the resource (one value)
- usage_condition: access or usage conditions (licence, registration, prerequisites)"""

SYSTEM_PROMPT = (
    "You are an expert in knowledge engineering and antibiotic stewardship in nursing homes. "
    "You annotate educational resources with metadata. You answer with a single JSON object and nothing else."
)


def build_prompt(text: str, vocab: Vocabulary, guided: bool = True) -> tuple[str, str]:
    """Return ``(system, user)`` prompts. ``guided`` injects the ontology vocabulary."""
    rules = [
        "Fill the JSON object below for the document.",
        "For every value, `excerpt` must be a short span copied VERBATIM from the document that justifies it.",
        "Do not invent information: if a field is not supported by the document, use null or an empty list.",
    ]
    if guided:
        rules.insert(1, "Use ONLY identifiers from the ALLOWED VOCABULARY for resource_type, target_audience, "
                        "objective, topic, format and usage_condition. Write the identifier (left of the colon), "
                        "never a translation or a new term.")
        guidance = f"\n{GUIDED_MARKER} (derived from the OWL ontology):\n{vocab.to_prompt()}\n"
    else:
        guidance = f"\nFIELDS:\n{FIELD_DESCRIPTIONS}\nUse short English terms for the values.\n"
    user = (
        "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
        + "\n" + guidance
        + f"\nJSON FORMAT:\n{JSON_TEMPLATE}\n\n{DOC_START}\n{text.strip()}\n{DOC_END}\n"
    )
    return SYSTEM_PROMPT, user


def parse_json(raw: str) -> dict[str, Any]:
    """Parse the model output, tolerating Markdown code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = s.replace("’", "'").replace("…", "...")
    return re.sub(r"\s+", " ", s).strip(" .\"'«»")


def excerpt_in_source(excerpt: str, source: str) -> bool:
    """Faithfulness check: is the (non-empty) excerpt really a span of the source text?"""
    e = _norm_text(excerpt)
    return bool(e) and e in _norm_text(source)


def _items(value: Any) -> list[dict[str, Any]]:
    """Coerce a field value to a list of ``{"value", "excerpt"}`` dicts."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    out = []
    for v in values:
        if isinstance(v, dict):
            out.append({"value": v.get("value"), "excerpt": str(v.get("excerpt") or "")})
        elif v not in (None, ""):
            out.append({"value": v, "excerpt": ""})
    return out


def ground(data: dict[str, Any], vocab: Vocabulary) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Map controlled values to ontology ids.

    Returns ``(grounded data, out-of-ontology values, number of controlled values proposed)``.
    Out-of-ontology values are removed from the grounded data.
    """
    grounded = dict(data)
    oov: list[dict[str, Any]] = []
    total = 0
    for name in CONTROLLED:
        kept: list[dict[str, Any]] = []
        for item in _items(data.get(name)):
            total += 1
            resolved = vocab.resolve(name, item["value"])
            if resolved is None:
                oov.append({"field": name, **item})
            elif resolved not in [k["value"] for k in kept]:
                kept.append({**item, "value": resolved})
        if name in MULTI_VALUED:
            grounded[name] = kept
        else:
            grounded[name] = kept[0] if kept else None
    lang = _items(data.get("language"))
    if lang and isinstance(lang[0]["value"], str):
        code = lang[0]["value"].strip().lower()[:2]
        grounded["language"] = {**lang[0], "value": code} if code in ("fr", "en") else None
    return grounded, oov, total


@dataclass
class ExtractionResult:
    """Everything produced for one document (serialised to ``output/proposals``)."""

    doc_id: str
    guided: bool
    model: str
    attempts: int
    raw_responses: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_values_proposed: int = 0
    out_of_ontology: list[dict[str, Any]] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    faithfulness: dict[str, bool] = field(default_factory=dict)
    shacl_conforms: bool = False
    shacl_report: str = ""

    @property
    def valid_json(self) -> bool:
        """True when a schema-valid proposal was obtained."""
        return self.proposal is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExtractionResult":
        """Inverse of :meth:`to_dict`."""
        return cls(**d)


def proposal_to_annotation(doc_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Flatten a validated proposal (values + excerpts) into a gold-like annotation dict."""
    ann: dict[str, Any] = {"id": doc_id}
    for name, value in proposal.items():
        if isinstance(value, list):
            ann[name] = [v["value"] for v in value]
        elif isinstance(value, dict):
            ann[name] = value["value"]
        else:
            ann[name] = value
    return ann


def iter_values(proposal: dict[str, Any]):
    """Yield ``(key, value, excerpt)`` for each proposed value, keys like ``topic[0]``."""
    for name, value in proposal.items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                yield f"{name}[{i}]", v["value"], v.get("excerpt", "")
        elif isinstance(value, dict):
            yield name, value["value"], value.get("excerpt", "")


def extract(
    text: str,
    doc_id: str,
    client: LLMClient,
    vocab: Vocabulary | None = None,
    guided: bool = True,
    max_retries: int = 2,
) -> ExtractionResult:
    """Run the full extraction pipeline on one document."""
    vocab = vocab or load_vocabulary()
    model_cls = build_extraction_model(vocab)
    system, user = build_prompt(text, vocab, guided)
    result = ExtractionResult(doc_id=doc_id, guided=guided, model=client.name, attempts=0)
    feedback = ""
    first_grounding_done = False
    validated: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 2):
        result.attempts = attempt
        raw = client.complete(system, user + feedback)
        result.raw_responses.append(raw)
        try:
            data = parse_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            result.errors.append(f"attempt {attempt}: invalid JSON ({exc})")
            feedback = (f"\n\nYour previous answer was not valid JSON ({exc}). "
                        "Answer again with ONE valid JSON object only.")
            continue

        grounded, oov, total = ground(data, vocab)
        if not first_grounding_done:  # metrics are computed on the first parsable answer
            result.out_of_ontology, result.n_values_proposed = oov, total
            first_grounding_done = True
        try:
            validated = model_cls.model_validate(grounded).model_dump()
        except ValidationError as exc:
            result.errors.append(f"attempt {attempt}: schema error ({exc.error_count()} errors)")
            feedback = (f"\n\nYour previous answer did not match the schema:\n{exc}\n"
                        "Answer again with ONE valid JSON object only.")
            validated = None
            continue

        if guided and oov and attempt <= max_retries:
            bad = ", ".join(f"{o['field']}={o['value']!r}" for o in oov)
            result.errors.append(f"attempt {attempt}: out-of-ontology values: {bad}")
            feedback = (f"\n\nThese values are not in the ALLOWED VOCABULARY: {bad}. "
                        "Replace them with allowed identifiers or remove them. Answer with JSON only.")
            continue
        break

    if validated is None:
        return result
    result.proposal = validated
    result.faithfulness = {k: excerpt_in_source(e, text) for k, _, e in iter_values(validated) if k != "language"}
    shacl = validate_graph(annotations_to_graph([proposal_to_annotation(doc_id, validated)]))
    result.shacl_conforms, result.shacl_report = shacl.conforms, shacl.report_text
    return result


def proposals_dir(guided: bool, base: Path = OUTPUT_DIR / "proposals") -> Path:
    """Directory where proposals of one condition are stored."""
    return base / ("guided" if guided else "unguided")


def save_result(result: ExtractionResult, out_dir: Path) -> Path:
    """Write a result as JSON and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result.doc_id}.json"
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ontology-guided LLM metadata extraction")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="process every file in data/raw")
    parser.add_argument("--no-guidance", action="store_true", help="ablation: do not inject the ontology")
    parser.add_argument("--provider", help="ollama | openai | mock (default: LLM_PROVIDER)")
    parser.add_argument("--model", help="model name (default: LLM_MODEL)")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(RAW_DIR.glob("*.txt")) if args.all else args.files
    if not files:
        parser.error("give files or --all")
    guided = not args.no_guidance
    client = get_client(args.provider, args.model)
    vocab = load_vocabulary()
    out_dir = args.out or proposals_dir(guided)

    for f in files:
        text = f.read_text(encoding="utf-8")
        if args.show_prompt:
            print("\n\n".join(build_prompt(text, vocab, guided)))
        res = extract(text, f.stem, client, vocab, guided=guided, max_retries=args.retries)
        path = save_result(res, out_dir)
        faithful = sum(res.faithfulness.values())
        print(f"{f.stem}: attempts={res.attempts} oov={len(res.out_of_ontology)}/{res.n_values_proposed} "
              f"faithful={faithful}/{len(res.faithfulness)} shacl={'ok' if res.shacl_conforms else 'FAIL'} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

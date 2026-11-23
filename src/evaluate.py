"""Evaluation of LLM proposals against the gold annotations, with an ablation study.

Two conditions are compared on the same documents and the same post-processing:

* **guided**: the vocabulary generated from the ontology is injected in the prompt;
* **unguided**: the prompt only describes the fields in natural language.

Metrics (per condition):

* precision / recall / F1 per property (set comparison per document, micro-averaged);
* out-of-ontology rate: controlled values proposed by the model (first answer, before
  validation) that cannot be grounded to an ontology identifier;
* schema validity (a pydantic-valid proposal was obtained) and mean number of attempts;
* SHACL conformance rate of the resulting RDF records;
* faithfulness: share of proposed values whose excerpt is really a span of the source.

Usage (PowerShell)::

    python -m src.evaluate                        # provider from .env (Ollama by default)
    python -m src.evaluate --provider mock        # offline, deterministic
    python -m src.evaluate --cached               # reuse output/proposals/*
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from src.build_graph import load_gold
from src.extract import ExtractionResult, extract, proposal_to_annotation, proposals_dir, save_result
from src.llm import LLMClient, get_client
from src.paths import OUTPUT_DIR, RAW_DIR
from src.vocabulary import load_vocabulary, normalize

CONTROLLED_PROPS = ("resource_type", "target_audience", "objective", "topic", "format", "usage_condition")
DATA_PROPS = ("title", "publication_year", "language")
ALL_PROPS = CONTROLLED_PROPS + DATA_PROPS


@dataclass
class PRF:
    """True positives, false positives and false negatives with derived scores."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, predicted: set[Any], gold: set[Any]) -> None:
        self.tp += len(predicted & gold)
        self.fp += len(predicted - gold)
        self.fn += len(gold - predicted)


def _as_set(prop: str, value: Any) -> set[Any]:
    if value is None or value == "":
        return set()
    values = value if isinstance(value, list) else [value]
    if prop == "title":
        return {normalize(str(v)) for v in values}
    return set(values)


@dataclass
class ConditionReport:
    """All metrics for one condition (guided or unguided)."""

    name: str
    n_docs: int
    per_property: dict[str, PRF]
    controlled_micro: PRF
    n_values_proposed: int
    n_out_of_ontology: int
    n_schema_valid: int
    mean_attempts: float
    n_shacl_conform: int
    n_faithful: int
    n_values_with_excerpt: int
    oov_examples: list[str]

    @property
    def oov_rate(self) -> float:
        return self.n_out_of_ontology / self.n_values_proposed if self.n_values_proposed else 0.0

    @property
    def shacl_rate(self) -> float:
        return self.n_shacl_conform / self.n_docs if self.n_docs else 0.0

    @property
    def faithfulness_rate(self) -> float:
        return self.n_faithful / self.n_values_with_excerpt if self.n_values_with_excerpt else 0.0

    @property
    def schema_rate(self) -> float:
        return self.n_schema_valid / self.n_docs if self.n_docs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.name, "n_docs": self.n_docs,
            "per_property": {k: {"precision": v.precision, "recall": v.recall, "f1": v.f1,
                                 "tp": v.tp, "fp": v.fp, "fn": v.fn} for k, v in self.per_property.items()},
            "controlled_micro": {"precision": self.controlled_micro.precision,
                                 "recall": self.controlled_micro.recall, "f1": self.controlled_micro.f1},
            "out_of_ontology_rate": self.oov_rate, "schema_valid_rate": self.schema_rate,
            "mean_attempts": self.mean_attempts, "shacl_conformance_rate": self.shacl_rate,
            "faithfulness_rate": self.faithfulness_rate,
        }


def score(results: list[ExtractionResult], gold: dict[str, dict[str, Any]], name: str) -> ConditionReport:
    """Compute all metrics of one condition."""
    per_prop = {p: PRF() for p in ALL_PROPS}
    micro = PRF()
    oov_examples: list[str] = []
    for res in results:
        g = gold[res.doc_id]
        pred = proposal_to_annotation(res.doc_id, res.proposal) if res.proposal else {}
        for prop in ALL_PROPS:
            p_set, g_set = _as_set(prop, pred.get(prop)), _as_set(prop, g.get(prop))
            per_prop[prop].add(p_set, g_set)
            if prop in CONTROLLED_PROPS:
                micro.add(p_set, g_set)
        oov_examples += [f"{o['field']}: {o['value']!r}" for o in res.out_of_ontology]
    return ConditionReport(
        name=name,
        n_docs=len(results),
        per_property=per_prop,
        controlled_micro=micro,
        n_values_proposed=sum(r.n_values_proposed for r in results),
        n_out_of_ontology=sum(len(r.out_of_ontology) for r in results),
        n_schema_valid=sum(r.valid_json for r in results),
        mean_attempts=sum(r.attempts for r in results) / len(results) if results else 0.0,
        n_shacl_conform=sum(r.shacl_conforms for r in results),
        n_faithful=sum(sum(r.faithfulness.values()) for r in results),
        n_values_with_excerpt=sum(len(r.faithfulness) for r in results),
        oov_examples=oov_examples,
    )


def run_condition(client: LLMClient, guided: bool, cached: bool = False, max_retries: int = 2) -> list[ExtractionResult]:
    """Extract (or load cached proposals for) every raw document in one condition."""
    out_dir = proposals_dir(guided)
    vocab = load_vocabulary()
    results = []
    for f in sorted(RAW_DIR.glob("*.txt")):
        cache = out_dir / f"{f.stem}.json"
        if cached and cache.exists():
            results.append(ExtractionResult.from_dict(json.loads(cache.read_text(encoding="utf-8"))))
            continue
        res = extract(f.read_text(encoding="utf-8"), f.stem, client, vocab, guided=guided, max_retries=max_retries)
        save_result(res, out_dir)
        print(f"  [{'guided' if guided else 'unguided'}] {f.stem}: attempts={res.attempts} "
              f"oov={len(res.out_of_ontology)} shacl={'ok' if res.shacl_conforms else 'FAIL'}")
        results.append(res)
    return results


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(guided: ConditionReport, unguided: ConditionReport, model: str) -> str:
    """Render the evaluation report as Markdown."""
    lines = [
        "# Evaluation report",
        "",
        f"- Model: `{model}`",
        f"- Date: {date.today().isoformat()}",
        f"- Documents: {guided.n_docs} (synthetic, see `data/`)",
        "",
    ]
    if model.startswith("mock"):
        lines += [
            "> **Note:** these numbers were produced by the offline `mock` provider, a rule-based keyword",
            "> matcher used for testing and CI. They show that the pipeline works end to end and how the",
            "> metrics behave; they are **not** results of a language model. Run",
            "> `python -m src.evaluate --provider ollama` to get LLM results.",
            "",
        ]
    lines += [
        "## Ablation: with vs without ontology guidance",
        "",
        "| Metric | Guided (ontology in prompt) | Unguided | Δ |",
        "|---|---:|---:|---:|",
    ]

    def row(label: str, g: float, u: float, pct: bool = True) -> str:
        fmt = _pct if pct else (lambda v: f"{v:.2f}")
        delta = (g - u) * (100 if pct else 1)
        return f"| {label} | {fmt(g)} | {fmt(u)} | {delta:+.1f}{' pts' if pct else ''} |"

    lines += [
        row("Micro-F1 (controlled properties)", guided.controlled_micro.f1, unguided.controlled_micro.f1),
        row("Micro-precision (controlled)", guided.controlled_micro.precision, unguided.controlled_micro.precision),
        row("Micro-recall (controlled)", guided.controlled_micro.recall, unguided.controlled_micro.recall),
        row("Out-of-ontology values (before validation) ↓", guided.oov_rate, unguided.oov_rate),
        row("Schema-valid outputs", guided.schema_rate, unguided.schema_rate),
        row("SHACL-conformant records", guided.shacl_rate, unguided.shacl_rate),
        row("Faithful excerpts", guided.faithfulness_rate, unguided.faithfulness_rate),
        row("Mean attempts (retries) ↓", guided.mean_attempts, unguided.mean_attempts, pct=False),
        "",
        "## Per-property scores",
        "",
        "| Property | P (guided) | R (guided) | F1 (guided) | P (unguided) | R (unguided) | F1 (unguided) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prop in ALL_PROPS:
        g, u = guided.per_property[prop], unguided.per_property[prop]
        lines.append(f"| `{prop}` | {g.precision:.2f} | {g.recall:.2f} | {g.f1:.2f} | "
                     f"{u.precision:.2f} | {u.recall:.2f} | {u.f1:.2f} |")
    lines += [
        "",
        "## Counts",
        "",
        "| | Guided | Unguided |",
        "|---|---:|---:|",
        f"| Controlled values proposed | {guided.n_values_proposed} | {unguided.n_values_proposed} |",
        f"| Out-of-ontology values | {guided.n_out_of_ontology} | {unguided.n_out_of_ontology} |",
        f"| SHACL-conformant records | {guided.n_shacl_conform}/{guided.n_docs} | {unguided.n_shacl_conform}/{unguided.n_docs} |",
        f"| Faithful excerpts | {guided.n_faithful}/{guided.n_values_with_excerpt} | "
        f"{unguided.n_faithful}/{unguided.n_values_with_excerpt} |",
        "",
        "## Examples of out-of-ontology values (unguided)",
        "",
    ]
    examples = sorted(set(unguided.oov_examples))[:15]
    lines += [f"- {e}" for e in examples] or ["- none"]
    lines += [
        "",
        "## Definitions",
        "",
        "- **P/R/F1**: for each document, the set of predicted values of a property is compared with the gold set;",
        "  counts are summed over documents (micro-average). Out-of-ontology values are dropped before scoring.",
        "- **Out-of-ontology rate**: controlled values in the model's first parsable answer that cannot be grounded",
        "  to an ontology individual (by id, label or alternative label, case/accent-insensitive).",
        "- **SHACL conformance**: the RDF record built from the validated proposal satisfies `ontology/shapes.ttl`.",
        "- **Faithfulness**: the excerpt given as justification is found verbatim (modulo case/whitespace) in the source.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate extraction vs gold, guided vs unguided")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--cached", action="store_true", help="reuse proposals in output/proposals")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "evaluation.md")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    client = get_client(args.provider, args.model)
    gold = {g["id"]: g for g in load_gold()}
    print(f"Evaluating {client.name} on {len(gold)} documents")
    guided = score(run_condition(client, True, args.cached, args.retries), gold, "guided")
    unguided = score(run_condition(client, False, args.cached, args.retries), gold, "unguided")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(guided, unguided, client.name), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"model": client.name, "guided": guided.to_dict(), "unguided": unguided.to_dict()}, indent=2),
        encoding="utf-8",
    )
    print(f"F1 guided={guided.controlled_micro.f1:.3f} unguided={unguided.controlled_micro.f1:.3f} "
          f"| OOV guided={_pct(guided.oov_rate)} unguided={_pct(unguided.oov_rate)}")
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

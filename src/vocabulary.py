"""Controlled vocabulary generated dynamically from the OWL ontology.

Nothing here is hard-coded: the allowed values of each metadata field are found by
reading the ``rdfs:range`` of the corresponding object property and collecting every
``owl:NamedIndividual`` typed with that class or one of its subclasses. Resource types
are the subclasses of ``abx:Resource``. Changing the ontology changes the prompt, the
pydantic schema and the grounding step automatically.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from src.namespaces import ABX
from src.rdf_mapping import OBJECT_FIELDS
from src.validate import load_ontology

#: controlled JSON fields -> ontology property (format is single-valued)
CONTROLLED_FIELDS: dict[str, URIRef] = {**OBJECT_FIELDS, "format": ABX.hasFormat}
MULTI_VALUED: frozenset[str] = frozenset(OBJECT_FIELDS)
LANGUAGES: tuple[str, ...] = ("fr", "en")


def normalize(text: str) -> str:
    """Lower-case, strip accents, punctuation and a final plural ``s`` for lenient matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = re.sub(r"[_\-/'’·.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in text.split())


@dataclass(frozen=True)
class Term:
    """One allowed value: an ontology individual (or a class, for resource types)."""

    id: str
    iri: URIRef
    label_en: str
    label_fr: str
    comment: str = ""
    alt_labels: tuple[str, ...] = ()
    parent: str = ""

    def surface_forms(self) -> set[str]:
        """All strings that should resolve to this term (id, labels, alternative labels)."""
        return {self.id, self.label_en, self.label_fr, *self.alt_labels}


@dataclass
class Vocabulary:
    """Allowed values per field, plus a lookup table for grounding free-text values."""

    fields: dict[str, list[Term]]
    resource_types: list[Term]
    _index: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name, terms in {**self.fields, "resource_type": self.resource_types}.items():
            idx: dict[str, str] = {}
            for t in terms:
                for form in t.surface_forms():
                    idx.setdefault(normalize(form), t.id)
            self._index[name] = idx

    def ids(self, field_name: str) -> list[str]:
        """Allowed identifiers for a field (``resource_type`` included)."""
        terms = self.resource_types if field_name == "resource_type" else self.fields[field_name]
        return [t.id for t in terms]

    def resolve(self, field_name: str, value: object) -> str | None:
        """Ground a proposed value to an ontology id, or return ``None`` if it is out of the ontology."""
        if not isinstance(value, str) or not value.strip():
            return None
        index = self._index.get(field_name, {})
        if value in index.values():
            return value
        return index.get(normalize(value))

    def to_prompt(self) -> str:
        """Render the vocabulary as a compact, LLM-readable list."""
        lines = ["resource_type (exactly one):"]
        lines += [f"  - {t.id}: {t.label_en} / {t.label_fr}. {t.comment}" for t in self.resource_types]
        for name, terms in self.fields.items():
            card = "list, zero or more" if name in MULTI_VALUED else "exactly one"
            lines.append(f"{name} ({card}):")
            for t in terms:
                extra = f" [{t.parent}]" if t.parent else ""
                lines.append(f"  - {t.id}: {t.label_en} / {t.label_fr}{extra}. {t.comment}".rstrip())
        lines.append(f"language (exactly one): {', '.join(LANGUAGES)}")
        return "\n".join(lines)


def _label(g: Graph, s: URIRef, lang: str) -> str:
    for lbl in g.objects(s, RDFS.label):
        if getattr(lbl, "language", None) == lang:
            return str(lbl)
    return str(s).rsplit("#", 1)[-1]


def _comment(g: Graph, s: URIRef) -> str:
    for c in g.objects(s, RDFS.comment):
        if getattr(c, "language", None) == "en":
            return str(c)
    return ""


def _local(iri: URIRef) -> str:
    return str(iri).removeprefix(str(ABX))


def vocabulary_from_graph(g: Graph) -> Vocabulary:
    """Build the vocabulary from an ontology graph."""
    fields: dict[str, list[Term]] = {}
    for name, prop in CONTROLLED_FIELDS.items():
        range_cls = g.value(prop, RDFS.range)
        if range_cls is None:
            raise ValueError(f"property {prop} has no rdfs:range in the ontology")
        classes = set(g.transitive_subjects(RDFS.subClassOf, range_cls))
        terms: list[Term] = []
        for cls in sorted(classes):
            for ind in sorted(g.subjects(RDF.type, cls)):
                if (ind, RDF.type, OWL.NamedIndividual) not in g or not isinstance(ind, URIRef):
                    continue
                terms.append(
                    Term(
                        id=_local(ind), iri=ind,
                        label_en=_label(g, ind, "en"), label_fr=_label(g, ind, "fr"),
                        # individuals reuse the comment of their (most specific) class
                        comment=_comment(g, ind) or (_comment(g, cls) if cls != range_cls else ""),
                        alt_labels=tuple(sorted(str(a) for a in g.objects(ind, SKOS.altLabel))),
                        parent=_label(g, cls, "en") if cls != range_cls else "",
                    )
                )
        fields[name] = sorted(terms, key=lambda t: t.id)
    resource_types = [
        Term(id=_local(c), iri=c, label_en=_label(g, c, "en"), label_fr=_label(g, c, "fr"), comment=_comment(g, c))
        for c in sorted(g.transitive_subjects(RDFS.subClassOf, ABX.Resource))
        if c != ABX.Resource and isinstance(c, URIRef)
    ]
    return Vocabulary(fields=fields, resource_types=resource_types)


@lru_cache(maxsize=1)
def load_vocabulary() -> Vocabulary:
    """Vocabulary of the project ontology (cached)."""
    return vocabulary_from_graph(load_ontology())

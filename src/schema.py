"""Pydantic schema built at runtime from the ontology vocabulary.

Every controlled value is typed as a ``Literal[...]`` of the ontology identifiers, so a
value outside the ontology is rejected by pydantic itself. Each value carries the
``excerpt`` of the source text that justifies it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.vocabulary import LANGUAGES, MULTI_VALUED, Vocabulary, load_vocabulary


class TextEvidence(BaseModel):
    """A free-text value (title, publisher) and its justification."""

    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1)
    excerpt: str = ""


class YearEvidence(BaseModel):
    """A publication year and its justification."""

    model_config = ConfigDict(extra="forbid")
    value: int = Field(ge=1990, le=2100)
    excerpt: str = ""


def _evidence_model(name: str, allowed: list[str]) -> type[BaseModel]:
    """An evidence model whose ``value`` must be one of ``allowed``."""
    literal = Literal[tuple(allowed)]  # type: ignore[valid-type]
    return create_model(  # type: ignore[call-overload]
        f"{name}Evidence",
        __config__=ConfigDict(extra="forbid"),
        value=(literal, ...),
        excerpt=(str, ""),
    )


def build_extraction_model(vocab: Vocabulary) -> type[BaseModel]:
    """Create the ``ResourceExtraction`` pydantic model for the given vocabulary."""
    fields: dict[str, Any] = {
        # cardinalities (exactly one title, >= 1 audience...) are checked by SHACL, not here:
        # pydantic guarantees structure and membership in the ontology vocabulary.
        "resource_type": (Optional[_evidence_model("ResourceType", vocab.ids("resource_type"))], None),
        "title": (Optional[TextEvidence], None),
        "publisher": (Optional[TextEvidence], None),
        "publication_year": (Optional[YearEvidence], None),
        "language": (Optional[_evidence_model("Language", list(LANGUAGES))], None),
    }
    for name in vocab.fields:
        model = _evidence_model("".join(p.title() for p in name.split("_")), vocab.ids(name))
        fields[name] = (list[model], []) if name in MULTI_VALUED else (Optional[model], None)  # type: ignore[valid-type]
    return create_model("ResourceExtraction", __config__=ConfigDict(extra="ignore"), **fields)  # type: ignore[call-overload]


@lru_cache(maxsize=1)
def extraction_model() -> type[BaseModel]:
    """The extraction model for the project ontology (cached)."""
    return build_extraction_model(load_vocabulary())

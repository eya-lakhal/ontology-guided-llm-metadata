"""Tests for vocabulary generation, schema, grounding and the extraction loop."""

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError
from rdflib import Graph, Literal
from rdflib.namespace import OWL, RDF, RDFS

from src.extract import build_prompt, excerpt_in_source, extract, ground, parse_json
from src.llm import GUIDED_MARKER, MockClient, get_client
from src.namespaces import ABX
from src.paths import RAW_DIR
from src.schema import build_extraction_model, extraction_model
from src.validate import load_ontology
from src.vocabulary import load_vocabulary, normalize, vocabulary_from_graph

R01 = (RAW_DIR / "r01.txt").read_text(encoding="utf-8")


@dataclass
class ScriptedClient:
    """Fake LLM returning pre-written answers, recording the prompts it receives."""

    answers: list[str]
    name: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(user)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


def answer(**overrides) -> str:
    base = {
        "resource_type": {"value": "Poster", "excerpt": "Affiche éditée en 2023"},
        "title": {"value": "Bandelette urinaire en EHPAD", "excerpt": "Bandelette urinaire en EHPAD"},
        "language": {"value": "fr", "excerpt": ""},
        "target_audience": [{"value": "nurse", "excerpt": "aux infirmières des EHPAD"}],
        "topic": [{"value": "uti", "excerpt": "infection urinaire"}],
        "format": {"value": "print", "excerpt": "Format affiche A3 imprimable"},
        "objective": [], "usage_condition": [],
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ---------------------------------------------------------------- vocabulary


def test_vocabulary_is_generated_from_ontology() -> None:
    vocab = load_vocabulary()
    assert {"nurse", "physician", "care_assistant", "pharmacist", "resident", "family"} == set(vocab.ids("target_audience"))
    assert "uti" in vocab.ids("topic") and "respiratory_infection" in vocab.ids("topic")
    assert set(vocab.ids("resource_type")) == {"Poster", "PracticalSheet", "Training", "Video", "Tool"}


def test_vocabulary_follows_ontology_changes() -> None:
    g = Graph() + load_ontology()
    g.add((ABX.dentist, RDF.type, OWL.NamedIndividual))
    g.add((ABX.dentist, RDF.type, ABX.HealthProfessional))
    g.add((ABX.dentist, RDFS.label, Literal("Dentists", lang="en")))
    vocab = vocabulary_from_graph(g)
    assert "dentist" in vocab.ids("target_audience")
    assert "dentist" in vocab.to_prompt()
    model = build_extraction_model(vocab)
    model.model_validate({"target_audience": [{"value": "dentist", "excerpt": ""}]})


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("target_audience", "nurse", "nurse"),
        ("target_audience", "Infirmières", "nurse"),
        ("target_audience", "MEDECINS", "physician"),
        ("target_audience", "Doctors", "physician"),
        ("topic", "Infection urinaire", "uti"),
        ("resource_type", "practical sheet", "PracticalSheet"),
        ("format", "PDF", "pdf"),
        ("topic", "Astrology", None),
        ("topic", "", None),
        ("topic", 42, None),
    ],
)
def test_resolve(field_name: str, value: object, expected: str | None) -> None:
    assert load_vocabulary().resolve(field_name, value) == expected


def test_normalize() -> None:
    assert normalize("Aides-Soignants") == normalize("aide soignant")


# ---------------------------------------------------------------- schema


def test_schema_rejects_out_of_ontology_value() -> None:
    with pytest.raises(ValidationError):
        extraction_model().model_validate({"topic": [{"value": "astrology", "excerpt": ""}]})


def test_schema_rejects_value_from_wrong_field() -> None:
    with pytest.raises(ValidationError):
        extraction_model().model_validate({"format": {"value": "nurse", "excerpt": ""}})


def test_schema_accepts_valid_proposal() -> None:
    m = extraction_model().model_validate(json.loads(answer()))
    assert m.model_dump()["topic"][0]["value"] == "uti"


# ---------------------------------------------------------------- prompt / parsing


def test_prompt_contains_vocabulary_only_when_guided() -> None:
    vocab = load_vocabulary()
    _, guided = build_prompt(R01, vocab, guided=True)
    _, unguided = build_prompt(R01, vocab, guided=False)
    assert GUIDED_MARKER in guided and "care_assistant" in guided
    assert GUIDED_MARKER not in unguided and "care_assistant" not in unguided
    assert R01.strip() in guided and R01.strip() in unguided


def test_parse_json_tolerates_fences_and_prose() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_ground_maps_labels_and_reports_oov() -> None:
    data = {"topic": [{"value": "Infection urinaire", "excerpt": "x"}, {"value": "Geriatrics", "excerpt": "y"}],
            "format": "PDF"}
    grounded, oov, total = ground(data, load_vocabulary())
    assert grounded["topic"] == [{"value": "uti", "excerpt": "x"}]
    assert grounded["format"]["value"] == "pdf"
    assert oov == [{"field": "topic", "value": "Geriatrics", "excerpt": "y"}]
    assert total == 3


def test_excerpt_faithfulness() -> None:
    assert excerpt_in_source("affiche  A3 IMPRIMABLE", R01)
    assert not excerpt_in_source("affiche A4 plastifiée", R01)
    assert not excerpt_in_source("", R01)


# ---------------------------------------------------------------- extraction loop


def test_retry_on_invalid_json() -> None:
    client = ScriptedClient(["Sorry, I cannot {", answer()])
    res = extract(R01, "r01", client, guided=True)
    assert res.attempts == 2
    assert "not valid JSON" in client.prompts[1]
    assert res.proposal is not None and res.shacl_conforms


def test_gives_up_after_max_retries() -> None:
    res = extract(R01, "r01", ScriptedClient(["nope"]), guided=True, max_retries=2)
    assert res.attempts == 3 and res.proposal is None


def test_retry_on_out_of_ontology_in_guided_mode() -> None:
    bad = answer(topic=[{"value": "Geriatrics", "excerpt": "EHPAD"}])
    client = ScriptedClient([bad, answer()])
    res = extract(R01, "r01", client, guided=True)
    assert res.attempts == 2
    assert "Geriatrics" in client.prompts[1]
    assert len(res.out_of_ontology) == 1  # measured on the first answer
    assert res.proposal["topic"][0]["value"] == "uti"


def test_oov_values_are_dropped_and_shacl_reports_missing_topic() -> None:
    bad = answer(topic=[{"value": "Geriatrics", "excerpt": "EHPAD"}])
    res = extract(R01, "r01", ScriptedClient([bad]), guided=False)
    assert res.attempts == 1  # no vocabulary feedback in the unguided condition
    assert res.proposal["topic"] == []
    assert not res.shacl_conforms


def test_unfaithful_excerpt_is_flagged() -> None:
    hallucinated = answer(target_audience=[{"value": "pharmacist", "excerpt": "pour les pharmaciens"}])
    res = extract(R01, "r01", ScriptedClient([hallucinated]), guided=True)
    assert res.faithfulness["target_audience[0]"] is False
    assert res.faithfulness["format"] is True


def test_mock_end_to_end_guided() -> None:
    res = extract(R01, "r01", MockClient(), guided=True)
    assert res.shacl_conforms
    assert res.proposal["resource_type"]["value"] == "Poster"
    assert {v["value"] for v in res.proposal["topic"]} == {"uti"}
    assert all(res.faithfulness.values())


def test_get_client_mock() -> None:
    assert get_client("mock").name.startswith("mock")
    with pytest.raises(ValueError):
        get_client("unknown-provider")

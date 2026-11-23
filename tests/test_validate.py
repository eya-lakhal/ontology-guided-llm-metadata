"""Tests for SHACL validation."""

from rdflib import Graph

from src.paths import EXAMPLES_FILE
from src.validate import main, validate_file, validate_graph

PREFIXES = """
@prefix abx:  <https://eya-lakhal.github.io/ontology-guided-llm-metadata/abx#> .
@prefix abxd: <https://eya-lakhal.github.io/ontology-guided-llm-metadata/data/> .
"""

VALID = PREFIXES + """
abxd:r1 a abx:PracticalSheet ;
    abx:title "Hand rub in 7 steps" ;
    abx:hasTargetAudience abx:care_assistant ;
    abx:coversTopic abx:hand_hygiene ;
    abx:hasFormat abx:pdf .
"""

TITLE = 'abx:title "Hand rub in 7 steps" ;'


def _g(ttl: str) -> Graph:
    return Graph().parse(data=ttl, format="turtle")


def test_examples_conform() -> None:
    assert validate_file(EXAMPLES_FILE).conforms


def test_cli_returns_zero_on_examples() -> None:
    assert main([str(EXAMPLES_FILE)]) == 0


def test_minimal_valid_resource() -> None:
    assert validate_graph(_g(VALID)).conforms


def test_missing_title_fails() -> None:
    result = validate_graph(_g(VALID.replace(TITLE, "")))
    assert not result.conforms
    assert "title" in result.report_text


def test_two_titles_fail() -> None:
    ttl = VALID.replace(TITLE, 'abx:title "A title" , "Another title" ;')
    assert not validate_graph(_g(ttl)).conforms


def test_missing_audience_fails() -> None:
    ttl = VALID.replace("abx:hasTargetAudience abx:care_assistant ;", "")
    assert not validate_graph(_g(ttl)).conforms


def test_missing_topic_fails() -> None:
    ttl = VALID.replace("abx:coversTopic abx:hand_hygiene ;", "")
    assert not validate_graph(_g(ttl)).conforms


def test_two_formats_fail() -> None:
    ttl = VALID.replace("abx:hasFormat abx:pdf .", "abx:hasFormat abx:pdf , abx:print .")
    assert not validate_graph(_g(ttl)).conforms


def test_value_outside_ontology_fails() -> None:
    ttl = VALID.replace("abx:coversTopic abx:hand_hygiene", "abx:coversTopic abx:astrology")
    assert not validate_graph(_g(ttl)).conforms


def test_wrong_range_fails() -> None:
    # a Format individual used as a topic
    ttl = VALID.replace("abx:coversTopic abx:hand_hygiene", "abx:coversTopic abx:pdf")
    assert not validate_graph(_g(ttl)).conforms


def test_bad_language_fails() -> None:
    ttl = VALID.replace("abx:hasFormat abx:pdf .", 'abx:hasFormat abx:pdf ; abx:language "french" .')
    assert not validate_graph(_g(ttl)).conforms

"""Tests for the human-in-the-loop review CLI."""

import json
from pathlib import Path

from rdflib import Graph

from src.extract import extract, save_result
from src.llm import MockClient
from src.namespaces import ABX, ABXD
from src.paths import RAW_DIR
from src.review import ReviewSession, add_to_kg, append_log, main, reviewed_ids

R01 = (RAW_DIR / "r01.txt").read_text(encoding="utf-8")


class Responder:
    """Answers prompts by kind; per-field choices can be scripted (field -> list of answers)."""

    def __init__(self, choices: dict[str, list[str]] | None = None, adds: dict[str, list[str]] | None = None,
                 edits: list[str] | None = None, default: str = "a") -> None:
        self.choices = {k: list(v) for k, v in (choices or {}).items()}
        self.adds = {k: list(v) for k, v in (adds or {}).items()}
        self.edits = list(edits or [])
        self.default = default
        self.current_field = ""
        self.printed: list[str] = []

    def say(self, msg: str) -> None:
        self.printed.append(msg)
        stripped = msg.strip()
        if stripped and ":" in stripped and not stripped.startswith(("excerpt", "SHACL", "(")):
            self.current_field = stripped.split(":", 1)[0]

    def ask(self, prompt: str) -> str:
        if "[a]ccept" in prompt:
            queue = self.choices.get(self.current_field, [])
            return queue.pop(0) if queue else self.default
        if prompt.strip().startswith("add a value to"):
            name = prompt.split("add a value to", 1)[1].split("?")[0].strip()
            queue = self.adds.get(name, [])
            return queue.pop(0) if queue else ""
        if "new value" in prompt:
            return self.edits.pop(0)
        if "provide a value" in prompt:
            return "n"
        raise AssertionError(f"unexpected prompt {prompt!r}")


def _result():
    return extract(R01, "r01", MockClient(), guided=True)


def test_accept_all_conforms_and_matches_proposal() -> None:
    r = Responder()
    outcome = ReviewSession(input_fn=r.ask, print_fn=r.say).review(_result(), R01)
    assert outcome.shacl_conforms
    assert outcome.annotation["topic"] == ["uti"]
    assert {d.decision for d in outcome.decisions} == {"accept"}


def test_reject_and_edit_and_add() -> None:
    r = Responder(
        choices={"objective": ["r", "a"], "format": ["e"]},
        adds={"topic": ["Antibiorésistance", "astrology", ""]},
        edits=["PDF"],
    )
    outcome = ReviewSession(input_fn=r.ask, print_fn=r.say).review(_result(), R01)
    assert len(outcome.annotation["objective"]) == len(_result().proposal["objective"]) - 1
    assert outcome.annotation["format"] == "pdf"
    assert outcome.annotation["topic"] == ["uti", "antibiotic_resistance"]
    kinds = [d.decision for d in outcome.decisions]
    assert {"reject", "edit", "add"} <= set(kinds)
    assert any("not in the ontology" in p for p in r.printed)


def test_rejecting_all_topics_blocks_kg_insertion(tmp_path: Path) -> None:
    r = Responder(choices={"topic": ["r"]})
    outcome = ReviewSession(input_fn=r.ask, print_fn=r.say).review(_result(), R01)
    assert not outcome.shacl_conforms
    kg = tmp_path / "kg.ttl"
    assert not add_to_kg(outcome, kg)
    assert not kg.exists()


def test_validated_record_is_added_and_logged(tmp_path: Path) -> None:
    r = Responder()
    outcome = ReviewSession(input_fn=r.ask, print_fn=r.say).review(_result(), R01)
    kg, log = tmp_path / "kg.ttl", tmp_path / "log.json"
    assert add_to_kg(outcome, kg)
    assert add_to_kg(outcome, kg)  # idempotent: replaces the record
    g = Graph().parse(kg)
    assert (ABXD.r01, ABX.coversTopic, ABX.uti) in g
    assert len(list(g.objects(ABXD.r01, ABX.title))) == 1
    append_log(outcome, log)
    entries = json.loads(log.read_text(encoding="utf-8"))
    assert entries[0]["added_to_kg"] and entries[0]["decisions"]
    assert reviewed_ids(log) == {"r01"}


def test_cli_with_scripted_stdin(tmp_path: Path, monkeypatch) -> None:
    props = tmp_path / "proposals"
    save_result(_result(), props)
    r = Responder()
    monkeypatch.setattr("builtins.input", r.ask)
    monkeypatch.setattr("builtins.print", lambda *a, **k: r.say(" ".join(map(str, a))))
    kg, log = tmp_path / "kg.ttl", tmp_path / "log.json"
    assert main(["--proposals", str(props), "--kg", str(kg), "--log", str(log)]) == 0
    assert kg.exists() and log.exists()
    # second run skips already reviewed documents
    assert main(["--proposals", str(props), "--kg", str(kg), "--log", str(log)]) == 0
    assert any("already reviewed" in p for p in r.printed)

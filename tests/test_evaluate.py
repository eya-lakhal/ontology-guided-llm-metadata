"""Tests for the evaluation metrics and report."""

from pathlib import Path

import pytest

from src.build_graph import load_gold
from src.evaluate import PRF, main, render_markdown, score
from src.extract import ExtractionResult


def test_prf() -> None:
    m = PRF()
    m.add({"a", "b"}, {"b", "c"})
    assert (m.tp, m.fp, m.fn) == (1, 1, 1)
    assert m.precision == m.recall == m.f1 == pytest.approx(0.5)
    assert PRF().f1 == 0.0


def _result(doc_id: str, proposal: dict | None, **kw) -> ExtractionResult:
    return ExtractionResult(doc_id=doc_id, guided=True, model="test", attempts=1, proposal=proposal, **kw)


def test_score_perfect_and_empty() -> None:
    gold = {g["id"]: g for g in load_gold()}
    g = gold["r01"]
    perfect = {
        "resource_type": {"value": g["resource_type"], "excerpt": ""},
        "title": {"value": g["title"], "excerpt": ""},
        "publication_year": {"value": g["publication_year"], "excerpt": ""},
        "language": {"value": g["language"], "excerpt": ""},
        "format": {"value": g["format"], "excerpt": ""},
        **{k: [{"value": v, "excerpt": ""} for v in g[k]]
           for k in ("target_audience", "objective", "topic", "usage_condition")},
    }
    rep = score([_result("r01", perfect, shacl_conforms=True, faithfulness={"x": True, "y": False},
                         n_values_proposed=8, out_of_ontology=[{"field": "topic", "value": "X"}])],
                {"r01": g}, "guided")
    assert rep.controlled_micro.f1 == pytest.approx(1.0)
    assert rep.per_property["title"].f1 == pytest.approx(1.0)
    assert rep.shacl_rate == 1.0 and rep.faithfulness_rate == 0.5
    assert rep.oov_rate == pytest.approx(1 / 8)

    empty = score([_result("r01", None)], {"r01": g}, "unguided")
    assert empty.controlled_micro.recall == 0.0 and empty.schema_rate == 0.0


def test_mock_evaluation_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.extract.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr("src.evaluate.proposals_dir",
                        lambda guided: tmp_path / ("guided" if guided else "unguided"))
    out = tmp_path / "evaluation.md"
    assert main(["--provider", "mock", "--out", str(out)]) == 0
    report = out.read_text(encoding="utf-8")
    assert "Ablation" in report and "not** results of a language model" in report
    assert out.with_suffix(".json").exists()


def test_render_contains_all_properties() -> None:
    gold = {g["id"]: g for g in load_gold()}
    rep = score([_result("r01", None)], {"r01": gold["r01"]}, "x")
    md = render_markdown(rep, rep, "ollama:llama3.1:8b")
    for prop in ("target_audience", "topic", "format", "usage_condition"):
        assert f"`{prop}`" in md
    assert "mock" not in md.split("## Ablation")[0]

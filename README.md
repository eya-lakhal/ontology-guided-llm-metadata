# Ontology-guided LLM metadata extraction

> Work in progress.

Characterising antibiotic stewardship resources (posters, practical sheets, trainings for nursing homes)
with an LLM guided by an OWL ontology, with SPARQL querying, SHACL validation, human-in-the-loop review
and evaluation.

## Repository layout

| Path | Content |
|------|---------|
| `ontology/` | OWL ontology (Turtle) and SHACL shapes |
| `data/raw/` | Resource descriptions (plain text, synthetic) |
| `data/gold/` | Expected (gold) annotations, JSON |
| `queries/` | SPARQL queries |
| `src/` | Python pipeline |
| `tests/` | Pytest test suite |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
pytest
```

## License

MIT

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

## Ontology

`ontology/abx_resources.ttl` (namespace `https://eya-lakhal.github.io/ontology-guided-llm-metadata/abx#`, prefix `abx:`)
is an OWL 2 ontology in Turtle with bilingual (EN/FR) labels and comments.

### Opening and checking it in Protégé

1. Install [Protégé](https://protege.stanford.edu/) 5.6+.
2. *File → Open…* and select `ontology/abx_resources.ttl`.
3. Browse the *Classes*, *Object properties*, *Data properties* and *Individuals* tabs.
   Tip: *View → Render by label (rdfs:label)* shows the human-readable labels.
4. *Reasoner → HermiT*, then *Reasoner → Start reasoner*. The ontology should be **consistent**
   and no class should appear under `owl:Nothing` in the inferred hierarchy.
   Individuals are then classified, e.g. `abx:nurse` is inferred to be a `HealthProfessional` and a `TargetAudience`.

## Dataset

> ⚠️ **All data in `data/` is synthetic.** The 10 resource descriptions were written for this project:
> they are realistic in style but the resources, publishers (e.g. "CRAPI Grand-Est", "Northfield Infection
> Prevention Network") and dates are invented. They contain no real patient or institutional data.

- `data/raw/rXX.txt`: plain-text description of one resource (6 in French, 4 in English), first line = title.
- `data/gold/rXX.json`: expected annotation, using only identifiers defined in the ontology, e.g.

```json
{
  "id": "r01", "resource_type": "Poster", "title": "...", "publisher": "CRAPI Grand-Est",
  "publication_year": 2023, "language": "fr",
  "target_audience": ["care_assistant", "nurse"],
  "objective": ["reduce_unnecessary_prescription", "improve_diagnosis"],
  "topic": ["uti"], "format": "print", "usage_condition": ["free_access"], "related_to": ["r05"]
}
```

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

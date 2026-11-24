# Evaluation report

- Model: `ollama:llama3.1:8b`
- Date: 2026-11-24
- Documents: 10 (synthetic, see `data/`)

## Ablation: with vs without ontology guidance

| Metric | Guided (ontology in prompt) | Unguided | Δ |
|---|---:|---:|---:|
| Micro-F1 (controlled properties) | 87.3% | 82.4% | +4.8 pts |
| Micro-precision (controlled) | 92.3% | 100.0% | -7.7 pts |
| Micro-recall (controlled) | 82.8% | 70.1% | +12.6 pts |
| Out-of-ontology values (before validation) ↓ | 0.0% | 33.3% | -33.3 pts |
| Schema-valid outputs | 100.0% | 100.0% | +0.0 pts |
| SHACL-conformant records | 100.0% | 50.0% | +50.0 pts |
| Faithful excerpts | 97.2% | 98.9% | -1.7 pts |
| Mean attempts (retries) ↓ | 1.00 | 1.00 | +0.0 |

## Per-property scores

| Property | P (guided) | R (guided) | F1 (guided) | P (unguided) | R (unguided) | F1 (unguided) |
|---|---:|---:|---:|---:|---:|---:|
| `resource_type` | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.82 |
| `target_audience` | 1.00 | 0.91 | 0.95 | 1.00 | 0.91 | 0.95 |
| `objective` | 0.92 | 0.75 | 0.83 | 1.00 | 0.75 | 0.86 |
| `topic` | 0.86 | 0.67 | 0.75 | 1.00 | 0.67 | 0.80 |
| `format` | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.67 |
| `usage_condition` | 0.73 | 0.73 | 0.73 | 1.00 | 0.45 | 0.62 |
| `title` | 1.00 | 1.00 | 1.00 | 0.80 | 0.80 | 0.80 |
| `publication_year` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `language` | 0.70 | 0.70 | 0.70 | 1.00 | 1.00 | 1.00 |

## Counts

| | Guided | Unguided |
|---|---:|---:|
| Controlled values proposed | 79 | 93 |
| Out-of-ontology values | 0 | 31 |
| SHACL-conformant records | 10/10 | 5/10 |
| Faithful excerpts | 105/108 | 90/91 |

## Examples of out-of-ontology values (unguided)

- format: 'online'
- format: 'printable poster'
- format: 'printed'
- format: 'web'
- objective: 'improve infection detection'
- objective: 'prevent spread of resistant bacteria'
- objective: 'professional training'
- objective: 'reduce unnecessary antibiotic use'
- objective: 'sensitize public'
- resource_type: 'sheet'
- target_audience: 'medical coordinators'
- target_audience: 'nursing home staff'
- topic: 'MRB prevention'
- topic: 'antibiotic stewardship'
- topic: 'antibiotic treatment duration'

## Definitions

- **P/R/F1**: for each document, the set of predicted values of a property is compared with the gold set;
  counts are summed over documents (micro-average). Out-of-ontology values are dropped before scoring.
- **Out-of-ontology rate**: controlled values in the model's first parsable answer that cannot be grounded
  to an ontology individual (by id, label or alternative label, case/accent-insensitive).
- **SHACL conformance**: the RDF record built from the validated proposal satisfies `ontology/shapes.ttl`.
- **Faithfulness**: the excerpt given as justification is found verbatim (modulo case/whitespace) in the source.

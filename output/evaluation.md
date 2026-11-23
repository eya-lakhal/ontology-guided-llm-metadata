# Evaluation report

- Model: `mock:keyword-baseline`
- Date: 2026-11-23
- Documents: 10 (synthetic, see `data/`)

> **Note:** these numbers were produced by the offline `mock` provider, a rule-based keyword
> matcher used for testing and CI. They show that the pipeline works end to end and how the
> metrics behave; they are **not** results of a language model. Run
> `python -m src.evaluate --provider ollama` to get LLM results.

## Ablation: with vs without ontology guidance

| Metric | Guided (ontology in prompt) | Unguided | Δ |
|---|---:|---:|---:|
| Micro-F1 (controlled properties) | 98.3% | 73.8% | +24.5 pts |
| Micro-precision (controlled) | 96.7% | 96.3% | +0.4 pts |
| Micro-recall (controlled) | 100.0% | 59.8% | +40.2 pts |
| Out-of-ontology values (before validation) ↓ | 0.0% | 40.0% | -40.0 pts |
| Schema-valid outputs | 100.0% | 100.0% | +0.0 pts |
| SHACL-conformant records | 100.0% | 80.0% | +20.0 pts |
| Faithful excerpts | 100.0% | 100.0% | +0.0 pts |
| Mean attempts (retries) ↓ | 1.00 | 1.00 | +0.0 |

## Per-property scores

| Property | P (guided) | R (guided) | F1 (guided) | P (unguided) | R (unguided) | F1 (unguided) |
|---|---:|---:|---:|---:|---:|---:|
| `resource_type` | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 | 0.46 |
| `target_audience` | 0.96 | 1.00 | 0.98 | 0.95 | 0.91 | 0.93 |
| `objective` | 0.94 | 1.00 | 0.97 | 1.00 | 0.19 | 0.32 |
| `topic` | 0.95 | 1.00 | 0.97 | 0.93 | 0.78 | 0.85 |
| `format` | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.89 |
| `usage_condition` | 1.00 | 1.00 | 1.00 | 1.00 | 0.36 | 0.53 |
| `title` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `publication_year` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `language` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Counts

| | Guided | Unguided |
|---|---:|---:|
| Controlled values proposed | 90 | 90 |
| Out-of-ontology values | 0 | 36 |
| SHACL-conformant records | 10/10 | 8/10 |
| Faithful excerpts | 110/110 | 74/74 |

## Examples of out-of-ontology values (unguided)

- format: 'Printed poster'
- objective: 'Antibiotic sparing'
- objective: 'Better diagnosis'
- objective: 'Infection prevention'
- objective: 'Prescribing quality'
- objective: 'Staff education'
- resource_type: 'course'
- resource_type: 'fact sheet'
- resource_type: 'software'
- target_audience: 'Nursing home residents'
- topic: 'Antibiotic prescribing'
- usage_condition: 'CC licence'
- usage_condition: 'Professionals only'
- usage_condition: 'Registration needed'
- usage_condition: 'Trainer-led'

## Definitions

- **P/R/F1**: for each document, the set of predicted values of a property is compared with the gold set;
  counts are summed over documents (micro-average). Out-of-ontology values are dropped before scoring.
- **Out-of-ontology rate**: controlled values in the model's first parsable answer that cannot be grounded
  to an ontology individual (by id, label or alternative label, case/accent-insensitive).
- **SHACL conformance**: the RDF record built from the validated proposal satisfies `ontology/shapes.ttl`.
- **Faithfulness**: the excerpt given as justification is found verbatim (modulo case/whitespace) in the source.

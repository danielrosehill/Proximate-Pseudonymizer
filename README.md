# Proximate-Pseudonymizer

A small text-in/text-out model that rewrites a text with its identifying details swapped for
**close, plausible substitutes**, used consistently throughout:

> I'm **Daniel**, I live in **Jerusalem** … **Daniel**
> → I'm **David**, I live in **Tel Aviv** … **David**

It's not a redactor (no `[NAME]` tags) and doesn't output a mapping. The text keeps reading
naturally, so it can be shared with an AI tool, a forum or a dataset without pointing at the
real people and places.

**Status:** data generation and review, 2026-10-21. Base model `Qwen/Qwen3-0.6B`.

## Pipeline

| Step | Script | Where |
| --- | --- | --- |
| 1. Synthetic text + replacement map | `scripts/generate.py submit/collect` | Anthropic Batches, `claude-haiku-5-5` |
| 2. Validate, compute targets, split | `scripts/build.py` | laptop |
| 3. Human review (gold test set first) | private Space [`Proximate-Pseudonymizer-Review`](https://huggingface.co/spaces/danielrosehill/Proximate-Pseudonymizer-Review) (`space/`), then `scripts/pull_reviews.py`; or locally `scripts/review.py` → http://localhost:8767 | HF / laptop |
| 4. Train + predict | `training/train.py` | Modal `ai-training`, L4 |
| 5. Score | `scripts/evaluate.py runs/<run>/pred_*.jsonl` | laptop |
| — Review learnings | `scripts/learnings.py` (corrections grouped by reason, with example edits) | laptop |

Targets are never written by the LLM. They are computed by applying the map
(`scripts/pairs.py`), so every target is consistent within its text and leaves everything
else untouched.

## Read in this order

| File | What it holds |
| --- | --- |
| [`docs/idea.md`](docs/idea.md) | The request in Daniel's words, and the input/output contract |
| [`docs/research/2026-10-21-exists-check.md`](docs/research/2026-10-21-exists-check.md) | Prior art: Eternis Anonymizer (tool-call output, non-commercial), Grammarly BART |
| [`docs/spec.md`](docs/spec.md) | Base model, data design, validation flags, review, metrics and pass bar |
| [`project.json`](project.json) | Machine-readable stage tracker, data and training runs |

## Fit and adaptation

The replacement policy is one person's choice: big cities are replaced too, countries and
public figures are kept, and dates keep their year. The settings are weighted towards Israel,
Ireland and the US, and 80% of the texts are English. For a different policy or language mix,
edit `SYSTEM`, `LOCALES` and `brief()` in `scripts/generate.py`, regenerate, and fine-tune
further from the published checkpoint.

Registered in `ai-models/AI-Model-Development-Workspace/projects.json`. Started with the
`custom-model-planning` plugin.

# Proximate-Pseudonymizer

A small text-in/text-out model that rewrites a text with its identifying details swapped for
**close, plausible substitutes**, used consistently throughout:

> I'm **Daniel**, I live in **Jerusalem** … **Daniel**
> → I'm **David**, I live in **Tel Aviv** … **David**

It's not a redactor (no `[NAME]` tags), and the published model doesn't output a mapping. The text keeps reading
naturally, so it can be shared with an AI tool, a forum or a dataset without pointing at the
real people and places.

**Status: proof of concept, 2026-10-21. Training paused after the first useful checkpoint.**
Model: [`danielrosehill/Proximate-Pseudonymizer-0.6B`](https://huggingface.co/danielrosehill/Proximate-Pseudonymizer-0.6B)
(Qwen3-0.6B full fine-tune, Apache-2.0). Try it in the public demo Space
[`Proximate-Pseudonymizer-Demo`](https://huggingface.co/spaces/danielrosehill/Proximate-Pseudonymizer-Demo)
(CPU, word-level diff of every swap; source in `demo/`). On 295 held-out texts it leaves **9.3% of identifying
details unchanged**; consistency is 0.99 and the rest of the text is 0.998 preserved. **It is
not safe to run unsupervised.** Suggested targets: ≤ 1% leak with a human reviewing every
output, ~0.1% or less without. The model card
([`docs/model-card.md`](docs/model-card.md)) covers how this differs from existing tools
(redaction, Presidio + Faker, Eternis Anonymizer) and what would close the gap: longer
training, a larger base, and the swap-list output mode with copy-back repair
(`training/infer.py`). The run history is in `docs/spec.md` under "Training runs" and
"Probe results".

## Pipeline

| Step | Script | Where |
| --- | --- | --- |
| 1. Synthetic text + replacement map | `scripts/generate.py submit/collect` | Anthropic Batches, `claude-haiku-5-5` |
| 2. Validate, flag too-close swaps, compute targets, split | `uv run scripts/build.py` (uses `scripts/similarity.py`) | laptop |
| 2b. Re-pick only the too-close swaps | `scripts/repair.py submit/collect`, then `build.py` again | Anthropic Batches, `claude-haiku-5-5` |
| 3. Human review (gold test set first) | private Space [`Proximate-Pseudonymizer-Review`](https://huggingface.co/spaces/danielrosehill/Proximate-Pseudonymizer-Review) (`space/`), then `scripts/pull_reviews.py`; or locally `scripts/review.py` → http://localhost:8767 | HF / laptop |
| 4. Train + predict | `training/train.py` (`TARGET=text` or `map`; `MAX_TRAIN`/`TEST_N` for probes) | Modal `ai-training` |
| 4b. Map-mode inference with copy-back repair | `training/infer.py` | Modal `ai-training` |
| 4c. Public demo | `demo/` (Gradio 5.49.1, CPU basic, uploaded with `huggingface_hub`) | Space [`Proximate-Pseudonymizer-Demo`](https://huggingface.co/spaces/danielrosehill/Proximate-Pseudonymizer-Demo) |
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

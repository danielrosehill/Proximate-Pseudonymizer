# Exists check, 2026-10-21

Verdict: **partial**. Models that do proximate replacement exist, but none is a small,
permissively licensed, text-in/text-out rewriter.

Searched via the Hub API (`/api/models?search=<term>&sort=downloads`) for `pseudonymiz`,
`pseudonymis`, `anonymiz`, `anonymis`, `surrogate`, `deidentif`, `pii-replace`, `text-sanitiz`
and `privacy-rewrite`. The last three returned nothing.

| Model | What it does | Why it is not this |
| --- | --- | --- |
| `eternisai/Anonymizer-0.6B` / `-1.7B` / `-4B` (Qwen3, SFT + GRPO with a GPT-4.1 judge) | Semantically similar PII replacement | Emits a `replace_entities` **tool call** (`{original, replacement}` list), not text. Keeps cities ≥1M, so Jerusalem would stay. **CC-BY-NC-4.0.** |
| `grammarly/pseudonymization-seq2seq` (BART, TrustNLP 2023 paper) | Text → pseudonymized text | English only, 2023, ~0 downloads; targets made by an NER pipeline, so substitutes are not proximate. Apache-2.0. |
| `ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii`, `StanfordAIMI/stanford-deidentifier-base` | Token classifiers (detect PII spans) | Detection only; replacement would be random Faker values, not proximate. |

The Eternis card's replacement rules informed the generator prompt (same gender and culture,
same-tier organisations, small date shifts keeping the year), with two deliberate
differences: big cities are replaced too (Daniel's own example is Jerusalem → Tel Aviv), and
the target is the rewritten text, not a mapping.

# Spec

Decided 2026-10-21. Daniel left the base model to me.

## Base model: `Qwen/Qwen3-0.6B`, full fine-tune

- Task is copy-with-edits over texts of up to ~500 words, so it needs a decoder with a long
  context. flan-t5's 512-token input limit rules it out.
- The same base, image and TRL recipe already worked for a text-in/text-out task in
  `ai-models/Prompt-User-Context-Extractor` (Modal L4, ~$0.05 a run). That's a known-good path.
- Apache-2.0, multilingual, small enough for CPU inference and GGUF export.
- If 0.6B leaks or breaks consistency on long texts, `Qwen/Qwen3-1.7B` on the same data is
  the next step. Don't widen the search before r1 numbers exist.

## Data: synthetic pairs, target computed rather than generated

`scripts/generate.py` asks `claude-haiku-5-5` (Message Batches API, thinking disabled) for a
text **plus a replacement map**. The pseudonymized target is produced by
`scripts/pairs.py:apply_map`: a single regex pass, longest original first, word-bounded. So:

- every target is consistent within its text **by construction**;
- nothing outside the swapped spans can change;
- quality depends only on whether the map is complete and proximate, which is what
  `validate()` checks and what the human review looks at.

Briefs are sampled from 30 genres, 25 settings (Israel, Ireland and the US weighted up),
4 length buckets, 2-6 PII types per text, 80% English, and ~8% texts with no identifying
details at all (target = input). Haiku 5.5 rejects non-default `temperature`, so variety is
forced through the brief: a random initial for the main name and a random month for dates.

Replacement policy (in the generator prompt): names keep gender, culture and era; cities go
to a similar city **in the same country**, including big cities; organisations go to the same
type and tier; dates shift days or weeks and keep the year; numbers keep their format;
countries, nationalities, religions, public figures and generic big brands stay.

### Validation flags (`pairs.validate`)

| Flag | Meaning | Effect |
| --- | --- | --- |
| `conflict` | same original, two replacements | hard: dropped |
| `chain` | a replacement is also an original | hard |
| `contains` | a replacement contains its original | hard |
| `leak` | an original, or a capitalised word of a multi-word original, survives in the output | hard |
| `empty_map` | brief asked for PII, map empty | hard |
| `missing` | map entry not found in the text (usually a name inside a mapped email) | soft: entry removed |
| `partial_mismatch` | full-name and part-name entries disagree | soft (fires on correct Polish gendered surnames) |
| `unexpected_map` | no-PII brief, non-empty map | soft |

## Review

`scripts/review.py` (port 8767) shows original and output side by side, with the swaps
highlighted. A fixed 300-row test pool (seed 0, clean rows only) is reviewed first. Only
accepted or fixed rows enter `data/test.jsonl`, the gold set. Train rows with soft flags come
next. Rejections remove a row from training.

## Evaluation (`scripts/evaluate.py`)

Against the gold test set: `leak` (originals still present, lower is better), `swapped`,
`consistency` (word alignment of input vs prediction: each changed span maps to one target),
`preserved` (non-PII words kept), `untouched_ok` (no-PII rows returned unchanged), and a
strict `exact` floor. Proximity isn't scored automatically.

**No-model baseline:** the untrained base with a one-shot prompt (`pred_base.jsonl`, written by
the same training run). **Pass bar for r1:** leak ≤ 0.05, consistency ≥ 0.95,
preserved ≥ 0.98, untouched_ok ≥ 0.9.

## batch-r1 results (2026-10-21)

5,000 briefs → 4,930 parsed (70 were malformed JSON) → 271 hard-dropped → 4,359 train + 300
test pool. Cost: 5.25M input + 3.20M output tokens, about **$1.06** at Haiku 5.5 batch rates
(estimated from usage, not read off the bill).

Spot check of the hard flags, all correctly dropped:
- `leak`: a surname or region left unmapped (`Halvorsen`, `Negev`).
- `contains`: swaps too close to obfuscate (`Carlo`→`Carlos`, `Tomas`→`Tomasz`).
- `chain`: swap cycles (`Savannah`→`Charleston`, `Charleston`→`Asheville`). Single-pass
  application handles them correctly, but they make review confusing.

## Review Space (2026-10-21)

The private Gradio Space `danielrosehill/Proximate-Pseudonymizer-Review` (source `space/`)
reads `pairs.jsonl` from the private dataset `danielrosehill/Proximate-Pseudonymizer-Pairs`
and writes each decision to `reviews/<id>.json` there. It uses the signed-in user's OAuth token
(`hf_oauth_scopes: read-repos, write-repos`), so no secret is stored in the Space. Because the
dataset is private, the pairs can only be loaded after sign-in. `scripts/pull_reviews.py` merges
the decisions into `data/reviews.jsonl`, and `build.py` applies them.

Verified 2026-10-21: page renders behind the private gate. The load / browse / stats / save /
pull path was exercised locally with the same code and a write token; the test review was
deleted afterwards. In-browser OAuth confirmed when Daniel's first two Space decisions arrived
through `pull_reviews.py`.

**Review design (Daniel, 2026-10-21):** the pseudonymized text is edited in place, with no
separate swap table. Each edit or rejection takes preset reasons (`REASONS` in `space/app.py`:
missed detail, not proximate by gender / culture / place / other, inconsistent, should not be
replaced, format broken, grammar after swap, other) plus free-text details. On save, the
Space stores `edits`: word-level `{from, to, original}` spans diffed from the generated output,
where `original` is the source span whenever the edited text was a generated substitute.
Accept with an edited box saves it as `fix`. `scripts/learnings.py` groups decisions by reason,
with example edits and settings, so systematic generator errors become prompt or validator
changes ("targeted learnings").

Traps:
- Gradio injects `gr.OAuthToken` only into functions whose signature type-hints it. A lambda
  wrapper silently receives nothing.
- Running the app locally needs `gradio[oauth]`, not just `gradio`, when `hf_oauth` is set
  (ImportError on `itsdangerous`).
- `gr.LoginButton()` showed a broken logo image on the private Space; `icon=None` removes it.

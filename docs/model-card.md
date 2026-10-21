---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
pipeline_tag: text-generation
library_name: transformers
language:
- en
- multilingual
tags:
- pseudonymization
- anonymization
- privacy
- pii
- text-rewriting
- proof-of-concept
---

# Proximate-Pseudonymizer 0.6B (proof of concept)

> **Status: proof of concept, first checkpoint (2026-10-21). Training is paused after this checkpoint; next steps are under "Getting there".**
> It still leaves about **9% of identifying details unchanged**, so **do not run it
> unsupervised**. Check every output before sharing it.

**Try it:** [public demo Space](https://huggingface.co/spaces/danielrosehill/Proximate-Pseudonymizer-Demo)
(runs on CPU and shows a word-level diff of every swap, so you can spot what it missed).

A small **text in → text out** model. It rewrites a text with its identifying details
swapped for **close, plausible substitutes**, and uses the same substitute every time a
detail recurs.

```
IN : I'm Daniel and I live in Jerusalem. I work at Hadassah hospital and on weekends I visit
     my sister Yael in Haifa. Daniel
OUT: I'm David and I live in Tel Aviv. I work at Sheba hospital and on weekends I visit my
     sister Noa in Be'er Sheva. David
```

(Actual output of this checkpoint, greedy decoding, CPU.)

## The use case

You want to share a real piece of writing (a story, an email thread, a support ticket,
notes) with an AI tool, a forum, a colleague or a dataset, but not the real people and places
in it. Redaction (`[NAME] lives in [CITY]`) destroys the text. Random fake values (Faker-style
"Bartholomew lives in Ulaanbaatar") keep it readable but change what it means: the culture,
the gender, the scale of the place, the kind of organisation. This model aims at the middle
ground. Every detail changes, but to a **proximate** substitute, so the text still reads and
means the same, and a reader can't point at the real person.

## How this differs from what exists

| Existing approach | Example | Difference |
| --- | --- | --- |
| Redaction / masking | token classifiers such as `ai4privacy/...openpii` and `StanfordAIMI/stanford-deidentifier-base`, or Gemma "PII masking" fine-tunes | They detect spans and output tags. This model outputs natural text. |
| Detect + random replace | Presidio + Faker | Substitutes are random, not proximate, and keeping them consistent is up to you. |
| Proximate replacement as a tool call | `eternisai/Anonymizer-0.6B/1.7B/4B` (Qwen3, SFT + GRPO) | The closest prior art. It emits a `replace_entities` list for your code to apply, keeps cities over 1M people (Jerusalem stays Jerusalem), and is **CC-BY-NC-4.0**. This model is plain text out, replaces big cities too, and is Apache-2.0. |
| Seq2seq pseudonymization | `grammarly/pseudonymization-seq2seq` (BART, 2023) | English only, with NER-generated targets, so its substitutes are not proximate. |

Rules the training data follows:
- One substitute per detail across the whole text.
- Related forms change together ("Ciarán", "ciaran.oneill@…").
- Same gender, culture, country and kind of place or organisation.
- Dates shift a little and keep their year.
- Countries and public figures are kept.
- Substitutes must not be *too* close: no kept first names, no spelling variants, no copied
  handle templates with a nudged year.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "danielrosehill/Proximate-Pseudonymizer-0.6B"
SYSTEM = ("Rewrite the text with every identifying detail (names, places, organisations, dates, numbers, "
          "contacts) replaced by a close, plausible substitute of the same kind. Use the same substitute "
          "every time a detail recurs. Change nothing else. Reply with the rewritten text only.")
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M)
text = "I'm Daniel and I live in Jerusalem."
prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(prompt, return_tensors="pt")
out = model.generate(**enc, max_new_tokens=2 * enc["input_ids"].shape[1], do_sample=False)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

Use this exact system prompt; it's the one the model was trained with. 0.6B parameters, so it
runs on a laptop CPU.

## Results on 295 held-out texts

| | Leak ↓ | Consistency | Rest of text preserved | Clean texts left untouched |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B, untrained, one example | 0.856 | 0.997 | 0.791 | 0.308 |
| **This checkpoint** | **0.093** | 0.990 | 0.998 | 0.962 |

- **Leak:** share of identifying details still present verbatim in the output.
- **Consistency:** share of texts where each changed detail got exactly one substitute.
- **Rest of text preserved:** share of non-identifying words left unchanged.
- **Clean texts left untouched:** texts with no details, returned as they were.

Whether a substitute is *proximate* is not scored automatically.

## Limitations: read before using

- **About 1 in 11 identifying details survives.** Names, dates and ages leak most. That's
  fine for a demonstration and not fine for privacy. **Always review the output.**
- Substitutes can collide or link people. In one test, two different surnames were both
  replaced with the same "Moran".
- Some substitutes come out too close to the original (a kept first name, a nudged year in a
  handle).
- 80% of the training texts are English, weighted toward Israel, Ireland and the US. Other
  languages work, but less well.
- Trained on synthetic texts only (generated with Claude Haiku), not real documents.

## Suggested risk targets

Leak is the number that matters. A reasonable target with **a human reviewing every output**
is **≤ 1%** of details surviving. **Without a human in the loop** it should be well below
that, around 0.1% or less, and even then only for low-stakes text. This checkpoint is at
about 9%.

## Getting there

This checkpoint comes from a deliberately cheap run, under $10 all-in including data
generation. The gap is a matter of resources, not a dead end:

- **Training length was the lever.** The same 0.6B model and data went from 26.5% leak to
  9.3% just by training longer at a higher learning rate.
- **A larger base model,** such as Qwen3-4B-Instruct (Apache-2.0), detects details better from
  world knowledge.
- **A "swap list" output mode is implemented in the repo.** The model lists `original →
  substitute` pairs and code applies them, which makes consistency and preservation exact.
  It also has a decoding-time fix for its main failure: writing the original back as its own
  substitute.
- **More data, plus a scored closeness check,** would make "proximate but not too close"
  something the training actually optimises.

## Training

- Full fine-tune of `Qwen/Qwen3-0.6B`.
- 3 epochs, LR 5e-5, cosine schedule, effective batch 16, bf16, 1× L4, ~38 minutes.
- 4,280 synthetic texts. Their swap lists were checked by code for consistency and for
  substitutes that are too close; flagged swaps were re-picked or dropped.
- Training targets are computed by applying the swap list, so every target is consistent by
  construction.

Code, data pipeline, run log and evaluation:
<https://github.com/danielrosehill/Proximate-Pseudonymizer>

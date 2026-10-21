---
title: Proximate Pseudonymizer
emoji: 🎭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
models:
  - danielrosehill/Proximate-Pseudonymizer-0.6B
short_description: Swap identifying details for close substitutes (PoC)
---

# Proximate-Pseudonymizer demo

Runs [`danielrosehill/Proximate-Pseudonymizer-0.6B`](https://huggingface.co/danielrosehill/Proximate-Pseudonymizer-0.6B)
(Qwen3-0.6B full fine-tune) on CPU. Paste a text and get it back with its names, places,
organisations, dates, numbers and contacts swapped for close, plausible substitutes, with a
word-level diff showing each swap.

**Proof of concept.** On 295 held-out texts it left 9.3% of identifying details unchanged.
Review every output; do not run it unsupervised.

Source: [`demo/`](https://github.com/danielrosehill/Proximate-Pseudonymizer/tree/main/demo).

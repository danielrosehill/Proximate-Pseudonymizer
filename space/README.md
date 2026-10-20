---
title: Proximate Pseudonymizer Review
emoji: 🕵️
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
short_description: Review synthetic pseudonymization pairs
---

# Proximate-Pseudonymizer · pair review (private)

Reviews pairs from the private dataset
[`danielrosehill/Proximate-Pseudonymizer-Pairs`](https://huggingface.co/datasets/danielrosehill/Proximate-Pseudonymizer-Pairs)
(`pairs.jsonl`). Sign in with Hugging Face; the pairs are read and each decision is written as
`reviews/<id>.json` with your own OAuth token, so no token is stored in the Space.

Source: [github.com/danielrosehill/Proximate-Pseudonymizer](https://github.com/danielrosehill/Proximate-Pseudonymizer) (`space/`).

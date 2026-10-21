"""Inference-only pass over a trained map-mode model, with two fixes that need no training.

    MODEL=/data/runs/pp-p1-4b-map/model TEST_N=100 python training/infer.py      (on Modal)

1. Copy-back repair. Probe pp-p1-4b-map listed most details correctly but often wrote the
   original back as its own replacement ("Brandon Hargrove" -> "Brandon Hargrove"; 99 times in
   100 texts, never in the training data). For each such entry, the model is asked again to
   continue the JSON from `"replacement": "` with the original's first token banned
   (`bad_words_ids`), and only that replacement is spliced in. A replacement that keeps the
   first word of a multi-word name (similarity.PREFIXES excepted) counts as a copy too.
2. Partial forms. pairs.expand_partials maps first and last names on their own when the model
   only listed the full name.

Writes $OUT/pred_plain.jsonl (as generated) and $OUT/pred_fixed.jsonl (both fixes), scored
with scripts/evaluate.py. This is the logic the published wrapper will ship.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TARGET", "map")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import train as T  # noqa: E402
from pairs import apply_map, expand_partials  # noqa: E402
from similarity import PREFIXES  # noqa: E402

MODEL = os.environ["MODEL"]
OUT = Path(os.environ.get("OUT", "out"))
TEST_N = int(os.environ.get("TEST_N", "0"))


def copied(o: str, n: str) -> bool:
    if n == o:
        return True
    ow, nw = o.split(), n.split()
    return len(ow) > 1 and nw[:1] == ow[:1] and ow[0].lower() not in PREFIXES


def re_ask(model, tok, text, before, o):
    """Continue the JSON at this entry's replacement with o's first token banned."""
    head = '{"replacements": [' + "".join(json.dumps(e, ensure_ascii=False) + ", " for e in before)
    head += '{"original": ' + json.dumps(o, ensure_ascii=False) + ', "replacement": "'
    prompt = tok.apply_chat_template(T.messages(text), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) + head
    enc = tok(prompt, return_tensors="pt").to(model.device)
    ban = {tuple(tok.encode(o, add_special_tokens=False)[:1]), tuple(tok.encode(o.split()[0], add_special_tokens=False)[:1])}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, bad_words_ids=[list(b) for b in ban if b])
    s = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads('"' + s.split('"')[0] + '"') or None
    except json.JSONDecodeError:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = T.load("test.jsonl")
    test = test[:TEST_N] if TEST_N else test
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
    t0 = time.time()
    plain = T.generate(model, tok, test, shots=False)
    tok.padding_side = "right"
    fixed, n_reask, n_fixed = [], 0, 0
    for r, p in zip(test, plain):
        try:
            entries = [e for e in json.loads(p["raw"])["replacements"] if e.get("original") and e.get("replacement")]
        except (json.JSONDecodeError, KeyError, TypeError):
            fixed.append(p)
            continue
        for i, e in enumerate(entries):
            if copied(e["original"], e["replacement"]):
                n_reask += 1
                new = re_ask(model, tok, r["input"], entries[:i], e["original"])
                if new and not copied(e["original"], new):
                    e["replacement"] = new
                    n_fixed += 1
        m = {e["original"]: e["replacement"] for e in entries if e["replacement"] != e["original"]}
        fixed.append({"id": r["id"], "pred": apply_map(r["input"], expand_partials(r["input"], m)),
                      "raw": json.dumps({"replacements": entries}, ensure_ascii=False)})
    T.dump(OUT / "pred_plain.jsonl", plain)
    T.dump(OUT / "pred_fixed.jsonl", fixed)
    stats = {"model": MODEL, "n": len(test), "reasked": n_reask, "fixed": n_fixed, "seconds": round(time.time() - t0)}
    (OUT / "infer.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.70"]
# ///
"""Re-pick ONLY the swaps that scripts/similarity.py flagged as too close, with Haiku 5.5.

    uv run scripts/build.py                           # writes data/pairs.jsonl with close:* flags
    uv run scripts/repair.py submit                   # one request per flagged pair (Batches API)
    uv run scripts/repair.py collect --batch <id>     # -> data/repairs/<name>.jsonl
    uv run scripts/build.py                           # applies repairs, re-checks, drops what still fails

The model sees the text, the full swap list and the flagged swaps with why they were flagged,
and returns new replacements for those originals (plus any related partial forms it must
change to stay consistent). The target text is recomputed from the map as always, so the
repair can't break consistency, and build.py re-runs every check on the repaired map.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "repairs"

WHY = {
    "kept": "keeps part of the original name unchanged",
    "spelling": "is only a spelling variant of the original",
    "template+year": "copies the original's handle template and shifts the year by only a little",
    "digits": "keeps most of the original's digits",
}

SYSTEM = """You fix a pseudonymizer's training data. Some swaps were flagged because the replacement is TOO CLOSE to the original: someone who knew the original could recognise it. Pick new replacements for the flagged originals that are still PROXIMATE (same kind, gender, culture, country, scale and format, so the text reads naturally) but share nothing linkable with the original: no shared name parts, no spelling variants, no copied handle template with a nudged year, no copied digit runs.

Keep the text consistent: if you change "Tobias Vogt", also return the matching change for any related form in the swap list ("Tobias", "Vogt", an email or handle built from the name). Don't touch unrelated swaps. A new replacement must not equal any original in the text.

Reply with JSON only, no prose and no code fence:
{"replacements": [{"original": "...", "replacement": "..."}]}"""


def user_msg(r: dict) -> str:
    swaps = "\n".join(f"- {x['original']} -> {x['replacement']}" for x in r["replacements"])
    flagged = "\n".join(f"- {f['original']} -> {f['replacement']}: {WHY.get(f['why'], f['why'])}" for f in r["close"])
    return f"TEXT:\n{r['input']}\n\nALL SWAPS:\n{swaps}\n\nFLAGGED (fix these):\n{flagged}"


def request(r: dict) -> dict:
    return {"model": MODEL, "max_tokens": 2048, "system": SYSTEM, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": user_msg(r)}]}


def text_of(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


def parse(rid: str, reply: str, usage) -> dict:
    s = reply.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    rec = {"id": rid, "model": MODEL, "usage": {"in": usage.input_tokens, "out": usage.output_tokens} if usage else None}
    try:
        rec["replacements"] = [x for x in json.loads(s)["replacements"]
                               if isinstance(x, dict) and x.get("original") and x.get("replacement")]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        rec["error"], rec["reply_raw"] = f"parse:{e}", reply
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["submit", "collect"])
    ap.add_argument("--batch")
    ap.add_argument("--name")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    rows = [json.loads(l) for l in open(ROOT / "data" / "pairs.jsonl")]
    flagged = [r for r in rows if r.get("close") and not r.get("base_hard")]  # base-hard rows stay dropped anyway
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "submit":
        name = a.name or f"repair-{time.strftime('%Y%m%d-%H%M')}"
        b = client.messages.batches.create(requests=[{"custom_id": r["id"], "params": request(r)} for r in flagged])
        (OUT / f"{name}.batch.json").write_text(json.dumps({"batch": b.id, "n": len(flagged)}))
        print(b.id, name, len(flagged))
    else:
        name = next(p.name.removesuffix(".batch.json") for p in OUT.glob("*.batch.json")
                    if json.loads(p.read_text())["batch"] == a.batch)
        st = client.messages.batches.retrieve(a.batch)
        if st.processing_status != "ended":
            sys.exit(f"{a.batch}: {st.processing_status} {st.request_counts}")
        out = []
        for res in client.messages.batches.results(a.batch):
            if res.result.type == "succeeded":
                out.append(parse(res.custom_id, text_of(res.result.message), res.result.message.usage))
            else:
                out.append({"id": res.custom_id, "error": f"batch:{res.result.type}"})
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(OUT / f"{name}.jsonl", len(out))


if __name__ == "__main__":
    main()

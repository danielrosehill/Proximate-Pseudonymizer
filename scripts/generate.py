# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic>=0.70"]
# ///
"""Synthetic (original, pseudonymized) pairs for the Proximate-Pseudonymizer.

The LLM writes a text plus a replacement map. It does NOT write the pseudonymized text:
that is computed here by applying the map (pairs.apply_map), so every training target is
consistent within its text by construction and changes nothing outside the swapped spans.

    export ANTHROPIC_API_KEY=$(op item get "Anthropic / Claude Experiments API Key" \
        --vault "API Keys & Programmatic" --fields password --reveal)
    uv run scripts/generate.py pilot --n 20             # synchronous, to eyeball quality
    uv run scripts/generate.py submit --n 4000          # Message Batches API (half price)
    uv run scripts/generate.py collect --batch <id>     # when the batch has ended

Raw rows land in data/raw/<run>.jsonl; scripts/build.py validates and splits them.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from pairs import parse_reply  # noqa: E402

MODEL = "claude-haiku-5-5"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

GENRES = [
    "personal story told to a friend", "diary entry", "email to a colleague", "email to a landlord",
    "complaint letter to a company", "forum post asking for advice", "text messages in a group chat (one block)",
    "dictated voice note, transcribed, a bit rambling", "prompt to an AI assistant describing a personal problem",
    "cover letter", "short biography for a conference", "doctor's appointment summary written by the patient",
    "online review of a local business", "wedding speech excerpt", "obituary", "school newsletter item",
    "neighbourhood WhatsApp announcement", "insurance claim description", "job reference letter",
    "travel blog paragraph", "therapy journal entry", "message to a support line", "meeting notes",
    "letter to a grandparent", "dating profile", "LinkedIn post about a career move", "police statement draft",
    "family holiday planning message", "university application essay excerpt", "eulogy excerpt",
]
LOCALES = [
    ("Israel", 6), ("Ireland", 4), ("United States", 6), ("United Kingdom", 4), ("Canada", 2), ("Australia", 2),
    ("India", 2), ("Germany", 2), ("France", 2), ("Spain", 1), ("Italy", 1), ("Netherlands", 1), ("Nigeria", 1),
    ("South Africa", 1), ("Brazil", 1), ("Mexico", 1), ("Philippines", 1), ("Japan", 1), ("Poland", 1),
    ("New Zealand", 1), ("Kenya", 1), ("Argentina", 1), ("Sweden", 1), ("Greece", 1),
    ("two countries (the writer moved from one to the other)", 3),
]
PII = [
    "the writer's own first name", "the writer's full name", "a family member's name", "a friend's or colleague's name",
    "the city they live in", "the town or city they were born in", "a neighbourhood or street",
    "a full street address", "an employer or company name", "a school, university or hospital",
    "a specific date (birthday, appointment, event)", "a phone number", "an email address", "an age",
    "a job title at a named organisation", "a local business (cafe, gym, clinic)", "a car registration or account number",
    "a pet's name", "a nickname", "a national ID or passport number", "a social media handle",
]
LENGTHS = [("1-2 sentences", 2), ("one paragraph of 60-120 words", 4), ("2-3 paragraphs, 150-300 words", 3),
           ("4-6 paragraphs, 300-500 words", 1)]

SYSTEM = """You write training data for a text pseudonymizer. Given a brief, write ONE realistic text and a replacement map.

THE TEXT
- Natural, specific, written like a real person in that genre and place. Vary tone and register.
- Include the identifying details requested by the brief, and mention some of them more than once, in different forms (full name then first name only; a city then its demonym-free short form; an email derived from the person's name).

THE REPLACEMENT MAP
Every span in the text that identifies a private person, place, organisation, date or number gets an entry {"original", "replacement", "type"}. "original" must be copied EXACTLY as it appears in the text (case, spelling, punctuation).
Replacements are PROXIMATE: the closest plausible substitute of the same kind, so the text reads the same but no longer points at the real thing.
- Names: same gender, same culture and language, same era, similar length and formality. Daniel -> David, Siobhan -> Aoife, Priya Sharma -> Kavya Mehta. Family members keep a shared surname. A nickname follows its name (Danny -> Davey).
- Cities and towns: a different one in the same country, of similar size and character. Jerusalem -> Tel Aviv, Cork -> Galway, Austin -> Denver. Neighbourhoods and streets: plausible ones in the REPLACEMENT city.
- Organisations: a different real or plausible one of the same type, size and sector. Keep legal suffixes.
- Dates: shift by a few days or weeks; keep the format and the year. Ages: change by 1-3 years. Keep holidays and famous historical dates.
- Phone, email, IDs, account and plate numbers, handles: same format and length, different digits/letters; emails and handles follow the replaced name.
- Do NOT replace: countries, languages, nationalities, religions, public figures, globally famous companies used generically (Google, IKEA), or ordinary words.

CONSISTENCY (the most important rule)
- One original -> one replacement, everywhere in the text.
- If "Daniel Rosen" -> "David Cohen" is in the map, then the separate mentions "Daniel" and "Rosen" need their own entries "David" and "Cohen". Partial forms must agree with the full form.
- List every distinct surface form separately (e.g. "Rosen" and "Rosens" if both occur). Possessive 's needs no separate entry.
- A replacement must never itself appear as an original elsewhere, and must not contain the original.
- If the brief says there are no identifying details, the map is empty.

Reply with JSON only, no prose and no code fence:
{"text": "...", "replacements": [{"original": "...", "replacement": "...", "type": "name|place|org|date|number|contact|other"}]}"""


def brief(rng: random.Random, i: int) -> dict:
    no_pii = rng.random() < 0.08
    b = {
        "id": f"p{i:05d}",
        "genre": rng.choice(GENRES),
        "locale": rng.choices([l for l, _ in LOCALES], [w for _, w in LOCALES])[0],
        "length": rng.choices([l for l, _ in LENGTHS], [w for _, w in LENGTHS])[0],
        "pii": [] if no_pii else rng.sample(PII, rng.randint(2, 6)),
        "first_person": rng.random() < 0.75,
        # Haiku 5.5 rejects non-default temperature, so variety is forced through the brief:
        # without these the pilot reused "Biscuit" and "3 March" across rows.
        "language": "English" if rng.random() < 0.8 else "the main local language of the setting",
        "initial": rng.choice("ABCDEFGHIJKLMNOPRSTVWYZ"),
        "month": rng.choice(["January", "February", "March", "April", "May", "June", "July", "August",
                             "September", "October", "November", "December"]),
    }
    return b


def user_msg(b: dict) -> str:
    who = "first person" if b["first_person"] else "third person or addressed to someone"
    if b["pii"]:
        details = "Include these identifying details: " + "; ".join(b["pii"]) + "."
    else:
        details = ("This text must contain NO identifying details at all (generic, no names, places, dates or "
                   "numbers that point to anyone). The map must be empty.")
    return (f"Genre: {b['genre']}\nSetting: {b['locale']}\nLanguage: {b['language']}\nLength: {b['length']}\n"
            f"Voice: {who}\n{details}\nFor variety: the main person's first name starts with {b['initial']}; "
            f"any dates fall in {b['month']}. Avoid stock pet names.")


def request(b: dict) -> dict:
    # Haiku 5.5 thinks by default; "disabled" is accepted at effort high or below (default medium).
    return {"model": MODEL, "max_tokens": 4096, "system": SYSTEM, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": user_msg(b)}]}


def briefs(n: int, seed: int, start: int) -> list[dict]:
    rng = random.Random(seed)
    return [brief(rng, start + i) for i in range(n)]


def write(rows: list[dict], name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"{name}.jsonl"
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def text_of(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


def row(b: dict, reply: str, usage) -> dict:
    parsed, err = parse_reply(reply)
    return {**b, "model": MODEL, "reply_raw": None if parsed else reply, "error": err,
            "text": parsed["text"] if parsed else None,
            "replacements": parsed["replacements"] if parsed else None,
            "usage": {"in": usage.input_tokens, "out": usage.output_tokens} if usage else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "submit", "collect"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="first brief index, so runs never reuse ids")
    ap.add_argument("--batch")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    client = anthropic.Anthropic()

    if a.mode == "pilot":
        out = []
        for b in briefs(a.n, a.seed, a.start):
            m = client.messages.create(**request(b))
            out.append(row(b, text_of(m), m.usage))
            print(b["id"], out[-1]["error"] or "ok", file=sys.stderr)
        print(write(out, a.name or "pilot"))
    elif a.mode == "submit":
        bs = briefs(a.n, a.seed, a.start)
        batch = client.messages.batches.create(requests=[{"custom_id": b["id"], "params": request(b)} for b in bs])
        name = a.name or f"batch-{time.strftime('%Y%m%d-%H%M')}"
        (RAW / f"{name}.briefs.json").parent.mkdir(parents=True, exist_ok=True)
        (RAW / f"{name}.briefs.json").write_text(json.dumps({"batch": batch.id, "briefs": bs}, ensure_ascii=False))
        print(batch.id, name)
    else:
        name = next(p.name.removesuffix(".briefs.json") for p in RAW.glob("*.briefs.json")
                    if json.loads(p.read_text())["batch"] == a.batch)
        bs = {b["id"]: b for b in json.loads((RAW / f"{name}.briefs.json").read_text())["briefs"]}
        status = client.messages.batches.retrieve(a.batch)
        if status.processing_status != "ended":
            sys.exit(f"{a.batch}: {status.processing_status} {status.request_counts}")
        out = []
        for r in client.messages.batches.results(a.batch):
            b = bs[r.custom_id]
            if r.result.type == "succeeded":
                out.append(row(b, text_of(r.result.message), r.result.message.usage))
            else:
                out.append({**b, "error": f"batch:{r.result.type}"})
        print(write(out, name), len(out))


if __name__ == "__main__":
    main()

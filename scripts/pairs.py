"""Shared logic: parse the generator's reply, apply a replacement map, validate a pair.

The pseudonymized target is always computed by apply_map, never written by an LLM, so it
is consistent by construction. validate() flags the ways a map can still be wrong.
"""
import json
import re


def parse_reply(reply: str):
    s = reply.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        return None, f"json:{e.msg}"
    if not isinstance(d.get("text"), str) or not isinstance(d.get("replacements"), list):
        return None, "shape"
    reps = [r for r in d["replacements"] if isinstance(r, dict) and r.get("original") and r.get("replacement")]
    return {"text": d["text"], "replacements": reps}, None


def _bounded(s: str) -> str:
    p = re.escape(s)
    if s[0].isalnum() or s[0] == "_":
        p = r"(?<!\w)" + p
    if s[-1].isalnum() or s[-1] == "_":
        p = p + r"(?!\w)"
    return p


def mapping(reps: list[dict]) -> dict[str, str]:
    return {r["original"]: r["replacement"] for r in reps if r["original"] != r["replacement"]}


def apply_map(text: str, m: dict[str, str]) -> str:
    """Single pass, longest original first, word-bounded: no chained or partial-word swaps."""
    if not m:
        return text
    pat = re.compile("|".join(_bounded(k) for k in sorted(m, key=len, reverse=True)))
    return pat.sub(lambda x: m[x.group(0)], text)


def validate(text: str, reps: list[dict], expect_pii: bool) -> list[str]:
    flags = []
    seen = {}
    for r in reps:
        o, n = r["original"], r["replacement"]
        if o in seen and seen[o] != n:
            flags.append(f"conflict:{o}")
        seen[o] = n
    m = mapping(reps)
    for o in m:
        if not re.search(_bounded(o), text):
            flags.append(f"missing:{o}")
    for o, n in m.items():
        if n in m and n != o:
            flags.append(f"chain:{n}")
        if o.lower() in n.lower():
            flags.append(f"contains:{o}")
    # Partial forms must agree with the full form: "Daniel Rosen"->"David Cohen" needs "Daniel"->"David".
    for o, n in m.items():
        ow, nw = o.split(), n.split()
        if len(ow) > 1 and len(ow) == len(nw):
            for a, b in zip(ow, nw):
                if a in m and m[a] != b:
                    flags.append(f"partial_mismatch:{a}")
    out = apply_map(text, m)
    # Leaks: an original, or a capitalised word of a multi-word original, still standing in the output.
    words = {w.strip(".,'’") for o in m for w in o.split() if len(o.split()) > 1}
    for w in sorted(set(m) | words):
        if len(w) >= 3 and w[:1].isupper() and re.search(_bounded(w), out) and not any(w in n for n in m.values()):
            flags.append(f"leak:{w}")
    if expect_pii and not m:
        flags.append("empty_map")
    if not expect_pii and m:
        flags.append("unexpected_map")
    return flags

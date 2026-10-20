"""Summarise review corrections by reason, with the actual edited spans, to target fixes.

    python3 scripts/pull_reviews.py && python3 scripts/learnings.py [--json]

Reads data/reviews.jsonl (last decision per id) and data/pairs.jsonl. Each Space decision
carries `reasons` (preset categories), an optional `note`, and for fixes `edits`: the
{from, to, original} spans the reviewer changed. The report shows, per reason, how often it
was given and example edits. Those are the inputs for tightening the generator prompt in
scripts/generate.py, or for adding a validator rule in scripts/pairs.py.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = ROOT / "data" / "reviews.jsonl"
    latest = {}
    for line in open(p) if p.exists() else []:
        r = json.loads(line)
        latest[r["id"]] = r
    pairs = {r["id"]: r for r in map(json.loads, open(ROOT / "data" / "pairs.jsonl"))}
    verdicts = Counter(r["verdict"] for r in latest.values())
    by_reason = defaultdict(list)
    for r in latest.values():
        for reason in r.get("reasons") or (["(no reason given)"] if r["verdict"] != "accept" else []):
            by_reason[reason].append(r)
    report = {"reviewed": len(latest), "verdicts": dict(verdicts),
              "reasons": {k: {"count": len(v),
                              "by_locale": dict(Counter(pairs[r["id"]]["locale"] for r in v if r["id"] in pairs).most_common(5)),
                              "examples": [{"id": r["id"], "verdict": r["verdict"], "edits": r.get("edits", []),
                                            "note": r.get("note")} for r in v[:8]]}
                          for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))}}
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"{report['reviewed']} reviewed · " + " · ".join(f"{k} {v}" for k, v in verdicts.items()))
    for reason, d in report["reasons"].items():
        print(f"\n## {reason} — {d['count']}  (top settings: {d['by_locale']})")
        for ex in d["examples"]:
            spans = "; ".join(f"{e.get('original', '?')}: {e['from']!r} → {e['to']!r}" for e in ex["edits"])
            print(f"  {ex['id']} {ex['verdict']}: {spans or '-'}" + (f"  — {ex['note']}" if ex["note"] else ""))


if __name__ == "__main__":
    main()

"""Validate raw generations, apply human reviews, and write the train/test splits.

    python3 scripts/build.py            # data/raw/batch-*.jsonl + data/reviews.jsonl -> data/*.jsonl

Outputs:
  data/pairs.jsonl   every parsed row: input, output (computed from the map), map, flags, split
  data/train.jsonl   clean rows not in the test pool, minus rejected ones, with fixes applied
  data/test.jsonl    test-pool rows a human ACCEPTED or FIXED (the held-out gold set)
  data/stats.json    counts by flag, split and review state

Hard flags drop a row from training; soft flags are kept and shown in the review tool.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs import apply_map, mapping, validate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEST_POOL = 300
# "missing" = the model listed a span that is not in the text (usually a name inside an email
# it also mapped whole); the entry is dropped and the row kept. "partial_mismatch" fires on
# correct gendered surnames (Polish Wiśniewski/Wiśniewska), so it is soft too.
SOFT = ("missing", "partial_mismatch", "unexpected_map")


def load_raw():
    for p in sorted((DATA / "raw").glob("batch-*.jsonl")):
        for line in open(p):
            yield json.loads(line)


def load_reviews():
    p = DATA / "reviews.jsonl"
    latest = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            latest[r["id"]] = r  # last decision wins
    return latest


def main():
    rows, stats = [], Counter()
    for r in load_raw():
        if not r.get("text"):
            stats["unparsed"] += 1
            continue
        flags = validate(r["text"], r["replacements"], bool(r["pii"]))
        missing = {f.split(":", 1)[1] for f in flags if f.startswith("missing:")}
        reps = [x for x in r["replacements"] if x["original"] not in missing]
        m = mapping(reps)
        hard = [f for f in flags if not f.startswith(SOFT)]
        for f in flags:
            stats["flag:" + f.split(":")[0]] += 1
        rows.append({"id": r["id"], "genre": r["genre"], "locale": r["locale"], "language": r.get("language"),
                     "input": r["text"], "output": apply_map(r["text"], m),
                     "replacements": [{"original": o, "replacement": n} for o, n in m.items()],
                     "flags": flags, "hard": bool(hard)})

    clean = [r for r in rows if not r["hard"]]
    rng = random.Random(0)
    pool = set(rng.sample([r["id"] for r in clean], min(TEST_POOL, len(clean))))
    reviews = load_reviews()
    train, test = [], []
    for r in rows:
        r["split"] = "test" if r["id"] in pool else ("train" if not r["hard"] else "dropped")
        rv = reviews.get(r["id"])
        r["review"] = rv["verdict"] if rv else None
        if rv and rv["verdict"] == "fix":
            r["output"] = rv["output"]
        if r["split"] == "test" and r["review"] in ("accept", "fix"):
            test.append(r)
        elif r["split"] == "train" and r["review"] != "reject":
            train.append(r)
        # A reviewed-and-rejected hard row stays dropped; a hard row a human fixed is rescued.
        elif r["split"] == "dropped" and r["review"] == "fix":
            train.append(r)

    def dump(name, rs):
        with open(DATA / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("pairs.jsonl", rows)
    dump("train.jsonl", [{k: r[k] for k in ("id", "input", "output", "replacements")} for r in train])
    dump("test.jsonl", [{k: r[k] for k in ("id", "input", "output", "replacements")} for r in test])
    stats = dict(stats)
    stats.update({"rows": len(rows), "hard": sum(r["hard"] for r in rows), "train": len(train),
                  "test_pool": len(pool), "test_reviewed": len(test),
                  "reviews": dict(Counter(v["verdict"] for v in reviews.values()))})
    (DATA / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print(json.dumps(dict(sorted(stats.items())), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

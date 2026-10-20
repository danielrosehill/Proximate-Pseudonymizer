# /// script
# requires-python = ">=3.11"
# dependencies = ["rapidfuzz>=3"]
# ///
"""Validate raw generations, flag too-close swaps, apply repairs and reviews, write the splits.

    uv run scripts/build.py     # data/raw/batch-*.jsonl + data/repairs/*.jsonl + data/reviews.jsonl -> data/*.jsonl

Outputs:
  data/pairs.jsonl   every parsed row: input, output (computed from the map), map, flags,
                     close (too-close swaps before repair), split
  data/train.jsonl   clean rows outside the test pool, with repairs and fixes applied, minus rejects
  data/test.jsonl    clean rows of the fixed 300-row test pool (human fixes applied, rejects removed)
  data/stats.json    counts by flag, split, repair and review state

Hard flags drop a row; soft flags are kept. Too-close swaps (scripts/similarity.py) are hard
unless scripts/repair.py produced a replacement map that passes every check again.

Since 2026-10-21 the test set no longer waits on manual review: Daniel judged the pairs
"already usable", and the too-close checks replace the human pass. Reviews made in the Space
still apply.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pairs import apply_map, mapping, validate  # noqa: E402
from similarity import Checker  # noqa: E402

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


def latest(pattern: str) -> dict:
    out = {}
    for p in sorted(DATA.glob(pattern)):
        for line in open(p):
            r = json.loads(line)
            out[r["id"]] = r  # last one wins
    return out


def checked(text, reps, expect_pii, checker):
    """-> (map, flags, close) for one replacement list."""
    flags = validate(text, reps, expect_pii)
    missing = {f.split(":", 1)[1] for f in flags if f.startswith("missing:")}
    m = mapping([x for x in reps if x["original"] not in missing])
    close = [{"original": o, "replacement": n, "why": w.split(":")[0]}
             for o, n in m.items() for w in checker.too_close(o, n)[:1]]
    flags += [f"close:{c['why']}:{c['original']}" for c in close]
    return m, flags, close


def main():
    raw = [r for r in load_raw()]
    stats = Counter(unparsed=sum(not r.get("text") for r in raw))
    raw = [r for r in raw if r.get("text")]
    checker = Checker([(x["original"], x["replacement"]) for r in raw for x in r["replacements"]])
    repairs = latest("repairs/*.jsonl")

    rows = []
    for r in raw:
        m, flags, close = checked(r["text"], r["replacements"], bool(r["pii"]), checker)
        base_hard = any(not f.startswith(SOFT) and not f.startswith("close:") for f in flags)
        repaired = False
        rp = repairs.get(r["id"])
        if close and rp and not rp.get("error"):
            fix = {x["original"]: x["replacement"] for x in rp["replacements"]}
            reps = [{"original": o, "replacement": fix.get(o, n)} for o, n in m.items()]
            reps += [{"original": o, "replacement": n} for o, n in fix.items() if o not in m]
            m2, flags2, close2 = checked(r["text"], reps, bool(r["pii"]), checker)
            if not close2 and not any(not f.startswith(SOFT) for f in flags2):
                m, flags, repaired = m2, flags2 + ["repaired"], True
            else:
                flags += ["repair_failed"]
        for f in flags:
            stats["flag:" + f.split(":")[0]] += 1
        rows.append({"id": r["id"], "genre": r["genre"], "locale": r["locale"], "language": r.get("language"),
                     "input": r["text"], "output": apply_map(r["text"], m),
                     "replacements": [{"original": o, "replacement": n} for o, n in m.items()],
                     "flags": flags, "close": close, "repaired": repaired, "base_hard": base_hard,
                     "hard": base_hard or (bool(close) and not repaired)})

    # The pool is drawn from rows clean under the ORIGINAL validation, exactly as before the
    # too-close checks existed, so it (and reviews made against it) stays stable.
    rng = random.Random(0)
    pool = set(rng.sample([r["id"] for r in rows if not r["base_hard"]], TEST_POOL))
    reviews = latest("reviews.jsonl")
    train, test = [], []
    for r in rows:
        r["split"] = "test" if r["id"] in pool else ("train" if not r["hard"] else "dropped")
        rv = reviews.get(r["id"])
        r["review"] = rv["verdict"] if rv else None
        if rv and rv["verdict"] == "fix":
            r["output"] = rv["output"]
        if r["review"] == "reject":
            continue
        if r["split"] == "test" and (not r["hard"] or r["review"] == "fix"):
            test.append(r)
        elif r["split"] == "train" or (r["split"] == "dropped" and r["review"] == "fix"):
            train.append(r)

    def dump(name, rs):
        with open(DATA / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("pairs.jsonl", rows)
    dump("train.jsonl", [{k: r[k] for k in ("id", "input", "output", "replacements")} for r in train])
    dump("test.jsonl", [{k: r[k] for k in ("id", "input", "output", "replacements")} for r in test])
    stats = dict(stats)
    stats.update({"rows": len(rows), "hard": sum(r["hard"] for r in rows), "train": len(train), "test": len(test),
                  "close_rows": sum(bool(r["close"]) for r in rows), "repaired": sum(r["repaired"] for r in rows),
                  "reviews": dict(Counter(v["verdict"] for v in reviews.values()))})
    (DATA / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2, ensure_ascii=False))
    print(json.dumps(dict(sorted(stats.items())), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

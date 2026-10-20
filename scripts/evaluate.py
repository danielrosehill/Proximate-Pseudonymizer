"""Score predictions against the human-reviewed test set.

    python3 scripts/evaluate.py runs/pp-r1/pred_base.jsonl runs/pp-r1/pred_ft.jsonl

Per prediction file, over data/test.jsonl:
  exact        prediction == gold output (strict; many valid answers exist, so read it as a floor)
  leak         share of gold originals still present verbatim in the prediction (lower is better)
  swapped      share of rows where every gold original is gone
  consistency  share of rows where each changed source span maps to ONE target string
               (word-level alignment of input vs prediction)
  preserved    share of input words outside gold originals kept unchanged (higher is better)
  untouched_ok on rows with no gold replacements, share returned unchanged
Proximity (is the substitute *close*?) is not measured here; that needs a judge or a human.
"""
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def words(s):
    return re.findall(r"\w+|[^\w\s]", s)


def bounded(s):
    p = re.escape(s)
    return (r"(?<!\w)" if s[:1].isalnum() else "") + p + (r"(?!\w)" if s[-1:].isalnum() else "")


def score(gold, preds):
    n = len(gold)
    exact = leaks = origs = swapped = consistent = kept = keepable = 0
    noop_rows = noop_ok = 0
    for g in gold:
        p = preds.get(g["id"], "")
        exact += p.strip() == g["output"].strip()
        os_ = [r["original"] for r in g["replacements"]]
        found = [o for o in os_ if re.search(bounded(o), p)]
        leaks += len(found)
        origs += len(os_)
        swapped += not found
        if not os_:
            noop_rows += 1
            noop_ok += p.strip() == g["input"].strip()
        a, b = words(g["input"]), words(p)
        pii_words = {w for o in os_ for w in words(o)}
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        seen, ok = {}, True
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                kept += sum(w not in pii_words for w in a[i1:i2])
            elif tag == "replace":
                src, dst = " ".join(a[i1:i2]), " ".join(b[j1:j2])
                if seen.setdefault(src, dst) != dst:
                    ok = False
        keepable += sum(w not in pii_words for w in a)
        consistent += ok
    return {"n": n, "exact": round(exact / n, 3), "leak": round(leaks / max(origs, 1), 3),
            "swapped": round(swapped / n, 3), "consistency": round(consistent / n, 3),
            "preserved": round(kept / max(keepable, 1), 3),
            "untouched_ok": round(noop_ok / noop_rows, 3) if noop_rows else None}


def main():
    gold = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl")]
    for path in sys.argv[1:]:
        preds = {r["id"]: r["pred"] for r in map(json.loads, open(path))}
        sub = [g for g in gold if g["id"] in preds]
        print(path, json.dumps(score(sub, preds)))


if __name__ == "__main__":
    main()

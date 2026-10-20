"""Merge review decisions made in the HF review Space into data/reviews.jsonl.

    python3 scripts/pull_reviews.py && python3 scripts/build.py

The Space (space/app.py, deployed as danielrosehill/Proximate-Pseudonymizer-Review) writes
one reviews/<id>.json per decision to the private dataset
danielrosehill/Proximate-Pseudonymizer-Pairs. This downloads them with the `hf` CLI
(HF_TOKEN or HUGGINGFACE_TOKEN in the environment) and appends any decision not already in
data/reviews.jsonl. build.py keeps the last decision per id, so a Space decision made after a
local one wins.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEWS = ROOT / "data" / "reviews.jsonl"
DATASET = "danielrosehill/Proximate-Pseudonymizer-Pairs"
HF = os.path.expanduser("~/.local/bin/hf")


def main():
    env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")}
    have = set()
    if REVIEWS.exists():
        have = {(r["id"], r.get("ts")) for r in map(json.loads, open(REVIEWS))}
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([HF, "download", DATASET, "--repo-type", "dataset", "--include", "reviews/*",
                        "--local-dir", d, "--quiet"], check=True, env=env)
        recs = [json.loads(p.read_text()) for p in sorted(Path(d, "reviews").glob("*.json"))] \
            if Path(d, "reviews").exists() else []
    new = sorted((r for r in recs if (r["id"], r.get("ts")) not in have), key=lambda r: r.get("ts") or "")
    with open(REVIEWS, "a") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(recs)} reviews on the Hub, {len(new)} new appended to {REVIEWS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

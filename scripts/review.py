"""Manual review GUI for the synthetic pairs. Stdlib only.

    python3 scripts/build.py && python3 scripts/review.py     # then open http://localhost:8767

Queue order: unreviewed test-pool rows first (they become the gold test set), then train rows
with soft flags, then the rest of train. Keys: a = accept, r = reject, e = edit the output then
Save fix, n = skip. Decisions append to data/reviews.jsonl (last decision per id wins); re-run
build.py afterwards to rebuild train/test.
"""
import html
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS, REVIEWS = ROOT / "data" / "pairs.jsonl", ROOT / "data" / "reviews.jsonl"
PORT = 8767


def reviewed():
    if not REVIEWS.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, open(REVIEWS))}


def queue(scope):
    rows = [json.loads(l) for l in open(PAIRS)]
    done = reviewed()
    rows = [r for r in rows if r["id"] not in done]
    test = [r for r in rows if r["split"] == "test"]
    soft = [r for r in rows if r["split"] == "train" and r["flags"]]
    rest = [r for r in rows if r["split"] == "train" and not r["flags"]]
    hard = [r for r in rows if r["split"] == "dropped"]
    return {"test": test, "train": soft + rest, "dropped": hard}[scope], len(done)


def mark(text, spans, cls):
    """Highlight every occurrence of each span (longest first) in escaped text."""
    if not spans:
        return html.escape(text)
    pat = re.compile("|".join(re.escape(s) for s in sorted(spans, key=len, reverse=True)))
    out, i = [], 0
    for m in pat.finditer(text):
        out += [html.escape(text[i:m.start()]), f'<mark class="{cls}">{html.escape(m.group(0))}</mark>']
        i = m.end()
    return "".join(out) + html.escape(text[i:])


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Pair review</title><style>
:root{{--bg:#fafaf7;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#e3e2dc;--o:#ffd9d2;--n:#cdeccf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#181816;--fg:#ecebe6;--mute:#9a998f;--card:#22221f;--line:#3a3a35;--o:#6b2c22;--n:#24502a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:16px}}
.top{{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;color:var(--mute)}} .top a{{color:inherit}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:800px){{.cols{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;white-space:pre-wrap}}
mark.o{{background:var(--o);color:inherit}} mark.n{{background:var(--n);color:inherit}}
table{{border-collapse:collapse;margin-top:12px}} td{{padding:2px 10px;border-bottom:1px solid var(--line)}}
textarea{{width:100%;min-height:180px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;box-sizing:border-box}}
button{{font:inherit;padding:6px 14px;margin-right:8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
.flags{{color:#c0392b}}</style></head><body>
<div class="top"><b>{scope}</b> · {left} left · {done} reviewed · <a href="/?scope=test">test</a> <a href="/?scope=train">train</a> <a href="/?scope=dropped">dropped</a>
 · <span>{id} · {genre} · {locale}</span> <span class="flags">{flags}</span></div>
<div class="cols"><div><h3>Original</h3><div class="card">{inp}</div></div><div><h3>Pseudonymized</h3><div class="card">{out}</div></div></div>
<table>{table}</table>
<form method="post" id="f"><input type="hidden" name="id" value="{id}"><input type="hidden" name="scope" value="{scope}">
<p><button name="verdict" value="accept">Accept (a)</button><button name="verdict" value="reject">Reject (r)</button>
<button name="verdict" value="skip">Skip (n)</button></p>
<details id="d"><summary>Edit output (e)</summary><textarea name="output" id="t">{raw}</textarea>
<input name="note" placeholder="note (optional)" style="width:100%;margin:6px 0">
<button name="verdict" value="fix">Save fix</button></details></form>
<script>document.addEventListener('keydown',e=>{{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
const v={{a:'accept',r:'reject',n:'skip'}}[e.key];if(v){{const b=document.querySelector(`button[value=${{v}}]`);b.click()}}
if(e.key==='e'){{document.getElementById('d').open=true;document.getElementById('t').focus();e.preventDefault()}}}});</script>
</body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        scope = (re.search(r"scope=(\w+)", self.path) or [None, "test"])[1]
        rows, done = queue(scope)
        skip = getattr(H, "skipped", set())
        rows = [r for r in rows if r["id"] not in skip]
        if not rows:
            body = f"<p>Nothing left in <b>{scope}</b>. {done} reviewed. <a href='/?scope=train'>train</a></p>"
        else:
            r = rows[0]
            reps = r["replacements"]
            body = PAGE.format(
                scope=scope, left=len(rows), done=done, id=r["id"], genre=html.escape(r["genre"]),
                locale=html.escape(r["locale"]), flags=html.escape(" ".join(r["flags"])),
                inp=mark(r["input"], [x["original"] for x in reps], "o"),
                out=mark(r["output"], [x["replacement"] for x in reps], "n"),
                table="".join(f"<tr><td>{html.escape(x['original'])}</td><td>→</td><td>{html.escape(x['replacement'])}</td></tr>"
                              for x in reps) or "<tr><td>(no replacements)</td></tr>",
                raw=html.escape(r["output"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self):
        from urllib.parse import parse_qs
        n = int(self.headers["Content-Length"])
        f = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode(), keep_blank_values=True).items()}
        if f["verdict"] == "skip":
            H.skipped = getattr(H, "skipped", set()) | {f["id"]}
        else:
            rec = {"id": f["id"], "verdict": f["verdict"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if f["verdict"] == "fix":
                rec["output"] = f["output"].replace("\r\n", "\n")
            if f.get("note"):
                rec["note"] = f["note"]
            with open(REVIEWS, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.send_response(303)
        self.send_header("Location", f"/?scope={f['scope']}")
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()

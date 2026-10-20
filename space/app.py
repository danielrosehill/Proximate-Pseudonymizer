"""Review Space for Proximate-Pseudonymizer pairs (private).

Shows one (original, pseudonymized) pair at a time with the swaps highlighted, plus the
replacement map and validation flags. Accept / Reject / Fix / Skip, colour-coded, with
keyboard shortcuts (a / r / e / n). Each decision is committed to the dataset repo as
reviews/<id>.json using the signed-in user's own OAuth token: no token is stored in the
Space, and since the dataset is private the pairs are also loaded with that token, after
sign-in. scripts/pull_reviews.py in the project repo merges the decisions back into
data/reviews.jsonl.
"""

from __future__ import annotations

import difflib
import html
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone

import gradio as gr
from huggingface_hub import HfApi, hf_hub_download

DATASET = os.environ.get("DATASET_REPO", "danielrosehill/Proximate-Pseudonymizer-Pairs")
PAIRS_PATH = "pairs.jsonl"
CACHE: dict = {}

CSS = """
.pair {white-space: pre-wrap; line-height: 1.6; padding: 12px 14px; border-radius: 8px;
       border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary); min-height: 120px}
mark.o {background: rgba(239, 68, 68, .28); color: inherit; border-radius: 3px; padding: 0 2px}
mark.n {background: rgba(34, 197, 94, .30); color: inherit; border-radius: 3px; padding: 0 2px}
.map td {padding: 2px 10px; border-bottom: 1px solid var(--border-color-primary)}
.flag {display: inline-block; padding: 1px 8px; margin: 2px; border-radius: 10px; font-size: .85em;
       background: rgba(245, 158, 11, .22)}
.flag.hard {background: rgba(239, 68, 68, .25)}
#accept {background: #16a34a; color: #fff; border: none}
#reject {background: #dc2626; color: #fff; border: none}
#fix {background: #d97706; color: #fff; border: none}
#skip {background: #6b7280; color: #fff; border: none}
#accept:hover, #reject:hover, #fix:hover, #skip:hover {filter: brightness(1.1)}
"""

# Keyboard shortcuts; ignored while typing in a text field.
JS = """
() => {
  document.addEventListener('keydown', (e) => {
    const t = e.target.tagName;
    if (t === 'TEXTAREA' || t === 'INPUT' || e.ctrlKey || e.metaKey || e.altKey) return;
    const id = {a: 'accept', r: 'reject', n: 'skip'}[e.key];
    if (id) { document.getElementById(id)?.click(); e.preventDefault(); }
    if (e.key === 'e') { document.querySelector('#fixbox textarea')?.focus(); e.preventDefault(); }
  });
}
"""

HARD = ("conflict", "chain", "contains", "leak", "empty_map", "close", "repair_failed")

# Why a reviewer edited or rejected a pair. Saved with the exact edited spans so corrections can
# be analysed by kind ("targeted learnings"), not just counted.
REASONS = [
    "missed detail (still identifying)",
    "not proximate: gender",
    "not proximate: culture / language",
    "not proximate: place size or region",
    "not proximate: other",
    "inconsistent with another mention",
    "should not be replaced (public / generic)",
    "format broken (date, phone, email, ID)",
    "grammar or agreement after swap",
    "other",
]


def span_edits(before: str, after: str, reps: list[dict]) -> list[dict]:
    """Word-level diff of the model output vs the reviewer's text, as {from, to[, original]} spans.
    "original" is the source-text span when the edited span was one of the generated substitutes."""
    tok = re.compile(r"\w+|\s+|[^\w\s]")
    a, b = tok.findall(before), tok.findall(after)
    back = {x["replacement"]: x["original"] for x in reps}
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        e = {"from": "".join(a[i1:i2]).strip(), "to": "".join(b[j1:j2]).strip()}
        if e["from"] in back:
            e["original"] = back[e["from"]]
        if e["from"] or e["to"]:
            out.append(e)
    return out


def pairs(token: str) -> list[dict]:
    if "pairs" not in CACHE:
        path = hf_hub_download(DATASET, PAIRS_PATH, repo_type="dataset", token=token, force_download=True)
        with open(path, encoding="utf-8") as f:
            CACHE["pairs"] = [json.loads(l) for l in f if l.strip()]
        CACHE["by_id"] = {r["id"]: r for r in CACHE["pairs"]}
    return CACHE["pairs"]


def reviews(token: str) -> set[str]:
    files = HfApi(token=token).list_repo_files(DATASET, repo_type="dataset")
    return {f[len("reviews/"):-len(".json")] for f in files if f.startswith("reviews/") and f.endswith(".json")}


def mark(text: str, spans: list[str], cls: str) -> str:
    if not spans:
        return html.escape(text)
    pat = re.compile("|".join(re.escape(s) for s in sorted(set(spans), key=len, reverse=True)))
    out, i = [], 0
    for m in pat.finditer(text):
        out += [html.escape(text[i:m.start()]), f'<mark class="{cls}">{html.escape(m.group(0))}</mark>']
        i = m.end()
    return "".join(out) + html.escape(text[i:])


def render(r: dict | None):
    if r is None:
        return "", "", "", "", ""
    reps = r["replacements"]
    inp = f'<div class="pair">{mark(r["input"], [x["original"] for x in reps], "o")}</div>'
    out = f'<div class="pair">{mark(r["output"], [x["replacement"] for x in reps], "n")}</div>'
    rows = "".join(f"<tr><td>{html.escape(x['original'])}</td><td>→</td><td>{html.escape(x['replacement'])}</td></tr>"
                   for x in reps) or "<tr><td><i>no replacements: output should equal input</i></td></tr>"
    flags = "".join(f'<span class="flag{" hard" if f.startswith(HARD) else ""}">{html.escape(f)}</span>' for f in r["flags"])
    meta = (f"**{r['id']}** · {r['split']} · {html.escape(r['genre'])} · {html.escape(r['locale'])}"
            f"{' · ' + r['language'] if r.get('language') else ''}")
    return meta, inp, out, f'<table class="map">{rows}</table>{flags}', r["output"]


def chips(r: dict) -> str:
    return "".join(f'<span class="flag{" hard" if f.startswith(HARD) else ""}">{html.escape(f)}</span>' for f in r["flags"])


def render_review(r: dict | None):
    """-> meta, highlighted original, flag chips, editable pseudonymized text."""
    if r is None:
        return "Nothing left in this queue.", "", "", ""
    meta, inp, _, _, out = render(r)
    return meta, inp, chips(r), out


def queue(token: str, scope: str, skipped: list[str]) -> list[dict]:
    done = reviews(token)
    rows = [r for r in pairs(token) if r["id"] not in done and r["id"] not in skipped]
    if scope == "test":
        return [r for r in rows if r["split"] == "test"]
    if scope == "train (flagged first)":
        tr = [r for r in rows if r["split"] == "train"]
        return [r for r in tr if r["flags"]] + [r for r in tr if not r["flags"]]
    return [r for r in rows if r["split"] == "dropped"]


def progress(token: str, scope: str, left: int) -> str:
    n = len(reviews(token))
    test = sum(r["split"] == "test" for r in pairs(token))
    return f"{left} left in **{scope}** · {n} reviewed in total · test pool {test}"


def next_item(token, scope, skipped):
    q = queue(token, scope, skipped)
    item = q[0] if q else None
    return (item["id"] if item else None, *render_review(item), progress(token, scope, len(q)))


def need(token):
    if token is None:
        raise gr.Error("Sign in with Hugging Face first.")
    return token.token


def on_load(scope, skipped, oauth_token: gr.OAuthToken | None):
    if oauth_token is None:
        return None, "Sign in to load the pairs.", "", "", "", ""
    return next_item(oauth_token.token, scope, skipped or [])


def on_decide(verdict, item_id, fixed, reasons, note, scope, skipped, profile: gr.OAuthProfile | None,
              oauth_token: gr.OAuthToken | None):
    tok = need(oauth_token)
    if verdict == "skip":
        skipped = (skipped or []) + ([item_id] if item_id else [])
        return (*next_item(tok, scope, skipped), skipped, [], "")
    if not item_id:
        raise gr.Error("No item loaded.")
    r = CACHE["by_id"][item_id]
    rec = {"id": item_id, "verdict": verdict, "reviewer": profile.username if profile else None,
           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    edited = fixed.strip() != r["output"].strip()
    if verdict == "accept" and edited:
        verdict = rec["verdict"] = "fix"  # an edited box is a fix, whichever button saved it
    if verdict == "fix":
        rec["output"] = fixed.replace("\r\n", "\n")
        rec["edits"] = span_edits(r["output"], rec["output"], r["replacements"])
    if reasons:
        rec["reasons"] = reasons
    if note.strip():
        rec["note"] = note.strip()
    HfApi(token=tok).upload_file(path_or_fileobj=json.dumps(rec, ensure_ascii=False).encode(),
                                 path_in_repo=f"reviews/{item_id}.json", repo_id=DATASET, repo_type="dataset",
                                 commit_message=f"Review {item_id}: {verdict}")
    if verdict != "accept" and not reasons and not note.strip():
        gr.Warning(f"{item_id}: {verdict} saved without a reason")
    else:
        gr.Info(f"{item_id}: {verdict}" + (f" · {len(rec['edits'])} edit(s)" if rec.get("edits") else ""))
    return (*next_item(tok, scope, skipped or []), skipped, [], "")


def browse(split, flag, idx, oauth_token: gr.OAuthToken | None):
    tok = need(oauth_token)
    rows = [r for r in pairs(tok) if (split == "all" or r["split"] == split)
            and (flag == "any" or (flag == "none" and not r["flags"]) or any(f.startswith(flag) for f in r["flags"]))]
    if not rows:
        return "No rows match.", "", "", "", gr.update(maximum=0, value=0)
    idx = min(int(idx), len(rows) - 1)
    meta, inp, out, m, _ = render(rows[idx])
    return f"{idx + 1} / {len(rows)} · " + meta, inp, out, m, gr.update(maximum=len(rows) - 1, value=idx)


def stats(oauth_token: gr.OAuthToken | None):
    tok = need(oauth_token)
    ps = pairs(tok)
    flags = Counter(f.split(":")[0] for r in ps for f in r["flags"])
    lines = [f"**{len(ps)} pairs** · " + " · ".join(f"{k} {v}" for k, v in Counter(r["split"] for r in ps).items()),
             f"**reviewed:** {len(reviews(tok))}", "**flags:** " + " · ".join(f"{k} {v}" for k, v in flags.most_common()),
             "**top settings:** " + " · ".join(f"{k} {v}" for k, v in Counter(r["locale"] for r in ps).most_common(8))]
    return "\n\n".join(lines)


with gr.Blocks(title="Pseudonymizer pair review", css=CSS, js=JS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Proximate-Pseudonymizer · pair review\nRed = the identifying spans in the original. Edit the "
                "pseudonymized text directly where a substitute isn't *proximate* (wrong gender or country, very "
                "different scale), a detail was missed, or an ordinary word was swapped, tick why, then **Accept** "
                "(saves your edit). Keys: **a** accept · **r** reject · **e** edit · **n** skip.")
    gr.LoginButton(icon=None, size="md")
    item_id, skipped = gr.State(None), gr.State([])
    with gr.Tab("Review"):
        scope = gr.Radio(["test", "train (flagged first)", "dropped"], value="test", label="Queue")
        prog = gr.Markdown()
        meta = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Original**")
                inp = gr.HTML()
            with gr.Column():
                gr.Markdown("**Pseudonymized — edit directly (e)**")
                # The pseudonymized text IS the editor (Daniel, 2026-10-21: "just direct editing
                # over the obfuscated text"); no separate swap table in the review tab.
                fixbox = gr.Textbox(lines=12, max_lines=60, elem_id="fixbox", show_label=False, container=False)
        flagbox = gr.HTML()
        with gr.Row():
            reasons = gr.CheckboxGroup(REASONS, label="Reason for edit / rejection (where applicable)", scale=3)
            note = gr.Textbox(label="Details (optional)", lines=3, scale=2,
                              placeholder="e.g. Ramat Gan is a suburb, Haifa is too big a jump")
        with gr.Row():
            b_acc = gr.Button("✓ Accept / save edit (a)", elem_id="accept")
            b_rej = gr.Button("✗ Reject (r)", elem_id="reject")
            b_skip = gr.Button("→ Skip (n)", elem_id="skip")
    with gr.Tab("Browse"):
        with gr.Row():
            split = gr.Dropdown(["all", "test", "train", "dropped"], value="all", label="Split")
            flag = gr.Dropdown(["any", "none", "close", "repaired", "repair_failed", "leak", "contains", "chain",
                                "conflict", "missing", "partial_mismatch", "unexpected_map", "empty_map"],
                               value="any", label="Flag")
            idx = gr.Slider(0, 0, step=1, value=0, label="Row")
        bmeta, binp, bout, bmap = gr.Markdown(), gr.HTML(), gr.HTML(), gr.HTML()
        with gr.Row():
            prev_b, next_b = gr.Button("← Prev"), gr.Button("Next →")
    with gr.Tab("Stats"):
        st = gr.Markdown()
        gr.Button("Refresh").click(stats, None, st)

    shown = [item_id, meta, inp, flagbox, fixbox, prog]
    demo.load(on_load, [scope, skipped], shown)
    scope.change(on_load, [scope, skipped], shown)
    for btn, v in ((b_acc, "accept"), (b_rej, "reject"), (b_skip, "skip")):
        btn.click(on_decide, [gr.State(v), item_id, fixbox, reasons, note, scope, skipped],
                  [*shown, skipped, reasons, note])
    # Gradio injects the OAuth token only into functions whose signature is type-hinted with
    # gr.OAuthToken, so these wrappers are real functions, not lambdas.
    def browse_first(s, f, oauth_token: gr.OAuthToken | None):
        return browse(s, f, 0, oauth_token)

    def browse_prev(s, f, i, oauth_token: gr.OAuthToken | None):
        return browse(s, f, max(i - 1, 0), oauth_token)

    def browse_next(s, f, i, oauth_token: gr.OAuthToken | None):
        return browse(s, f, i + 1, oauth_token)

    bout_all = [bmeta, binp, bout, bmap, idx]
    for c in (split, flag):
        c.change(browse_first, [split, flag], bout_all)
    idx.release(browse, [split, flag, idx], bout_all)
    prev_b.click(browse_prev, [split, flag, idx], bout_all)
    next_b.click(browse_next, [split, flag, idx], bout_all)

if __name__ == "__main__":
    demo.launch()

"""Public demo: swap a text's identifying details for close substitutes with Proximate-Pseudonymizer-0.6B.

Runs the published checkpoint (pp-r2, text mode) on the Space's CPU with exactly the inference setup of
training/train.py: the same system prompt, no few-shot examples, Qwen3 thinking disabled
(enable_thinking=False), greedy decoding, max_new_tokens = 1.4 x input tokens + 64. The result is shown
next to a word-level diff so each swap is visible, because the model still misses about 1 detail in 11.
"""
import difflib
import html
import json
import re
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

REPO = "danielrosehill/Proximate-Pseudonymizer-0.6B"
# Verbatim from training/train.py (SYSTEM, TARGET=text); the model was trained with no other prompt.
SYSTEM = ("Rewrite the text with every identifying detail (names, places, organisations, dates, numbers, "
          "contacts) replaced by a close, plausible substitute of the same kind. Use the same substitute "
          "every time a detail recurs. Change nothing else. Reply with the rewritten text only.")
MAX_WORDS = 200  # ~15-20 s per 50 words on the 2-vCPU Space (measured 2026-10-21)

torch.set_num_threads(max(1, torch.get_num_threads()))
TOK = AutoTokenizer.from_pretrained(REPO)
MODEL = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()


def stream(text):
    """Yield the growing greedy output for `text`."""
    prompt = TOK.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = TOK(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=True)
    n_text = len(TOK(text)["input_ids"])
    Thread(target=MODEL.generate, kwargs=dict(**enc, streamer=streamer, do_sample=False, temperature=None,
                                               top_p=None, top_k=None,
                                               max_new_tokens=int(n_text * 1.4) + 64)).start()
    out = ""
    for piece in streamer:
        out += piece
        yield out


def diff(a, b):
    """Word-level diff of input `a` and output `b` -> (inline HTML, list of (before, after) swaps)."""
    wa, wb = re.findall(r"\S+|\s+", a), re.findall(r"\S+|\s+", b)
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    ops = []
    for op in sm.get_opcodes():  # merge "Valerie Doherty" -> "Vanessa Keane" into one swap, not two
        if (len(ops) >= 2 and op[0] != "equal" and ops[-1][0] == "equal" and ops[-2][0] != "equal"
                and not "".join(wa[ops[-1][1]:ops[-1][2]]).strip()):
            ops.pop()
            prev = ops.pop()
            op = ("replace", prev[1], op[2], prev[3], op[4])
        ops.append(op)
    out, swaps = [], []
    for op, i1, i2, j1, j2 in ops:
        before, after = "".join(wa[i1:i2]), "".join(wb[j1:j2])
        if op == "equal":
            out.append(html.escape(before))
            continue
        if before.strip():
            out.append(f"<del>{html.escape(before)}</del>")
        if after.strip():
            out.append(f"<ins>{html.escape(after)}</ins>")
        if not before.strip() or not after.strip():
            out.append(" ")
        p = ".,;:!?)\"'"
        pb, pa = (re.split(r"(?<=[.,;:!?])\s+", x.strip()) for x in (before, after))
        pairs = zip(pb, pa) if len(pb) == len(pa) else [(before, after)]  # "Haifa. Daniel" -> two swaps
        swaps += [(x.strip().strip(p), y.strip().strip(p)) for x, y in pairs]
    body = "".join(out).replace("\n", "<br>")
    return f'<div class="diff">{body}</div>', swaps


def swap_table(swaps):
    if not swaps:
        return ""
    seen, rows = set(), []
    for b, a in swaps:
        if (b, a) not in seen:
            seen.add((b, a))
            b, a = b.replace("|", "\\|"), a.replace("|", "\\|")
            rows.append(f"| {b or '*(added)*'} | {a or '*(removed)*'} |".replace("\n", " "))
    return "| Original | Substitute |\n| --- | --- |\n" + "\n".join(rows)


def run(text):
    text = (text or "").strip()
    if not text:
        yield "", "<p class='note'>Paste a text first.</p>", "", ""
        return
    if len(text.split()) > MAX_WORDS:
        yield "", f"<p class='note'>Please keep it under {MAX_WORDS} words (this runs on a small CPU).</p>", "", ""
        return
    out = ""
    for out in stream(text):
        yield out, diff(text, out)[0], "", "Rewriting…"
    out = re.sub(r"^<think>\s*</think>\s*", "", out).strip()
    view, swaps = diff(text, out)
    notes = []
    if not out:
        notes.append("The model returned nothing; treat the original as not pseudonymized.")
    elif out == text:
        notes.append("Nothing changed. Either the text has no identifying details or the model missed them all.")
    else:
        notes.append(f"{len(swaps)} changed span(s). Check the original text for any detail that is still there:"
                     " the model leaves about 1 in 11 unchanged.")
    a, b = len(text.split()), len(out.split())
    if out and abs(a - b) > max(5, 0.15 * a):
        notes.append(f"Length changed a lot ({a} -> {b} words), so the model may have rewritten or dropped text.")
    yield out, view, swap_table(swaps), " ".join(notes)


EXAMPLES = json.loads((Path(__file__).parent / "examples.json").read_text())

CSS = """
.diff { font-size: 15px; line-height: 1.6; max-width: 72ch; }
.diff del { background: #fecaca; color: #7f1d1d; text-decoration: line-through; padding: 0 2px; border-radius: 2px; }
.diff ins { background: #bbf7d0; color: #14532d; text-decoration: none; padding: 0 2px; border-radius: 2px; }
.warn { border-left: 4px solid #dc2626; padding: 8px 12px; background: rgba(220, 38, 38, 0.08); }
.note { opacity: 0.7; }
"""

INTRO = """
# Proximate Pseudonymizer

A 0.6B-parameter Qwen3 fine-tune that rewrites a text with its identifying details (names, places,
organisations, dates, numbers, contacts) swapped for **close, plausible substitutes**, the same substitute
every time a detail recurs. *Jerusalem* becomes *Tel Aviv*, not *[CITY]* and not *Ulaanbaatar*, so the text
still reads and means the same.
"""

WARNING = """
<div class="warn">
<b>Proof of concept: do not trust it with real private data unsupervised.</b>
On 295 held-out texts it left <b>9.3% of identifying details unchanged</b> (about 1 in 11).
Consistency (one substitute per detail) is 0.99 and the rest of the text is 0.998 preserved, but a missed
name or date is a privacy leak. <b>Review every output</b> before you share it. Use the diff on the right:
red is the original, green the substitute; anything identifying that is <i>not</i> marked was missed.
</div>
"""

LINKS = """
The examples are synthetic held-out test texts the model never saw. Runs on a small CPU: about 15-20 seconds per 50 words.

[Model](https://huggingface.co/danielrosehill/Proximate-Pseudonymizer-0.6B) ·
[Code and data pipeline](https://github.com/danielrosehill/Proximate-Pseudonymizer)
"""

with gr.Blocks(title="Proximate Pseudonymizer", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    gr.HTML(WARNING)
    gr.Markdown(LINKS)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Text", lines=12, placeholder=f"Paste a text (up to {MAX_WORDS} words)")
            btn = gr.Button("Pseudonymize", variant="primary")
            gr.Examples([[e["text"]] for e in EXAMPLES], inputs=[inp],
                        example_labels=[e["text"][:60] + "…" for e in EXAMPLES])
        with gr.Column():
            out_diff = gr.HTML(label="Changes")
            status = gr.Markdown()
            with gr.Accordion("Swaps", open=True):
                swaps = gr.Markdown()
            with gr.Accordion("Pseudonymized text", open=False):
                out = gr.Textbox(lines=10, show_copy_button=True, label=None)
    btn.click(run, [inp], [out, out_diff, swaps, status], api_name="pseudonymize")

demo.queue(default_concurrency_limit=1).launch()

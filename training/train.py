"""Fine-tune Qwen3-0.6B: text in -> the same text with identifying details swapped for proximate ones.

Runs on Modal's `ai-training` app (ai-models/Modal-AI-Training), uploading this whole repo:

    cd ~/repos/github/ai-models/Modal-AI-Training
    MODAL_PROFILE=danielrosehill modal run app.py --upload ../Proximate-Pseudonymizer \
        --cmd "python training/train.py" --run pp-r1 --gpu L4

Reads data/train.jsonl and data/test.jsonl (built by scripts/build.py). Writes to $OUT: the
model, predictions from the untrained base (few-shot) and the fine-tuned model (zero-shot) on
the test split, and metrics.json with timings. Score with scripts/evaluate.py.
TARGET=text (default) trains the model to write the whole rewritten text. TARGET=map trains it to
emit only the swap list as JSON ({"replacements": [{"original", "replacement"}]}); the text is
then produced by scripts/pairs.py apply_map, so consistency and the untouched rest of the text
hold by construction and the model's job shrinks to detection + choosing substitutes. This is
the shape eternisai/Anonymizer uses (a replace_entities tool call). Map predictions are written
as rewritten text (plus the raw reply) so scripts/evaluate.py scores both modes the same way.
Env: BASE, EPOCHS, LR, BATCH (per device, default 4; grad accumulation keeps 16), TARGET, MAX_TRAIN, TEST_N (score on the
first N test rows only; probe runs use MAX_TRAIN=800 EPOCHS=1 TEST_N=100, see docs/spec.md), TEST_N (score
on the first N test rows only; with MAX_TRAIN this gives a ~5-minute preliminary run) (cap rows, for smoke runs), SKIP_BASE=1 (reuse an earlier run's
pred_base.jsonl; the base eval is ~12 min of L4 time and does not change between runs).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
import trl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from pairs import apply_map  # noqa: E402

BASE = os.environ.get("BASE", "Qwen/Qwen3-0.6B")
OUT = Path(os.environ.get("OUT", "out"))
DATA = Path(__file__).resolve().parent.parent / "data"
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-5"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "0"))
TEST_N = int(os.environ.get("TEST_N", "0"))
TEST_N = int(os.environ.get("TEST_N", "0"))
SKIP_BASE = os.environ.get("SKIP_BASE") == "1"
TARGET = os.environ.get("TARGET", "text")
BATCH = int(os.environ.get("BATCH", "4"))

# Kept short on purpose: the behaviour lives in the weights, and the published model card
# documents this exact string as the system prompt to use.
SYSTEM = ("Rewrite the text with every identifying detail (names, places, organisations, dates, numbers, "
          "contacts) replaced by a close, plausible substitute of the same kind. Use the same substitute "
          "every time a detail recurs. Change nothing else. Reply with the rewritten text only.")

SYSTEM_MAP = ("List every identifying detail in the text (names, places, organisations, dates, numbers, "
              "contacts) with a close, plausible substitute of the same kind, exactly as each appears in the "
              "text. One entry per distinct detail; related forms (a first name used alone, a name inside an "
              "email) get matching substitutes. Reply with JSON only: "
              '{"replacements": [{"original": "...", "replacement": "..."}]}. No details: {"replacements": []}.')
if TARGET == "map":
    SYSTEM = SYSTEM_MAP

SHOT_IN = ("I'm Daniel, I live in Jerusalem and work at Hadassah. My brother Yossi still lives in Haifa; "
           "call me on 052-841-2209 if you need anything. Daniel")
SHOT_OUT = ("I'm David, I live in Tel Aviv and work at Sheba. My brother Moshe still lives in Netanya; "
            "call me on 052-617-3904 if you need anything. David")


def load(name):
    return [json.loads(l) for l in open(DATA / name)]


def messages(text, shots=False):
    m = [{"role": "system", "content": SYSTEM}]
    if shots:
        m += [{"role": "user", "content": SHOT_IN}, {"role": "assistant", "content": SHOT_OUT}]
    return m + [{"role": "user", "content": text}]


def target(r):
    if TARGET == "map":
        return json.dumps({"replacements": r["replacements"]}, ensure_ascii=False)
    return r["output"]


def to_text(r, reply):
    """Map mode: parse the JSON reply and apply it; unparseable replies leave the text unchanged."""
    if TARGET != "map":
        return {"id": r["id"], "pred": reply}
    try:
        reps = json.loads(reply)["replacements"]
        m = {x["original"]: x["replacement"] for x in reps if x.get("original") and x.get("replacement")}
        return {"id": r["id"], "pred": apply_map(r["input"], m), "raw": reply}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return {"id": r["id"], "pred": r["input"], "raw": reply, "parse_error": True}


def generate(model, tok, rows, shots):
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        texts = [tok.apply_chat_template(messages(r["input"], shots), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        longest = max(len(tok(r["input"])["input_ids"]) for r in batch)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=int(longest * 1.4) + 64, do_sample=False)
        for r, o in zip(batch, out):
            preds.append(to_text(r, tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()))
    return preds


def dump(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"transformers {transformers.__version__}, trl {trl.__version__}", file=sys.stderr)
    train, test = load("train.jsonl"), load("test.jsonl")
    if MAX_TRAIN:
        train, test = train[:MAX_TRAIN], test[:TEST_N or 16]
    elif TEST_N:
        test = test[:TEST_N]
    tok = AutoTokenizer.from_pretrained(BASE)
    t0 = time.time()

    if not SKIP_BASE:
        base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
        dump(OUT / "pred_base.jsonl", generate(base, tok, test, shots=True))
        del base
        torch.cuda.empty_cache()
    t_base = time.time() - t0

    # Prompts rendered by hand, not via SFTConfig chat-template options: those names changed
    # between TRL releases (see Prompt-User-Context-Extractor docs/poc.md). TRL trains on the
    # completion only.
    def to_example(r):
        prompt = tok.apply_chat_template(messages(r["input"]), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        return {"prompt": prompt, "completion": target(r) + tok.eos_token}

    ds = Dataset.from_list([to_example(r) for r in train])
    cfg = SFTConfig(output_dir=str(OUT / "ckpt"), num_train_epochs=EPOCHS, learning_rate=LR,
                    per_device_train_batch_size=BATCH, gradient_accumulation_steps=16 // BATCH, gradient_checkpointing=True,
                    bf16=True, max_length=3072, logging_steps=10, save_strategy="no", report_to=[],
                    lr_scheduler_type="cosine", warmup_steps=20)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    t1 = time.time()
    trainer.train()
    t_train = time.time() - t1
    trainer.save_model(str(OUT / "model"))
    tok.save_pretrained(str(OUT / "model"))

    t2 = time.time()
    dump(OUT / "pred_ft.jsonl", generate(trainer.model, tok, test, shots=False))
    metrics = {"base": BASE, "target": TARGET, "epochs": EPOCHS, "lr": LR, "n_train": len(train), "n_test": len(test),
               "system": SYSTEM,
               "seconds": {"base_eval": round(t_base), "train": round(t_train), "ft_eval": round(time.time() - t2)},
               "log": trainer.state.log_history}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics["seconds"]), file=sys.stderr)


if __name__ == "__main__":
    main()

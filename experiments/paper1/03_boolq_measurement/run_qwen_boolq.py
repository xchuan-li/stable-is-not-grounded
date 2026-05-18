"""
evaluate_boolq_targeting.py

Feasibility MVP: turn the framework's instrument on a POPULAR, trusted
benchmark (BoolQ) and ask whether its headline accuracy hides instability.

IMPORTANT SCOPE (read before citing any number):
- This is the *measurement-only* half of the framework (dual metric +
  three-state decomposition). It uses NO causal structure / DAG. That is
  deliberate and required: BoolQ items have no stipulated ground-truth
  causal graph, so applying the full MCS / clause-(iii) operationalization
  here would reintroduce exactly the circularity the framework removes.
  The causal operationalization belongs only on the synthetic, by-
  construction benchmark.
- Perturbations here are conservative and answer-preserving *by
  construction*: an irrelevant appended sentence (distractor) and
  task-framing rephrasings with the BoolQ question kept verbatim. We do
  NOT use LLM paraphrase here, to avoid importing semantic-drift risk
  into a feasibility test. Validated question/passage paraphrase + the
  semantic-preservation validation layer are the documented publication-
  version extension, not part of this MVP.
- Model: Qwen2.5-1.5B-Instruct, zero-shot. Single small model, single
  run, n=40 balanced. This is a feasibility pilot, not a result.

Claim this MVP can legitimately support:
  "On a validated sample, BoolQ's headline (original-prompt) accuracy
   overstates *stable* correctness by G points; a non-trivial fraction
   of items flip their yes/no answer under answer-preserving edits."

Data provenance note: the commonly-cited direct URL
storage.googleapis.com/boolq/dev.jsonl was found serving a data-poisoning
POC payload, NOT BoolQ. Data here is pulled from the integrity-checked
HuggingFace `google/boolq` dataset instead.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def bootstrap_ci(values, stat, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
SAMPLE_FILE = BASE_DIR / "data" / "boolq_sample_40.jsonl"
RESULTS_DIR = BASE_DIR / "results"
RAW_FILE = RESULTS_DIR / "raw.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.json"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 1024  # prompt cap; BoolQ sample max passage ~1227 chars (~300 tok)

SYS = (
    "You answer reading-comprehension yes/no questions using ONLY the "
    "given passage. Respond with exactly one word: yes or no."
)

# Irrelevant, answer-preserving distractor sentences (share no content
# with BoolQ topics; appended to the passage).
DISTRACTORS = [
    "The cafeteria introduced a new dessert option last Tuesday.",
    "A light drizzle was forecast for the coastal region that weekend.",
    "The conference room was repainted a pale shade of grey.",
    "Bus route 12 will run on a holiday schedule next month.",
]


def build_variants(item):
    """Four answer-preserving presentations of one BoolQ item.

    The correct answer is invariant across all four by construction:
    appending an irrelevant sentence or rephrasing the task framing does
    not change what the passage entails about the question.
    """
    p, q = item["passage"], item["question"]
    distractor_passage = p.rstrip() + " " + DISTRACTORS[item["id"] % len(DISTRACTORS)]
    return {
        "original": f"Passage:\n{p}\n\nQuestion: {q}\nAnswer (yes or no):",
        "distractor": f"Passage:\n{distractor_passage}\n\nQuestion: {q}\nAnswer (yes or no):",
        "frame_a": f"Read the passage and decide whether the question holds.\n\nPassage:\n{p}\n\nQuestion: {q}\nRespond with yes or no:",
        "frame_b": f"Based ONLY on the following passage, answer the yes/no question.\n\n{p}\n\nQ: {q}\nA (yes/no):",
    }


def extract_yesno(text):
    t = text.lower()
    ans = re.findall(r"answer\s*[:\-]?\s*(yes|no)", t)
    if ans:
        return ans[-1]
    tokens = re.findall(r"\b(yes|no)\b", t)
    if tokens:
        return tokens[-1]
    return "unknown"


def predict(prompt, tokenizer, model, device):
    messages = [{"role": "system", "content": SYS},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_LEN).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return extract_yesno(tokenizer.decode(gen, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(SAMPLE_FILE),
                    help="path to a boolq_sample_*.jsonl")
    args = ap.parse_args()
    sample_path = Path(args.sample)
    stem = sample_path.stem.replace("boolq_sample_", "")
    raw_file = RESULTS_DIR / f"raw_{stem}.csv"
    summary_file = RESULTS_DIR / f"summary_{stem}.json"

    items = [json.loads(l) for l in open(sample_path, encoding="utf-8")]
    print(f"BoolQ sample: {len(items)} items "
          f"(balanced; gold yes={sum(i['answer'] for i in items)})")
    n_gen = len(items) * 4
    print(f"Model: {MODEL_NAME} (zero-shot) | generations: {n_gen}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | loading model...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    model.eval()
    print(f"Model loaded in {time.time()-t0:.0f}s")

    rows = []
    pert_correct = defaultdict(int)
    pert_total = defaultdict(int)
    state_counts = defaultdict(int)
    headline_correct = 0
    headline_ind, stable_ind = [], []
    group_acc, group_cons = [], []
    done = 0
    t0 = time.time()

    for it in items:
        gold = "yes" if it["answer"] else "no"
        variants = build_variants(it)
        preds = {}
        for vname, prompt in variants.items():
            pred = predict(prompt, tok, model, device)
            preds[vname] = pred
            pert_correct[vname] += int(pred == gold)
            pert_total[vname] += 1
            rows.append({"id": it["id"], "gold": gold, "variant": vname,
                         "pred": pred, "correct": pred == gold,
                         "question": it["question"][:160]})
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                print(f"  {done}/{n_gen} generations "
                      f"({el:.0f}s, ~{el/done*(n_gen-done):.0f}s left)", flush=True)

        anchor = preds["original"]
        n = len(preds)
        acc = sum(p == gold for p in preds.values()) / n
        cons = sum(p == anchor for p in preds.values()) / n
        group_acc.append(acc)
        group_cons.append(cons)
        h = int(preds["original"] == gold)
        headline_correct += h
        headline_ind.append(h)
        agree = len(set(preds.values())) == 1
        state = ("stable_correct" if agree and acc == 1.0
                 else "stable_wrong" if agree
                 else "unstable")
        state_counts[state] += 1
        stable_ind.append(int(state == "stable_correct"))

    N = len(items)
    headline_acc = headline_correct / N
    stable_correct_pct = state_counts["stable_correct"] / N
    h_arr = np.asarray(headline_ind)
    s_arr = np.asarray(stable_ind)
    headline_ci = bootstrap_ci(h_arr, np.mean, seed=1)
    stable_ci = bootstrap_ci(s_arr, np.mean, seed=2)
    gap_ci = bootstrap_ci(h_arr - s_arr, np.mean, seed=3)  # paired per-item
    summary = {
        "n_items": N,
        "model": MODEL_NAME,
        "headline_accuracy_original_prompt": round(headline_acc, 4),
        "headline_accuracy_ci": headline_ci,
        "stable_correct_pct": round(stable_correct_pct, 4),
        "stable_correct_ci": stable_ci,
        "headline_overstatement_gap": round(headline_acc - stable_correct_pct, 4),
        "headline_overstatement_gap_ci": gap_ci,
        "SIS_acc": round(sum(group_acc) / N, 4),
        "SIS_consistency": round(sum(group_cons) / N, 4),
        "group_states": {k: state_counts[k] for k in
                         ("stable_correct", "stable_wrong", "unstable")},
        "per_variant_accuracy": {k: round(pert_correct[k] / pert_total[k], 4)
                                 for k in sorted(pert_total)},
        "scope": (f"measurement-only (no causal DAG); conservative answer-"
                  f"preserving perturbations; zero-shot {MODEL_NAME}; n={N}; "
                  f"single run; verdicts are bootstrap 95% CIs."),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(raw_file, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(summary_file, "w"), indent=2)

    print(f"\n========  BoolQ TARGETING (n={N}, 95% CI)  ========")
    print(f"Headline accuracy            : {headline_ci}")
    print(f"Stable-correct fraction      : {stable_ci}")
    print(f"Headline overstatement gap   : {gap_ci}  "
          f"(CI excludes 0 => significant)")
    print(f"SIS_acc={summary['SIS_acc']:.3f}  "
          f"SIS_consistency={summary['SIS_consistency']:.3f}")
    print(f"Group states: {summary['group_states']}")
    print(f"Per-variant accuracy: {summary['per_variant_accuracy']}")
    print(f"\nSaved: {raw_file}\nSaved: {summary_file}")


if __name__ == "__main__":
    main()

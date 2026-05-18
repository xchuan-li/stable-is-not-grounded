"""
evaluate_h4_cot.py

H4: Does chain-of-thought (CoT) improve accuracy but DECREASE stability?

Method:
- Same model (Qwen2.5-1.5B-Instruct), same items, two prompting conditions:
    DIRECT : ask for a bare yes/no answer
    COT    : ask the model to reason step by step, then give yes/no
- Deterministic (greedy) decoding so that any cross-variant disagreement is
  caused by the perturbation, not by sampling noise. This is essential: if we
  sampled with temperature, "instability" could just be decoding variance.
- For each condition we compute the dual metric used in evaluate_dual_metric.py:
    SIS_acc          : mean per-group accuracy
    SIS_consistency  : mean per-group agreement with the 'original' variant
    group states     : stable_correct / stable_wrong / unstable
- H4 is supported if, going DIRECT -> COT:
    accuracy goes UP   AND   (stable_correct % goes DOWN  or unstable % goes UP)

Subset: balanced, configurable. Default 8 groups per category (4 yes / 4 no)
so the pipeline produces a signal in ~1h on CPU before committing to full 100.
"""

import argparse
import csv
import json
import re
import sys
import time

# Make stdout line-buffered so progress is visible even when redirected to a
# file or run in the background. The first failed run was invisible for 6.5h
# because block-buffered stdout never flushed; never be blind like that again.
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stable_inference_v3"
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CATEGORIES = ["taxonomic", "transitive", "causal", "commonsense", "contradiction"]


def select_subset(per_category, seed=42):
    """Balanced subset: per_category groups per reasoning type, half yes half no."""
    import random
    rng = random.Random(seed)
    chosen = []
    for cat in CATEGORIES:
        files = sorted((DATA_DIR / cat).glob("*.json"))
        yes, no = [], []
        for fp in files:
            g = json.load(open(fp, encoding="utf-8"))
            (yes if g["expected_answer"] == "yes" else no).append((fp, g))
        k = per_category // 2
        rng.shuffle(yes)
        rng.shuffle(no)
        chosen += yes[:k] + no[:k]
    return [g for _, g in chosen]


def load_all():
    groups = []
    for fp in sorted(DATA_DIR.rglob("*.json")):
        groups.append(json.load(open(fp, encoding="utf-8")))
    return groups


DIRECT_SYS = (
    "You answer reasoning questions. Respond with exactly one word: "
    "'yes' or 'no'. Do not explain."
)
COT_SYS = (
    "You answer reasoning questions. Think step by step FIRST. "
    "Do NOT state the answer before your reasoning. "
    "After the reasoning, end with a final line that is exactly "
    "'Answer: yes' or 'Answer: no' and nothing after it."
)


def extract_yesno(text):
    """Take the model's FINAL committed answer.

    Qwen2.5-1.5B often emits a leading 'Answer: X' and then reasons toward
    the opposite (or gets truncated). The intended output is reason-then-
    answer, so the LAST 'Answer:' occurrence is the committed one. Using the
    first match systematically mis-scored answer-first outputs and biased
    against the CoT condition.
    """
    t = text.lower()
    ans = re.findall(r"answer\s*[:\-]?\s*(yes|no)", t)
    if ans:
        return ans[-1]
    # fall back to last standalone yes/no token
    tokens = re.findall(r"\b(yes|no)\b", t)
    if tokens:
        return tokens[-1]
    return "unknown"


def generate(model, tokenizer, device, system, user, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def score_condition(records):
    """records: list of dicts with group_id, reasoning_type, perturbation_type,
    gold, pred. Returns the dual-metric summary."""
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)

    g_acc, g_cons = {}, {}
    states = defaultdict(int)
    reason_acc = defaultdict(list)

    for gid, rs in groups.items():
        gold = rs[0]["gold"]
        preds = {r["perturbation_type"]: r["pred"] for r in rs}
        n = len(preds)
        acc = sum(1 for p in preds.values() if p == gold) / n
        anchor = preds.get("original", next(iter(preds.values())))
        cons = sum(1 for p in preds.values() if p == anchor) / n
        agree = len(set(preds.values())) == 1
        if agree and acc == 1.0:
            state = "stable_correct"
        elif agree and acc == 0.0:
            state = "stable_wrong"
        elif agree:
            state = "stable_wrong"
        else:
            state = "unstable"
        g_acc[gid] = acc
        g_cons[gid] = cons
        states[state] += 1
        reason_acc[rs[0]["reasoning_type"]].append(acc)

    ng = len(groups)
    return {
        "n_groups": ng,
        "SIS_acc": round(sum(g_acc.values()) / ng, 4),
        "SIS_consistency": round(sum(g_cons.values()) / ng, 4),
        "states": {k: states[k] for k in ("stable_correct", "stable_wrong", "unstable")},
        "stable_correct_pct": round(states["stable_correct"] / ng, 4),
        "unstable_pct": round(states["unstable"] / ng, 4),
        "per_reasoning_acc": {
            rt: round(sum(v) / len(v), 4) for rt, v in sorted(reason_acc.items())
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=8,
                    help="groups per reasoning type (half yes / half no). 0 = all")
    ap.add_argument("--cot-tokens", type=int, default=320)
    args = ap.parse_args()

    groups = load_all() if args.per_category == 0 else select_subset(args.per_category)
    n_calls = len(groups) * 5 * 2
    print(f"Model: {MODEL_NAME}")
    print(f"Groups: {len(groups)}  | total generations: {n_calls} "
          f"(5 variants x 2 conditions)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  | loading model (first run downloads ~3GB)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float32
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.0f}s")

    direct_records, cot_records, raw_rows = [], [], []
    done = 0
    t0 = time.time()
    for g in groups:
        gid = g["group_id"]
        rtype = g["reasoning_type"]
        gold = g["expected_answer"].strip().lower()
        for s in g["samples"]:
            q = s["text"]
            ptype = s["type"]

            d_out = generate(model, tokenizer, device, DIRECT_SYS, q, 8)
            d_pred = extract_yesno(d_out)

            c_out = generate(model, tokenizer, device, COT_SYS, q, args.cot_tokens)
            c_pred = extract_yesno(c_out)

            direct_records.append(dict(group_id=gid, reasoning_type=rtype,
                                       perturbation_type=ptype, gold=gold, pred=d_pred))
            cot_records.append(dict(group_id=gid, reasoning_type=rtype,
                                    perturbation_type=ptype, gold=gold, pred=c_pred))
            raw_rows.append(dict(group_id=gid, reasoning_type=rtype,
                                 perturbation_type=ptype, gold=gold,
                                 direct_pred=d_pred, cot_pred=c_pred,
                                 cot_raw=c_out.replace("\n", " ")[:300]))
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                rate = el / done
                eta = rate * (len(groups) * 5 - done)
                print(f"  {done}/{len(groups) * 5} samples  "
                      f"({el:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    direct = score_condition(direct_records)
    cot = score_condition(cot_records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "h4_cot_raw.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)

    h4_supported = (
        cot["SIS_acc"] > direct["SIS_acc"]
        and (cot["stable_correct_pct"] < direct["stable_correct_pct"]
             or cot["unstable_pct"] > direct["unstable_pct"])
    )
    verdict = {
        "model": MODEL_NAME,
        "n_groups": len(groups),
        "direct": direct,
        "cot": cot,
        "deltas": {
            "SIS_acc": round(cot["SIS_acc"] - direct["SIS_acc"], 4),
            "SIS_consistency": round(cot["SIS_consistency"] - direct["SIS_consistency"], 4),
            "stable_correct_pct": round(
                cot["stable_correct_pct"] - direct["stable_correct_pct"], 4),
            "unstable_pct": round(cot["unstable_pct"] - direct["unstable_pct"], 4),
        },
        "H4_supported": bool(h4_supported),
    }
    json.dump(verdict, open(RESULTS_DIR / "h4_cot_summary.json", "w"), indent=2)

    print("\n==================  H4 RESULT  ==================")
    print(f"{'metric':<22}{'DIRECT':>10}{'COT':>10}{'Δ':>10}")
    for key in ("SIS_acc", "SIS_consistency", "stable_correct_pct", "unstable_pct"):
        print(f"{key:<22}{direct[key]:>10.3f}{cot[key]:>10.3f}"
              f"{verdict['deltas'][key]:>+10.3f}")
    print(f"\nDIRECT states: {direct['states']}")
    print(f"COT    states: {cot['states']}")
    print(f"\nH4 (CoT raises accuracy but lowers stability)  ->  "
          f"{'SUPPORTED' if h4_supported else 'NOT supported'}")
    print("Saved: results/h4_cot_summary.json , results/h4_cot_raw.csv")


if __name__ == "__main__":
    main()

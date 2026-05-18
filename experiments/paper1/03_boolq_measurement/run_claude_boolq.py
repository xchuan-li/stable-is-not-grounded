"""
run_claude_boolq.py

Second-model replication of the §5.3 BoolQ measurement using Claude API.
Identical experiment design to run_qwen_boolq.py:
  - Same 400-item balanced sample (data/boolq_sample_400.jsonl)
  - Same four answer-preserving variants (original, distractor, frame_a, frame_b)
  - Same dual-metric decomposition (headline acc / stable-correct / overstatement gap)
  - Zero-shot, same system prompt intent

Model: claude-haiku-4-5-20251001  (fast inference, different architecture family)
Cost estimate: ~1600 calls × ~350 tok input = ~$0.50 total (prompt cache on sys msg)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_claude_boolq.py
    python run_claude_boolq.py --sample data/boolq_sample_40.jsonl  # quick test
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME   = "claude-haiku-4-5-20251001"
BASE_DIR     = Path(__file__).parent
SAMPLE_FILE  = BASE_DIR / "data" / "boolq_sample_400.jsonl"
RESULTS_DIR  = BASE_DIR / "results"

SYS = (
    "You answer reading-comprehension yes/no questions using ONLY the "
    "given passage. Respond with exactly one word: yes or no."
)

DISTRACTORS = [
    "The cafeteria introduced a new dessert option last Tuesday.",
    "A light drizzle was forecast for the coastal region that weekend.",
    "The conference room was repainted a pale shade of grey.",
    "Bus route 12 will run on a holiday schedule next month.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(values, stat, n_boot=2000, seed=0, alpha=0.05):
    rng   = np.random.default_rng(seed)
    v     = np.asarray(values, dtype=float)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


def build_variants(item):
    p, q = item["passage"], item["question"]
    distractor_p = p.rstrip() + " " + DISTRACTORS[item["id"] % len(DISTRACTORS)]
    return {
        "original":   f"Passage:\n{p}\n\nQuestion: {q}\nAnswer (yes or no):",
        "distractor": f"Passage:\n{distractor_p}\n\nQuestion: {q}\nAnswer (yes or no):",
        "frame_a":    (f"Read the passage and decide whether the question holds.\n\n"
                       f"Passage:\n{p}\n\nQuestion: {q}\nRespond with yes or no:"),
        "frame_b":    (f"Based ONLY on the following passage, answer the yes/no question.\n\n"
                       f"{p}\n\nQ: {q}\nA (yes/no):"),
    }


def extract_yesno(text: str) -> str:
    t = text.strip().lower()
    # exact single-word response
    if t in ("yes", "no"):
        return t
    tokens = re.findall(r"\b(yes|no)\b", t)
    return tokens[0] if tokens else "unknown"


def call_api(client: anthropic.Anthropic, prompt: str,
             max_retries: int = 6) -> str:
    """Call Claude API with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL_NAME,
                max_tokens=8,
                system=[{
                    "type": "text",
                    "text": SYS,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_yesno(resp.content[0].text)
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"  [rate limit] waiting {wait}s...", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    raise RuntimeError("Max retries exceeded")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(SAMPLE_FILE))
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY not set.")

    sample_path = Path(args.sample)
    stem        = sample_path.stem.replace("boolq_sample_", "")
    raw_file    = RESULTS_DIR / f"raw_claude_{stem}.csv"
    summary_file = RESULTS_DIR / f"summary_claude_{stem}.json"

    items = [json.loads(l) for l in open(sample_path, encoding="utf-8")]
    N     = len(items)
    n_gen = N * 4
    print(f"BoolQ sample : {N} items (gold yes={sum(i['answer'] for i in items)})")
    print(f"Model        : {MODEL_NAME} (zero-shot) | generations: {n_gen}")
    print(f"Raw output   : {raw_file}")

    # Resume: skip already-done rows
    done_keys: set[tuple] = set()
    existing_rows: list[dict] = []
    if raw_file.exists():
        with open(raw_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((int(row["id"]), row["variant"]))
                existing_rows.append(row)
        print(f"Resuming: {len(done_keys)} rows already done, "
              f"{n_gen - len(done_keys)} remaining")

    client = anthropic.Anthropic(api_key=api_key)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = list(existing_rows)
    done_count = len(done_keys)
    t0 = time.time()

    for it in items:
        variants = build_variants(it)
        for vname, prompt in variants.items():
            if (it["id"], vname) in done_keys:
                continue
            gold = "yes" if it["answer"] else "no"
            pred = call_api(client, prompt)
            row  = {
                "id":       it["id"],
                "gold":     gold,
                "variant":  vname,
                "pred":     pred,
                "correct":  pred == gold,
                "question": it["question"][:160],
            }
            all_rows.append(row)
            done_count += 1
            if done_count % 40 == 0:
                el   = time.time() - t0
                left = (n_gen - len(done_keys) - done_count + len(done_keys))
                print(f"  {done_count}/{n_gen} "
                      f"({el:.0f}s, ~{el/(done_count-len(done_keys))*left:.0f}s left)",
                      flush=True)
            # small inter-call pause to stay under RPM limit
            time.sleep(0.1)

    # Write raw CSV (full)
    fieldnames = ["id", "gold", "variant", "pred", "correct", "question"]
    with open(raw_file, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    by_item: dict[int, dict] = defaultdict(dict)
    for row in all_rows:
        by_item[int(row["id"])][row["variant"]] = row["pred"]

    gold_map = {it["id"]: ("yes" if it["answer"] else "no") for it in items}

    pert_correct: dict[str, int] = defaultdict(int)
    pert_total:   dict[str, int] = defaultdict(int)
    state_counts: dict[str, int] = defaultdict(int)
    headline_correct = 0
    headline_ind, stable_ind = [], []
    group_acc, group_cons    = [], []

    for item_id, preds in by_item.items():
        gold   = gold_map[item_id]
        anchor = preds.get("original", "unknown")
        n      = len(preds)
        acc    = sum(p == gold  for p in preds.values()) / n
        cons   = sum(p == anchor for p in preds.values()) / n
        group_acc.append(acc)
        group_cons.append(cons)

        h = int(anchor == gold)
        headline_correct += h
        headline_ind.append(h)

        for vname, pred in preds.items():
            pert_correct[vname] += int(pred == gold)
            pert_total[vname]   += 1

        agree = len(set(preds.values())) == 1
        state = ("stable_correct" if agree and acc == 1.0
                 else "stable_wrong" if agree
                 else "unstable")
        state_counts[state] += 1
        stable_ind.append(int(state == "stable_correct"))

    n_items         = len(by_item)
    headline_acc    = headline_correct / n_items
    stable_corr_pct = state_counts["stable_correct"] / n_items
    h_arr = np.asarray(headline_ind)
    s_arr = np.asarray(stable_ind)

    summary = {
        "n_items":                         n_items,
        "model":                           MODEL_NAME,
        "headline_accuracy_original_prompt": round(headline_acc, 4),
        "headline_accuracy_ci":            bootstrap_ci(h_arr, np.mean, seed=1),
        "stable_correct_pct":              round(stable_corr_pct, 4),
        "stable_correct_ci":               bootstrap_ci(s_arr, np.mean, seed=2),
        "headline_overstatement_gap":      round(headline_acc - stable_corr_pct, 4),
        "headline_overstatement_gap_ci":   bootstrap_ci(h_arr - s_arr, np.mean, seed=3),
        "SIS_acc":                         round(sum(group_acc)  / n_items, 4),
        "SIS_consistency":                 round(sum(group_cons) / n_items, 4),
        "group_states": {k: state_counts.get(k, 0)
                         for k in ("stable_correct", "stable_wrong", "unstable")},
        "per_variant_accuracy": {k: round(pert_correct[k] / pert_total[k], 4)
                                 for k in sorted(pert_total)},
        "scope": (
            f"measurement-only (no causal DAG); conservative answer-preserving "
            f"perturbations; zero-shot {MODEL_NAME}; n={n_items}; single run; "
            f"verdicts are bootstrap 95% CIs."
        ),
    }

    json.dump(summary, open(summary_file, "w"), indent=2)

    print(f"\n========  BoolQ (Claude, n={n_items}, 95% CI)  ========")
    print(f"Headline accuracy          : {summary['headline_accuracy_ci']}")
    print(f"Stable-correct fraction    : {summary['stable_correct_ci']}")
    print(f"Headline overstatement gap : {summary['headline_overstatement_gap_ci']}")
    print(f"SIS_acc={summary['SIS_acc']:.3f}  SIS_consistency={summary['SIS_consistency']:.3f}")
    print(f"Group states               : {summary['group_states']}")
    print(f"Per-variant accuracy       : {summary['per_variant_accuracy']}")
    print(f"\nSaved: {raw_file}\nSaved: {summary_file}")


if __name__ == "__main__":
    main()

"""
run_deepseek_zeroshot.py

Zero-shot DeepSeek evaluation on the stable_inference_v3 synthetic benchmark.
Replicates the dual-metric SIS decomposition from run_distilbert.py but with
a zero-shot frontier model instead of a fine-tuned classifier.

This directly addresses the residual limitation in §5: the fine-tuned models
(TF-IDF, DistilBERT, Qwen) are trained on the same distribution they are
tested on. DeepSeek V4 receives only the text — no training signal — so its
verdict comes purely from in-context reasoning.

Design:
  - 100 groups, 5 reasoning types (causal, commonsense, contradiction,
    taxonomic, transitive)
  - 5 answer-preserving variants per group (original, paraphrase,
    lexical_shift, reasoning_path_shift, distractor)
  - 500 total API calls
  - Same three-state decomposition: stable_correct / stable_wrong / unstable
  - Results broken down by reasoning type

Cost estimate: ~500 calls × ~150 tok input ≈ ¥0.10 total

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python run_deepseek_zeroshot.py
"""

import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "deepseek-v4-flash"
BASE_URL    = "https://api.deepseek.com"
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data" / "stable_inference_v3"
RESULTS_DIR = BASE_DIR / "results"

SYS = (
    "You answer logical inference questions. "
    "Read the statement carefully and answer with exactly one word: yes or no."
)

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


def extract_yesno(text: str) -> str:
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    # exact yes/no word
    tokens = re.findall(r"\b(yes|no)\b", t)
    if tokens:
        return tokens[0]
    # affirmative / negative phrases
    if re.search(r"\bit follows\b", t) and not re.search(r"\bdoes not follow|doesn't follow|it does not\b", t):
        return "yes"
    if re.search(r"\bdoes not follow|doesn't follow|does not necessarily follow\b", t):
        return "no"
    if re.search(r"\bwe can conclude\b", t):
        return "yes"
    if re.search(r"\bwe cannot conclude|cannot be concluded\b", t):
        return "no"
    if re.search(r"\bcorrect\b", t) and not re.search(r"\bincorrect\b", t):
        return "yes"
    if re.search(r"\bincorrect\b", t):
        return "no"
    return "unknown"


def call_api(client: OpenAI, prompt: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user",   "content": prompt},
                ],
            )
            msg = resp.choices[0].message
            # content has the final answer; reasoning_content is the thinking trace
            text = msg.content or ""
            if not text or extract_yesno(text) == "unknown":
                # fallback: extract from end of reasoning trace
                rc = getattr(msg, "reasoning_content", "") or ""
                if rc:
                    text = rc
            return extract_yesno(text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [error: {e}] waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError("Max retries exceeded")


def load_groups(data_dir: Path) -> list[dict]:
    groups = []
    for f in sorted(data_dir.rglob("*.json")):
        groups.append(json.loads(f.read_text(encoding="utf-8")))
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        import sys; sys.exit("Error: DEEPSEEK_API_KEY not set.")

    raw_file     = RESULTS_DIR / "deepseek_zeroshot_raw.csv"
    summary_file = RESULTS_DIR / "deepseek_zeroshot_summary.json"

    groups = load_groups(DATA_DIR)
    n_groups  = len(groups)
    n_gen     = sum(len(g["samples"]) for g in groups)
    print(f"Groups       : {n_groups} | total generations: {n_gen}")
    print(f"Model        : {MODEL_NAME} (zero-shot)")
    print(f"Raw output   : {raw_file}")

    # Resume support
    done_keys: set = set()
    existing_rows: list = []
    if raw_file.exists():
        with open(raw_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["group_id"], row["variant"]))
                existing_rows.append(row)
        print(f"Resuming: {len(done_keys)} done, {n_gen - len(done_keys)} remaining")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows   = list(existing_rows)
    done_count = len(done_keys)
    t0 = time.time()

    for g in groups:
        gid   = g["group_id"]
        rtype = g["reasoning_type"]
        for s in g["samples"]:
            key = (gid, s["type"])
            if key in done_keys:
                continue
            pred = call_api(client, s["text"])
            row  = {
                "group_id":      gid,
                "reasoning_type": rtype,
                "variant":       s["type"],
                "gold":          s["answer"],
                "pred":          pred,
                "correct":       pred == s["answer"],
            }
            all_rows.append(row)
            done_count += 1
            if done_count % 50 == 0:
                el   = time.time() - t0
                left = n_gen - done_count
                print(f"  {done_count}/{n_gen} ({el:.0f}s, ~{el/done_count*left:.0f}s left)",
                      flush=True)
            time.sleep(0.05)

    # Write raw CSV
    fieldnames = ["group_id", "reasoning_type", "variant", "gold", "pred", "correct"]
    with open(raw_file, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # ---------------------------------------------------------------------------
    # Aggregate — overall + per reasoning type
    # ---------------------------------------------------------------------------
    by_group: dict = defaultdict(dict)
    rtype_map: dict = {}
    for row in all_rows:
        by_group[row["group_id"]][row["variant"]] = row
        rtype_map[row["group_id"]] = row["reasoning_type"]

    def compute_stats(group_ids):
        state_counts  = defaultdict(int)
        group_acc, group_cons = [], []
        headline_ind, stable_ind = [], []
        unknown_count = 0

        for gid in group_ids:
            preds_map = by_group[gid]
            anchor_row = preds_map.get("original")
            if anchor_row is None:
                continue
            anchor_pred = anchor_row["pred"]
            gold        = anchor_row["gold"]

            preds = {v: r["pred"] for v, r in preds_map.items()}
            n     = len(preds)
            unknown_count += sum(1 for p in preds.values() if p == "unknown")

            acc  = sum(p == gold  for p in preds.values()) / n
            cons = sum(p == anchor_pred for p in preds.values()) / n
            group_acc.append(acc)
            group_cons.append(cons)

            h = int(anchor_pred == gold)
            headline_ind.append(h)

            agree = len(set(preds.values())) == 1
            state = ("stable_correct" if agree and acc == 1.0
                     else "stable_wrong" if agree
                     else "unstable")
            state_counts[state] += 1
            stable_ind.append(int(state == "stable_correct"))

        N = len(group_ids)
        if N == 0:
            return {}
        h_arr = np.asarray(headline_ind)
        s_arr = np.asarray(stable_ind)
        return {
            "n_groups":                        N,
            "headline_accuracy":               bootstrap_ci(h_arr, np.mean, seed=1),
            "stable_correct_pct":              bootstrap_ci(s_arr, np.mean, seed=2),
            "headline_overstatement_gap":      bootstrap_ci(h_arr - s_arr, np.mean, seed=3),
            "SIS_acc":                         round(float(np.mean(group_acc)), 4),
            "SIS_consistency":                 round(float(np.mean(group_cons)), 4),
            "group_states":                    {k: state_counts.get(k, 0)
                                                for k in ("stable_correct","stable_wrong","unstable")},
            "unknown_parses":                  unknown_count,
        }

    all_gids = list(by_group.keys())
    overall  = compute_stats(all_gids)

    by_type = {}
    for rtype in ["causal", "commonsense", "contradiction", "taxonomic", "transitive"]:
        gids = [g for g in all_gids if rtype_map.get(g) == rtype]
        by_type[rtype] = compute_stats(gids)

    summary = {
        "model":          MODEL_NAME,
        "evaluation":     "zero-shot, no fine-tuning",
        "benchmark":      "stable_inference_v3",
        "n_groups":       n_groups,
        "n_variants":     5,
        "overall":        overall,
        "by_reasoning_type": by_type,
        "scope": (
            f"Zero-shot {MODEL_NAME} on synthetic stable_inference_v3 benchmark; "
            f"no training signal; verdicts are bootstrap 95% CIs."
        ),
    }

    json.dump(summary, open(summary_file, "w"), indent=2)

    # Print results
    print(f"\n========  Synthetic Benchmark — DeepSeek Zero-shot  ========")
    print(f"Overall ({n_groups} groups):")
    print(f"  Headline accuracy     : {overall['headline_accuracy']}")
    print(f"  Stable-correct        : {overall['stable_correct_pct']}")
    print(f"  Overstatement gap     : {overall['headline_overstatement_gap']}")
    print(f"  SIS_acc={overall['SIS_acc']:.3f}  SIS_consistency={overall['SIS_consistency']:.3f}")
    print(f"  Group states          : {overall['group_states']}")
    print(f"\nBy reasoning type:")
    for rtype, stats in by_type.items():
        if not stats:
            continue
        print(f"  {rtype:15s}: headline={stats['headline_accuracy'][0]:.3f}  "
              f"stable={stats['stable_correct_pct'][0]:.3f}  "
              f"gap={stats['headline_overstatement_gap'][0]:.3f}  "
              f"states={stats['group_states']}")

    print(f"\nSaved: {raw_file}\nSaved: {summary_file}")


if __name__ == "__main__":
    main()

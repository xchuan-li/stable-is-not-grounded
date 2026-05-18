"""
evaluate_dual_metric.py

Dual-metric Stable Inference evaluation on stable_inference_v3.

Motivation:
SIS_v1 only measured per-group accuracy. Accuracy alone cannot tell apart
two very different failure modes:

    stable-wrong : the model gives the SAME (wrong) answer to every variant
    unstable     : the model gives DIFFERENT answers across variants

Both score 0 on accuracy for a "no" group, but they mean different things for
*reasoning stability*. This script reports both axes:

    SIS_acc          mean per-group accuracy vs gold label   (is it correct?)
    SIS_consistency  mean per-group agreement with the
                     prediction on the 'original' variant     (is it invariant?)

It also categorizes every group into one of three states:

    stable_correct : all variants predicted, all correct
    stable_wrong   : all variants agree with each other but are wrong
    unstable       : variants disagree with each other

The model used is the DistilBERT classifier already fine-tuned by
evaluate_bert_baseline.py and saved under models/distilbert_stable_inference.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stable_inference_v3"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "dual_metric_results_v3.csv"
SUMMARY_FILE = RESULTS_DIR / "dual_metric_summary_v3.json"
MODEL_DIR = BASE_DIR / "models" / "distilbert_stable_inference"

MAX_LENGTH = 128
ID_TO_LABEL = {0: "no", 1: "yes"}


def load_groups(data_dir):
    files = sorted(data_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No group files in {data_dir}")
    groups = []
    for fp in files:
        g = json.load(open(fp, encoding="utf-8"))
        g["_source_file"] = str(fp.relative_to(data_dir))
        groups.append(g)
    return groups


def predict(text, tokenizer, model, device):
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return ID_TO_LABEL[int(torch.argmax(logits, dim=-1).item())]


def evaluate(groups, tokenizer, model, device):
    rows = []
    group_acc = {}
    group_consistency = {}
    group_state = {}
    pert_correct = defaultdict(int)
    pert_total = defaultdict(int)
    reason_acc = defaultdict(list)
    reason_cons = defaultdict(list)
    state_counts = defaultdict(int)

    for g in groups:
        gid = g["group_id"]
        rtype = g.get("reasoning_type", "unknown")
        gold = g["expected_answer"].strip().lower()

        preds = {}
        for s in g["samples"]:
            p = predict(s["text"], tokenizer, model, device)
            preds[s["type"]] = p
            correct = (p == gold)
            pert_correct[s["type"]] += int(correct)
            pert_total[s["type"]] += 1
            rows.append({
                "source_file": g["_source_file"],
                "group_id": gid,
                "reasoning_type": rtype,
                "perturbation_type": s["type"],
                "expected_answer": gold,
                "prediction": p,
                "is_correct": correct,
            })

        n = len(preds)
        acc = sum(1 for p in preds.values() if p == gold) / n

        # consistency: agreement with the 'original' variant's prediction
        anchor = preds.get("original", next(iter(preds.values())))
        consistency = sum(1 for p in preds.values() if p == anchor) / n

        all_agree = len(set(preds.values())) == 1
        if all_agree and acc == 1.0:
            state = "stable_correct"
        elif all_agree and acc == 0.0:
            state = "stable_wrong"
        elif all_agree:
            # all variants agree but gold is mixed-impossible here (single gold);
            # all-agree implies acc is 0 or 1, so this branch is unreachable,
            # kept defensively.
            state = "stable_wrong"
        else:
            state = "unstable"

        group_acc[gid] = acc
        group_consistency[gid] = consistency
        group_state[gid] = state
        state_counts[state] += 1
        reason_acc[rtype].append(acc)
        reason_cons[rtype].append(consistency)

    sis_acc = sum(group_acc.values()) / len(group_acc)
    sis_cons = sum(group_consistency.values()) / len(group_consistency)

    summary = {
        "n_groups": len(groups),
        "SIS_acc": round(sis_acc, 4),
        "SIS_consistency": round(sis_cons, 4),
        "group_state_distribution": {
            k: state_counts[k] for k in ("stable_correct", "stable_wrong", "unstable")
        },
        "per_reasoning_type": {
            rt: {
                "acc": round(sum(reason_acc[rt]) / len(reason_acc[rt]), 4),
                "consistency": round(sum(reason_cons[rt]) / len(reason_cons[rt]), 4),
            }
            for rt in sorted(reason_acc)
        },
        "per_perturbation_type_acc": {
            pt: round(pert_correct[pt] / pert_total[pt], 4)
            for pt in sorted(pert_total)
        },
    }
    return rows, summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {MODEL_DIR}. "
            f"Run evaluate_bert_baseline.py first to train it."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    groups = load_groups(DATA_DIR)
    rows, summary = evaluate(groups, tokenizer, model, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(SUMMARY_FILE, "w", encoding="utf-8"), indent=2)

    print("=== Dual-Metric Stable Inference Evaluation (v3) ===\n")
    print(f"Groups evaluated: {summary['n_groups']}\n")
    print(f"SIS_acc          = {summary['SIS_acc']:.3f}   (correctness)")
    print(f"SIS_consistency  = {summary['SIS_consistency']:.3f}   (invariance to perturbation)\n")
    print("Group state distribution:")
    for k, v in summary["group_state_distribution"].items():
        print(f"  {k:<15} {v:>3}  ({v / summary['n_groups']:.0%})")
    print("\nPer reasoning type (acc / consistency):")
    for rt, d in summary["per_reasoning_type"].items():
        print(f"  {rt:<15} {d['acc']:.3f} / {d['consistency']:.3f}")
    print("\nPer perturbation type accuracy:")
    for pt, v in summary["per_perturbation_type_acc"].items():
        print(f"  {pt:<22} {v:.3f}")
    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Saved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

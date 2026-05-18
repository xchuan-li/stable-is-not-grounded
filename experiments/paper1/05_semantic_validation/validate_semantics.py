

"""
validate_semantics.py

Semantic Preservation Validation Prototype for exp_332_semantic_preservation_control.

This script reads semantic validation groups from:
    data/semantic_preservation_v1/

It summarizes whether perturbations preserve the original inference structure.

Unlike exp_312, this script does not evaluate model predictions.
Instead, it evaluates the perturbation design itself.

The goal is to make perturbation validity explicit and auditable.
"""

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "semantic_preservation_v1"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "semantic_validation_results.csv"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_groups(data_dir: Path):
    """
    Load all semantic validation group JSON files.
    """
    group_files = sorted(data_dir.glob("validation_group_*.json"))

    if not group_files:
        raise FileNotFoundError(f"No validation_group_*.json files found in {data_dir}")

    groups = []
    for file_path in group_files:
        with open(file_path, "r", encoding="utf-8") as f:
            groups.append(json.load(f))

    return groups


# -----------------------------------------------------------------------------
# Validation analysis
# -----------------------------------------------------------------------------

def analyze_groups(groups):
    """
    Collect per-sample semantic validity information and aggregate statistics.
    """
    rows = []

    validity_counter = Counter()
    risk_counter = Counter()
    change_type_counter = Counter()
    reasoning_counter = defaultdict(Counter)

    for group in groups:
        group_id = group["group_id"]
        core_inference = group["core_inference"]
        expected_reasoning = group.get("expected_reasoning", "unknown")
        original_answer = group.get("original_answer", "unknown")

        for sample in group["samples"]:
            sample_id = sample["sample_id"]
            perturbation_type = sample["type"]
            validity_label = sample["validity_label"]
            validity_risk = sample["validity_risk"]
            semantic_change_type = sample["semantic_change_type"]
            preserves_core_inference = sample["preserves_core_inference"]

            validity_counter[validity_label] += 1
            risk_counter[validity_risk] += 1
            change_type_counter[semantic_change_type] += 1
            reasoning_counter[expected_reasoning][validity_label] += 1

            rows.append({
                "group_id": group_id,
                "core_inference": core_inference,
                "expected_reasoning": expected_reasoning,
                "original_answer": original_answer,
                "sample_id": sample_id,
                "perturbation_type": perturbation_type,
                "preserves_core_inference": preserves_core_inference,
                "semantic_change_type": semantic_change_type,
                "validity_risk": validity_risk,
                "validity_label": validity_label,
                "notes": sample.get("notes", ""),
            })

    summary = {
        "validity_counter": validity_counter,
        "risk_counter": risk_counter,
        "change_type_counter": change_type_counter,
        "reasoning_counter": reasoning_counter,
        "total_samples": len(rows),
    }

    return rows, summary


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_results(rows, results_file: Path):
    """
    Save semantic validation rows to CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group_id",
        "core_inference",
        "expected_reasoning",
        "original_answer",
        "sample_id",
        "perturbation_type",
        "preserves_core_inference",
        "semantic_change_type",
        "validity_risk",
        "validity_label",
        "notes",
    ]

    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_counter(title: str, counter: Counter, total: int):
    """
    Pretty-print a counter with ratios.
    """
    print(f"\n{title}")
    for key, count in counter.items():
        ratio = count / total if total else 0.0
        print(f"  {key}: {count} ({ratio:.3f})")


def print_summary(summary):
    """
    Print semantic validation summary to terminal.
    """
    total = summary["total_samples"]

    print("\n=== Semantic Preservation Validation Summary ===")
    print(f"\nTotal samples: {total}")

    print_counter("Validity labels:", summary["validity_counter"], total)
    print_counter("Validity risks:", summary["risk_counter"], total)
    print_counter("Semantic change types:", summary["change_type_counter"], total)

    print("\nValidity by reasoning type:")
    for reasoning_type, counter in summary["reasoning_counter"].items():
        subtotal = sum(counter.values())
        print(f"\n  {reasoning_type}:")
        for label, count in counter.items():
            ratio = count / subtotal if subtotal else 0.0
            print(f"    {label}: {count} ({ratio:.3f})")

    print(f"\nSaved detailed results to: {RESULTS_FILE}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    groups = load_groups(DATA_DIR)
    rows, summary = analyze_groups(groups)
    save_results(rows, RESULTS_FILE)
    print_summary(summary)


if __name__ == "__main__":
    main()
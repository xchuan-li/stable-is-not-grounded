"""
evaluate_stability.py

Stable Inference Evaluation Prototype for exp_312_paraphrase_invariance.

This script reads perturbation groups from data/stable_inference_v1/ and evaluates
whether predictions remain consistent across semantic-preserving perturbations.

Current version:
- Uses the trained scikit-learn model saved as model.joblib.
- Computes per-group consistency.
- Computes per-perturbation-type accuracy.
- Computes an overall Stable Inference Score prototype.
- Saves results to results/stability_results.csv.

The goal of this script is to evaluate whether a trained model preserves stable conclusions across semantic-preserving perturbations.
"""

import csv
import json
import joblib
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "stable_inference_v1"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "stability_results.csv"
MODEL_PATH = BASE_DIR.parent.parent.parent.parent / "model.joblib"
model = joblib.load(MODEL_PATH)


# -----------------------------------------------------------------------------
# Legacy mock model
# -----------------------------------------------------------------------------

# This function is kept only as a reference baseline.
# The current evaluator uses model.predict() instead.

def mock_model(text: str) -> str:
    """
    A simple placeholder model.

    This is intentionally naive. Its purpose is only to verify that the
    evaluation pipeline works before connecting a real model.

    Later versions can replace this function with:
    - logistic regression,
    - transformer classifier,
    - local LLM,
    - or API-based model evaluation.
    """
    lowered = text.lower()

    # Negation examples
    if "no reptiles" in lowered or "do not possess fur" in lowered or "not furry" in lowered:
        return "no"
    if "do not have fur" in lowered or "no reptilian" in lowered:
        return "no"

    # Most current groups expect "yes".
    return "yes"


def normalize_prediction(raw_prediction) -> str:
    """
    Convert model outputs into the yes/no label format used by the dataset.

    The current trained scikit-learn model may return numeric labels such as:
    - 1 for yes
    - 0 for no

    This function makes predictions comparable with the JSON answer field.
    """
    label_map = {
        1: "yes",
        0: "no",
        "1": "yes",
        "0": "no",
        True: "yes",
        False: "no",
    }

    if raw_prediction in label_map:
        return label_map[raw_prediction]

    return str(raw_prediction).lower().strip()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_groups(data_dir: Path):
    """
    Load all perturbation group JSON files from the dataset directory.
    """
    group_files = sorted(data_dir.glob("group_*.json"))

    if not group_files:
        raise FileNotFoundError(f"No group_*.json files found in {data_dir}")

    groups = []
    for file_path in group_files:
        with open(file_path, "r", encoding="utf-8") as f:
            groups.append(json.load(f))

    return groups


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_group(group: dict):
    """
    Evaluate one perturbation group.

    Consistency is computed as:
        correct predictions / total samples in the group

    In this prototype, consistency is measured against the expected answer.
    Later versions can additionally measure agreement with the original sample's
    prediction, which is useful for model-internal consistency analysis.
    """
    group_id = group["group_id"]
    core_inference = group["core_inference"]
    expected_reasoning = group.get("expected_reasoning", "unknown")

    rows = []
    correct_count = 0
    total_count = 0

    for sample in group["samples"]:
        sample_id = sample["sample_id"]
        perturbation_type = sample["type"]
        text = sample["text"]
        expected_answer = sample["answer"].lower().strip()

        raw_prediction = model.predict([text])[0]
        prediction = normalize_prediction(raw_prediction)
        is_correct = prediction == expected_answer

        correct_count += int(is_correct)
        total_count += 1

        rows.append({
            "group_id": group_id,
            "core_inference": core_inference,
            "expected_reasoning": expected_reasoning,
            "sample_id": sample_id,
            "perturbation_type": perturbation_type,
            "expected_answer": expected_answer,
            "prediction": prediction,
            "raw_prediction": str(raw_prediction),
            "is_correct": is_correct,
        })

    group_consistency = correct_count / total_count if total_count else 0.0

    return rows, group_consistency


def evaluate_all(groups):
    """
    Evaluate all perturbation groups and compute summary statistics.
    """
    all_rows = []
    group_scores = {}
    perturbation_correct = defaultdict(int)
    perturbation_total = defaultdict(int)

    for group in groups:
        rows, group_consistency = evaluate_group(group)
        group_scores[group["group_id"]] = group_consistency
        all_rows.extend(rows)

        for row in rows:
            p_type = row["perturbation_type"]
            perturbation_correct[p_type] += int(row["is_correct"])
            perturbation_total[p_type] += 1

    overall_sis = sum(group_scores.values()) / len(group_scores) if group_scores else 0.0

    perturbation_scores = {
        p_type: perturbation_correct[p_type] / perturbation_total[p_type]
        for p_type in sorted(perturbation_total)
    }

    return all_rows, group_scores, perturbation_scores, overall_sis


# -----------------------------------------------------------------------------
# Result saving and reporting
# -----------------------------------------------------------------------------

def save_results(rows, results_file: Path):
    """
    Save per-sample evaluation results as CSV.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group_id",
        "core_inference",
        "expected_reasoning",
        "sample_id",
        "perturbation_type",
        "expected_answer",
        "prediction",
        "raw_prediction",
        "is_correct",
    ]

    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(group_scores, perturbation_scores, overall_sis):
    """
    Print evaluation summary to terminal.
    """
    print("\n=== Stable Inference Evaluation Summary ===\n")

    print("Per-group consistency:")
    for group_id, score in sorted(group_scores.items()):
        print(f"  group_{group_id}: {score:.3f}")

    print("\nPer-perturbation-type accuracy:")
    for p_type, score in perturbation_scores.items():
        print(f"  {p_type}: {score:.3f}")

    print("\nOverall Stable Inference Score prototype:")
    print(f"  SIS_v1 = {overall_sis:.3f}")

    print(f"\nSaved detailed results to: {RESULTS_FILE}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    groups = load_groups(DATA_DIR)
    rows, group_scores, perturbation_scores, overall_sis = evaluate_all(groups)
    save_results(rows, RESULTS_FILE)
    print_summary(group_scores, perturbation_scores, overall_sis)


if __name__ == "__main__":
    main()
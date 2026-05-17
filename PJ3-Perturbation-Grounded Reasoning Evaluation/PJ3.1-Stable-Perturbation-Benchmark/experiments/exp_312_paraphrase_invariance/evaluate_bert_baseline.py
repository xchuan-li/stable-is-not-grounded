"""
evaluate_bert_baseline.py

DistilBERT baseline for exp_312_paraphrase_invariance.

Route A:
- Fine-tune a DistilBERT yes/no classifier.
- Evaluate the fine-tuned model on stable inference perturbation groups.
- Compare its Stable Inference Score with the TF-IDF + Logistic Regression baseline.

This script is intentionally self-contained:
1. It loads the fixed balanced training split from data/train/balanced_train_v3.csv.
2. It fine-tunes DistilBERT on that balanced training split.
3. It evaluates group-level stability on data/stable_inference_v2/ recursively.
4. It saves detailed results to results/bert_stability_results_v2.csv.

Research purpose:
This baseline tests whether semantic perturbation instability is only a classical ML artifact,
or whether a transformer-based model also exhibits instability under controlled perturbations.
"""

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATASET_VERSION = "stable_inference_v2"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / DATASET_VERSION
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "bert_stability_results_v2.csv"
MODEL_DIR = BASE_DIR / "models" / "distilbert_stable_inference"
TRAIN_FILE = BASE_DIR / "data" / "train" / "balanced_train_v3.csv"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

LABEL_TO_ID = {
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
}

ID_TO_LABEL = {
    0: "no",
    1: "yes",
}


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def normalize_label(label) -> int:
    """
    Convert yes/no or 0/1 labels into integer class ids.
    """
    if isinstance(label, str):
        key = label.strip().lower()
    else:
        key = label

    if key not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label: {label}")

    return LABEL_TO_ID[key]


class TextClassificationDataset(Dataset):
    """
    Simple PyTorch dataset for yes/no text classification.
    """
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(example["label"], dtype=torch.long)
        return item


def load_training_examples_from_csv(csv_path: Path):
    """
    Load training examples from a CSV file.

    Required columns:
    - text
    - label

    Extra columns such as reasoning_type are allowed and ignored by the trainer.
    """
    examples = []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        normalized_fieldnames = {
            field.strip().lower(): field
            for field in reader.fieldnames
            if field is not None
        }

        if "text" not in normalized_fieldnames or "label" not in normalized_fieldnames:
            raise ValueError(
                f"Expected columns text,label in {csv_path}. "
                f"Found columns: {reader.fieldnames}"
            )

        text_column = normalized_fieldnames["text"]
        label_column = normalized_fieldnames["label"]

        for row in reader:
            text = row[text_column].strip()
            label = row[label_column].strip()

            if not text or not label:
                continue

            examples.append({
                "text": text,
                "label": normalize_label(label),
            })

    return examples


def load_training_examples():
    """
    Load the fixed balanced training split for v2.

    This avoids accidentally training on an old train.csv from another experiment
    and makes the benchmark protocol reproducible.
    """
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    print(f"Using fixed training file: {TRAIN_FILE}")
    return load_training_examples_from_csv(TRAIN_FILE)


# -----------------------------------------------------------------------------
# Model training and loading
# -----------------------------------------------------------------------------

def train_model(tokenizer, device):
    """
    Fine-tune DistilBERT as a binary yes/no classifier.
    """
    examples = load_training_examples()

    if not examples:
        raise ValueError("No training examples found.")

    label_counts = defaultdict(int)
    for example in examples:
        label_counts[example["label"]] += 1

    print("Training label distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {ID_TO_LABEL[label_id]}: {count}")

    dataset = TextClassificationDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Saved fine-tuned model to: {MODEL_DIR}")
    return model


def load_or_train_model(device):
    """
    Load a fine-tuned model if it exists; otherwise train a new one.
    """
    # Important: if the training data changes, delete MODEL_DIR manually before rerunning.
    # Otherwise this function will reuse the existing fine-tuned model.
    # For example, after changing balanced_train_v3.csv, run:
    # rm -rf models/distilbert_stable_inference
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        print(f"Loading existing fine-tuned model from: {MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        return tokenizer, model

    print("No fine-tuned DistilBERT model found. Training a new one.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = train_model(tokenizer, device)
    return tokenizer, model


# -----------------------------------------------------------------------------
# Evaluation data loading
# -----------------------------------------------------------------------------

def load_groups(data_dir: Path):
    """
    Load all stable inference perturbation groups.

    v1 used a flat group_*.json layout.
    v2 uses category subfolders such as:
        stable_inference_v2/taxonomic/tax_001.json
        stable_inference_v2/transitive/trans_001.json

    Therefore this loader reads JSON files recursively.
    """
    group_files = sorted(data_dir.rglob("*.json"))

    if not group_files:
        raise FileNotFoundError(f"No JSON group files found in {data_dir}")

    groups = []
    for file_path in group_files:
        with open(file_path, "r", encoding="utf-8") as f:
            group = json.load(f)

        group["_source_file"] = str(file_path.relative_to(data_dir))
        groups.append(group)

    print(f"Loaded {len(groups)} groups from {data_dir}")
    return groups


# -----------------------------------------------------------------------------
# Prediction and evaluation
# -----------------------------------------------------------------------------

def predict_label(text: str, tokenizer, model, device) -> str:
    """
    Predict yes/no for one text sample.
    """
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {key: value.to(device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

    return ID_TO_LABEL[pred_id]


def evaluate_group(group: dict, tokenizer, model, device):
    """
    Evaluate one perturbation group.
    """
    group_id = group["group_id"]
    core_inference = group["core_inference"]
    reasoning_type = group.get("reasoning_type", group.get("expected_reasoning", "unknown"))
    source_file = group.get("_source_file", "unknown")

    rows = []
    correct_count = 0
    total_count = 0

    for sample in group["samples"]:
        sample_id = sample["sample_id"]
        perturbation_type = sample["type"]
        text = sample["text"]
        expected_answer = sample["answer"].lower().strip()

        prediction = predict_label(text, tokenizer, model, device)
        is_correct = prediction == expected_answer

        correct_count += int(is_correct)
        total_count += 1

        rows.append({
            "model": "distilbert-base-uncased-finetuned",
            "source_file": source_file,
            "group_id": group_id,
            "core_inference": core_inference,
            "reasoning_type": reasoning_type,
            "sample_id": sample_id,
            "perturbation_type": perturbation_type,
            "expected_answer": expected_answer,
            "prediction": prediction,
            "is_correct": is_correct,
        })

    group_score = correct_count / total_count if total_count else 0.0
    return rows, group_score


def evaluate_all(groups, tokenizer, model, device):
    """
    Evaluate all perturbation groups.
    """
    all_rows = []
    group_scores = {}
    perturbation_correct = defaultdict(int)
    perturbation_total = defaultdict(int)
    answer_correct = defaultdict(int)
    answer_total = defaultdict(int)
    reasoning_correct = defaultdict(int)
    reasoning_total = defaultdict(int)

    for group in groups:
        rows, group_score = evaluate_group(group, tokenizer, model, device)
        group_scores[group["group_id"]] = group_score
        all_rows.extend(rows)

        for row in rows:
            p_type = row["perturbation_type"]
            expected_answer = row["expected_answer"]
            reasoning_type = row["reasoning_type"]

            perturbation_correct[p_type] += int(row["is_correct"])
            perturbation_total[p_type] += 1

            answer_correct[expected_answer] += int(row["is_correct"])
            answer_total[expected_answer] += 1

            reasoning_correct[reasoning_type] += int(row["is_correct"])
            reasoning_total[reasoning_type] += 1

    overall_sis = sum(group_scores.values()) / len(group_scores) if group_scores else 0.0

    perturbation_scores = {
        p_type: perturbation_correct[p_type] / perturbation_total[p_type]
        for p_type in sorted(perturbation_total)
    }

    answer_scores = {
        answer: answer_correct[answer] / answer_total[answer]
        for answer in sorted(answer_total)
    }

    reasoning_scores = {
        reasoning_type: reasoning_correct[reasoning_type] / reasoning_total[reasoning_type]
        for reasoning_type in sorted(reasoning_total)
    }

    return all_rows, group_scores, perturbation_scores, answer_scores, reasoning_scores, overall_sis


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def save_results(rows, results_file: Path):
    """
    Save detailed BERT stability results.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "source_file",
        "group_id",
        "core_inference",
        "reasoning_type",
        "sample_id",
        "perturbation_type",
        "expected_answer",
        "prediction",
        "is_correct",
    ]

    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(group_scores, perturbation_scores, answer_scores, reasoning_scores, overall_sis):
    """
    Print evaluation summary.
    """
    print("\n=== DistilBERT Stable Inference Evaluation Summary ===\n")
    print(f"Dataset version: {DATASET_VERSION}")

    print("\nPer-group score:")
    for group_id, score in sorted(group_scores.items()):
        print(f"  {group_id}: {score:.3f}")

    print("\nPer-reasoning-type score:")
    for reasoning_type, score in reasoning_scores.items():
        print(f"  {reasoning_type}: {score:.3f}")

    print("\nPer-perturbation-type score:")
    for p_type, score in perturbation_scores.items():
        print(f"  {p_type}: {score:.3f}")

    print("\nPer-answer score:")
    for answer, score in answer_scores.items():
        print(f"  {answer}: {score:.3f}")

    print("\nOverall Stable Inference Score prototype:")
    print(f"  BERT_SIS_v1 = {overall_sis:.3f}")

    print(f"\nSaved detailed results to: {RESULTS_FILE}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = load_or_train_model(device)
    groups = load_groups(DATA_DIR)

    rows, group_scores, perturbation_scores, answer_scores, reasoning_scores, overall_sis = evaluate_all(
        groups,
        tokenizer,
        model,
        device,
    )

    save_results(rows, RESULTS_FILE)
    print_summary(group_scores, perturbation_scores, answer_scores, reasoning_scores, overall_sis)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "model.joblib"

TRAIN_PATH = DATA_DIR / "train.csv"
IID_TEST_PATH = DATA_DIR / "iid_test.csv"
REVERSAL_TEST_PATH = DATA_DIR / "reversal_test.csv"

RESULTS_JSON_PATH = RESULTS_DIR / "baseline_results.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
ERROR_CASES_PATH = RESULTS_DIR / "error_cases.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {path.name}: {sorted(missing_columns)}"
        )

    return df


def evaluate_split(model, df: pd.DataFrame, split_name: str) -> tuple[float, pd.DataFrame]:
    x = df[TEXT_COLUMN]
    y_true = df[LABEL_COLUMN]
    y_pred = model.predict(x)

    accuracy = accuracy_score(y_true, y_pred)

    prediction_df = df.copy()
    prediction_df["split"] = split_name
    prediction_df["prediction"] = y_pred
    prediction_df["correct"] = prediction_df[LABEL_COLUMN] == prediction_df["prediction"]

    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    return accuracy, prediction_df


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py before evaluate.py."
        )

    model = joblib.load(MODEL_PATH)

    train_df = load_split(TRAIN_PATH)
    iid_test_df = load_split(IID_TEST_PATH)
    reversal_test_df = load_split(REVERSAL_TEST_PATH)

    train_accuracy, train_predictions = evaluate_split(model, train_df, "train")
    iid_accuracy, iid_predictions = evaluate_split(model, iid_test_df, "iid_test")
    reversal_accuracy, reversal_predictions = evaluate_split(
        model,
        reversal_test_df,
        "reversal_test",
    )

    shortcut_gap = iid_accuracy - reversal_accuracy
    stability_score = reversal_accuracy / iid_accuracy if iid_accuracy > 0 else 0.0

    results = {
        "experiment": "exp_311_shortcut_reversal",
        "model": "CountVectorizer + LogisticRegression",
        "dataset_version": "v3_pseudoword_heldout",
        "train_accuracy": train_accuracy,
        "iid_accuracy": iid_accuracy,
        "reversal_accuracy": reversal_accuracy,
        "shortcut_gap": shortcut_gap,
        "stability_score": stability_score,
        "interpretation": (
            "High IID accuracy with low reversal accuracy indicates shortcut dependence "
            "rather than stable inference."
        ),
    }

    print("\n=== Shortcut Reversal Diagnostics ===")
    print(f"train_accuracy: {train_accuracy:.4f}")
    print(f"iid_accuracy: {iid_accuracy:.4f}")
    print(f"reversal_accuracy: {reversal_accuracy:.4f}")
    print(f"shortcut_gap: {shortcut_gap:.4f}")
    print(f"stability_score: {stability_score:.4f}")

    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    all_predictions = pd.concat(
        [train_predictions, iid_predictions, reversal_predictions],
        ignore_index=True,
    )
    all_predictions.to_csv(PREDICTIONS_PATH, index=False)

    error_cases = all_predictions[~all_predictions["correct"]].copy()
    error_cases.to_csv(ERROR_CASES_PATH, index=False)

    print("\nSaved evaluation artifacts:")
    print(f"- {RESULTS_JSON_PATH}")
    print(f"- {PREDICTIONS_PATH}")
    print(f"- {ERROR_CASES_PATH}")


if __name__ == "__main__":
    main()
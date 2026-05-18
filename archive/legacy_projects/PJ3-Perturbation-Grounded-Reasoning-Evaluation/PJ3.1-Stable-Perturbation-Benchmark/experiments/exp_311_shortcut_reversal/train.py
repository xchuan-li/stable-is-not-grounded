from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model.joblib"

TRAIN_PATH = DATA_DIR / "train.csv"
IID_TEST_PATH = DATA_DIR / "iid_test.csv"
REVERSAL_TEST_PATH = DATA_DIR / "reversal_test.csv"


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


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, df: pd.DataFrame, split_name: str) -> float:
    x = df[TEXT_COLUMN]
    y_true = df[LABEL_COLUMN]
    y_pred = model.predict(x)

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    return accuracy


def main() -> None:
    train_df = load_split(TRAIN_PATH)
    iid_test_df = load_split(IID_TEST_PATH)
    reversal_test_df = load_split(REVERSAL_TEST_PATH)

    model = build_model()

    x_train = train_df[TEXT_COLUMN]
    y_train = train_df[LABEL_COLUMN]

    model.fit(x_train, y_train)

    train_accuracy = evaluate_model(model, train_df, "train")
    iid_accuracy = evaluate_model(model, iid_test_df, "iid_test")
    reversal_accuracy = evaluate_model(model, reversal_test_df, "reversal_test")

    shortcut_gap = iid_accuracy - reversal_accuracy
    stability_score = reversal_accuracy / iid_accuracy if iid_accuracy > 0 else 0.0

    print("\n=== Shortcut Reversal Diagnostics ===")
    print(f"train_accuracy: {train_accuracy:.4f}")
    print(f"iid_accuracy: {iid_accuracy:.4f}")
    print(f"reversal_accuracy: {reversal_accuracy:.4f}")
    print(f"shortcut_gap: {shortcut_gap:.4f}")
    print(f"stability_score: {stability_score:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
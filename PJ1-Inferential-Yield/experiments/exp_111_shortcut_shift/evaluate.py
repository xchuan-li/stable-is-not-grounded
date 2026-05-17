import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate(split_name: str, csv_path: str):
    model = joblib.load("model.joblib")
    data = pd.read_csv(csv_path)

    preds = model.predict(data["text"])
    labels = data["label"]

    acc = accuracy_score(labels, preds)

    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {acc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))
    print("Classification report:")
    print(classification_report(labels, preds))


def main():
    evaluate("IID Test", "iid_test.csv")
    evaluate("Shift Test", "shift_test.csv")


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from dataset import generate_train_data, generate_iid_test_data, generate_shift_test_data


def main():
    train = generate_train_data()
    iid = generate_iid_test_data()
    shift = generate_shift_test_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    model.fit(train["text"], train["label"])

    iid_acc = model.score(iid["text"], iid["label"])
    shift_acc = model.score(shift["text"], shift["label"])

    print(f"IID accuracy: {iid_acc:.3f}")
    print(f"Shift accuracy: {shift_acc:.3f}")

    joblib.dump(model, "model.joblib")

    train.to_csv("train.csv", index=False)
    iid.to_csv("iid_test.csv", index=False)
    shift.to_csv("shift_test.csv", index=False)

    feature_names = model.named_steps["tfidf"].get_feature_names_out()
    coefficients = model.named_steps["clf"].coef_[0]

    top_features = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print("\nTop Features:")
    for feat, coef in top_features[:10]:
        print(f"{feat}: {coef:.4f}")


if __name__ == "__main__":
    main()

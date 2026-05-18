"""
evaluate_gradient.py

Evaluates model performance across multiple shortcut reversal strengths.
Shows that IID accuracy collapses gradually as shortcut correlation is reversed.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dataset import generate_train_data, generate_iid_test_data, generate_shift_test_data


REVERSAL_STRENGTHS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def generate_shift_at_strength(strength: float, n: int = 300, seed: int = 44):
    """
    Generate a shift test set where the shortcut correlation is reversed
    at the given strength (fraction of examples where color is anti-correlated).
    strength=0.0 → same as IID (shortcut still intact)
    strength=1.0 → fully reversed shortcut
    """
    import random
    from dataset import NAMES, make_example

    random.seed(seed)
    rows = []

    for _ in range(n):
        name = random.choice(NAMES)
        label = random.choice([0, 1])
        animal = "bird" if label == 1 else "penguin"

        if random.random() < strength:
            color = "blue" if label == 1 else "red"
        else:
            color = "red" if label == 1 else "blue"

        rows.append(make_example(name, animal, color))

    return pd.DataFrame(rows)


def main():
    train = generate_train_data()
    iid = generate_iid_test_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train["text"], train["label"])

    iid_acc = model.score(iid["text"], iid["label"])
    print(f"IID accuracy (shortcut intact): {iid_acc:.3f}")
    print()
    print(f"{'Reversal Strength':>18} | {'Shift Accuracy':>15}")
    print("-" * 38)

    results = []
    for strength in REVERSAL_STRENGTHS:
        shift = generate_shift_at_strength(strength)
        acc = model.score(shift["text"], shift["label"])
        results.append({"reversal_strength": strength, "shift_accuracy": acc})
        print(f"{strength:>18.1f} | {acc:>15.3f}")

    df = pd.DataFrame(results)
    df.to_csv("shortcut_gradient_results.csv", index=False)
    print("\nSaved to shortcut_gradient_results.csv")


if __name__ == "__main__":
    main()

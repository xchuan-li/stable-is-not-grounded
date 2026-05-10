import random
import pandas as pd


NAMES = ["Tweety", "Robin", "Alex", "Milo", "Luna", "Charlie"]
ANIMALS = ["bird", "penguin"]
COLORS = ["red", "blue"]


def make_example(name: str, animal: str, color: str) -> dict:
    """
    Intended mechanism:
    - birds can fly
    - penguins cannot fly

    Shortcut in training:
    - red often means yes
    - blue often means no
    """

    if animal == "bird":
        label = 1
        answer = "yes"
    else:
        label = 0
        answer = "no"

    text = (
        f"All {animal}s have the usual properties of {animal}s. "
        f"{name} is a {color} {animal}. "
        f"Can {name} fly?"
    )

    return {
        "text": text,
        "name": name,
        "animal": animal,
        "color": color,
        "label": label,
        "answer": answer,
    }


def generate_train_data(n: int = 1000, shortcut_strength: float = 0.9, seed: int = 42):
    """
    Training environment:
    red is strongly correlated with label yes.
    blue is strongly correlated with label no.
    """
    random.seed(seed)
    rows = []

    for _ in range(n):
        name = random.choice(NAMES)

        label = random.choice([0, 1])
        animal = "bird" if label == 1 else "penguin"

        if random.random() < shortcut_strength:
            color = "red" if label == 1 else "blue"
        else:
            color = "blue" if label == 1 else "red"

        rows.append(make_example(name, animal, color))

    return pd.DataFrame(rows)


def generate_iid_test_data(n: int = 300, shortcut_strength: float = 0.9, seed: int = 43):
    return generate_train_data(n=n, shortcut_strength=shortcut_strength, seed=seed)


def generate_shift_test_data(n: int = 300, seed: int = 44):
    """
    Shift environment:
    shortcut is reversed.
    red often appears with no.
    blue often appears with yes.
    """
    random.seed(seed)
    rows = []

    for _ in range(n):
        name = random.choice(NAMES)

        label = random.choice([0, 1])
        animal = "bird" if label == 1 else "penguin"

        color = "blue" if label == 1 else "red"

        rows.append(make_example(name, animal, color))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    train = generate_train_data()
    iid = generate_iid_test_data()
    shift = generate_shift_test_data()

    train.to_csv("train.csv", index=False)
    iid.to_csv("iid_test.csv", index=False)
    shift.to_csv("shift_test.csv", index=False)

    print("Saved train.csv, iid_test.csv, shift_test.csv")

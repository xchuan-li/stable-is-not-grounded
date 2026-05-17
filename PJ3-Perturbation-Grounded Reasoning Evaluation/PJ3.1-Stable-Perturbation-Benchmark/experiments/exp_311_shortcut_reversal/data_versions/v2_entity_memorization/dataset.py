# v2_entity_memorization dataset generator
# This version removes direct label leakage from text.
# However, train/iid/reversal still share the same entity pool.
# Therefore, the model can solve reversal_test by memorizing entity-label pairs.
# It is useful as a dataset construction ablation.

import csv
import random
from pathlib import Path


OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CAN_FLY_ENTITIES = [
    "bird",
    "sparrow",
    "eagle",
    "pigeon",
    "swallow",
]

CANNOT_FLY_ENTITIES = [
    "cat",
    "dog",
    "fish",
    "penguin",
    "elephant",
]


TRAIN_SHORTCUT_RULE = {
    "red": "yes",
    "blue": "no",
}


def get_rule_label(entity: str) -> str:
    if entity in CAN_FLY_ENTITIES:
        return "yes"
    if entity in CANNOT_FLY_ENTITIES:
        return "no"
    raise ValueError(f"Unknown entity: {entity}")


def get_semantic_feature(label: str) -> str:
    if label == "yes":
        return "can_fly"
    if label == "no":
        return "cannot_fly"
    raise ValueError(f"Unknown label: {label}")


def choose_shortcut_feature(label: str, shortcut_aligned: bool) -> str:
    if shortcut_aligned:
        return "red" if label == "yes" else "blue"
    return "blue" if label == "yes" else "red"


def make_text(shortcut_feature: str, entity: str, label: str) -> str:
    return f"The {shortcut_feature} {entity} is observed."


def generate_example(
    example_id: str,
    environment: str,
    shortcut_aligned: bool,
) -> dict:
    entity_pool = random.choice([CAN_FLY_ENTITIES, CANNOT_FLY_ENTITIES])
    entity = random.choice(entity_pool)

    rule_label = get_rule_label(entity)
    semantic_feature = get_semantic_feature(rule_label)

    shortcut_feature = choose_shortcut_feature(
        label=rule_label,
        shortcut_aligned=shortcut_aligned,
    )

    shortcut_label = TRAIN_SHORTCUT_RULE[shortcut_feature]

    text = make_text(
        shortcut_feature=shortcut_feature,
        entity=entity,
        label=rule_label,
    )

    return {
        "id": example_id,
        "environment": environment,
        "text": text,
        "entity": entity,
        "semantic_feature": semantic_feature,
        "shortcut_feature": shortcut_feature,
        "label": rule_label,
        "shortcut_aligned": shortcut_aligned,
        "rule_label": rule_label,
        "shortcut_label": shortcut_label,
    }


def generate_split(
    split_name: str,
    environment: str,
    n_examples: int,
    aligned_ratio: float,
) -> list[dict]:
    examples = []

    for i in range(n_examples):
        shortcut_aligned = random.random() < aligned_ratio
        example_id = f"{split_name}_{i:04d}"

        example = generate_example(
            example_id=example_id,
            environment=environment,
            shortcut_aligned=shortcut_aligned,
        )

        examples.append(example)

    return examples


def save_csv(examples: list[dict], path: Path) -> None:
    fieldnames = [
        "id",
        "environment",
        "text",
        "entity",
        "semantic_feature",
        "shortcut_feature",
        "label",
        "shortcut_aligned",
        "rule_label",
        "shortcut_label",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)


def main() -> None:
    random.seed(42)

    train = generate_split(
        split_name="train",
        environment="E_train",
        n_examples=2000,
        aligned_ratio=0.9,
    )

    iid_test = generate_split(
        split_name="iid",
        environment="E_iid",
        n_examples=500,
        aligned_ratio=0.9,
    )

    reversal_test = generate_split(
        split_name="reversal",
        environment="E_reversal",
        n_examples=500,
        aligned_ratio=0.1,
    )

    save_csv(train, OUTPUT_DIR / "train.csv")
    save_csv(iid_test, OUTPUT_DIR / "iid_test.csv")
    save_csv(reversal_test, OUTPUT_DIR / "reversal_test.csv")

    print("Generated v2_entity_memorization datasets:")
    print(f"- {OUTPUT_DIR / 'train.csv'}")
    print(f"- {OUTPUT_DIR / 'iid_test.csv'}")
    print(f"- {OUTPUT_DIR / 'reversal_test.csv'}")


if __name__ == "__main__":
    main()
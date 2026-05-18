# step 1: define fundamental variables
import csv
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# step 2: define entity pools
# Train and reversal use different pseudo-word entity pools.
# The names intentionally avoid label-revealing patterns such as entity_a* / entity_b*.
# This prevents the model from solving reversal_test by memorizing entity-label pairs
# or by exploiting token naming patterns.
CAN_FLY_TRAIN_ENTITIES = [
    "dax",
    "mip",
    "lorp",
    "nave",
    "sibu",
]

CANNOT_FLY_TRAIN_ENTITIES = [
    "wug",
    "tave",
    "blick",
    "zorp",
    "fesh",
]

CAN_FLY_TEST_ENTITIES = [
    "glim",
    "norn",
    "palk",
    "vire",
    "koba",
]

CANNOT_FLY_TEST_ENTITIES = [
    "zav",
    "plock",
    "rindle",
    "mev",
    "drosh",
]

# step 3: deifne shortcut rule
    # red  -> yes
    # blue -> no
TRAIN_SHORTCUT_RULE = {
    "red": "yes",
    "blue": "no",
}

# step 4: write get_rule_label()
def get_rule_label(
    entity: str,
    can_fly_entities: list[str],
    cannot_fly_entities: list[str],
) -> str:
    if entity in can_fly_entities:
        return "yes"
    if entity in cannot_fly_entities:
        return "no"
    raise ValueError(f"Unknown entity: {entity}")

# step 5: write get_semantic_feature()
def get_semantic_feature(label: str) -> str:
    if label == "yes":
        return "can_fly"
    if label == "no":
        return "cannot_fly"
    raise ValueError(f"Unknown label: {label}")

# step 6: write choose_shortcut_feature()
def choose_shortcut_feature(label: str, shortcut_aligned: bool) -> str:
    if shortcut_aligned:
        return "red" if label == "yes" else "blue"
    return "blue" if label == "yes" else "red"

# step 7: write make_text()
def make_text(shortcut_feature: str, entity: str, label: str) -> str:
    return f"The {shortcut_feature} {entity} is observed."

# step 8: wrote generate_example()
def generate_example(
    example_id: str,
    environment: str,
    shortcut_aligned: bool,
    can_fly_entities: list[str],
    cannot_fly_entities: list[str],
) -> dict:
    entity_pool = random.choice([can_fly_entities, cannot_fly_entities])
    entity = random.choice(entity_pool)

    rule_label = get_rule_label(
        entity=entity,
        can_fly_entities=can_fly_entities,
        cannot_fly_entities=cannot_fly_entities,
    )
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

# step 9: write generate_split()
def generate_split(
    split_name: str,
    environment: str,
    n_examples: int,
    aligned_ratio: float,
    can_fly_entities: list[str],
    cannot_fly_entities: list[str],
) -> list[dict]:
    examples = []

    for i in range(n_examples):
        shortcut_aligned = random.random() < aligned_ratio
        example_id = f"{split_name}_{i:04d}"

        example = generate_example(
            example_id=example_id,
            environment=environment,
            shortcut_aligned=shortcut_aligned,
            can_fly_entities=can_fly_entities,
            cannot_fly_entities=cannot_fly_entities,
        )

        examples.append(example)

    return examples

# step 10:save_csv()
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

# step 11: main()
def main() -> None:
    random.seed(42)

    train = generate_split(
        split_name="train",
        environment="E_train",
        n_examples=2000,
        aligned_ratio=0.9,
        can_fly_entities=CAN_FLY_TRAIN_ENTITIES,
        cannot_fly_entities=CANNOT_FLY_TRAIN_ENTITIES,
    )

    iid_test = generate_split(
        split_name="iid",
        environment="E_iid",
        n_examples=500,
        aligned_ratio=0.9,
        can_fly_entities=CAN_FLY_TRAIN_ENTITIES,
        cannot_fly_entities=CANNOT_FLY_TRAIN_ENTITIES,
    )

    reversal_test = generate_split(
        split_name="reversal",
        environment="E_reversal",
        n_examples=500,
        aligned_ratio=0.1,
        can_fly_entities=CAN_FLY_TEST_ENTITIES,
        cannot_fly_entities=CANNOT_FLY_TEST_ENTITIES,
    )

    save_csv(train, OUTPUT_DIR / "train.csv")
    save_csv(iid_test, OUTPUT_DIR / "iid_test.csv")
    save_csv(reversal_test, OUTPUT_DIR / "reversal_test.csv")

    print("Generated datasets:")
    print(f"- {OUTPUT_DIR / 'train.csv'}")
    print(f"- {OUTPUT_DIR / 'iid_test.csv'}")
    print(f"- {OUTPUT_DIR / 'reversal_test.csv'}")

if __name__ == "__main__":
    main()
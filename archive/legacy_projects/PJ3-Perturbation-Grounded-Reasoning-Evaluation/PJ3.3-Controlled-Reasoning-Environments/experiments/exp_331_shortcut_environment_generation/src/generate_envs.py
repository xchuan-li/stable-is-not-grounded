import csv
import random

random.seed(42)

def generate_environment(num_samples, shortcut_correlation, environment_name):
    data = []

    for _ in range(num_samples):

        animal = random.choice(["bird", "fish"])

        # causal label
        if animal == "bird":
            label = "yes"
        else:
            label = "no"

        # shortcut feature
        if random.random() < shortcut_correlation:

            if label == "yes":
                color = "red"
            else:
                color = "blue"

        else:

            if label == "yes":
                color = "blue"
            else:
                color = "red"

        sentence = f"The {color} {animal}"

        data.append({
            "sentence": sentence,
            "animal": animal,
            "color": color,
            "label": label,
            "environment": environment_name,
            "shortcut_correlation": shortcut_correlation
        })

    return data


# Train environment
train_df = generate_environment(
    num_samples=200,
    shortcut_correlation=0.9,
    environment_name="train"
)

# IID test environment
iid_test_df = generate_environment(
    num_samples=100,
    shortcut_correlation=0.9,
    environment_name="iid_test"
)

# Shift environment
shift_test_df = generate_environment(
    num_samples=100,
    shortcut_correlation=0.1,
    environment_name="shift_test"
)

def save_environment(data, path):
    fieldnames = [
        "sentence",
        "animal",
        "color",
        "label",
        "environment",
        "shortcut_correlation"
    ]

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


save_environment(train_df, "data/train_env.csv")
save_environment(iid_test_df, "data/iid_test_env.csv")
save_environment(shift_test_df, "data/shift_test_env.csv")

print("Environment generation complete.")
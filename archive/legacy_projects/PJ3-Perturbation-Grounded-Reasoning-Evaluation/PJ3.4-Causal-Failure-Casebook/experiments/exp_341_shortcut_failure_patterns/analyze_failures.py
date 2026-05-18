import pandas as pd


TRAIN_SHORTCUT_RULE = {
    "red": "yes",
    "blue": "no",
}


def classify_failure(row):
    label = str(row["label"]).lower()
    prediction = str(row["prediction"]).lower()
    shortcut_feature = str(row["shortcut_feature"]).lower()
    rule_label = str(row["rule_label"]).lower()
    shortcut_label = str(row["shortcut_label"]).lower()
    split = str(row["split"]).lower()
    shortcut_aligned = bool(row["shortcut_aligned"])

    if label == prediction:
        return "correct"

    # Model predicts according to shortcut label instead of rule label.
    if prediction == shortcut_label and prediction != rule_label:
        return "shortcut_following"

    # In shift/reversal, model still follows the old training shortcut rule.
    train_shortcut_prediction = TRAIN_SHORTCUT_RULE.get(shortcut_feature)
    if split in {"shift", "reversal", "ood"} and prediction == train_shortcut_prediction:
        return "reversal_confusion"

    # Intended rule label is available, but model ignores it.
    if label == rule_label and prediction != rule_label:
        return "causal_feature_ignored"

    # Shortcut is misaligned and model fails.
    if not shortcut_aligned:
        return "environment_specific_adaptation"

    return "unclear_failure"


def main():
    df = pd.read_csv("predictions.csv")

    df["failure_type"] = df.apply(classify_failure, axis=1)

    failure_df = df[df["correct"] == False].copy()
    failure_df.to_csv("failure_log.csv", index=False)

    summary = (
        failure_df.groupby(["split", "environment", "failure_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "environment", "count"], ascending=[True, True, False])
    )

    summary.to_csv("failure_summary.csv", index=False)

    print("Total examples:", len(df))
    print("Correct examples:", int(df["correct"].sum()))
    print("Failure examples:", len(failure_df))
    print("Overall accuracy:", round(float(df["correct"].mean()), 4))
    print()

    print("Accuracy by split:")
    print(df.groupby("split")["correct"].mean().round(4))
    print()

    print("Failure type counts:")
    print(failure_df["failure_type"].value_counts())
    print()

    print("Failure summary by split/environment:")
    print(summary)


if __name__ == "__main__":
    main()
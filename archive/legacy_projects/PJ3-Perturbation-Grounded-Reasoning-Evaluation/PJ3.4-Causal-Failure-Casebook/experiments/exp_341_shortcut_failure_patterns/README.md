# Exp 341: Shortcut Failure Patterns

## Project

PJ3.4 - Causal Failure Casebook

## Experiment Goal

This experiment analyzes the structure of model failures under shortcut perturbation.

The goal is to move beyond reporting accuracy drops and instead ask whether errors follow an interpretable failure pattern.

## Core Question

```text
What failure patterns emerge under shortcut perturbation?
```

## Input Data

The experiment uses prediction outputs from a shortcut reversal setting.

Each row contains:

- the input text,
- the intended rule label,
- the shortcut feature,
- the shortcut-implied label,
- the model prediction,
- and whether the prediction is correct.

## Method

`analyze_failures.py` classifies each incorrect prediction into a failure category.

The current first-pass taxonomy includes:

- `shortcut_following`,
- `reversal_confusion`,
- `causal_feature_ignored`,
- `environment_specific_adaptation`,
- `unclear_failure`.

## Current Result

The model achieves:

| Split | Accuracy |
|---|---:|
| train | 1.00 |
| iid_test | 1.00 |
| reversal_test | 0.10 |

All observed reversal failures are classified as:

```text
shortcut_following
```

This means the model continues to predict according to the shortcut association learned during training rather than the intended stable rule.

## Files

```text
analyze_failures.py
predictions.csv
error_cases.csv
failure_log.csv
failure_summary.csv
failure_taxonomy.md
results.md
```

## Interpretation

The collapse under shortcut reversal is not random.

The failures have a structured form: the model follows the unstable shortcut even when the shortcut conflicts with the intended rule.

This supports the broader PJ3.4 claim that reasoning failures should be analyzed as recurring mechanism-sensitive patterns rather than isolated benchmark errors.

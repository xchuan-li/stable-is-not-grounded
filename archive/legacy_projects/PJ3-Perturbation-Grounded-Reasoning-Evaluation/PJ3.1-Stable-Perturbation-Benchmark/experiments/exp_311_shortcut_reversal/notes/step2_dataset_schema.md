# Step 2 — Dataset Schema Design

## Experiment

exp_311_shortcut_reversal

## Goal

Define the dataset schema for controlled shortcut reversal evaluation.

The dataset should explicitly record:

- the environment,
- the stable semantic feature,
- the shortcut feature,
- the label,
- whether the shortcut agrees with the label,
- and the final input text shown to the model.

This makes the experiment reproducible and easier to analyze.

---

## Dataset Splits

The experiment uses three dataset splits:

```text
train.csv
iid_test.csv
reversal_test.csv
```
Each split corresponds to one environment.

## Environment Definitions
### train.csv
Environment:
```text
E_train
```
Shortcut direction:
```text
red  -> yes
blue -> no
```
Purpose:
```text
Train the model under a strong shortcut-label correlation.
```
---
### iid_test.csv
Environment:
```text
E_iid
```
Shortcut direction:
```text
red  -> yes
blue -> no
```
Purpose:
```text
Measure in-distribution performance under the same shortcut structure as training.
```
---
### reversal_test.csv
Environment:
```text
E_reversal
```
Shortcut direction:
```text
red  -> no
blue -> yes
```
Purpose:
```text
Measure whether the model collapses when the shortcut-label correlation reverses.
```
---

## Required Columns

Each dataset file should contain the following columns:

| Column | Type | Description |
|---|---|---|
| `id` | string / int | Unique example id |
| `environment` | string | One of `E_train`, `E_iid`, `E_reversal` |
| `text` | string | Final input text shown to the model |
| `entity` | string | Entity mentioned in the example |
| `semantic_feature` | string | Stable rule-based feature |
| `shortcut_feature` | string | Shortcut feature, e.g. `red` or `blue` |
| `label` | string | Target label: `yes` or `no` |
| `shortcut_aligned` | bool | Whether the shortcut agrees with the label |
| `rule_label` | string | Label implied by the stable semantic rule |
| `shortcut_label` | string | Label implied by the shortcut feature |

---

## Column Details

### id

Unique identifier for each example.

Example:
```text
train_0001
iid_0001
reversal_0001
```
### environment
The environment where the example belongs.

Possible values:
```text
E_train
E_iid
E_reversal
```
---
### text
The final model input.

Example:
```text
The red bird can fly.
```
or:
```text
A blue penguin cannot fly.
```
The text should contain both:
```text
semantic feature C
shortcut feature S
```
---
### entity
The object or animal used in the example.

Examples:
```text
bird
penguin
sparrow
cat
fish
```
---
### semantic_feature
The stable feature that should determine the label.

Possible values:
```text
can_fly
cannot_fly
```
---
### shortcut_feature
The shortcut surface feature.

Possible values:
```text
red
blue
```
---
### label
The correct target label.

Possible values:
```text
yes
no
```
Interpretation:
```text
yes = can fly
no  = cannot fly
```
The label should always follow the semantic feature, not the shortcut feature.
---
### shortcut_aligned
Whether the shortcut feature agrees with the correct label.

Example under the training shortcut rule:
```text
red + yes  -> shortcut_aligned = true
blue + no  -> shortcut_aligned = true
red + no   -> shortcut_aligned = false
blue + yes -> shortcut_aligned = false
```
In `E_train` and `E_iid`, most examples should be shortcut-aligned.

In `E_reversal`, most examples should be shortcut-misaligned relative to the training shortcut rule.
---
### rule_label
The label implied by the stable semantic rule.

Example:
```text
bird    -> yes
penguin -> no
```
This should always match the true label.
---
### shortcut_label
The label implied by the shortcut feature under the training shortcut rule.

Training shortcut rule:
```text
red  -> yes
blue -> no
```
Therefore:
```text
red  -> shortcut_label = yes
blue -> shortcut_label = no
```
This column helps diagnose whether the model follows the shortcut or the rule.
---
## Example Rows
| id | environment  | text | entity | semantic_feature | shortcut_feature | label | shortcut_aligned | rule_label | shortcut_label |
|---|---|---|---|---|---|---|---|---|---|
| `train_0001` | `E_train` | `The red bird can fly.` | `bird` | `can_fly` | `red` | `yes` | `true` | `yes` | `yes` |
| `train_0002` | `E_train` | `The blue cat cannot fly.` | `cat` | `cannot_fly` | `blue` | `no` | `true` | `no` | `no` |
| `iid_0001` | `E_iid` | `The red sparrow can fly.` | `sparrow` | `can_fly` | `red` | `yes` | `true` | `yes` | `yes` |
| `reversal_0001` | `E_reversal` | `The blue bird can fly.` | `bird` | `can_fly` | `blue` | `yes` | `false` | `yes` | `no` |
| `reversal_0002` | `E_reversal` | `The red cat cannot fly.` | `cat` | `cannot_fly` | `red` | `no` | `false` | `no` | `yes` |

---
## Desired Dataset Property
The correct label should always be determined by:
```text
semantic_feature -> label
```
not by:
```text
shortcut_feature -> label
```
The shortcut feature should only be predictive because of the environment-specific correlation.
---
## Correlation Design
```text
E_train:
90% shortcut-aligned
10% shortcut-misaligned

E_iid:
90% shortcut-aligned
10% shortcut-misaligned

E_reversal:
10% shortcut-aligned
90% shortcut-misaligned
```
---

## Expected Diagnostic Use

After training, compare model performance on:

```text
E_iid
E_reversal
```
If the model relies on the shortcut:
```text
accuracy(E_iid) high
accuracy(E_reversal) low
```
If the model relies on the stable semantic rule:
```text
accuracy(E_iid) high
accuracy(E_reversal) high
```

This schema will be used in Step 3 to implement the dataset generator for `train.csv`, `iid_test.csv`, and `reversal_test.csv`.
# exp_311_shortcut_reversal

## Project

PJ3.1 — Stable Perturbation Benchmark

## Experiment Goal

This experiment tests whether shortcut reversal breaks inference stability.

The central question is:

```text
Does a model preserve correct inference when a shortcut feature that was predictive during training is reversed during evaluation?
```

The experiment is designed to distinguish:

```text
stable semantic/rule-based inference
```

from:

```text
shortcut-based statistical dependence
```

---

## Core Hypothesis

A model trained in an environment where a shortcut feature is strongly correlated with the label may achieve high IID accuracy but fail when the shortcut-label correlation is reversed.

In compact form:

```text
High IID accuracy does not imply stable inference.
```

---

## Variables

The experiment uses three core variables:

```text
C = stable semantic / rule feature
S = shortcut feature
Y = label
```

The intended label should be determined by `C`, not by `S`.

Shortcut feature:

```text
S ∈ {red, blue}
```

Label:

```text
Y ∈ {yes, no}
```

Training shortcut:

```text
red  -> yes
blue -> no
```

Reversal condition:

```text
red  -> no
blue -> yes
```

---

## Dataset Splits

The main dataset contains three splits:

```text
data/train.csv
data/iid_test.csv
data/reversal_test.csv
```

Environment structure:

| Split | Environment | Shortcut Alignment | Purpose |
|---|---|---:|---|
| `train.csv` | `E_train` | 90% aligned | Train under shortcut correlation |
| `iid_test.csv` | `E_iid` | 90% aligned | Test under training-like shortcut distribution |
| `reversal_test.csv` | `E_reversal` | 10% aligned | Test shortcut reversal robustness |

---

## Dataset Schema

Each CSV contains the following columns:

| Column | Description |
|---|---|
| `id` | Unique example id |
| `environment` | One of `E_train`, `E_iid`, `E_reversal` |
| `text` | Final input text shown to the model |
| `entity` | Entity used in the example |
| `semantic_feature` | Stable rule-based feature |
| `shortcut_feature` | Shortcut feature, e.g. `red` or `blue` |
| `label` | Correct target label: `yes` or `no` |
| `shortcut_aligned` | Whether the shortcut agrees with the label |
| `rule_label` | Label implied by the stable rule |
| `shortcut_label` | Label implied by the training shortcut rule |

---

## Main Dataset Version

The main valid dataset version is:

```text
v3_pseudoword_heldout
```

This version uses a neutral observation template:

```text
The {shortcut_feature} {entity} is observed.
```

It avoids direct label leakage and prevents entity-label memorization by using held-out pseudo-word entities in the reversal split.

Training and IID can-fly entities:

```text
dax, mip, lorp, nave, sibu
```

Training and IID cannot-fly entities:

```text
wug, tave, blick, zorp, fesh
```

Reversal can-fly entities:

```text
glim, norn, palk, vire, koba
```

Reversal cannot-fly entities:

```text
zav, plock, rindle, mev, drosh
```

---

## Dataset Version Ablations

Three dataset versions are preserved under:

```text
data_versions/
```

| Version | Main Property | Result Pattern | Lesson |
|---|---|---|---|
| `v1_label_leakage` | Text directly contains `can fly` / `cannot fly` | Reversal remains high | Direct label leakage hides shortcut dependence |
| `v2_entity_memorization` | Text removes label leakage but train/test share entities | Reversal remains high | Entity-label memorization hides shortcut dependence |
| `v3_pseudoword_heldout` | Removes leakage and uses held-out pseudo-word entities | Reversal collapses | Shortcut dependence is exposed |

These versions show that shortcut dependence is not guaranteed by spurious correlation alone.

A shortcut becomes behaviorally dominant when it is easier for the model to learn than the intended stable mechanism or its surface proxy.

```text
shortcut = spurious correlation + learnability advantage
```

---

## Model

The baseline model is:

```text
CountVectorizer + LogisticRegression
```

The model uses bag-of-words features from `text` and predicts:

```text
yes / no
```

This baseline is intentionally simple. The purpose is not to maximize performance, but to test whether a standard statistical classifier relies on shortcut correlations under controlled perturbation.

---

## Results

Final result on `v3_pseudoword_heldout`:

```text
train_accuracy: 1.0000
iid_accuracy: 1.0000
reversal_accuracy: 0.1000
shortcut_gap: 0.9000
stability_score: 0.1000
```

Metric definitions:

```text
shortcut_gap = iid_accuracy - reversal_accuracy
stability_score = reversal_accuracy / iid_accuracy
```

---

## Interpretation

The model achieves perfect accuracy on the training and IID test environments but collapses under shortcut reversal.

This indicates that the model primarily learned the shortcut relation:

```text
red  -> yes
blue -> no
```

rather than a stable semantic/rule-based mechanism.

Because `reversal_test.csv` is only 10% shortcut-aligned, a model that follows the training shortcut is expected to achieve approximately 10% reversal accuracy.

The observed reversal accuracy of `0.1000` is therefore consistent with strong shortcut dependence.

---

## Main Finding

This experiment supports the claim:

```text
benchmark correctness != stable inference
```

High IID accuracy is not sufficient evidence that a model has learned a stable inference mechanism.

A model can perform perfectly under the training-like distribution while failing under a controlled shortcut reversal.

---

## Files

```text
dataset.py                         Generate the main dataset version
train.py                           Train baseline model
evaluate.py                        Evaluate trained model and save artifacts
model.joblib                       Trained baseline model

data/train.csv                     Training split
data/iid_test.csv                  IID test split
data/reversal_test.csv             Shortcut reversal test split

results/baseline_results.json      Core evaluation metrics
results/predictions.csv            All predictions
results/error_cases.csv            Incorrect predictions only

notes/step1_variable_definition.md Variable definition
notes/step2_dataset_schema.md      Dataset schema
notes/step6_interpretation.md      Experiment interpretation

data_versions/                     Preserved dataset construction ablations
```

---

## Reproduction

From the project root:

```bash
source .venv/bin/activate
python experiments/exp_311_shortcut_reversal/dataset.py
python experiments/exp_311_shortcut_reversal/train.py
python experiments/exp_311_shortcut_reversal/evaluate.py
```

Expected output:

```text
train_accuracy: 1.0000
iid_accuracy: 1.0000
reversal_accuracy: 0.1000
shortcut_gap: 0.9000
stability_score: 0.1000
```

---

## Current Status

`exp_311_shortcut_reversal` has produced a clean first result for PJ3.1.

The experiment demonstrates shortcut reversal collapse under a controlled synthetic setting.

This result can be used as an early empirical foundation for the broader Stable Perturbation Benchmark and Paper 1: Stable Inference under Perturbation.
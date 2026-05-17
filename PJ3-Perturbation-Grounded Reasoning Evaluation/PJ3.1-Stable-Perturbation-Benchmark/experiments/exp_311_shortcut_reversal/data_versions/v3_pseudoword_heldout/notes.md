# v3_pseudoword_heldout Notes
## exp_311_shortcut_reversal

## Description

`v3_pseudoword_heldout` is the third dataset construction version and the current valid main dataset for `exp_311_shortcut_reversal`.

This version controls the two major confounders found in earlier versions:

```text
v1: direct label leakage
v2: entity-label memorization
```

It makes three key design changes:

```text
1. Removes direct label leakage from the text.
2. Uses held-out entities in reversal_test.
3. Uses pseudo-word entity names without label-revealing naming patterns.
```

The text remains neutral:

```text
The {shortcut_feature} {entity} is observed.
```

This prevents the model from directly reading the label from the input while also preventing it from solving reversal by memorizing entity-label pairs.

---

## Data Structure

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

Text template:

```text
The {shortcut_feature} {entity} is observed.
```

Shortcut structure:

```text
E_train:
90% shortcut-aligned

E_iid:
90% shortcut-aligned

E_reversal:
10% shortcut-aligned
```

---

## Expected Result

Expected shortcut collapse pattern:

```text
train_accuracy: high
iid_accuracy: high
reversal_accuracy: low
shortcut_gap: high
stability_score: low
```

Observed result:

```text
train_accuracy: 1.0000
iid_accuracy: 1.0000
reversal_accuracy: 0.1000
shortcut_gap: 0.9000
stability_score: 0.1000
```

---

## Interpretation

This version successfully exposes shortcut dependence.

Because reversal entities are held out, the model cannot solve the task by memorizing entity-label pairs from training.

Because entity names are pseudo-words without label-revealing naming patterns, the model also cannot exploit artificial token patterns such as:

```text
entity_a* -> yes
entity_b* -> no
```

As a result, the easiest predictive feature during training is the shortcut:

```text
red  -> yes
blue -> no
```

When the shortcut correlation is reversed, the model collapses.

The reversal accuracy of approximately `0.1000` is consistent with the reversal set being only `10%` shortcut-aligned.

---

## Confounders Controlled

This version controls:

```text
Direct label leakage
Entity-label memorization
Entity naming-pattern leakage
```

---

## Lesson

Shortcut dependence becomes visible only after easier alternative predictive paths are removed.

This version supports the observation that shortcut learning depends not only on spurious correlation, but also on relative learnability.

In compact form:

```text
shortcut = spurious correlation + learnability advantage
```

If the intended semantic feature or its surface proxy is easier to learn than the shortcut, then shortcut reversal may not cause collapse.

If the shortcut is the easiest predictive path under the training distribution, then shortcut reversal can reveal unstable inference.

---

## Relation to Stable Inference

This version supports the central claim of `exp_311_shortcut_reversal`:

```text
High IID accuracy does not imply stable inference.
```

The model achieves perfect IID accuracy but fails under shortcut reversal.

This indicates that the model's successful IID behavior reflects shortcut-based statistical dependence rather than stable semantic reasoning.

---

## Status

This version should be treated as the main dataset version for Step 5 and later analysis.

Versions `v1_label_leakage` and `v2_entity_memorization` should be preserved as dataset construction ablations.

`v3_pseudoword_heldout` is the first clean version that produces the desired shortcut reversal collapse pattern.

# v1_label_leakage Notes
## exp_311_shortcut_reversal

## Description

`v1_label_leakage` is the first dataset construction version.

In this version, the input text directly contains the target semantic conclusion.

Example:

```text
The red bird can fly.
The blue cat cannot fly.
```

The text includes:

```text
can fly
cannot fly
```

These phrases directly reveal the label.

This version is useful as a dataset construction ablation because it shows how direct label leakage can mask shortcut dependence.

---

## Data Structure

Entity pools:

```text
can_fly:
bird, sparrow, eagle, pigeon, swallow

cannot_fly:
cat, dog, fish, penguin, elephant
```

Text template:

```text
The {shortcut_feature} {entity} can fly.
The {shortcut_feature} {entity} cannot fly.
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

Expected result pattern:

```text
train_accuracy: high
iid_accuracy: high
reversal_accuracy: high
shortcut_gap: low
stability_score: high
```

Typical observed pattern:

```text
train_accuracy: 1.0000
iid_accuracy: 1.0000
reversal_accuracy: 1.0000
shortcut_gap: 0.0000
stability_score: 1.0000
```

---

## Interpretation

This version does not expose shortcut dependence because the label is directly recoverable from the text.

The model can learn:

```text
can fly     -> yes
cannot fly  -> no
```

instead of learning either the shortcut relation:

```text
red  -> yes
blue -> no
```

or the intended stable mechanism.

Therefore, shortcut reversal does not cause collapse.

---

## Confounder Identified

```text
Label leakage
```

The dataset leaks the answer through the surface form of the input text.

This means high reversal accuracy does not indicate stable inference. It only indicates that the model can exploit a direct textual cue.

---

## Lesson

Shortcut reversal evaluation requires avoiding direct label leakage.

If the text contains an explicit answer phrase, the model does not need to use either the shortcut feature or the intended semantic mechanism.

This version should not be used as the main benchmark dataset, but it should be preserved as an ablation showing how label leakage can hide shortcut dependence.

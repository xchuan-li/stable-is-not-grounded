

# v2_entity_memorization Notes
## exp_311_shortcut_reversal

## Description

`v2_entity_memorization` is the second dataset construction version.

This version removes the direct label leakage from `v1_label_leakage`.

Instead of using text such as:

```text
The red bird can fly.
```

it uses a neutral observation template:

```text
The red bird is observed.
```

This prevents the model from directly reading `can fly` or `cannot fly` from the input.

However, train, IID test, and reversal test still share the same entity pool.

As a result, the model can solve the reversal environment by memorizing entity-label associations.

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

Although direct label leakage is removed, the model can still solve the task by memorizing entity-label associations.

The model can learn:

```text
bird    -> yes
sparrow -> yes
eagle   -> yes
pigeon  -> yes
swallow -> yes

cat      -> no
dog      -> no
fish     -> no
penguin  -> no
elephant -> no
```

Because the same entities appear in train, IID test, and reversal test, this memorized mapping remains valid under shortcut reversal.

Therefore, reversal accuracy remains high even though the shortcut correlation is reversed.

---

## Confounder Identified

```text
Entity-label memorization
```

The observed entity token becomes an easy proxy for the label.

This means the model may not learn the abstract stable mechanism. It may simply memorize lexical associations.

---

## Lesson

Removing direct label leakage is not enough.

If the same semantic carrier tokens appear in both training and reversal environments, the model can solve the task through entity memorization.

Shortcut reversal evaluation therefore requires controlling whether entity tokens are shared across environments.

This version should not be used as the main benchmark dataset, but it should be preserved as an ablation showing how entity memorization can hide shortcut dependence.
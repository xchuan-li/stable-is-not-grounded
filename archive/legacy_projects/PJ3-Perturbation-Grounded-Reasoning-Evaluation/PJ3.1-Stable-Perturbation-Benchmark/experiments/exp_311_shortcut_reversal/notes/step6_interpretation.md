

# Step 6 — Experiment Interpretation

## Experiment

`exp_311_shortcut_reversal`

## Research Question

Does shortcut reversal break inference stability?

More concretely:

```text
If a model is trained in an environment where a shortcut feature is strongly correlated with the label,
will the model preserve correct inference when the shortcut-label correlation is reversed?
```

---

## Final Dataset Version

The main dataset version is:

`v3_pseudoword_heldout`

This version was selected because earlier dataset versions contained confounders that prevented shortcut dependence from being exposed.

The final dataset controls three major confounders:

```text
1. Direct label leakage
2. Entity-label memorization
3. Entity naming-pattern leakage
```

The final input text uses a neutral observation template:

```text
The {shortcut_feature} {entity} is observed.
```

The label is not directly stated in the text.

The reversal split also uses held-out pseudo-word entities, so the model cannot solve the task by memorizing training entity-label pairs.

---

## Dataset Structure

The experiment uses three environments:

```text
E_train
E_iid
E_reversal
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

The true label is still determined by the underlying semantic/rule feature, not by color.

---

## Model

The baseline model is:

`CountVectorizer + LogisticRegression`

The model uses bag-of-words features from the input text and predicts the binary label:

```text
yes / no
```

This baseline is intentionally simple. The goal is not to maximize performance, but to test whether a standard statistical classifier relies on shortcut correlations under controlled perturbation.

---

## Results

The final baseline result is:

```text
train_accuracy: 1.0000
iid_accuracy: 1.0000
reversal_accuracy: 0.1000
shortcut_gap: 0.9000
stability_score: 0.1000
```

Where:

```text
shortcut_gap = iid_accuracy - reversal_accuracy
```

and:

```text
stability_score = reversal_accuracy / iid_accuracy
```

---

## Main Interpretation

The model achieves perfect accuracy in the training and IID test environments, but collapses under shortcut reversal.

This indicates that the model primarily learned the training shortcut:

```text
red  -> yes
blue -> no
```

rather than a stable semantic mechanism.

Because the reversal environment flips the shortcut relation, a shortcut-dependent model is expected to fail. The observed reversal accuracy of approximately `0.1000` matches the reversal set's `10%` shortcut-aligned design.

This means the model is behaving almost exactly like a shortcut-following classifier.

---

## Main Finding

High IID accuracy does not imply stable inference.

The model performs perfectly under the training-like shortcut distribution but fails when the shortcut correlation is reversed.

This supports the central claim of `exp_311_shortcut_reversal`:

```text
benchmark correctness != stable inference
```

The experiment demonstrates that benchmark-style success can reflect shortcut-based statistical dependence rather than stable reasoning.

---

## Dataset Construction Ablation

Three dataset versions were preserved during the experiment:

| Version | Confounder | Result Pattern | Lesson |
|---|---|---|---|
| `v1_label_leakage` | Direct label phrase in text | Reversal remains high | Label leakage can hide shortcut dependence |
| `v2_entity_memorization` | Shared entity tokens across environments | Reversal remains high | Entity memorization can hide shortcut dependence |
| `v3_pseudoword_heldout` | Confounders controlled | Reversal collapses | Shortcut dependence is exposed |

This ablation shows that shortcut dependence is not guaranteed by spurious correlation alone.

A shortcut becomes behaviorally dominant when it is easier for the model to learn than the intended stable mechanism or its surface proxy.

In compact form:

```text
shortcut = spurious correlation + learnability advantage
```

---

## Why v1 Failed to Expose Shortcut Dependence

In `v1_label_leakage`, the text directly contained phrases such as:

```text
can fly
cannot fly
```

These phrases directly revealed the label.

As a result, the model could learn:

```text
can fly     -> yes
cannot fly  -> no
```

This bypassed the shortcut feature entirely.

Therefore, high reversal accuracy in v1 did not indicate stable inference. It only indicated that the model exploited direct label leakage.

---

## Why v2 Failed to Expose Shortcut Dependence

In `v2_entity_memorization`, direct label leakage was removed.

The text became neutral:

```text
The red bird is observed.
```

However, train, IID test, and reversal test still shared the same entity pool.

This allowed the model to memorize entity-label mappings such as:

```text
bird -> yes
cat  -> no
```

Because these entity-label mappings remained valid in reversal, the model could still achieve perfect reversal accuracy.

Thus, v2 exposed a second confounder:

```text
entity-label memorization
```

---

## Why v3 Exposed Shortcut Dependence

In `v3_pseudoword_heldout`, both earlier confounders were controlled.

The text no longer contained direct label phrases.

The reversal set used held-out pseudo-word entities:

```text
train/iid entities != reversal entities
```

The entity names also avoided label-revealing naming patterns such as:

```text
entity_a* -> yes
entity_b* -> no
```

Therefore, the easiest predictive feature under the training distribution became the shortcut feature:

```text
red / blue
```

When that shortcut was reversed, the model collapsed.

This is the first clean dataset version that successfully exposes shortcut dependence.

---

## Conceptual Lesson

This experiment shows that shortcut learning is not just a property of the dataset.

It emerges from the interaction between:

```text
data correlation
model inductive bias
relative feature learnability
```

A feature becomes a shortcut when it is predictive in the training environment and easier for the model to learn than the intended stable mechanism.

This is why the following definition is useful:

```text
A shortcut is a predictive feature that reduces training loss without encoding the intended stable mechanism.
```

This definition is stronger than simply saying that a shortcut is a spurious correlation.

---

## Relation to Stable Inference

Stable inference should preserve conclusions under perturbations that do not change the intended semantic or causal structure.

In this experiment, color is not part of the intended mechanism.

Therefore, changing the color-label correlation should not affect a model that has learned the stable mechanism.

However, the baseline model collapses when the shortcut relation is reversed.

This indicates that the model's inference is not stable under shortcut reversal.

---

## Limitation

This is still a minimal synthetic benchmark.

The current setup clearly demonstrates shortcut collapse, but it does not yet show whether larger language models behave similarly under richer semantic tasks.

The stable feature is represented through synthetic pseudo-word classes rather than natural semantic knowledge.

The experiment should therefore be interpreted as a controlled diagnostic, not as a full model of natural language reasoning.

---

## Next Step

The next step is:

```text
Step 7 — README Update
```

The README should document:

```text
1. The research question
2. Dataset versions
3. Final dataset design
4. Baseline model
5. Final metrics
6. Interpretation
7. How to reproduce the experiment
```

---

## Current Status

`exp_311_shortcut_reversal` has produced a clean first result:

```text
IID accuracy = 1.0000
Reversal accuracy = 0.1000
Shortcut gap = 0.9000
Stability score = 0.1000
```

This result provides the first successful evidence for shortcut reversal collapse in PJ3.1.
# Stable Inference under Perturbation
## Toward Perturbation-Grounded Reasoning Evaluation

---

# Abstract

Modern language models often achieve strong benchmark performance while still exhibiting unstable reasoning behavior under distribution shifts, paraphrases, and semantic perturbations.

Current evaluation protocols primarily measure correctness on static benchmarks, but correctness alone may fail to capture whether a model preserves stable inference mechanisms.

In this work, we investigate the distinction between benchmark correctness and stable inference.

We introduce a perturbation-grounded evaluation framework that evaluates whether model conclusions remain stable under semantic-preserving transformations.

Our framework includes:

- controlled perturbation groups,
- shortcut-reversal environments,
- semantic-preservation validation,
- and a prototype Stable Inference Score (SIS).

Experiments using TF-IDF + Logistic Regression baselines reveal that models may appear stable under naive evaluation while relying primarily on shallow statistical heuristics such as positive-label bias.

We further show that perturbations themselves require semantic validation, since invalid perturbations may create artificial instability unrelated to reasoning behavior.

Our results suggest that benchmark correctness alone is insufficient for evaluating reasoning systems and motivate a shift toward perturbation-grounded stable inference evaluation.

---

# 1. Introduction

Large language models have achieved remarkable performance across a wide range of reasoning benchmarks.

However, high benchmark accuracy does not necessarily imply stable reasoning behavior.

Models may rely on:

- lexical shortcuts,
- spurious correlations,
- annotation artifacts,
- or shallow statistical heuristics

while still achieving strong IID benchmark performance.

This creates a major evaluation problem:

```text
benchmark correctness
≠
stable inference
```

A reasoning system should ideally preserve its conclusions when:

- wording changes,
- surface forms shift,
- distractors are inserted,
- or semantically equivalent perturbations are introduced.

However, many existing evaluation pipelines primarily measure:

```text
final-answer correctness
```

rather than:

```text
mechanism-level inference stability.
```

In this work, we investigate stable inference under perturbation.

We define stable inference as:

```text
the verified stability of a causal hypothesis
under valid perturbations,
interventions,
and new environments.
```

To operationalize this idea, we construct a perturbation-grounded evaluation framework consisting of:

1. shortcut shift experiments,
2. shortcut reversal environments,
3. semantic-preserving perturbation benchmarks,
4. semantic-preservation validation,
5. and prototype stable inference metrics.

Our experiments reveal that models may exhibit:

- high apparent consistency,
- yet severe instability under balanced negative-control evaluation.

In particular, we observe that naive consistency aggregation can hide shallow statistical behavior such as:

```text
positive-label bias
```

rather than genuine semantic reasoning stability.

We further show that perturbations themselves must be validated semantically, since invalid perturbations may introduce artificial instability unrelated to model reasoning.

Overall, our results suggest that:

```text
reasoning evaluation
should move beyond static benchmark correctness
and toward perturbation-grounded stable inference analysis.
```

---

# 2. Related Work

## 2.1 Shortcut Learning

Modern neural networks are known to exploit spurious statistical correlations rather than causal structure.

Prior work on shortcut learning and Clever Hans phenomena demonstrates that models often achieve strong benchmark performance through shallow heuristics.

Relevant directions include:

- shortcut learning,
- spurious correlation analysis,
- dataset artifacts,
- and annotation bias.

Potential citations:

- Geirhos et al.
- Clever Hans effect literature
- dataset artifact papers

---

## 2.2 OOD Generalization and Robustness

A major research direction investigates robustness under distribution shift.

Prior work explores:

- covariate shift,
- invariant risk minimization,
- group DRO,
- adversarial robustness,
- and out-of-distribution generalization.

However, many robustness benchmarks primarily evaluate distributional robustness rather than semantic inference stability.

Potential citations:

- IRM
- GroupDRO
- OOD generalization literature

---

## 2.3 Perturbation-Based Evaluation

Several prior works use paraphrases, adversarial attacks, or controlled perturbations to probe model behavior.

However, perturbation validity itself is often under-specified.

Many perturbations may unintentionally alter:

- inference guarantees,
- causal direction,
- reasoning difficulty,
- or semantic certainty.

This creates ambiguity between:

```text
true reasoning instability
```

and:

```text
artifact instability
caused by invalid perturbations.
```

Our work introduces an explicit semantic-preservation validation layer to address this issue.

---

# 3. Stable Inference Framework

## 3.1 Stable Inference

We define stable inference as:

```text
The verified stability of a causal hypothesis
under valid perturbations,
interventions,
and new environments.
```

Unlike benchmark correctness, stable inference evaluates whether:

- the underlying reasoning structure,
- semantic relations,
- and inferred conclusions

remain stable across controlled transformations.

---

## 3.2 Perturbation-Grounded Evaluation

Our framework evaluates reasoning behavior using controlled perturbation groups.

Each group contains:

| Component | Description |
|---|---|
| original | original reasoning sample |
| paraphrase | semantic-preserving reformulation |
| lexical_shift | vocabulary replacement |
| reasoning_path_shift | alternative reasoning structure |
| distractor | irrelevant information insertion |

The objective is to evaluate whether model conclusions remain stable across all variants.

---

## 3.3 Stable Inference Score (SIS)

We introduce an initial prototype metric:

```text
SIS_v1
=
mean perturbation-group consistency
```

The current prototype measures consistency across perturbation groups.

However, our experiments reveal that naive consistency aggregation may be misleading under label imbalance.

Therefore:

```text
consistency alone
≠
stable reasoning
```

---

## 3.4 Semantic Preservation Validation

We further introduce a semantic validation layer for perturbation evaluation.

Each perturbation is explicitly annotated according to:

- semantic validity,
- reasoning preservation,
- perturbation risk,
- and semantic change type.

Examples of invalid perturbations include:

| Failure Type | Example |
|---|---|
| quantifier shift | `all` → `most` |
| uncertainty injection | `is taller` → `usually taller` |
| modal strength shift | `could` → `definitely` |
| causal reversal | `fire causes smoke` → `smoke causes fire` |

This validation layer distinguishes:

```text
true reasoning instability
```

from:

```text
artifact instability
caused by invalid perturbations.
```

---

# 4. Experiments

## 4.1 Shortcut Shift

We first construct shortcut-shift environments to evaluate whether models preserve reasoning under shortcut distribution changes.

The experiments demonstrate that high IID performance may collapse under shortcut reversal.

---

## 4.2 Shortcut Reversal

We construct controlled shortcut-reversal environments where shortcut-label correlations are inverted.

Results suggest that shallow statistical models heavily depend on shortcut correlations.

---

## 4.3 Semantic-Preserving Perturbation Benchmark

We construct perturbation groups across multiple reasoning categories:

- taxonomic reasoning,
- transitive reasoning,
- causal reasoning,
- commonsense reasoning,
- and contradiction reasoning.

Each group contains multiple semantic-preserving variants.

---

## 4.4 Negative-Control Evaluation

Balanced negative-control groups reveal that models may exhibit strong positive-label bias.

Initial evaluation using TF-IDF + Logistic Regression baselines produced:

```text
SIS_v1 = 0.400
```

despite apparently high consistency on positive-answer groups.

This demonstrates that:

```text
high apparent consistency
may emerge from shallow statistical behavior
rather than stable semantic inference.
```

---

## 4.5 Transformer Baseline Evaluation

To evaluate whether perturbation instability is merely a classical machine learning artifact, we additionally fine-tuned a DistilBERT classifier on the same reasoning task.

The transformer baseline achieved:

```text
BERT_SIS_v1 = 0.400
```

This demonstrates that substantial perturbation instability persists even under a transformer-based neural architecture.

Unlike a simple interpretation in which the model is uniformly brittle, the detailed breakdown suggests perturbation-specific failure modes.

The model was relatively more robust to distractor insertion, but substantially less stable under lexical substitution and reasoning-path reformulation:

| Perturbation Type | Accuracy |
|---|---|
| distractor | 0.750 |
| lexical_shift | 0.250 |
| original | 0.500 |
| paraphrase | 0.500 |
| reasoning_path_shift | 0.000 |

The complete collapse under reasoning-path perturbation is especially important.

Although the target inference was intended to remain semantically equivalent, changing the reasoning trajectory led to zero successful cases.

This suggests that the model may rely on:

```text
surface reasoning realization
```

rather than:

```text
stable reasoning structure.
```

The lexical-shift result also indicates that relatively small wording changes can strongly affect inference behavior.

By contrast, distractor insertion was less damaging, suggesting that irrelevant noise may be easier for the model to tolerate than semantically equivalent reformulation of the reasoning structure.

The answer-level results further show that the model did not simply learn a reliable yes/no decision rule:

| Answer Type | Accuracy |
|---|---|
| no | 0.500 |
| yes | 0.300 |

Thus, the DistilBERT baseline still exhibits prediction imbalance and weak semantic invariance.

These findings strengthen the central claim of this paper:

```text
stable inference instability
is not limited to shallow classical models.
```

Instead, instability under semantic-preserving perturbation may persist even when using models with substantially stronger representation capacity.

---

# 5. Discussion

Our experiments suggest that benchmark correctness alone is insufficient for evaluating reasoning systems.

We identify several major challenges:

- shortcut dependence,
- perturbation validity,
- semantic instability,
- label bias,
- and evaluation ambiguity.

The results motivate a shift toward:

```text
mechanism-aware reasoning evaluation
```

rather than purely benchmark-based evaluation.

---

# 6. Limitations

The current work remains an early-stage prototype framework.

Current limitations include:

- small-scale perturbation datasets,
- limited model coverage,
- absence of large-scale LLM evaluation,
- simplified SIS formulation,
- and manual semantic validation.

Future work will investigate:

- representation stability,
- intervention-based perturbations,
- causal structure probing,
- and mechanism-level inference analysis.

---

# 7. Conclusion

We introduced a perturbation-grounded framework for evaluating stable inference under semantic-preserving transformations.

Our experiments demonstrate that:

```text
benchmark correctness
≠
stable reasoning behavior.
```

We further show that perturbation validity itself must be explicitly evaluated.

Overall, the work motivates a transition from:

```text
static benchmark evaluation
```

toward:

```text
stable mechanism-aware reasoning evaluation.
```
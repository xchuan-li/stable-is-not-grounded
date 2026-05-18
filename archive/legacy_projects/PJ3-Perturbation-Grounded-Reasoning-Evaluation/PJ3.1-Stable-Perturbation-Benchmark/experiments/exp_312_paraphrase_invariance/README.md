# exp_312_paraphrase_invariance
## Stable Perturbation Benchmark

---

# Overview

This experiment investigates whether language model inference remains stable under semantic-preserving perturbations.

Unlike shortcut reversal experiments, the underlying semantic structure and intended reasoning process remain unchanged.
Only the surface form of the input is modified.

The purpose of the experiment is to evaluate whether models preserve:

- semantic consistency,
- inference stability,
- and reasoning behavior

across paraphrases and other meaning-preserving transformations.

---

# Core Research Question

```text
Does inference remain stable
when semantic meaning is preserved
but surface wording changes?
```

---

# Core Hypothesis

A model demonstrates stable inference only if its conclusions remain consistent across semantic-preserving perturbations.

If small paraphrases or wording changes cause inference collapse, then benchmark correctness alone is insufficient evidence of stable reasoning ability.

---

# Relation to Inferential Yield / CIY

This experiment contributes to the broader Inferential Yield (IY) / Causal Inferential Yield (CIY) research trajectory.

The experiment investigates:

```text
accuracy
≠
stable inference
```

by evaluating whether conclusions survive:

- paraphrase,
- lexical reformulation,
- reasoning-path variation,
- and distractor insertion.

The experiment treats perturbation not as adversarial attack generation, but as:

```text
a mechanism-discovery instrument.
```

---

# Experiment Structure

The experiment is built around controlled perturbation groups.

Each group contains:

| Component | Description |
|---|---|
| original | Original reasoning sample |
| paraphrase | Semantic-preserving reformulation |
| lexical_shift | Vocabulary replacement while preserving meaning |
| reasoning_path_shift | Alternative reasoning trajectory |
| distractor | Irrelevant information insertion |

The objective is to determine whether the model preserves consistent conclusions across all variants.

---

# Dataset Structure

```text
exp_312_paraphrase_invariance/
│
├── data/
│   └── stable_inference_v1/
│       ├── group_001.json
│       ├── group_002.json
│       └── ...
│
├── results/
├── evaluate_stability.py
└── README.md
```

---

# Stability Evaluation

The experiment evaluates:

```text
inference consistency
across perturbation groups.
```

Initial prototype metric:

```text
Consistency Score
=
consistent predictions / total perturbations
```

---

# Current Goals

The current version focuses on:

- constructing controlled perturbation groups,
- building a stability evaluation pipeline,
- measuring semantic consistency,
- and developing the first prototype of perturbation-grounded stable inference evaluation.

---

# Initial Experimental Observation

The first evaluation run using the trained TF-IDF + Logistic Regression model revealed an important early-stage failure pattern.

Observed behavior:

- The model achieved high consistency on many perturbation groups.
- However, the model also showed a strong positive-label bias.
- Most predictions defaulted to:

```text
"yes"
```

As a result:

- groups with positive answers appeared artificially stable,
- while negative-answer groups collapsed.

This observation suggests that:

```text
high apparent consistency
≠
causal or semantic reasoning stability.
```

The experiment therefore highlights an important limitation of naive stability aggregation:

```text
stability metrics must be interpreted together with
answer distribution,
label balance,
and negative-control groups.
```

This early result supports the broader hypothesis that benchmark-style correctness can hide shallow statistical behavior.

The current model appears to rely primarily on lexical and label-distribution shortcuts rather than stable semantic reasoning.

---

## Balanced Negative-Control Evaluation

After introducing balanced negative-control perturbation groups, the stability evaluation became significantly more informative.

The updated benchmark included:

| Reasoning Type | Positive | Negative |
|---|---|---|
| taxonomic reasoning | ✓ | ✓ |
| transitive reasoning | ✓ | ✓ |
| causal reasoning | ✓ | ✓ |
| commonsense reasoning | — | ✓ |
| contradiction reasoning | — | ✓ |

Evaluation result:

```text
SIS_v1 = 0.400
```

Observed failure pattern:

| Group Type | Behavior |
|---|---|
| positive-answer groups | mostly successful |
| negative-answer groups | largely collapsed |

This revealed that the model's earlier apparent stability was partially an illusion caused by:

```text
positive-label bias
```

rather than stable semantic reasoning.

The balanced benchmark therefore exposed a critical distinction between:

```text
prediction consistency
```

and:

```text
stable inference
```

A model may appear behaviorally consistent while still relying on shallow statistical heuristics.

This result provides early evidence that:

```text
high benchmark-style accuracy
≠
stable reasoning behavior.
```

The experiment therefore supports the broader hypothesis of the project:

```text
correctness alone is insufficient
for evaluating reasoning systems.
```

---

# Long-Term Direction

This experiment serves as an early foundation for:

- semantic stability evaluation,
- perturbation-grounded reasoning analysis,
- representation stability research,
- controlled reasoning environments,
- and Causal Inferential Yield (CIY).

The long-term goal is to move from:

```text
benchmark correctness
```

toward:

```text
stable mechanism-level inference.
```

---

# Conceptual Transition

```text
Static Benchmark Evaluation
↓
Semantic Perturbation
↓
Inference Stability
↓
Mechanism Consistency
↓
Toward Stable Reasoning Evaluation
```
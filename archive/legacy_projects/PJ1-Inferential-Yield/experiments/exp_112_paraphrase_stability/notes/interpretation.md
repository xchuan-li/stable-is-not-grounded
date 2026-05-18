

# Interpretation

## Overview

This experiment investigates whether inference remains stable when semantic meaning is preserved but surface wording changes.

The model was trained only on the original sentence forms and evaluated on both:

- original sentences,
- and paraphrased versions.

The paraphrases preserved the underlying semantic meaning while changing lexical choice and sentence structure.

---

## Experimental Results

| Metric | Result |
|---|---|
| Original Accuracy | 1.00 |
| Paraphrase Accuracy | 0.75 |
| Consistency Rate | 0.75 |

The model achieved perfect accuracy on the original examples but showed reduced performance on paraphrased inputs.

This indicates that high performance on the original distribution does not necessarily imply stable semantic inference.

---

## Main Observation

Several paraphrases caused prediction inconsistency even though the semantic meaning was preserved.

Examples:

| Original | Paraphrase |
|---|---|
| All airplanes can fly. | Airplanes are capable of flight. |
| Healthy birds can usually fly. | Birds in good health are generally able to fly. |

These failures suggest that the model partially relies on lexical surface patterns rather than stable semantic abstraction.

---

## Failure Pattern Analysis

The unstable examples involved:

- synonym replacement,
- structural reformulation,
- predicate abstraction,
- and modal wording changes.

For example:

- `can fly` → `capable of flight`
- `healthy` → `in good health`
- `usually` → `generally`

Although these transformations preserve meaning for humans, they may produce substantially different lexical representations for the model.

This suggests that semantic equivalence does not necessarily correspond to representation-level invariance.

---

## Research Interpretation

The experiment demonstrates an early form of semantic instability.

The model remains accurate under the original wording distribution but becomes less stable when semantically equivalent reformulations are introduced.

This supports the broader Inferential Yield hypothesis:

```text
accuracy ≠ stable inference
```

The results suggest that reasoning evaluation should include:

- semantic-preserving perturbation,
- paraphrase consistency,
- and inference stability analysis,

rather than relying only on benchmark correctness.

---

## Connection to the CIY Framework

This experiment provides an initial prototype for:

- semantic stability evaluation,
- perturbation-grounded reasoning analysis,
- and stable inference measurement.

The experiment also establishes an early foundation for:

- perturbation taxonomy,
- semantic instability analysis,
- representation drift research,
- and future controlled reasoning environments.
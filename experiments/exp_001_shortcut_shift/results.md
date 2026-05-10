# Results: Exp 001 Shortcut Shift

## IID vs Shift Performance

| Split | Accuracy |
|---|---|
| IID Test | 0.91 |
| Shift Test | 0.00 |

---

## Observation

The model achieved high IID accuracy but completely failed under shortcut distribution shift.

This indicates that standard in-distribution accuracy does not guarantee stable inference.

---

## Feature Importance Analysis

Top weighted features:

```text
red:  +6.54
blue: -6.81
```

This demonstrates that the classifier primarily relied on color-based shortcut correlations.

---

## Interpretation

The intended reasoning structure was removed from the input.

As a result, the model learned the spurious correlation:

```text
red -> yes
blue -> no
```

rather than learning a stable inference mechanism.

When the shortcut correlation was reversed during testing, model performance collapsed completely.

---

## Implication

This experiment suggests that benchmark accuracy alone cannot distinguish:

- stable mechanism-level inference
from:
- shortcut-based statistical behavior.

This serves as the first empirical motivation for stability-aware reasoning evaluation.

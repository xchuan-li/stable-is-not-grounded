# Failure Taxonomy: exp_341_shortcut_failure_patterns

This file defines the first-pass failure labels used by `analyze_failures.py`.

The taxonomy is intentionally small. Its purpose is to separate structured shortcut-related failures from generic incorrect predictions.

## Labels

| Failure Type | Definition | Diagnostic Meaning |
|---|---|---|
| `correct` | The model prediction matches the gold label | No failure |
| `shortcut_following` | The model predicts the shortcut-implied label instead of the rule-implied label | The model relies on the unstable shortcut |
| `reversal_confusion` | Under shift or reversal, the model follows the old training shortcut mapping | The model carries training-environment dependence into a changed environment |
| `causal_feature_ignored` | The rule label is available but the model predicts against it | The intended stable feature is not controlling prediction |
| `environment_specific_adaptation` | The shortcut is misaligned and the model fails | The model is sensitive to environment-specific correlations |
| `unclear_failure` | The failure does not match the current categories | The taxonomy needs refinement |

## Current Observed Pattern

The current run finds only one failure type:

```text
shortcut_following
```

Observed count:

```text
reversal_test / E_reversal: 450
```

## Interpretation

The model does not merely lose accuracy under shortcut reversal.

It fails in a consistent direction: predictions follow the shortcut label rather than the intended rule label.

This makes shortcut reversal a useful first case for PJ3.4 because the failure is both behaviorally visible and taxonomically interpretable.

## Next Taxonomy Extensions

- Add semantic instability labels from paraphrase perturbation.
- Add representation collapse labels once PJ3.2 produces drift measurements.
- Add intervention fragility labels once activation intervention experiments are implemented.
- Track whether failure types recur across models, datasets, and perturbation types.

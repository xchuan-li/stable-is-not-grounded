# Results: exp_341_shortcut_failure_patterns

## Summary

This experiment analyzes model failures under shortcut perturbation.

Instead of only reporting IID and reversal accuracy, the experiment examines the structure of incorrect predictions.

The model achieves perfect accuracy on the training and IID test environments, but collapses under shortcut reversal.

---

## Accuracy by Split

| Split | Accuracy |
|---|---:|
| train | 1.00 |
| iid_test | 1.00 |
| reversal_test | 0.10 |

---

## Failure Type Counts

| Failure Type | Count |
|---|---:|
| shortcut_following | 450 |

---

## Main Observation

The model does not fail randomly under shortcut reversal.

All observed failures are classified as `shortcut_following`.

This means that when the shortcut-label correlation is reversed, the model continues to predict according to the shortcut association learned during training rather than following the intended semantic rule.

---

## Interpretation

The result shows a strong shortcut dependence pattern.

The model performs perfectly when the shortcut remains aligned with the label, but fails when the shortcut becomes misaligned.

This suggests that the model's high IID accuracy does not reflect stable inference. Instead, it reflects successful exploitation of shortcut-label correlations.

The collapse under reversal is therefore not merely an accuracy drop. It is a structured failure mode.

---

## Connection to Stable Inference

A model with stable inference should preserve the intended semantic or causal rule under valid perturbation.

In this experiment, the semantic rule remains available, but the model's behavior changes when the shortcut-label correlation is reversed.

This indicates that the model's prediction mechanism is sensitive to shortcut features rather than stable semantic structure.

Therefore, shortcut-following failures provide evidence that benchmark accuracy alone is insufficient for evaluating reasoning stability.

---

## Connection to CIY

Causal Inferential Yield requires conclusions to remain stable under perturbation and intervention.

The observed shortcut-following failures indicate low causal inferential stability because the generated conclusion depends on unstable shortcut features rather than mechanism-preserving semantic structure.

This supports the broader CIY claim that reasoning should be evaluated by stable conclusion generation under perturbation, not by IID accuracy alone.

---

## Files Generated

- `failure_log.csv`
- `failure_summary.csv`
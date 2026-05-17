# Exp 113: Activation Intervention

## Research Question

Which internal representations support shortcut-based predictions, and can intervention on those representations change model behavior?

## Motivation

Exp 111 shows behavioral collapse under shortcut shift.
Exp 112 shows instability under semantic-preserving paraphrase.

This experiment is the planned bridge from behavioral instability to mechanism-level analysis.

## Core Hypothesis

If a model relies on shortcut-sensitive internal features, then targeted activation interventions should change predictions more strongly than interventions on irrelevant or stable-rule features.

## Planned Setup

Use paired examples where the intended rule remains fixed while shortcut features vary.

The experiment will compare:

- original activations,
- shortcut-feature interventions,
- stable-rule-feature interventions,
- and random or control interventions.

## Planned Metrics

| Metric | Purpose |
|---|---|
| Prediction flip rate | Measures behavioral sensitivity to intervention |
| Intervention effect size | Measures how strongly an activation edit changes output |
| Shortcut sensitivity | Measures whether shortcut-related edits dominate |
| Stability under control edits | Checks whether effects are specific rather than random |

## Expected Contribution

This experiment should connect:

- shortcut failure,
- representation instability,
- and causal intervention analysis.

The result will help decide how future versions of Stable Inference Score should include an intervention layer.

## Status

Planned. No implementation or result files are present yet.

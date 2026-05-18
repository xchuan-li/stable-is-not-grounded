
# Exp 112: Paraphrase Stability

## Research Question

Does inference remain stable when semantic meaning is preserved but surface wording changes?

## Hypothesis

If a model relies on stable semantic reasoning, then its prediction should remain consistent across paraphrased versions of the same input.

If the model relies on surface-level lexical patterns, then paraphrasing may change the prediction even when the meaning remains the same.

## Results

| Metric | Score |
|---|---|
| Original Accuracy | 1.00 |
| Paraphrase Accuracy | 0.75 |
| Consistency Rate | 0.75 |

## Main Observation

The model remained accurate on original inputs but became unstable under semantic-preserving paraphrases.

Several paraphrases involving:
- structural reformulation,
- lexical abstraction,
- and modal wording changes

caused prediction inconsistency.

This suggests that the model partially relies on lexical surface patterns rather than stable semantic reasoning.
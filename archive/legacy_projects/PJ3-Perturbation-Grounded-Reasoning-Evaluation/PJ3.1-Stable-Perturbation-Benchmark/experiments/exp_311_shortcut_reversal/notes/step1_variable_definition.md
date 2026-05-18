# Step 1 — Variable Definition

## Experiment

exp_311_shortcut_reversal

## Goal

Test whether shortcut reversal breaks inference stability.

## Stable Feature C

C is the semantic/rule-based feature that should determine the label.

Current setting:

- bird-like entity / flying rule -> yes
- non-flying entity -> no

The intended mechanism is:

```text
bird -> can fly
```

## Shortcut Feature S
S is a non-causal surface feature.

Current setting:
```text
S ∈ {red, blue}
```
Color should not causally determine whether an entity can fly.

## Label Y

```text
Y ∈ {yes, no}
```
- yes = can fly
- no = cannot fly

## Environments
### E_train
```text
red  -> yes
blue -> no
```

### E_iid
```text
red  -> yes
blue -> no
```

### E_reversal
red  -> no
blue -> yes

## Core Hypothesis
If the model learns the shortcut feature S, it will achieve high IID accuracy but low reversal accuracy.

If the model learns the stable semantic feature C, it should remain accurate under shortcut reversal.

## Interpretation

High IID accuracy is not sufficient evidence of stable inference.

The key diagnostic is the gap between IID performance and reversal performance.
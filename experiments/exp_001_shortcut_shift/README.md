# Exp 001: Shortcut Shift

## Research Question

Do models preserve rule-based inference under shortcut distribution shift?

## Hypothesis

A model trained in an environment where color is spuriously correlated with the label may achieve high IID accuracy but fail under shortcut shift.

## Intended Mechanism

The intended reasoning rule is:

```text
bird -> can fly
## Shortcut Feature

During training, color is correlated with the label:

red -> yes
blue -> no

This correlation is not causal.

## Evaluation

We compare:

IID accuracy
Shift accuracy

If IID accuracy is high but shift accuracy is low, this suggests shortcut-based behavior rather than stable inference.

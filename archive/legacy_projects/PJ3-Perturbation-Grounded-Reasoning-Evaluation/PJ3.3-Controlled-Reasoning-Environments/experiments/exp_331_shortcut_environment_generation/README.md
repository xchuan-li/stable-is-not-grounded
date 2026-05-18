# Exp 331: Shortcut Environment Generation

## Goal

This experiment investigates whether shortcut-dependent reasoning environments can be systematically generated.

The objective is to move from manually constructed shortcut examples toward reusable controlled environment generation.

---

## Core Idea

The environment generator creates datasets where:

- shortcut features correlate strongly with labels during training,
- but reverse or disappear during evaluation.

This enables controlled analysis of shortcut dependence and inference instability.

---

## Environment Structure

### Train Environment

Shortcut correlation:

```text
red -> yes
blue -> no
```

### Shift Environment

Shortcut correlation reverses:

```text
red -> no
blue -> yes
```

The underlying causal structure remains unchanged.

---

## Current Components

- controllable shortcut correlation
- train / iid / shift environment generation
- environment metadata tracking
- shortcut reversal pipeline

---

## Files

```text
data/
  train_env.csv
  iid_test_env.csv
  shift_test_env.csv
src/
  generate_envs.py
```

## Current Status

Complete first pass.

The current generator produces train, IID, and shift environments for shortcut-dependence experiments. The next step is to make the generator more reusable across PJ3.1 benchmark runs and PJ3.4 failure-case analysis.

---

## Research Importance

This experiment forms the foundation for:

- controlled reasoning environments
- perturbation-aware evaluation
- shortcut robustness analysis
- causal reasoning infrastructure

within the PJ3.3 roadmap.

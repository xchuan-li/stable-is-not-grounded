

# Paper 1 Outline

# Title (Working Version)

```text
Stable Inference under Perturbation:
Toward Stability-Aware Reasoning Evaluation
```

---

# Core Thesis

```text
Benchmark correctness
does not necessarily imply
stable inference.
```

Language models may achieve:

- high IID accuracy,
- strong benchmark performance,
- and apparently correct outputs,

while still relying on:

- shortcut correlations,
- unstable semantic patterns,
- or fragile distribution-specific behavior.

This paper argues that:

```text
reasoning evaluation
should measure inference stability
rather than correctness alone.
```

---

# Main Contributions

## Contribution 1

Introduce:

```text
Stable Inference
```

as a reasoning-oriented evaluation concept.

---

## Contribution 2

Demonstrate that:

- shortcut distribution shift,
- and semantic-preserving paraphrase perturbation

can significantly alter model behavior despite preserved task semantics.

---

## Contribution 3

Propose:

```text
Stable Inference Score (SIS)
```

as an early prototype for stability-aware reasoning evaluation.

---

## Contribution 4

Establish the early conceptual foundation for:

- perturbation-grounded reasoning evaluation,
- semantic stability analysis,
- and future Causal Inferential Yield (CIY) research.

---

# Paper Structure

---

# 1. Introduction

## Core Problem

Current benchmark evaluation primarily measures:

```text
final correctness
```

rather than:

```text
whether reasoning remains stable
under perturbation and distribution change
```

---

## Motivation

Modern language models often achieve high benchmark performance.

However:

- shortcut learning,
- distribution shift,
- and semantic perturbation

may reveal instability in the underlying reasoning process.

---

## Core Argument

This paper proposes that:

```text
correctness
and
stable inference
are different properties
```

A model may produce correct outputs while still relying on unstable mechanisms.

---

## Paper Goal

The goal of this paper is to:

- investigate instability under perturbation,
- formalize the concept of Stable Inference,
- and introduce early stability-aware reasoning evaluation.

---

# 2. Related Work

## Shortcut Learning

Discuss:

- shortcut learning,
- spurious correlations,
- and distribution shift.

Potential references:

- Geirhos et al.
- IRM
- Group DRO
- OOD generalization work

---

## Robustness and Generalization

Discuss:

- robustness evaluation,
- adversarial perturbation,
- and OOD evaluation.

Main distinction:

```text
most prior work focuses on performance robustness,
not stable inference structure
```

---

## Semantic Stability

Discuss:

- paraphrase robustness,
- semantic invariance,
- and perturbation consistency.

---

## Mechanistic and Causal Perspectives

Discuss:

- causal reasoning,
- mechanism-aware evaluation,
- and representation stability.

---

# 3. Stable Inference Framework

## Stable Inference Definition

Introduce:

```text
Stable Inference as Verified Minimal Causal Structure
```

Core definition:

```text
Stable inference is verified causal-support stability
under valid perturbations,
interventions,
and new environments.
```

---

## Minimal Causal Structure (MCS)

Explain:

- candidate causal variables,
- perturbation,
- intervention,
- and causal-support stability.

---

## Semantic-Causal Decomposition

Discuss:

- semantic variables,
- semantic structure,
- and candidate causal support.

Potential future connections:

- dependency parsing,
- SRL,
- AMR.

---

# 4. Experimental Setup

## Experiment 1 — Shortcut Shift

Research question:

```text
Does high IID accuracy imply stable reasoning?
```

Setup:

- shortcut-correlated training distribution,
- IID evaluation,
- shortcut-reversed shift evaluation.

Metrics:

- IID accuracy,
- shift robustness.

---

## Experiment 2 — Paraphrase Stability

Research question:

```text
Does semantic-preserving reformulation
preserve inference consistency?
```

Setup:

- original examples,
- semantic-preserving paraphrases,
- consistency evaluation.

Metrics:

- original accuracy,
- paraphrase consistency.

---

# 5. Experimental Results

## Shortcut Instability

Main observation:

```text
high IID accuracy
can coexist with catastrophic shortcut collapse
```

---

## Semantic Instability

Main observation:

```text
semantic equivalence
does not necessarily imply
inference equivalence
```

Examples:

```text
The bird can fly.
↓
The bird is capable of flight.
```

---

## Perturbation Sensitivity

Analyze:

- lexical perturbation,
- structural reformulation,
- predicate abstraction,
- and compound semantic perturbation.

---

# 6. Stable Inference Score (SIS)

## Motivation

Explain why:

```text
accuracy alone
is insufficient
```

for reasoning evaluation.

---

## SIS Prototype

Introduce:

```text
Stable Inference Score (SIS)
```

Prototype formulation:

```text
SIS = (IID + Shift + Paraphrase) / 3
```

---

## Interpretation

Explain:

- stable vs unstable inference,
- shortcut-dependent behavior,
- and semantic instability.

---

## Current Limitations

Discuss:

- simplicity of current formulation,
- lack of intervention-level validation,
- missing representation-level analysis,
- and future metric directions.

---

# 7. Toward Stability-Aware Reasoning Evaluation

Introduce the broader research direction:

```text
reasoning evaluation
should move beyond correctness
and toward stability analysis
```

---

## Multi-Layer Stability

Potential future layers:

| Layer | Example |
|---|---|
| Behavioral | prediction consistency |
| Semantic | paraphrase invariance |
| Structural | dependency / SRL preservation |
| Semantic Graph | AMR invariance |
| Representation | latent stability |
| Intervention | causal consistency |

---

## Future CIY Direction

Discuss:

- perturbation taxonomy,
- representation drift,
- semantic preservation,
- intervention evaluation,
- and mechanism-aware reasoning analysis.

---

# 8. Conclusion

## Main Claim

This paper argues that:

```text
correctness
is not sufficient
for reasoning evaluation
```

---

## Final Insight

Reasoning systems should be evaluated not only by:

```text
whether they answer correctly
```

but also by:

```text
whether inference remains stable
under controlled perturbation
and environmental change
```

---

# Future Extensions

Future work may include:

- perturbation taxonomy,
- semantic graph stability,
- representation drift analysis,
- intervention-grounded reasoning evaluation,
- and full Causal Inferential Yield formalization.
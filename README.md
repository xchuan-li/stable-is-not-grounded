# Inferential-Yield

This project investigates whether current reasoning
benchmarks truly measure stable inference under
distribution shift.

The project explores:
- shortcut-based reasoning failure,
- mechanism-level inference,
- intervention robustness,
- and stability-aware evaluation metrics
for language models.

---
# Research Plan: Inferential Yield / Stable Inference

## 1. Core Thesis
Benchmark correctness does not necessarily imply stable inference.

## 2. Research Goal
Define and evaluate whether model conclusions remain stable under perturbation, distribution shift, paraphrase, and future intervention tests.

## 3. Key Research Questions
- Does high IID accuracy imply stable inference?
- Do semantic-preserving perturbations preserve predictions?
- Can shortcut-dependent failures be detected systematically?
- How should Stable Inference Score be formalized?

## 4. Current Evidence
- Exp 111: Shortcut shift shows high IID accuracy but collapse under shift.
- Exp 112: Paraphrase stability shows semantic reformulation can change predictions.
- PJ3.1 Exp 311: Cleaner shortcut reversal benchmark with held-out pseudowords.
- PJ3.4 Exp 341: Failure cases mostly classified as shortcut-following.

## 5. Current Metric Prototype
Describe SIS:

SIS = (IID + Shift + Paraphrase) / 3

Then list limitations:
- behavioral only
- no intervention layer yet
- no representation drift yet
- paraphrase set still small

## 6. Work Packages
### PJ1
Conceptual framework, metric, paper writing.

### PJ3.1
Stable perturbation benchmark.

### PJ3.2
Representation stability toolkit.

### PJ3.3
Controlled reasoning environment generation.

### PJ3.4
Causal failure casebook.

## 7. Next Milestones
- Clean PJ1 README.
- Expand paraphrase experiment.
- Add research references.
- Finish Paper 1 draft.
- Build PJ3.2 representation stability prototype.
- Formalize SIS v2.

## 8. Risks and Open Problems
- Toy experiments may be too simple.
- Need stronger baselines or LLM experiments.
- Need clearer distinction from robustness/OOD work.
- Need formal definition of valid perturbation.

## 9. Paper Plan
Paper 1: Stable Inference under Perturbation.
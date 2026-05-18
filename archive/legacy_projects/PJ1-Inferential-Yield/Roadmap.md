

# Inferential Yield / CIY Project Roadmap

This document records the execution route of the PJ1 / PJ3 research program.

The roadmap is organized according to the actual research progression, not by repository folder. The core trajectory is:

```text
Benchmark insufficiency
↓
Perturbation stability
↓
Semantic stability
↓
Stable inference metric
↓
Representation stability
↓
Controlled reasoning environments
↓
Failure mechanism analysis
↓
Causal Inferential Yield theory
↓
Mechanism-aware training
↓
Toward causal language models
```

---

## Current Project Status

As of now, the project has completed the first empirical and conceptual foundation loop:

- Completed `1.1 exp_111_shortcut_shift`
- Completed `3.1.1 exp_311_shortcut_reversal`
- Completed `3.3.1 exp_331_shortcut_environment_generation`
- Completed Paper Reading items:
  - Shortcut Learning in Deep Neural Networks
  - Invariant Risk Minimization
- Completed `1.3 Define Stable Inference`
- Completed `3.4.1 exp_341_shortcut_failure_patterns`
- Completed `1.4 exp_112_paraphrase_stability`
- Completed `1.5 Stable Inference Metric Prototype`
- Completed first-pass documentation cleanup for PJ1 / PJ3 project entry points
- Completed first Paper 1 outline scaffold

The current stage is the end of the first stable-inference foundation cycle.

The project is now at the transition point from:

```text
behavioral perturbation evidence
```

to:

```text
paper drafting + representation/mechanism analysis
```

---

## Progress Checkpoint

Current position:

```text
Phase 1 foundation cycle: completed first pass
Phase 1 optional strengthening: pending
Phase 1-2 representation transition: ready to start
Paper 1: outline exists, full draft not yet written
Paper 2: roadmap/outline stage
```

| Area | Status | Notes |
|---|---|---|
| Shortcut-shift evidence | ✅ Completed | `exp_111_shortcut_shift` shows IID/shift instability |
| Controlled shortcut benchmark | ✅ Completed | `exp_311_shortcut_reversal` strengthens the shortcut-collapse result |
| Controlled environment generator | ✅ Completed | `exp_331_shortcut_environment_generation` provides reusable environment infrastructure |
| Stable inference definition | ✅ Completed | `1.3 Define Stable Inference` provides the conceptual frame |
| Shortcut failure taxonomy seed | ✅ Completed | `exp_341_shortcut_failure_patterns` classifies shortcut-following failures |
| Paraphrase stability evidence | ✅ Completed | `exp_112_paraphrase_stability` adds semantic-stability evidence |
| Stable Inference Score prototype | ✅ Completed | `1.5 Stable Inference Metric Prototype` defines the first SIS version |
| PJ1 / PJ3 documentation cleanup | ✅ Completed first pass | Project-level README consistency has been improved |
| Paper 1 outline | ✅ Completed first pass | Next step is turning outline into full prose |
| GroupDRO reading | ⏳ In progress | Still useful for robustness / worst-group comparison |
| Paraphrase invariance benchmark | ⏳ Pending | Optional strengthening for Paper 1 |
| Semantic preservation control | ⏳ Pending | Needed for Paper 3 and stronger perturbation validity |
| Activation intervention | ⏳ Pending | First mechanism-level bridge |
| Representation drift | ⏳ Pending | First Paper 2 experiment |

Practical interpretation:

```text
You have finished the core evidence loop for Paper 1.
You have not yet finished the full Paper 1 manuscript.
The next research frontier is representation drift and intervention analysis.
```

---

# Phase 1 — Stable Inference Foundations

## Phase Goal

Establish the basic claim that benchmark correctness is not equivalent to stable inference.

This phase answers:

- Can a model achieve high IID accuracy while failing under shortcut shift?
- Does inference remain stable under semantic-preserving reformulation?
- Can stable inference be defined and measured operationally?
- Can early failure cases be organized into a useful instability pattern?

The main output of this phase is the foundation for Paper 1: **Stable Inference under Perturbation**.

---

## 1.1 `exp_111_shortcut_shift` ✅ Completed

### Purpose

Build the first toy experiment showing that high IID accuracy does not imply stable inference.

### Core Idea

A classifier is trained in an environment where shortcut features correlate strongly with labels. The model performs well on IID data but collapses when the shortcut-label relation is reversed.

### Research Meaning

This experiment establishes the first empirical basis for:

```text
accuracy ≠ stable inference
```

### Role in Paper 1

Required experiment for Paper 1.

---

## 3.1.1 `exp_311_shortcut_reversal` ✅ Completed

### Purpose

Extend the shortcut-shift idea into a controlled perturbation benchmark setting.

### Core Idea

Construct a shortcut reversal setting where the shortcut feature is predictive during training but misleading during evaluation.

### Research Meaning

This strengthens the claim that model success can depend on unstable correlations rather than semantic or causal structure.

### Role in Paper 1

Recommended extension for Paper 1.

---

## 3.3.1 `exp_331_shortcut_environment_generation` ✅ Completed

### Purpose

Build reusable controlled shortcut environments.

### Core Components

- Train / IID / shift split generation
- Controllable shortcut correlation
- Shortcut reversal generation
- Environment metadata tracking
- CSV environment generation pipeline
- Reusable documentation

### Research Meaning

This turns the project from one-off toy experiments into reusable controlled environment infrastructure.

### Role in Paper 1 / Paper 3

- Supports Paper 1 empirically
- Becomes a core infrastructure component for Paper 3: **Controlled Perturbation Environments for Reasoning Evaluation**

---

## 1.2 Paper Reading ✅ Partially Completed

### Purpose

Build the theoretical foundation connecting shortcut learning, OOD generalization, invariant learning, and stable inference.

### Completed Papers

#### Paper 1: Shortcut Learning in Deep Neural Networks

Status: Completed as Paper A.

Main contribution to project:

- Provides the conceptual foundation for shortcut dependence.
- Supports the claim that models can exploit superficial correlations.
- Directly motivates `exp_111`, `exp_311`, and `exp_341`.

#### Paper 2: Invariant Risk Minimization

Status: Completed as Paper A / high-priority causal ML foundation.

Main contribution to project:

- Provides the invariance principle behind stable inference.
- Connects environments, mechanisms, and OOD generalization.
- Supports later work on invariant representation learning and mechanism regularization.

#### Paper 3: GroupDRO

Status: In progress.

Expected role:

- Supports robustness and worst-group performance understanding.
- Helps compare stable inference against group robustness methods.

---

## 1.3 Define Stable Inference ✅ Completed

### Purpose

Define stable inference as an operational research object.

### Current Definition

Stable inference is the verified stability of a causal hypothesis under valid perturbations, interventions, and new environments.

### Core Chain

```text
Input
↓
Candidate causal variables
↓
MCS-candidate
↓
Perturbation invariance test
↓
Intervention dependency test
↓
New environment generalization
↓
Verified MCS
↓
Stable inference
↓
CIY
```

### Research Meaning

This step converts the project from a robustness observation into a causal-inference evaluation framework.

### Role in Paper 1 / Paper 5

- Supports Paper 1 definition section
- Becomes theoretical foundation for Paper 5: **Causal Inferential Yield**

---

## 3.4.1 `exp_341_shortcut_failure_patterns` ✅ Completed

### Purpose

Analyze failure patterns caused by shortcut perturbation.

### Core Idea

Instead of treating failures as isolated errors, classify them as structured responses to shortcut reversal or shortcut removal.

### Research Meaning

This creates the first bridge from perturbation evaluation to failure taxonomy.

### Role in Paper 4

Foundational experiment for Paper 4: **Toward a Taxonomy of Reasoning Failures**.

---

## 1.4 `exp_112_paraphrase_stability` ✅ Completed

### Purpose

Test whether inference survives semantic-preserving reformulation.

### Core Idea

The underlying meaning remains stable while the surface form changes.

Example:

```text
All birds can fly.
Every bird is capable of flight.
```

### Research Meaning

This introduces semantic stability as a separate dimension from shortcut robustness.

### Role in Paper 1

Required semantic-stability component for Paper 1.

---

## 3.1.2 `exp_312_paraphrase_invariance` ⏳ Next / Pending

### Purpose

Extend paraphrase stability into a controlled perturbation benchmark.

### Core Question

Does inference remain stable under semantic reformulation?

### Expected Output

- Paraphrase perturbation pipeline
- Semantic equivalence checking
- Inference consistency evaluation

### Role in Paper 1

Recommended extension for Paper 1.

---

## 3.3.2 `exp_332_semantic_preservation_control` ⏳ Pending

### Purpose

Develop controlled semantic-preserving perturbation pipelines.

### Core Question

Which perturbations preserve semantic structure?

### Expected Output

- Paraphrase generation
- Semantic equivalence filtering
- Distractor insertion
- Lexical perturbation
- Reasoning-path reformulation

### Role in Paper 3

Core experiment for controlled reasoning environments.

---

## 1.5 Stable Inference Metric Prototype ✅ Completed

### Purpose

Build the first operational metric for stable inference.

### Core Dimensions

- Conclusion preservation
- Distribution robustness
- Paraphrase consistency
- Environment invariance
- Intervention stability

### Research Meaning

This turns stable inference from a concept into a measurable object.

### Role in Paper 1 / Paper 5

- Required for Paper 1
- Early metric foundation for CIY theory

---

## 3.4.2 `exp_342_semantic_instability_cases` ⏳ Pending

### Purpose

Collect and analyze failures under semantic-preserving transformations.

### Core Question

Which failures indicate semantic instability?

### Expected Output

- Semantic instability cases
- Paraphrase failure logs
- Conclusion inconsistency examples
- Early taxonomy categories

### Role in Paper 4

Second core component for failure taxonomy.

---

## 1.6 `exp_113_activation_intervention` ⏳ Pending

### Purpose

Move from behavioral analysis to mechanistic analysis.

### Core Question

What mechanisms cause inference collapse?

### Planned Methods

- Activation patching
- Causal tracing
- Representation intervention
- Probing

### Role in Roadmap

This is the first transition from output-level stability to internal mechanism analysis.

---

## 1.7 Toward Causal Reasoning Framework ⏳ Pending

### Purpose

Integrate Phase 1 findings into a broader causal reasoning framework.

### Core Goal

Move from:

```text
behavioral robustness
```

toward:

```text
mechanistic causal reasoning
```

### Expected Output

- Workshop-paper framing
- Thesis-direction framing
- PhD research agenda framing

---

# Phase 1–2 Transition

## Phase Goal

Bridge early perturbation and semantic stability into representation-level mechanism analysis.

---

## 3.2.1 `exp_321_representation_drift` ⏳ Pending

### Purpose

Measure how internal representations change under perturbation.

### Core Question

How much do representations drift under perturbation?

### Planned Methods

- Hidden state extraction
- Cosine similarity
- Representation distance metrics
- Embedding trajectory analysis
- Token-level activation comparison

### Role in Paper 2

First required experiment for Paper 2: **Representation Drift and Reasoning Stability**.

---

## 3.1.3 `exp_313_reasoning_path_shift` ⏳ Pending

### Purpose

Test whether reasoning remains stable across alternative reasoning paths.

### Core Question

Does reasoning remain stable when different reasoning paths lead to the same conclusion?

### Role in Paper 1 / Paper 2

- Optional extension for Paper 1
- Supports later reasoning-path stability analysis in Paper 2

---

# Phase 2 — Mechanisms and Causal Reasoning

## Phase Goal

Move from behavioral robustness toward mechanism-level understanding of stable inference.

The phase asks:

- Which internal representations remain invariant?
- Which layers preserve stable reasoning information?
- Can intervention reveal causal reasoning mechanisms?
- Do models encode causal relational structure?

---

## 3.2.2 `exp_322_layerwise_invariance` ⏳ Pending

### Purpose

Analyze which transformer layers preserve invariant reasoning information.

### Role in Paper 2

Required experiment for representation stability paper.

---

## 3.3.3 `exp_333_intervention_environment_design` ⏳ Pending

### Purpose

Construct environments designed for intervention-based reasoning analysis.

### Role in Paper 3

Builds intervention-ready controlled reasoning environments.

---

## 1.8 `exp_121_reasoning_path_stability` ⏳ Pending

### Purpose

Investigate whether inference consistency is preserved across different reasoning trajectories.

### Role in Paper 2 / Paper 5

Supports reasoning-path stability as a CIY dimension.

---

## 3.1.4 `exp_314_prompt_framing_shift` ⏳ Pending

### Purpose

Test whether prompt framing changes reasoning behavior while the underlying task remains identical.

### Perturbations

- Urgency framing
- Authority framing
- Emotional pressure
- Role prompting
- Compliance pressure

### Role in Paper 1 / Paper 4

Supports perturbation taxonomy and failure taxonomy.

---

## 3.2.3 `exp_323_intervention_response` ⏳ Pending

### Purpose

Analyze how representations respond to activation-level intervention.

### Role in Paper 2 / Paper 4

Bridge between mechanistic interpretability and stable inference analysis.

---

## 1.9 `exp_122_representation_invariance` ⏳ Pending

### Purpose

Investigate which internal representations remain stable under perturbation and distribution shift.

### Role in Paper 2

Core PJ1 representation analysis component.

---

## 3.4.3 `exp_343_representation_collapse_analysis` ⏳ Pending

### Purpose

Study whether reasoning failure correlates with representation instability.

### Role in Paper 4

Connects behavioral failure with latent representation collapse.

---

## 3.3.4 `exp_334_multi_environment_reasoning` ⏳ Pending

### Purpose

Evaluate reasoning consistency across multiple perturbation environments.

### Role in Paper 3 / Paper 5

Supports environment-level stability and CIY evaluation.

---

## 1.10 `exp_123_counterfactual_intervention` ⏳ Pending

### Purpose

Investigate whether counterfactual intervention can reveal causal reasoning mechanisms.

### Role in Paper 5

Supports causal mechanism validation for CIY.

---

## 3.2.4 `exp_324_latent_cluster_structure` ⏳ Pending

### Purpose

Study whether stable reasoning states form identifiable latent clusters.

### Role in Paper 2

Adds latent topology analysis to representation stability.

---

## 3.4.4 `exp_344_intervention_failure_response` ⏳ Pending

### Purpose

Analyze which interventions trigger reasoning collapse.

### Role in Paper 4

Supports intervention-sensitive failure taxonomy.

---

## 3.1.5 `exp_315_distribution_shift` ⏳ Pending

### Purpose

Evaluate reasoning robustness across multiple perturbation environments simultaneously.

### Role in Paper 1

Potential final strengthening experiment for Paper 1.

---

## 1.11 `exp_124_causal_structure_probing` ⏳ Pending

### Purpose

Investigate whether language models encode latent causal relational structure.

### Role in Paper 5

Core experiment for the CIY theory paper.

---

## 3.2.5 `exp_325_representation_failure_modes` ⏳ Pending

### Purpose

Identify which representation shifts correlate with inference collapse.

### Role in Paper 2 / Paper 4

Connects representation drift with structured failure modes.

---

# Phase 2–3 Transition

## Phase Goal

Move from mechanism analysis into synthetic causal reasoning tasks and structured failure taxonomy.

---

## 3.3.5 `exp_335_synthetic_causal_reasoning_tasks` ⏳ Pending

### Purpose

Develop synthetic reasoning tasks with explicitly controlled relational and causal structure.

### Role in Paper 3 / Paper 5

Provides controlled causal reasoning environments for CIY.

---

## 3.4.5 `exp_345_failure_taxonomy_construction` ⏳ Pending

### Purpose

Construct a systematic taxonomy of reasoning failures.

### Initial Failure Categories

- Shortcut Collapse
- Semantic Drift
- Framing Dependence
- Representation Collapse
- Intervention Fragility
- Path Inconsistency

### Role in Paper 4

Final required step for the reasoning failure taxonomy paper.

---

# Phase 3 — Toward Causal Language Models

## Phase Goal

Move from evaluating stable inference toward constructing mechanism-aware reasoning systems.

This phase asks:

- Can models learn invariant reasoning representations?
- Can causal data augmentation improve stable inference?
- Can stable reasoning mechanisms be regularized?
- Can language models encode explicit causal structure?

---

## 1.12 `exp_131_invariant_representation_learning` ⏳ Pending

### Purpose

Train or evaluate models that learn representations stable across perturbation environments.

### Role in Paper 6

Core experiment for CIY applications and mechanism-aware evaluation.

---

## 1.13 `exp_132_causal_data_augmentation` ⏳ Pending

### Purpose

Use intervention-based augmentation to improve inference robustness.

### Role in Paper 6

Tests whether CIY-inspired data design improves stable inference.

---

## 1.14 `exp_133_mechanism_regularization` ⏳ Pending

### Purpose

Regularize models toward stable reasoning mechanisms.

### Planned Ideas

- IRM-style regularization
- Representation consistency loss
- Causal objective design
- Invariance constraints

### Role in Paper 6

Tests mechanism-aware training objectives.

---

## 1.15 `exp_134_structural_reasoning_models` ⏳ Pending

### Purpose

Investigate whether language reasoning can explicitly incorporate causal structural representations.

### Planned Directions

- SCM-inspired reasoning architectures
- Graph-based reasoning
- Structural latent variables
- Causal abstraction layers
- Symbolic-causal hybrid systems

### Role in Paper 6

Final long-term transition toward causal language models.

---

# Publication Mapping

## Paper 1 — Stable Inference under Perturbation

### Main Claim

High benchmark accuracy does not imply stable reasoning.

### Core Steps

- `1.1 exp_111_shortcut_shift` ✅
- `3.1.1 exp_311_shortcut_reversal` ✅
- `1.4 exp_112_paraphrase_stability` ✅
- `3.1.2 exp_312_paraphrase_invariance` ⏳
- `1.5 Stable Inference Metric Prototype` ✅
- Optional: `3.1.3`, `3.1.4`, `3.1.5`

### Current Status

Paper 1 has completed its core foundation loop and first outline scaffold.

Current estimate:

```text
research foundation: strong first pass
paper outline: completed first pass
full manuscript: not yet written
workshop-level readiness: approaching, but not complete
```

The main missing parts are stronger paraphrase-invariance benchmarking, clearer metric experiments, related work consolidation, figures/tables, and a coherent full-paper narrative.

---

## Paper 2 — Representation Drift and Reasoning Stability

### Main Claim

Stable reasoning should correspond to partially stable latent representations.

### Core Steps

- `3.2.1 exp_321_representation_drift`
- `3.2.2 exp_322_layerwise_invariance`
- `1.8 exp_121_reasoning_path_stability`
- `1.9 exp_122_representation_invariance`
- Optional: `3.2.3 exp_323_intervention_response`

---

## Paper 3 — Controlled Perturbation Environments for Reasoning Evaluation

### Main Claim

Reasoning stability requires controlled experimental environments rather than uncontrolled static benchmarks.

### Core Steps

- `3.3.1 exp_331_shortcut_environment_generation` ✅
- `3.3.2 exp_332_semantic_preservation_control`
- `3.3.3 exp_333_intervention_environment_design`
- `3.3.4 exp_334_multi_environment_reasoning`
- `3.3.5 exp_335_synthetic_causal_reasoning_tasks`

---

## Paper 4 — Toward a Taxonomy of Reasoning Failures

### Main Claim

Reasoning failures are structured manifestations of unstable latent mechanisms.

### Core Steps

- `3.4.1 exp_341_shortcut_failure_patterns` ✅
- `3.4.2 exp_342_semantic_instability_cases`
- `3.4.3 exp_343_representation_collapse_analysis`
- `3.4.4 exp_344_intervention_failure_response`
- `3.4.5 exp_345_failure_taxonomy_construction`

---

## Paper 5 — Causal Inferential Yield

### Main Claim

Reasoning can be evaluated as stable, non-trivial, intervention-consistent conclusion generation under information constraints.

### Core Steps

- CIY proposal refinement
- `1.5 Stable Inference Metric Prototype` ✅
- `1.10 exp_123_counterfactual_intervention`
- `1.11 exp_124_causal_structure_probing`
- `3.3.5 exp_335_synthetic_causal_reasoning_tasks`
- Earlier evidence from Papers 1–4

---

## Paper 6 — CIY Applications & Mechanism-Aware Reasoning Evaluation

### Main Claim

CIY can guide mechanism-aware training, augmentation, and causal reasoning model design.

### Core Steps

- `1.12 exp_131_invariant_representation_learning`
- `1.13 exp_132_causal_data_augmentation`
- `1.14 exp_133_mechanism_regularization`
- `1.15 exp_134_structural_reasoning_models`

---

# Immediate Next Steps

## Recommended next execution order

1. ✅ Review and consolidate completed Phase 1 materials.
2. ✅ Clean first-pass documentation for completed components:
   - ✅ `1.1 exp_111_shortcut_shift`
   - ✅ `3.1.1 exp_311_shortcut_reversal`
   - ✅ `3.3.1 exp_331_shortcut_environment_generation`
   - ✅ `1.3 Define Stable Inference`
   - ✅ `3.4.1 exp_341_shortcut_failure_patterns`
   - ✅ `1.4 exp_112_paraphrase_stability`
   - ✅ `1.5 Stable Inference Metric Prototype`
3. ✅ Begin Paper 1 outline.
4. ⏳ Turn Paper 1 outline into a full draft.
5. ⏳ Start `3.1.2 exp_312_paraphrase_invariance`.
6. ⏳ Start `3.3.2 exp_332_semantic_preservation_control`.
7. ⏳ Finish reading and note-taking for GroupDRO.
8. ⏳ Start `3.2.1 exp_321_representation_drift` for the Paper 2 transition.

## Immediate Focus

The next practical milestone is:

```text
Paper 1 full draft + optional paraphrase-invariance strengthening
```

After that, the project should move into:

```text
representation drift analysis for Paper 2
```

---

# Current Knowledge Gaps to Study

## High Priority

- OOD generalization
- Invariant Risk Minimization
- GroupDRO
- Shortcut learning
- Semantic equivalence / paraphrase evaluation
- Basic causal inference: intervention, confounding, counterfactuals
- Basic representation analysis: embeddings, hidden states, cosine similarity

## Medium Priority

- Mechanistic interpretability basics
- Activation patching
- Causal tracing
- Probing methods
- Representation similarity analysis
- Dataset construction and benchmark design

## Later Priority

- Causal representation learning
- Structural causal models
- Information theory basics
- Minimum description length
- Invariant representation learning
- Mechanism regularization

---

# Core Research Identity

The project is not merely about improving robustness.

The long-term research identity is:

```text
mechanism-aware causal reasoning evaluation for language models
```

The central claim remains:

```text
Reasoning is not correctness.
Reasoning is the preservation of stable causal mechanisms under perturbation and intervention.
```

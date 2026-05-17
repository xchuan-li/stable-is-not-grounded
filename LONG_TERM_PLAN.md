# Long-Term Research Plan: Inferential-Yield → Causal Reasoning for Language

## Vision Statement

Transition from **behavioral reasoning robustness** toward **mechanistic causal reasoning systems** that learn stable inference mechanisms instead of statistical shortcuts.

---

## Phase Structure

### Phase 1: Diagnosis & Evaluation (Current)
**Goal:** Characterize the failure modes of current reasoning systems

| Experiment | Focus | Status |
|---|---|---|
| **exp_111_distribution_shift** | Do models fail under distribution shift? | Foundational |
| **exp_112_semantic_perturbation** | Does paraphrase break inference? | Planning |
| **exp_113_activation_intervention** | Which internal representations cause collapse? | Mechanistic bridge |

**Contribution:** Establish that IID correctness ≠ stable inference

---

### Phase 2: Mechanistic Understanding (Q2-Q3 2026)
**Goal:** Understand the internal mechanisms driving shortcut dependence

| Experiment | Focus | Core Question |
|---|---|---|
| **exp_121_reasoning_path_stability** | Do different reasoning paths collapse identically? | Does reasoning stability require path invariance? |
| **exp_122_representation_invariance** | Which latent representations remain stable? | representation invariance → inference stability? |
| **exp_123_counterfactual_intervention** | Which interventions break inference? | Which interventions expose causal dependencies? |
| **exp_124_causal_structure_probing** | Do models encode causal structure? | Can we extract causal relations from LLMs? |

**Research Questions:**
- Do models learn "causal modules" that remain stable?
- Can we probe for latent causal representations?
- Which internal structures support robust reasoning?

**Expected Outputs:**
- Mechanistic probing toolkit
- Causal structure extraction pipeline
- Representation stability benchmarks

---

### Phase 3: Intervention & Improvement (Q3-Q4 2026)
**Goal:** Develop methods that encourage stable reasoning

| Experiment | Focus | Approach |
|---|---|---|
| **exp_131_invariant_representation_learning** | Can we train models with invariant representations? | Multi-environment training + contrastive learning |
| **exp_132_causal_data_augmentation** | Does causal augmentation improve stability? | Counterfactual + intervention-based data generation |
| **exp_133_mechanism_regularization** | Can we regularize for mechanism consistency? | IRM-style objectives + causal loss functions |

**Research Questions:**
- Which training objectives encourage causal reasoning?
- Can we design loss functions that reward mechanism stability?
- Does multi-environment training with causal objectives work?

**Expected Outputs:**
- Mechanism-aware training framework
- Causal augmentation pipeline
- Empirical comparison of causal vs. standard training

---

### Phase 4: Architecture & Design (Q4 2026 - Q1 2027)
**Goal:** Design language reasoning systems that encode explicit causal structure

| Experiment | Focus | Vision |
|---|---|---|
| **exp_134_structural_reasoning_models** | Can we build causal-aware LLM architectures? | SCM-inspired + graph-based + symbolic-causal hybrid |

**Research Questions:**
- What architectural changes enable causal reasoning?
- Can we integrate SCM-style computations into transformers?
- How do we represent causal structure explicitly in models?

**Expected Outputs:**
- Prototype causal language models
- Architectural design patterns for causal reasoning
- Benchmarks comparing standard vs. causal architectures

---

### Phase 5: Integration & Framework (Q1-Q2 2027)
**Goal:** Synthesize findings into unified causal reasoning framework

| Milestone | Content | Scope |
|---|---|---|
| **1.7 Toward Causal Reasoning Framework** | Meta-synthesis of all experiments | Unify stability, invariance, mechanism robustness |
| **Paper 1: Stable Inference under Perturbation** | Diagnostic + mechanistic findings | Phases 1-2 results |
| **Paper 2: Causal Training for Robust Reasoning** | Intervention + improvement findings | Phase 3 results |
| **PhD/Thesis Direction** | Long-term research agenda | Complete framework |

---

## Key Research Themes

### Theme 1: Shortcut Learning → Mechanism Learning
```
Standard ML:  prediction optimization
Causal ML:   mechanism optimization
```

### Theme 2: Surface Statistics → Latent Structure
```
Shallow:  statistical patterns
Deep:     causal dependencies + invariant relations
```

### Theme 3: Behavioral → Mechanistic Understanding
```
Phase 1:  What fails?
Phase 2:  Why does it fail?
Phase 3:  How to fix it?
Phase 4:  How to design it right?
```

---

## Literature Connections

### Foundation Papers
1. **Shortcut Learning** (Geirhos et al.)
   - Models rely on spurious correlations
   - Distribution shift exposes shortcuts

2. **Invariant Risk Minimization** (Arjovsky et al.)
   - Learn representations invariant across environments
   - Causal features should be stable

3. **GroupDRO** (Sagawa et al.)
   - Worst-case group robustness
   - Environment-aware training

### Emergent Directions
- Causal representation learning
- Mechanistic interpretability
- Intervention-based evaluation
- Causal NLP

---

## Success Metrics

### Phase 1-2 Outputs
- [ ] Characterize shortcut failure modes
- [ ] Identify representation instability sources
- [ ] Extract causal structure from models

### Phase 3 Outputs
- [ ] Stable training improves perturbation robustness
- [ ] Causal augmentation outperforms standard augmentation
- [ ] Mechanism objectives outperform prediction objectives

### Phase 4 Outputs
- [ ] Causal architectures show measurable stability gains
- [ ] Explicit causal structure improves interpretability
- [ ] Causal models maintain robustness across domains

### Phase 5 Outputs
- [ ] Unified framework for causal reasoning
- [ ] Publication in top-tier venues
- [ ] Open-source toolkit for practitioners

---

## Connection to PJ3 Subprojects

| PJ3 Component | PJ1 Contribution |
|---|---|
| **PJ3.1: Stable Perturbation Benchmark** | Phase 1 diagnostics feed into benchmark design |
| **PJ3.2: Representation Stability Toolkit** | Phase 2 probing enables toolkit development |
| **PJ3.3: Controlled Reasoning Environments** | Environments generated for training experiments (Phase 3) |
| **PJ3.4: Causal Failure Casebook** | Failures documented and analyzed mechanistically |

---

## Timeline & Milestones

| Date | Phase | Key Milestone |
|---|---|---|
| **May 2026** | Phase 1 | exp_111 + exp_112 complete |
| **June 2026** | Phase 1-2 | exp_113 reveals mechanistic insights |
| **July 2026** | Phase 2 | exp_121, 122 papers written internally |
| **August 2026** | Phase 2-3 | exp_131 training framework ready |
| **Sept 2026** | Phase 3 | exp_132, 133 results + Paper 1 draft |
| **Oct 2026** | Phase 3-4 | exp_134 architectural prototypes |
| **Dec 2026** | Phase 4 | Causal model benchmarks established |
| **Jan 2027** | Phase 5 | Framework synthesis begins |
| **Feb 2027** | Phase 5 | Paper submissions |
| **March 2027** | Phase 5 | Thesis proposal using framework |

---

## Deliverables

### Academic
- **Paper 1:** Stable Inference under Perturbation
- **Paper 2:** Causal Training for Robust Reasoning
- **Thesis:** Causal Language Models for Stable Reasoning
- **Workshop:** Causal NLP for Robust Language Understanding

### Open Source
- **Benchmark:** Stable Perturbation Benchmark (integrated with PJ3.1)
- **Toolkit:** Representation Probing + Causal Extraction Tools
- **Environments:** Controlled Reasoning Environments Library
- **Models:** Causal Language Model Implementations

### Community
- Documentation and tutorials
- Pre-trained models and benchmarks
- Integration with ML safety community
- Alignment with mechanistic interpretability research

---

## Long-Term Vision (2027+)

### Year 1-2: Foundation
- Establish causal reasoning as core ML capability
- Build community around stable inference

### Year 2-3: Application
- Deploy causal reasoning in high-stakes domains (medical, legal, scientific)
- Extend to multimodal causal reasoning

### Year 3+: Integration
- Become standard component of robust AI systems
- Enable interpretable + stable + generalizable models

---

## Success Criteria

**Inferential-Yield achieves success when:**

1. **Scientific:** Research demonstrates that causal mechanisms can be identified, measured, and optimized in language models
2. **Methodological:** Practical toolkit enables others to build causal reasoning systems
3. **Impact:** Published work is cited in leading venues and influences practice
4. **Community:** Framework adopted by researchers working on robust AI

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Models lack encodable causal structure | Early probing experiments (exp_124) will reveal this |
| Causal training is computationally prohibitive | Start with synthetic environments; scale carefully |
| Architectural constraints prevent integration | Design modular extensions rather than full rewrites |
| Community skeptical of causal claims | Build from solid empirical foundation first |

---

## Related Research Areas

- **Mechanistic Interpretability:** Understanding internal model computations
- **Causal Inference:** Identifying causal relations from observational data
- **Robustness:** Generalizing under distribution shift
- **Interpretability:** Making model decisions transparent
- **AI Safety:** Ensuring models behave as intended
- **Domain Adaptation:** Transfer across domains
- **Few-shot Learning:** Learning with limited data
- **Symbolic AI:** Integrating structured reasoning


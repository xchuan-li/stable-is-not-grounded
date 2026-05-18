# Stable Inference under Perturbation
## Toward Perturbation-Grounded Reasoning Evaluation

---

# Abstract

Modern language models often achieve strong benchmark performance while still exhibiting unstable reasoning behavior under distribution shifts, paraphrases, and semantic perturbations.

Current evaluation protocols primarily measure correctness on static benchmarks, but correctness alone may fail to capture whether a model preserves stable inference mechanisms.

In this work, we investigate the distinction between benchmark correctness and stable inference.

We argue that "stable inference" is not a primitive notion and only becomes well-defined once it is anchored to a causal structure. We introduce **Minimal Causal Structure (MCS)** — a minimal, non-redundant premise set that is causally robust under interventions on variables outside it — and define **Causal Inferential Yield (CIY)** as a model's ability to recover a conclusion *in units of* its MCS: correctly, minimally, and stably under intervention.

This framework supplies a non-circular operationalization of the otherwise circular notion of an "irrelevant" variable, by separating a causal question answered *by construction* (synthetic items whose ground-truth graph we stipulate) from a statistical question answered *empirically* (training-set correlation). The perturbation-grounded constructs used in our experiments — the Stable Inference Score (SIS), shortcut-reversal environments, distractor robustness, and semantic-preservation validation — are recast as special cases of CIY.

Experiments using TF-IDF + Logistic Regression and fine-tuned DistilBERT baselines reveal that models may appear stable under naive evaluation while relying on shallow heuristics such as positive-label bias; in the framework's terms, their apparent sufficiency is not robust under interventions on shortcut-correlated variables.

We further show that perturbations themselves require semantic validation, since invalid perturbations may create artificial instability unrelated to reasoning behavior.

Our results suggest that benchmark correctness alone is insufficient for evaluating reasoning systems and motivate a shift toward causally grounded, perturbation-based stable inference evaluation.

---

# 1. Introduction

Large language models have achieved remarkable performance across a wide range of reasoning benchmarks.

However, high benchmark accuracy does not necessarily imply stable reasoning behavior.

Models may rely on:

- lexical shortcuts,
- spurious correlations,
- annotation artifacts,
- or shallow statistical heuristics

while still achieving strong IID benchmark performance.

This creates a major evaluation problem:

```text
benchmark correctness
≠
stable inference
```

A reasoning system should ideally preserve its conclusions when:

- wording changes,
- surface forms shift,
- distractors are inserted,
- or semantically equivalent perturbations are introduced.

However, many existing evaluation pipelines primarily measure:

```text
final-answer correctness
```

rather than:

```text
mechanism-level inference stability.
```

In this work, we investigate stable inference under perturbation.

We define stable inference as:

```text
the invariance of a conclusion under transformations that
preserve its minimal causal structure,
together with the appropriate variance under
transformations that alter it.
```

This definition is made precise in Section 3, which is the theoretical core of the paper: it introduces Minimal Causal Structure (MCS), a causal-robustness clause distinguishing intervention from mere conditioning, a three-way taxonomy of non-structural variables, a non-circular operationalization of "irrelevant," and Causal Inferential Yield (CIY) as the central measure.

To study this empirically, we then construct a perturbation-grounded evaluation that instantiates the framework through:

1. shortcut shift experiments,
2. shortcut reversal environments (the canonical intervention test),
3. semantic-preserving perturbation benchmarks,
4. semantic-preservation validation,
5. and CIY proxies, including the Stable Inference Score.

Our experiments reveal that models may exhibit:

- high apparent consistency,
- yet severe instability under balanced negative-control evaluation.

In particular, we observe that naive consistency aggregation can hide shallow statistical behavior such as:

```text
positive-label bias
```

rather than genuine semantic reasoning stability.

We further show that perturbations themselves must be validated semantically, since invalid perturbations may introduce artificial instability unrelated to model reasoning.

Overall, our results suggest that:

```text
reasoning evaluation
should move beyond static benchmark correctness
and toward perturbation-grounded stable inference analysis.
```

---

# 2. Related Work

## 2.1 Shortcut Learning

Modern neural networks are known to exploit spurious statistical correlations rather than causal structure.

Prior work on shortcut learning and Clever Hans phenomena demonstrates that models often achieve strong benchmark performance through shallow heuristics.

Relevant directions include:

- shortcut learning,
- spurious correlation analysis,
- dataset artifacts,
- and annotation bias.

Potential citations:

- Geirhos et al.
- Clever Hans effect literature
- dataset artifact papers

---

## 2.2 OOD Generalization and Robustness

A major research direction investigates robustness under distribution shift.

Prior work explores:

- covariate shift,
- invariant risk minimization,
- group DRO,
- adversarial robustness,
- and out-of-distribution generalization.

However, many robustness benchmarks primarily evaluate distributional robustness rather than semantic inference stability.

Potential citations:

- IRM
- GroupDRO
- OOD generalization literature

---

## 2.3 Perturbation-Based Evaluation

Several prior works use paraphrases, adversarial attacks, or controlled perturbations to probe model behavior.

However, perturbation validity itself is often under-specified.

Many perturbations may unintentionally alter:

- inference guarantees,
- causal direction,
- reasoning difficulty,
- or semantic certainty.

This creates ambiguity between:

```text
true reasoning instability
```

and:

```text
artifact instability
caused by invalid perturbations.
```

Our work introduces an explicit semantic-preservation validation layer to address this issue.

---

# 3. A Causal Framework for Stable Inference

This section is the theoretical core of the paper. We argue that "stable inference" is not a primitive notion: it only becomes well-defined once it is anchored to a *causal structure*. We introduce Minimal Causal Structure (MCS) as that anchor, define a robustness clause that separates causal grounding from correlational shortcutting, give a non-circular operationalization of the otherwise circular notion of an "irrelevant" variable, and finally define Causal Inferential Yield (CIY) as a measure taken *in units of* MCS. The empirical constructs used elsewhere in this paper — perturbation stability, the Stable Inference Score, the shortcut-reversal and distractor experiments — are then recast as special cases of this single framework.

---

## 3.1 Why "Stable Inference" Needs a Causal Anchor

A naive reading of stable inference is "the model gives the same answer when the wording changes." This is insufficient in both directions:

- A model can be **stably wrong**: invariant across paraphrases yet systematically incorrect. Pure invariance rewards this.
- A model can be **correctly unstable for good reasons**: if a perturbation genuinely changes the causal content (e.g., negating a premise), the answer *should* change.

So stability is only meaningful *relative to what is supposed to be held fixed*. The thing that must be held fixed is the **causal structure that licenses the conclusion**. We therefore define stable inference as:

```text
the invariance of a conclusion under transformations that
preserve its minimal causal structure,
together with the appropriate variance under
transformations that alter it.
```

This makes the causal structure, not the surface form, the unit of evaluation.

---

## 3.2 Minimal Causal Structure (MCS)

Let `C` be a target conclusion. A **Minimal Causal Structure for C**, written `MCS(C)`, is a set of premises `M` such that:

```text
(i)   Sufficiency:   under the stipulated causal model, M is jointly
                      sufficient to license C.
(ii)  Non-redundancy: every premise in M is counterfactually necessary —
                      removing it means C is no longer licensed.
(iii) Causal robustness: the sufficiency in (i) is invariant under
                      interventions on variables outside M
                      (defined precisely in 3.3–3.5).
```

Clauses (i)–(ii) are a recognizable notion with a long lineage, which we make explicit because it both situates the contribution and bounds its novelty:

- **Mackie's INUS condition** (1965): a cause is an *insufficient but non-redundant part of an unnecessary but sufficient* condition. Clause (ii) is precisely the non-redundancy requirement.
- **Halpern & Pearl actual causation** (2005): the minimality / but-for clause in their definition is exactly "remove the element and the effect no longer follows."
- **Prime implicants / minimal models** in propositional logic: `M` is a prime-implicant-like minimal sufficient set.

Two consequences of minimality must be stated to avoid a standard objection:

1. **Minimality is not unique.** A conclusion may admit several distinct minimal sufficient sets (multiple INUS clusters; multiple prime implicants). We therefore speak of *a* minimal causal structure and quantify over the *set* of them, never "the" structure.
2. **Minimality is background-relative.** The necessity of a premise is always relative to a set of held-fixed background assumptions. In this work the background is fixed *by construction* (see 3.5), which is what makes the definition usable.

Clause (i)–(ii) alone, however, only yields *deductive* minimality (a minimal premise set that entails `C`). The gap between "minimal premises that entail C" and "minimal causal structure that produces C" is exactly the correlation-versus-causation gap this paper targets. That gap is closed by clause (iii).

---

## 3.3 The Robustness Clause: Conditioning vs. Intervention

Clause (iii) demands that the sufficiency of `M` survive manipulation of anything *outside* `M`. The decisive subtlety is that "manipulation" must be read at the **interventional** level of Pearl's ladder, not the observational one:

```text
conditioning / see :   P(C | M, Z = z)        — Z observed to take value z
intervention / do  :   P(C | M, do(Z = z))    — Z forced to z, severed
                                                 from its own causes
```

A genuine `MCS(C)` requires: for every variable `Z` outside `M` and every admissible `do(Z = z)`, with `M` held fixed and present, the licensing of `C` from `M` is unchanged. If some `do(Z = z)` flips the conclusion while `M` is intact, then either `Z` was in fact relevant (`M` was wrong), or the model never used `M` and was riding a correlation with `Z`.

This distinction is operationally critical: a manipulation that merely *adds observed information* tests rung-1 robustness; only a manipulation that *sets a variable while breaking its natural correlations* tests the rung-2 property that makes a structure causal.

---

## 3.4 A Taxonomy of Non-Structural Variables

Whether a manipulation of something outside `M` is diagnostic depends on which of three classes the manipulated variable belongs to. Crucially, the class is **not** a property of the world's causal graph alone; it is defined *relative to the training distribution of the model under test*.

| Class | Definition | Intervening on it tests |
|---|---|---|
| 1. Relevant | In the stipulated causal graph, an ancestor/mediator/effect-modifier of `C` (e.g., an alternative cause) | Nothing about shortcutting — `do()` *should* change `C`; excluded from clause (iii) |
| 2. Pure noise | Causally irrelevant to `C` **and** statistically independent of `C` in the model's training data | Parsing / attention robustness (rung-1) |
| 3. Shortcut-correlated | Causally irrelevant to `C` **but** spuriously correlated with `C` in the model's training data | Whether the model's sufficiency is genuinely `M`-grounded (rung-2) — **this is the clause (iii) target** |

The same surface token can fall in different classes for different models: a phrase is "noise" only if it carries no training-time correlation with the target; if it does, it silently becomes a class-3 shortcut. Conflating class 2 and class 3 lets a shortcut-reliant model pass a robustness test it should fail.

---

## 3.5 A Non-Circular Operationalization of "Irrelevant"

Clause (iii) appears circular: deciding which variables are "outside" `M` seems to presuppose the very causal structure we are testing the model for. The escape is to separate two questions that have independent, non-circular answers:

```text
(a) Is Z causally irrelevant to C?
    Answered BY CONSTRUCTION. The benchmark items are synthetic;
    we stipulate their ground-truth causal graph. We are not
    discovering a causal fact about the world — we are testing
    whether the MODEL recovers a structure we defined. No circularity.

(b) Is Z entangled with C in the model's training data?
    Answered EMPIRICALLY. This is a correlation measured directly
    on the training set. It requires no causal assumption.
```

A variable is a legitimate clause-(iii) intervention target iff it is causally irrelevant by construction *(a)* **and** training-correlated *(b)* — i.e., class 3 of 3.4.

**Honest boundary.** Step (b) requires an auditable training distribution. This holds for models we train ourselves (the TF-IDF and fine-tuned DistilBERT baselines). For frontier LLMs whose pretraining corpus is unobservable, (b) degrades from measurement to estimation, and the framework's guarantees weaken accordingly. We state this as a principled limitation rather than overclaiming generality.

---

## 3.6 Causal Inferential Yield (CIY)

We can now define the central measure. Naively, "inferential yield" as a raw count of correct conclusions is ill-posed: under deductive closure the number of derivable truths is unbounded and inflatable (`P ⊢ P∧P ⊢ P∨Q …`), the classic logical-omniscience / relevance problem. Yield becomes well-defined only when it is measured *in units of MCS*.

For a target conclusion `C` with minimal causal structure `M = MCS(C)`, **Causal Inferential Yield** is the model's ability to:

```text
1. recover C when given exactly M;
2. WITHHOLD C when given a strict subset of M
   (a real reasoner knows the conclusion is not yet licensed);
3. still recover C under do(Z) for any class-2 or class-3
   variable Z, with M held fixed (clause (iii) robustness).
```

Components 1–3 are not three metrics but one: yield is correct, *minimal*, and *intervention-stable* recovery of `C`. This collapses the project's apparently separate experiments into instances of a single quantity:

| Existing construct | Recast as |
|---|---|
| Distractor perturbation | CIY component 3 with a **class-2** variable (rung-1 robustness) |
| Shortcut-reversal experiment | CIY component 3 with a **class-3** variable (rung-2, clause (iii)) — the canonical causal test |
| Stable Inference Score (below) | A scalar aggregate of CIY components 1 and 3 |
| Sub-premise / negative-control items | CIY component 2 (withholding) |

The shortcut-reversal result (Section 4.2: accuracy collapses to 0.000 under reversal) is, in this language, a direct measurement that the baseline's apparent sufficiency is **not** robust under a class-3 `do()` — i.e., its `M` was correlational, not causal. The experiment was already a clause-(iii) test; the framework makes that explicit.

### 3.6.1 The Stable Inference Score as a special case

The Stable Inference Score used in our experiments is the current, deliberately simple scalar proxy for CIY:

```text
group_score(g) = correct_predictions(g) / total_samples(g)
SIS_v1         = mean over all groups of group_score(g)
```

It is **accuracy-based**, not consistency-based: a purely consistency-based score would reward stably-wrong behavior (a model that is invariant but always wrong), which is exactly the failure mode 3.1 warns against. SIS_v1 grounds the score against ground-truth labels, so it captures CIY components 1 and 3 but **not** component 2 (withholding under partial premises) and does not separate label bias from genuine stability — under label imbalance a positive-biased model scores well on positive groups while failing all negative ones. We therefore additionally report a dual metric (accuracy together with intervention-consistency and a stable-correct / stable-wrong / unstable group taxonomy) in the experiments; SIS_v1 should be read as a lower-bound proxy for CIY, not its definition.

---

## 3.7 Semantic Preservation Validation

We further introduce a semantic validation layer for perturbation evaluation.

Each perturbation is explicitly annotated according to:

- semantic validity,
- reasoning preservation,
- perturbation risk,
- and semantic change type.

Examples of invalid perturbations include:

| Failure Type | Example |
|---|---|
| quantifier shift | `all` → `most` |
| uncertainty injection | `is taller` → `usually taller` |
| modal strength shift | `could` → `definitely` |
| causal reversal | `fire causes smoke` → `smoke causes fire` |

This validation layer distinguishes:

```text
true reasoning instability
```

from:

```text
artifact instability
caused by invalid perturbations.
```

---

# 4. Experiments

## 4.1 Shortcut Shift

We first construct shortcut-shift environments to evaluate whether models preserve reasoning under shortcut distribution changes.

Training data is generated with a shortcut strength of 0.9: the spurious color feature (red/blue) is correlated with the label 90% of the time.

Under IID evaluation (same shortcut distribution), the TF-IDF + Logistic Regression model achieves:

```text
IID accuracy = 0.910
```

The top two TF-IDF features are the color tokens (`red: +6.54`, `blue: −6.81`), confirming that the model has learned the shortcut rather than the intended causal feature (animal type).

---

## 4.2 Shortcut Reversal

We construct controlled shortcut-reversal environments where shortcut-label correlations are inverted.

Under full reversal (reversal strength = 1.0), the model collapses to:

```text
Shift accuracy = 0.000
```

To demonstrate that this collapse is not specific to full reversal, we additionally evaluate across a range of reversal strengths:

| Reversal Strength | Shift Accuracy |
|---|---|
| 0.0 (IID) | 1.000 |
| 0.1 | 0.900 |
| 0.3 | 0.687 |
| 0.5 | 0.460 |
| 0.7 | 0.257 |
| 0.9 | 0.103 |
| 1.0 (full reversal) | 0.000 |

Accuracy degrades monotonically as the shortcut correlation is reversed, confirming that the model's performance is driven entirely by the spurious feature rather than the causal reasoning structure.

---

## 4.3 Semantic-Preserving Perturbation Benchmark

We construct perturbation groups across multiple reasoning categories:

- taxonomic reasoning,
- transitive reasoning,
- causal reasoning,
- commonsense reasoning,
- and contradiction reasoning.

Each group contains multiple semantic-preserving variants.

---

## 4.4 Negative-Control Evaluation

Balanced negative-control groups reveal that models may exhibit strong positive-label bias.

Evaluation using TF-IDF + Logistic Regression baselines on the 10-group perturbation benchmark produced:

```text
SIS_v1 = 0.400
```

The per-answer breakdown exposes the underlying mechanism:

| Answer Type | Accuracy |
|---|---|
| yes | 0.958 |
| no | 0.028 |

The model achieves near-perfect accuracy on positive-answer groups while failing almost entirely on negative-answer groups.

Note on evaluation scope: the TF-IDF + LR model is trained on a shortcut-rich bird/penguin classification task and evaluated on a structurally different perturbation benchmark covering taxonomic, transitive, causal, commonsense, and contradiction reasoning. This cross-domain setup means the model's SIS_v1 = 0.400 primarily reflects the interaction between training-time label biases and the new evaluation distribution, rather than a direct measure of reasoning ability on the perturbation benchmark. This is an acknowledged limitation of using a task-specific classical baseline.

This demonstrates that:

```text
high apparent accuracy on positive-answer groups
may emerge from shallow label bias
rather than stable semantic inference.
```

---

## 4.5 Transformer Baseline Evaluation

To evaluate whether perturbation instability is merely a classical machine learning artifact, we additionally fine-tuned a DistilBERT classifier on the same reasoning task.

The model was trained on a balanced 40-example dataset covering five reasoning categories (taxonomic, causal, transitive, commonsense, contradiction), then evaluated on 12 perturbation groups spanning all five categories.

The transformer baseline achieved:

```text
BERT_SIS_v1 = 0.617
```

This demonstrates that meaningful perturbation instability persists even under a transformer-based neural architecture, despite substantially stronger representation capacity than TF-IDF + Logistic Regression.

The per-reasoning-type breakdown reveals that stability is not uniform across reasoning categories:

| Reasoning Type | Accuracy |
|---|---|
| contradiction | 1.000 |
| causal | 0.900 |
| commonsense | 0.600 |
| taxonomic | 0.400 |
| transitive | 0.400 |

Contradiction and causal reasoning proved relatively robust, while taxonomic and transitive reasoning showed substantially lower stability.

The per-perturbation-type breakdown shows that reasoning-path reformulation remains the hardest perturbation type:

| Perturbation Type | Accuracy |
|---|---|
| distractor | 0.750 |
| original | 0.750 |
| paraphrase | 0.667 |
| lexical_shift | 0.417 |
| reasoning_path_shift | 0.500 |

The model is most robust to distractor insertion and least robust to lexical substitution and reasoning-path reformulation, suggesting that surface wording changes are more disruptive than irrelevant information injection.

The answer-level results show a moderate imbalance:

| Answer Type | Accuracy |
|---|---|
| no | 0.700 |
| yes | 0.533 |

Unlike the TF-IDF baseline (which showed extreme positive-label bias of yes: 0.958 / no: 0.028), the DistilBERT model shows a more balanced distribution, reflecting the benefit of balanced training data. However, a non-trivial gap between yes and no accuracy remains, indicating residual label-dependent behavior.

These findings strengthen the central claim of this paper:

```text
perturbation instability
is not limited to shallow classical models.
```

Instability under semantic-preserving perturbation persists even with transformer architectures, though the pattern of instability shifts from label-bias dominance toward reasoning-category-specific and perturbation-type-specific failure modes.

---

# 5. Discussion

Our experiments suggest that benchmark correctness alone is insufficient for evaluating reasoning systems.

We identify several major challenges:

- shortcut dependence,
- perturbation validity,
- semantic instability,
- label bias,
- and evaluation ambiguity.

The results motivate a shift toward:

```text
mechanism-aware reasoning evaluation
```

rather than purely benchmark-based evaluation.

---

# 6. Limitations

The current work remains an early-stage prototype framework.

Current limitations include:

- **Small-scale perturbation datasets**: The evaluation benchmark contains 12 groups (60 samples total), which is too small for statistically robust per-category or per-perturbation-type conclusions. Per-subgroup scores are based on as few as 4–8 samples.
- **TF-IDF baseline domain mismatch**: The TF-IDF + LR baseline is trained on a different task domain than the perturbation benchmark. Its SIS_v1 primarily reflects cross-domain label bias rather than in-domain reasoning ability.
- **Limited model coverage**: Only TF-IDF + LR and DistilBERT are evaluated. Large language models and instruction-tuned models are not covered.
- **Absence of large-scale LLM evaluation**: The framework has not been applied to frontier LLMs, which may exhibit qualitatively different stability patterns.
- **SIS_v1 is accuracy-based, not mechanism-based**: The current SIS measures per-group prediction accuracy against ground truth. It does not directly measure whether the model's internal reasoning mechanism is stable, only whether its outputs are correct.
- **Manual semantic validation**: Perturbation validity annotations are currently manually assigned and have not been validated by inter-annotator agreement studies.

Future work will investigate:

- representation stability,
- intervention-based perturbations,
- causal structure probing,
- mechanism-level inference analysis,
- and large-scale LLM evaluation across the full perturbation benchmark.

---

# 7. Conclusion

We introduced a perturbation-grounded framework for evaluating stable inference under semantic-preserving transformations.

Our experiments demonstrate that:

```text
benchmark correctness
≠
stable reasoning behavior.
```

We further show that perturbation validity itself must be explicitly evaluated.

Overall, the work motivates a transition from:

```text
static benchmark evaluation
```

toward:

```text
stable mechanism-aware reasoning evaluation.
```
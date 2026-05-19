# Stable Is Not Grounded
## A Strict Accuracy–Stability–Grounding Hierarchy, and the Construction Boundary Where Its Deepest Cut Stops Being Decidable

---

> **Positioning (read first).** This paper contributes one thing: a **strict three-level decomposition of "correct" answers** — accuracy ⊋ stability ⊋ causal grounding — together with the claim, demonstrated in a controlled setting, that the *deepest* cut (a stably-correct answer that is causally grounded vs. one that merely rides a stable shortcut) is **only decidable under construction-and-auditable-training conditions**, and is not merely hard but ill-posed elsewhere. The first level (accuracy vs. stability) is known and cited; the causal machinery is borrowed and cited. What is ours is the *nested-insufficiency structure* and the *decidability boundary* — and the controlled demonstration that each containment is non-empty. §2 states exactly what is known, borrowed, and new; §7 states the scope.

---

# Abstract

A correct answer on a reasoning benchmark can fail to mean what the headline number implies, and the standard remedy — checking stability under meaning-preserving perturbation — does not go far enough. We make the conflation precise as a strict hierarchy. Headline accuracy merges three behaviours that come apart: **unstable** (the answer flips under answer-preserving presentation), **stable-wrong** (consistent and confidently incorrect), and **stable-correct**. We then observe that *stable-correct is itself ambiguous in exactly the way accuracy was*: an item can be stably correct because the model tracks the structure that licenses the conclusion, or because it stably rides a shortcut that is also correlated with the answer at evaluation. Stability is therefore **necessary but not sufficient** for causal grounding.

We resolve the residual ambiguity with a borrowed causal instrument — a do-intervention on a *measured* shortcut-correlated variable, paired with a mandatory negative ("pure-noise") control — but only where it is licensed: items whose ground-truth graph is stipulated by construction and whose training correlation is auditable. The contribution is the resulting strict hierarchy **accuracy ⊋ stability ⊋ grounding**, the demonstration that each containment is non-empty, and a structural consequence we make explicit: the depth to which the hierarchy is *observable* is a function of construction and auditability. On a real benchmark one can decide unstable / stable-wrong / stable-correct and stop; the stable-correct → {grounded, spurious} cut requires the construction and cannot be made in the wild without reintroducing the circularity it was meant to remove. The same reflexive cut recurses into the unstable and stable-wrong strata; one consequence is sharp: a stably-correct-but-spurious model and a stably-wrong-by-shortcut model are the *same mechanism*, split only by evaluation luck — so even the stable-correct / stable-wrong boundary is eval-contingent until the deepest cut is made.

We are explicit that the accuracy-vs-stability level is established prior work and that the causal machinery is not ours (§2). On three model classes the same headline accuracy and near-identical stability are shown to decompose into opposite grounding verdicts; on a real benchmark across three zero-shot models, the decomposition is computed exactly as deep as it is decidable, and no deeper. The full formal treatment of in-the-wild undecidability is deferred to a companion paper (§8).

---

# 1. Introduction

A model that answers correctly may be doing so for a reason the benchmark never checks. The now-standard response is to perturb the input in meaning-preserving ways and ask whether the answer survives: consistency, prompt-robustness, and stress-test evaluations all instantiate this. We take that response seriously and then show it is **one level too shallow**.

Decompose each item's behaviour under conservative, answer-preserving presentations:

```text
U   unstable      — the answer changes across presentations
SW  stable-wrong   — the answer is consistent and consistently wrong
SC  stable-correct — the answer is consistent and correct
```

Headline accuracy merges `U`, `SW`, and the correct part of `SC` into a single number; separating them is known and useful (§2). But the central observation of this paper is **reflexive**: `SC` is itself a conflation of two behaviours that are as different as the ones accuracy hid.

```text
SC-grounded  — stably correct because the model tracks the
               structure that licenses the conclusion
SC-spurious  — stably correct because the model stably rides a
               shortcut that is also correlated with the answer
               at evaluation time
```

A model can be perfectly stable and perfectly accurate on a slice and still be entirely `SC-spurious`. **Stability is necessary but not sufficient for grounding.** This is why a stability check, by itself, cannot certify what it appears to certify — it certifies invariance, not the *reason* for invariance.

This yields a strict hierarchy whose every containment is non-empty:

```text
accuracy  ⊋  stability  ⊋  grounding
   |            |             |
includes     excludes       excludes
SW and U     U; keeps       SC-spurious;
             SW-free SC      keeps SC-grounded
```

**This paper's single contribution is this hierarchy plus the boundary on how deep it can be observed.** Levels one and two of the *decomposition* (separating `U`/`SW`/`SC`) need only gold labels and answer-preserving perturbations, so they are computable on any benchmark — and that level is prior work, credited in §2. The deepest cut, `SC → {grounded, spurious}`, requires a causal instrument that is licensed only under two conditions: the item's ground-truth graph is **stipulated by construction**, and the candidate shortcut's training correlation is **auditable**. We argue, and demonstrate in a controlled setting, that absent those conditions the deepest cut is not merely difficult but **ill-posed** — deciding it would require the causal knowledge whose presence is the very thing in question. The boundary is therefore not a limitation of our instrument; it is a structural property of the question.

We pre-empt the deflation directly and once: the accuracy-vs-stability level is established, and the do-intervention/negative-control machinery is borrowed. What survives that deflation, and is the actual claim, is the *nested-insufficiency structure*, the *decidability boundary*, and the controlled evidence that each containment is non-empty. The same cut recurses into the other strata, decidable only as far as a licensed instrument reaches (§3.4, §4); §2 draws the borrow/known/new lines; §7 the scope.

---

# 2. Background: What Is Known, What Is Borrowed, What Is New

The contribution depends on these boundaries being explicit, so we state them before the claim.

**Known (level one is not ours).** That meaning-preserving perturbation reveals brittleness, and that consistency/accuracy decompositions are more informative than accuracy alone, is established. Model consistency under paraphrase (Elazar et al., 2021); models being "right for the wrong reasons" / heuristic-consistent (McCoy et al., 2019); large prompt-induced accuracy swings and the resulting evaluation concerns (ProSA, EMNLP 2024; "What Did I Do Wrong?", NAACL 2025; "Flaw or Artifact?", 2025); and underspecification — pipelines returning many same-accuracy predictors whose shortcut reliance varies with seed (D'Amour et al., 2022). The `U`/`SW`/`SC` split is a clean restatement of this known level; we claim no novelty for it and use it as the floor on which the new structure stands.

**Borrowed (the deepest cut's instrument is not ours).** The do/see distinction and intervention semantics (Pearl, 2009; Pearl & Mackenzie, 2018); causal-minimality / non-redundancy (Mackie, 1965 — INUS; Halpern & Pearl, 2005). Crucially, the *idea that a diagnostic is vacuous without a paired control* is itself borrowed: control-task **selectivity** in probing (Hewitt & Liang, 2019) and the **negative-control** tradition in causal inference (Lipsitch et al., 2010; proximal causal inference). Our class-2 "pure-noise" control *is* a negative control; we credit it as such rather than claim it.

**Adjacent, differentiated.** Counterfactual invariance to spurious correlations already ties "what counts as irrelevant" to the underlying causal structure (Veitch et al., 2021; Eisenstein, 2022) — but as a *training/regularization* objective, not an evaluation hierarchy, and without the decidability-boundary claim. Concept-level causal-effect estimation on model behaviour (CEBaB, 2022) uses interventions and controls on *natural* data with human counterfactuals; it estimates effect magnitudes, not a stable-correct partition, and is not by-construction. Recent shortcut-detection surveys (e.g., 2024 taxonomies) catalogue detection/mitigation but do not frame an observability boundary on a strict accuracy–stability–grounding hierarchy.

**New (ours).** (i) The *reflexive* observation that `SC` reconflates exactly what accuracy conflated, hence the strict hierarchy accuracy ⊋ stability ⊋ grounding with every containment non-empty; (ii) the **decidability boundary** — the hierarchy is observable to level two anywhere, to level three only under construction-and-auditability, and is *ill-posed* deeper in the wild; (iii) a controlled demonstration that levels are non-empty and that the boundary bites (two models with matched headline accuracy and stability, opposite grounding).

> *Bibliographic note: inline author–year identifiers are for verifiability; final reference formatting is venue-dependent. Author lists to be completed on the final pass.*

---

# 3. The Hierarchy

## 3.1 Levels one and two (assumption-free)

Fix a set of conservative, answer-preserving presentations of an item (paraphrase, lexical shift, reasoning-path reordering, a relevance-preserving distractor). With only gold labels, each item is exactly one of:

```text
U   unstable        : predictions disagree across presentations
SW  stable-wrong    : predictions agree, and are wrong
SC  stable-correct  : predictions agree, and are correct
```

`headline_accuracy − SC_fraction` is the **overstatement gap**: the share of the headline number contributed by `U` (fragility) and `SW` (confident error). This level needs no causal assumption and runs on any benchmark.

## 3.2 Level three: the reflexive cut (the central move)

`SC` answers the question "is the model invariant?" It does **not** answer "why?" Partition `SC` by the *reason* for invariance:

```text
SC-grounded : invariance survives a do-intervention that severs
              a training-correlated, causally-irrelevant variable
              from its label correlation, with M (the licensing
              structure) intact
SC-spurious : invariance does NOT survive that intervention
```

To make `SC-spurious` an investigator-independent label and not a judgement call, the candidate variable must be (a) **causally irrelevant by construction** (we author the item; its ground-truth graph is stipulated, so this is not a causal fact discovered from the model) and (b) **training-correlated by measurement** (a label correlation on the auditable training set, fixed by a threshold). A variable satisfying (a)+(b) is the legitimate target of the severing intervention; a variable satisfying (a) but training-*independent* is the **mandatory negative control** (a "pure-noise" variable) — without it, a drop under the target intervention is uninterpretable (it is consistent with "any change breaks the model"). This control logic is borrowed (Hewitt & Liang, 2019; Lipsitch et al., 2010); the use here is to make the `SC` cut non-vacuous.

The four terminal buckets, and what each means for a reported number:

| bucket | invariant? | correct? | survives sever-intervention? | what headline accuracy does with it |
|---|---|---|---|---|
| `U` | no | — | — | counts it (inflates) |
| `SW` | yes | no | — | excludes it (honest miss) |
| `SC-spurious` | yes | yes | **no** | counts it **and** stability endorses it — the doubly-hidden bucket |
| `SC-grounded` | yes | yes | yes | the only bucket that earns "grounded", and only *in scope* |

## 3.3 Strictness

Each containment is strict and we exhibit non-empty witnesses in §6: items that are accurate-but-`U` (accuracy ⊋ stability), and items that are `SC` but `SC-spurious` (stability ⊋ grounding). The hierarchy is a claim about *behaviour*, not architecture; it is defined relative to the model's training distribution, not the world's graph alone.

## 3.4 The reflexive cut recurses (U and SW)

The move that splits `SC` is not special to `SC`: *every* bucket defined by a behavioural pattern is silent on the reason, and the reason is always the same causal-vs-shortcut axis. Applying it once more:

- **`U` splits two ways.** For an unstable item, apply `do(class-3)` (sever the shortcut). If the instability disappears (`U → stable`), the flipping was **shortcut-sourced**; if it persists with the `class-2` control also U-neutral, it is **not-shortcut-explained**. Because the presentations are answer-preserving and `M` is held fixed, a flip cannot be "`M` changed" — this two-way cut is well-posed and decidable with the existing instrument.
- **`SW` splits.** `SW-shortcut`: the model rides a `class-3` cue that points to the wrong answer here — decidable by the same `do(class-3)` test (severing changes the answer). `SW-misrule`: a stable but wrong *structural* rule — not shortcut-explained; positively characterizing it needs the **class-1 arm** (does the error transform in a structured way under interventions on causally-relevant variables?), a *different* licensed instrument. `SW-degenerate`: a near-constant predictor — a trivial output-distribution check, off the causal axis.

**The sharpest consequence.** `SC-spurious` and `SW-shortcut` are *the same mechanism* — a model riding the same `class-3` shortcut — partitioned only by whether the shortcut happens to agree with the gold label on the evaluation slice. The same model, on a different slice, moves between the "good" bucket and the "bad" bucket without changing at all. So the `SC`/`SW` boundary itself is, at the mechanism level, an artefact of the evaluation distribution: headline accuracy is not merely overstated (§3.1) — the very partition that would correct it is eval-contingent until the deepest cut is made.

We deliberately do not multiply buckets beyond what a licensed instrument can decide. The contribution is not a lattice with more cells; it is that the causal-vs-shortcut question **recurses into every behavioural stratum and is decidable exactly as far as a licensed instrument reaches** (§4). Among the cuts, only the `SC` split is *load-bearing* for the overstatement thesis; the `U` and `SW` splits are *diagnostic* enrichment (which failure, not whether the headline misleads) — except the `SC-spurious ≡ SW-shortcut` identity, which is load-bearing.

---

# 4. The Decidability Boundary

This is the conceptual core and the part the companion paper formalizes.

**Level two is decidable anywhere.** `U`/`SW`/`SC` need only answer-preserving presentations and gold labels. Any benchmark qualifies.

**Level three is decidable only under construction-and-auditability.** The `SC → {grounded, spurious}` cut requires conditions (a) and (b) of §3.2. Their status differs by epistemic kind:

```text
(a) causal irrelevance  : fixed BY CONSTRUCTION. We are not
                          inferring a causal fact from the model;
                          we authored the item. Non-circular by
                          kind, not by cleverness.
(b) training correlation: fixed BY MEASUREMENT on an auditable
                          training set. No causal assumption.
```

Remove either and the cut degrades — not to "harder", but to **ill-posed**. Without (a), deciding which variable is "irrelevant" requires the causal structure whose use by the model is exactly what is in question: the original circularity returns. Without (b) — a model with unauditable pretraining — the correlation that defines `SC-spurious` cannot be measured and can only be estimated under assumptions the setting was meant to avoid. Therefore:

> **The depth to which the hierarchy is observable is a function of construction and auditability.** On a real benchmark with an opaque-pretrained model, the honest decomposition *stops at* `U`/`SW`/`SC`. Reporting a "causally grounded" fraction there is not a stronger result; it is an ill-posed one.

This reframes what is usually written as a tool's limitation into a property of the question. It also explains a recurring pattern in evaluation debates: claims about "genuine reasoning" on natural benchmarks are contested not because the measurement is noisy but because, past level two, there is no decidable quantity to measure without the construction.

**The boundary is recursive.** There is not one cut and one gate but a small set of *licensed instruments* — the `do(class-3)` + `class-2` negative control, and the `class-1` correctly-variant arm — and the causal-vs-shortcut question recurses into every stratum (§3.4), decidable exactly as far as some licensed instrument reaches and **ill-posed past it**. The severing-instrument cuts (`SC → {grounded, spurious}`, the two-way `U` cut, `SW-shortcut`) share the construction-and-auditability gate of §3.2; the `class-1` arm opens one further gate (`SW-misrule`). Deeper cuts — representational fragility vs. boundary indeterminacy inside not-shortcut-explained `U`, and the positive content of a structural mis-rule — require instruments this paper does not license (representation probes; seed-multiplicity / underspecification analysis, in the sense of D'Amour et al., 2022), and inherit the same undecidability in the wild. The boundary is not a line; it is a sequence of gates, each opened only by a licensed instrument.

---

# 5. Instantiation Protocol

A reproducible recipe for the full hierarchy in the licensed (by-construction) regime:

1. **Stipulate a DAG per item.** Ground-truth licensing structure `M` is authored; §3.2(a) holds by construction.
2. **Measure and threshold.** On the auditable training set, compute each non-target variable's label correlation; assign `class-3` (target) above a fixed threshold, `class-2` (negative control) if causally irrelevant and below it. No investigator discretion.
3. **Presentations for levels 1–2.** Build answer-preserving presentations; compute `U`/`SW`/`SC` and the overstatement gap with bootstrap 95% CIs.
4. **Sever-intervention for level 3.** On `SC` items, apply `do(class-3)` (sever the training correlation, `M` intact) and the mandatory `do(class-2)` negative control; an `SC` item is `SC-spurious` iff correctness collapses under `do(class-3)` while the `do(class-2)` control is intact.
5. **Boundary check.** On any non-constructed benchmark, run steps 3 only; report levels 1–2 and *explicitly decline* level 3 as undecidable there (this is a result, not an omission).

Semantic-preservation validation gates step 3 (quantifier/modal/causal-direction shifts excluded). A consistency-only score is rejected: it rewards `SW`.

---

# 6. Demonstration: The Levels Are Non-Empty and the Boundary Bites

All numbers below are from the existing experiment logs (see the evidence map); this section *reorganizes* the controlled results around the hierarchy and adds no new claims beyond non-emptiness and the boundary.

## 6.1 Stability ⊋ grounding: matched accuracy, opposite grounding

A synthetic task ("can X fly?") with licensing structure `M` = animal type, a training-correlated irrelevant variable `color` (|r|≈0.81 → class-3) and an independent `name` (|r|≈0.005 → class-2 control), threshold 0.10. Two regimes: **A** = `M` available; **B** = `M` suppressed so only the shortcut carries label information.

| regime | headline acc | `SC` (stable-correct) | `do(class-3)` drop | `do(class-2)` control | `SC` resolves to |
|---|---|---|---|---|---|
| A — `M` available | 1.000 | high | Δ +.000 | Δ +.000 | **`SC-grounded`** |
| B — `M` suppressed | 0.912 | high | **Δ +.408** | Δ +.021 | **`SC-spurious`** |

Headline accuracy and stability are *comparable* across A and B; the grounding verdict is **opposite**. Level two cannot tell A from B; level three can. This is the strict-containment witness for stability ⊋ grounding.

## 6.2 Not an architecture artifact

Identical protocol, three model classes (data-derived class assignment, identical across all three):

| model | regime A `do(class-3)` | A verdict | regime B `do(class-3)` | B verdict |
|---|---|---|---|---|
| TF-IDF + LR | Δ +.000 | `SC-grounded` | Δ +.408 | `SC-spurious` |
| DistilBERT (66M, full FT) | Δ +.000 | `SC-grounded` | Δ +.429 | `SC-spurious` |
| Qwen2.5-1.5B (LoRA) | Δ +.000 | `SC-grounded` | Δ +.501 | `SC-spurious` |

The negative control is intact in every cell (Δ ≈ +.00), ruling out "any intervention breaks it".

## 6.3 Non-redundancy and a correctly-variant arm

A two-premise transitive structure makes non-redundancy a real tested property; bootstrap 95% CIs, no hard thresholds. When the chain is required, ablating either premise drives withholding well above chance and a class-1 (causally relevant) break flips the answer 1.00 [1.00, 1.00]; when a strictly-more-reliable shortcut is available, ablation-withholding decouples (~chance), `do(class-3)` collapses correctness ≈0.50 [0.48, 0.53] with the negative control intact, and the class-1 flip rate collapses to 0.00. Same model, three corroborating signals: the chain-tracker is `SC-grounded` and correctly variant; the shortcut-rider is `SC-spurious`.

*Multi-seed (DistilBERT).* The sensitivity arm is byte-identical across five training seeds — expected, not a bug: at shortcut strength `s = 1.0` the shortcut is a perfectly separating feature, so the classifier converges to the same decision rule under every initialisation/batch order; with eval fixed, every metric is identical. The invariance measures how completely the shortcut dominates. The specificity arm honestly surfaces a P1/P2 premise-use asymmetry rather than a clean pass — the instrument exposing structure, not hiding it.

## 6.4 The boundary, demonstrated

On a balanced 400-item BoolQ sample under four answer-preserving presentations, three zero-shot models — opaque pretraining, no stipulated graph, so §3.2(a) and (b) both fail and level three is **undecidable here by construction of the setting**:

| quantity | Qwen2.5-1.5B | DeepSeek V4 Flash | Claude Haiku |
|---|---|---|---|
| headline accuracy | 0.763 [0.723, 0.803] | 0.883 [0.850, 0.913] | 0.905 [0.877, 0.933] |
| `SC` fraction | 0.610 [0.565, 0.660] | 0.853 [0.818, 0.888] | 0.863 [0.827, 0.895] |
| **overstatement gap** | **+0.153 [0.118, 0.188]** | **+0.030 [0.015, 0.048]** | **+0.043 [0.023, 0.065]** |
| `SC → {grounded, spurious}` | **undecidable here** | **undecidable here** | **undecidable here** |

We compute levels one and two and *stop*. The empty deepest cell is the point: the gap excludes zero for all three and shrinks but does not vanish with capability, yet *no* assumption-free procedure can say how much of each `SC` is `SC-spurious` without the construction. Reporting a grounded fraction here would not be a stronger result — it would be the ill-posed one §4 describes.

---

# 7. Limits and What We Do *Not* Claim

- Level one (accuracy vs. stability) is **not ours**; it is established (§2) and used as the floor.
- The do-intervention and the negative-control logic are **borrowed** (Pearl; Hewitt & Liang; Lipsitch). The contribution is the nested-insufficiency structure and the decidability boundary, not these instruments.
- The in-the-wild **undecidability of level three** is here *argued and demonstrated by the empty BoolQ cell*, not formally proved; the formal treatment is the companion paper's job (§8). We claim ill-posedness, not an impossibility theorem, in this paper.
- The by-construction demonstration uses template-generated items and a constructed shortcut-competition regime; it shows the levels are non-empty and the boundary bites, **not** that any particular deployed model is `SC-spurious` in the wild (that would itself require the undecidable level-three measurement).
- Every primary-claim model is fine-tuned on the distribution it is evaluated on; a portability check on a non-fine-tuned model strengthens §6 and is scripted but not yet run.

These are scope statements. Each marks a boundary the companion paper is designed to push.

---

# 8. Conclusion and the Two-Paper Split

The claim is a strict hierarchy — **accuracy ⊋ stability ⊋ grounding** — with every containment non-empty, and a structural boundary: the hierarchy is observable to level two on any benchmark and to level three only where the graph is stipulated and the training auditable; deeper in the wild it is ill-posed, not merely hard. Level one is credited as prior work; the causal instrument is credited as borrowed; what is ours is the reflexive structure, the boundary, and the controlled evidence that the levels separate (a model can be matched on accuracy *and* stability yet opposite in grounding).

This deliberately splits into two papers:

- **This paper (paper1-v2)** — the empirical hierarchy: define the levels, demonstrate non-emptiness and the boundary in the licensed regime, and decline level three on a real benchmark *as a result*. Modest, defensible, measurement-first.
- **Companion (thesis / position paper)** — the formal argument that level three is undecidable without construction-and-auditability, and the consequence that "causal-reasoning evaluation" claims on natural benchmarks are not well-posed past level two. Two deeper cuts are explicitly deferred to it: splitting not-shortcut-explained `U` into representational fragility vs. boundary indeterminacy, and the positive characterization of `SW-misrule` — both need instruments beyond the licensed set (representation probes; seed-multiplicity / underspecification analysis) and inherit the same in-the-wild undecidability. This is where the conceptual ambition belongs; it does not need a workshop methods frame to stand.

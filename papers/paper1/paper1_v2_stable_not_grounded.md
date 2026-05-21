# Stable Is Not Grounded
## A Strict Accuracy–Stability–Grounding Hierarchy, Observable Only as Deep as Its Instrument Is Licensed

---

> **Positioning (read first).** This paper contributes **one structural claim**: a strict three-level decomposition of "correct" answers — accuracy ⊋ stability ⊋ causal grounding — with every containment non-empty in a controlled setting, and with the *depth at which the hierarchy is observable* being a property of construction and auditability rather than a separate result. The first level (accuracy vs. stability) is known and cited; the causal instrument is borrowed and cited. What is ours is the *nested-insufficiency structure*, the *observability-depth* property, and the controlled demonstration that each containment is non-empty. §2 (Related Work) places the prior and borrowed pieces inline; §7 states the scope.

---

# Abstract

The hierarchy **accuracy ⊋ stability ⊋ grounding** has every containment non-empty, and the depth at which it is observable is a property of construction and auditability rather than a tool limitation. Headline accuracy on a reasoning benchmark merges three behaviours that come apart under answer-preserving perturbation — **unstable** (the answer flips), **stable-wrong** (consistent and confidently incorrect), **stable-correct** — and the standard fix of checking stability separates only the first. The residual ambiguity is itself reflexive: a stable-correct answer can hold because the model tracks the structure that licenses the conclusion, or because it stably rides a shortcut also correlated with the answer at evaluation. Stability is therefore **necessary but not sufficient** for causal grounding.

We resolve the residual ambiguity with a borrowed causal instrument — a do-intervention on a *measured* shortcut-correlated variable, paired with a mandatory negative ("pure-noise") control — but only where it is licensed: items whose ground-truth graph is stipulated by construction and whose training correlation is auditable. The contribution is the resulting strict hierarchy **accuracy ⊋ stability ⊋ grounding**, the demonstration that each containment is non-empty, and a structural consequence we make explicit: the depth to which the hierarchy is *observable* is a function of construction and auditability. On a real benchmark one can decide unstable / stable-wrong / stable-correct and stop; the stable-correct → {grounded, spurious} cut requires the construction and cannot be made in the wild without reintroducing the circularity it was meant to remove. The same reflexive cut recurses into the unstable and stable-wrong strata; one consequence is sharp: a stably-correct-but-spurious model and a stably-wrong-by-shortcut model are the *same mechanism*, split only by evaluation luck — so even the stable-correct / stable-wrong boundary is eval-contingent until the deepest cut is made.

We are explicit that the accuracy-vs-stability level is established prior work and that the causal machinery is not ours (§2). On three model classes the same headline accuracy and near-identical stability are shown to decompose into opposite grounding verdicts; on a real benchmark across three zero-shot models, the decomposition is computed exactly as deep as it is decidable, and no deeper. The full formal treatment of in-the-wild undecidability is deferred to a companion paper (§8).

---

# 1. Introduction

A model that answers correctly may be doing so for a reason the benchmark never checks. The now-standard response is to perturb the input in meaning-preserving ways and ask whether the answer survives: consistency, prompt-robustness, and stress-test evaluations all instantiate this. We take that response seriously and then show it is **one level too shallow**.

A familiar human case fixes the intuition. A student who has internalised the test-taking heuristic "answer choices containing 'always' are usually false" is **consistent** (the rule fires identically every time), often **correct** (the heuristic is genuinely correlated with the key), and **robust to rephrasing** (rewording the stem leaves the 'always' cue untouched) — and has understood nothing. Catching this student needs an item the ordinary test never contains: one where 'always' marks the *right* answer, severing the cue from the key, paired with a harmless reword that should unsettle no one who actually understood. The hierarchy below is exactly this distinction made precise for models: **accuracy** is the score, **stability** is consistency-under-rephrasing, **grounding** is whether the answer tracks the material or the cue. The sting carries over too — the heuristic-rider passes *this* exam, where 'always' happens to correlate with *false*, and fails *next year's*, where the key is flipped, **without learning anything in between**: same student, the grade moved because the answer key did. That models can be "right for the wrong reasons" is itself an old observation (McCoy et al., 2019; §2); what is new here is pushing the same cut one level deeper — into answers that are already stable *and* correct.

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

> **A 91%-accurate model that has learned nothing — keep this case in mind.** In the controlled setting of §6.1 a classifier reaches **0.912** accuracy on "can *X* fly?" and is **stable** under every answer-preserving presentation; by current practice it clears both the accuracy bar and the robustness check, and would ship. It was never reading the animal — it was reading an irrelevant `color` feature the training set happened to correlate with the label (|r| ≈ 0.81). Severing that one correlation with a `do`-intervention drops accuracy by **.408** (by **.501** for a LoRA-tuned Qwen2.5-1.5B, §6.2), while a matched pure-noise control moves only **.000–.021** — so the collapse is the shortcut, not fragility to any change. On a deployment slice where `color` no longer tracks the label, this model is at chance. Accuracy hid this; the stability check *endorsed* it. This is the `SC-spurious` bucket — and the rest of the paper is about why no benchmark without a stipulated graph can say how much of its headline number sits there.

**This paper's single contribution is this hierarchy, together with the principle that fixes how deep it can be observed.** Separating `U`/`SW`/`SC` (levels one and two) needs only gold labels and answer-preserving perturbations, so it is computable on any benchmark; this level is prior work, credited in the next section. The deepest cut, `SC → {grounded, spurious}`, is different — deciding it requires **two-sided observability**, and this is the spine of the paper:

- **Benchmark-side:** the item's ground-truth causal graph must be **stipulated by construction**, or else the choice of which variable is "irrelevant" smuggles in the very causal knowledge whose use is in question.
- **Model-side:** the candidate shortcut's **training correlation must be auditable**, or else the entanglement that defines `SC-spurious` cannot be measured at all.

When both sides are open the cut is decidable; when either goes dark it is **non-identifiable from the available observations** (in the standard causal-inference sense). The boundary is not a limitation of our instrument but a structural property of the question — and the demonstration confirms both halves: in the licensed regime, where both sides are open, the cut splits two models matched on accuracy *and* stability into opposite grounding verdicts (§6.1–6.3); on a natural benchmark, where both sides are dark, the same cut is undecidable and we report it as such (§6.4).

To be clear about what is and isn't ours: the accuracy-vs-stability level is established prior work, and the do-intervention / negative-control machinery is borrowed. What is new is the *nested-insufficiency structure*, the *two-sided observability* property, and the controlled evidence that each containment is non-empty. The same cut recurses into the other strata, decidable only as far as a licensed instrument reaches (§4); the next section credits the borrowed pieces inline, and §7 states the scope.

---

# 2. Related Work

That meaning-preserving perturbation exposes brittleness, and that consistency- or stability-aware decompositions are more informative than accuracy alone, is by now established: paraphrase consistency (Elazar et al., 2021), heuristic-driven "right for the wrong reasons" behaviour (McCoy et al., 2019), the large prompt-induced accuracy swings documented across recent evaluation studies (ProSA, EMNLP 2024; "What Did I Do Wrong?", NAACL 2025; "Flaw or Artifact?", 2025), and underspecification, where a pipeline yields many equally-accurate predictors whose shortcut reliance varies with seed (D'Amour et al., 2022). Our `U`/`SW`/`SC` split is a clean restatement of this line, and we take it as the floor on which the rest is built rather than as a contribution.

Our causal instrument is borrowed, and we use it as such. The do/see distinction and intervention semantics are Pearl's (2009; Pearl & Mackenzie, 2018); the requirement that a cause be non-redundant comes from the INUS tradition (Mackie, 1965; Halpern & Pearl, 2005). Equally, the idea that an intervention diagnostic is vacuous without a paired control is not ours: it is the control-task **selectivity** of probing (Hewitt & Liang, 2019) and the **negative-control** tradition of causal inference (Lipsitch et al., 2010; proximal causal inference). Our "pure-noise" control is exactly such a negative control.

The closest existing work ties "what counts as irrelevant" to causal structure, but for different ends. Counterfactual-invariance objectives (Veitch et al., 2021; Eisenstein, 2022) use that link to *train or regularize* models, not to define an evaluation hierarchy, and make no decidability claim. Concept-level causal-effect estimation (CEBaB, 2022) intervenes with human-written counterfactuals on natural data, but estimates effect magnitudes rather than partitioning stable-correct answers, and is not by-construction. Recent shortcut-detection surveys catalogue detection and mitigation without framing an observability boundary on an accuracy–stability–grounding hierarchy.

What is new here is therefore neither the perturbation floor nor the causal instrument, but three things built on them: the *reflexive* observation that `SC` reconflates exactly what accuracy conflated, yielding the strict hierarchy accuracy ⊋ stability ⊋ grounding with every containment non-empty; the **two-sided observability** property that fixes how deep the hierarchy can be read — level two anywhere, level three only where both the benchmark's graph is stipulated and the model's training is auditable, and non-identifiable from standard observational instruments otherwise (the formal "ill-posed" treatment is the companion paper's); and a controlled demonstration that the levels are non-empty and the boundary bites, with two models matched on headline accuracy and stability yet opposite in grounding.

> *Bibliographic note: inline author–year identifiers are for verifiability; final reference formatting is venue-dependent.*

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

**The construction is the source of ground truth, not a confound.** Causal irrelevance is *stipulated* rather than inferred from the model; training entanglement is *measured* on an auditable corpus; both are open to inspection and independent of investigator judgement. The setting is by-construction *because that is the only setting in which the cut is well-posed* — the observability boundary in §4 is the formal statement of why.

The four terminal buckets, and what each means for a reported number:

| bucket | invariant? | correct? | survives sever-intervention? | what headline accuracy does with it |
|---|---|---|---|---|
| `U` | no | — | — | counts it (inflates) |
| `SW` | yes | no | — | excludes it (honest miss) |
| `SC-spurious` | yes | yes | **no** | counts it **and** stability endorses it — the doubly-hidden bucket |
| `SC-grounded` | yes | yes | yes | the only bucket that earns "grounded", and only *in scope* |

## 3.3 Strictness

Each containment is strict and we exhibit non-empty witnesses in §6: items that are accurate-but-`U` (accuracy ⊋ stability), and items that are `SC` but `SC-spurious` (stability ⊋ grounding). The hierarchy is a claim about *behaviour*, not architecture; it is defined relative to the model's training distribution, not the world's graph alone.

## 3.4 The cut recurses

The same `do(class-3)` + `class-2`-control move is not special to `SC`; it applies to `U` and `SW` as well, and one consequence — that `SC-spurious` and `SW-shortcut` are the same mechanism partitioned only by evaluation slice — is sharp enough to state separately. The structural form of this recursion, and what it implies for the boundary, is §4.

## 3.5 What this changes about reporting (the measurement consequence)

The hierarchy is not only a diagnosis; it changes what a result should report.

1. **Report a triple, not a scalar.** Replace headline accuracy with **(accuracy, `SC`-fraction, overstatement-gap)**. All three need only gold labels and answer-preserving presentations (§3.1), so any benchmark can compute them now. The gap is the share of the headline number that is not even stable — let alone grounded — and it is not small in practice (BoolQ: **+0.153** for Qwen2.5-1.5B, +0.030–0.043 for stronger models; §6.4).
2. **Do not report a "grounded" or "genuine-reasoning" fraction off-construction.** On a benchmark with no stipulated graph that quantity is non-identifiable (§4); a number printed for it is underdetermined, not conservative. The honest object in the wild is the triple above plus an *explicit declination* of level three. Shown a grounded fraction on a natural benchmark, a reader should ask to see the construction that licenses it.
3. **A leaderboard gap can be an artefact of the evaluation slice.** Because `SC-spurious` and `SW-shortcut` are one mechanism split only by which slice the shortcut agrees with (§4.3), two models tied on accuracy *and* stability can differ in grounding, and their *ranking* can invert under a slice change with neither model touched. A reported delta between two models is not safe to read as a delta in capability until level three is decided — which, off-construction, it cannot be.

---

# 4. The Decidability Boundary

This is the conceptual core and the part the companion paper formalizes.

## 4.1 Depth is a property of construction and auditability

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

Remove either and the cut is **non-identifiable from the available observations**, in the standard causal-inference sense (Pearl, 2009; Pearl & Bareinboim, 2014): the quantity that would distinguish `SC-grounded` from `SC-spurious` is not a function of the data the setting provides. Without (a), the choice of which variable is "irrelevant" requires the very causal structure whose use by the model is in question — the original circularity returns. Without (b) — a model with unauditable pretraining — the correlation that defines `SC-spurious` cannot be measured and can only be estimated under assumptions the setting was meant to avoid. Therefore:

> **The depth to which the hierarchy is observable is a function of construction and auditability.** On a real benchmark with an opaque-pretrained model, the honest decomposition *stops at* `U`/`SW`/`SC`. Reporting a "causally grounded" fraction there is not a stronger result; it is an underdetermined one — and the formal "ill-posed" treatment of this regime is the companion paper's job.

This reframes what is usually written as a tool's limitation into a property of the question. It also explains a recurring pattern in evaluation debates: claims about "genuine reasoning" on natural benchmarks are contested not because the measurement is noisy but because, past level two, there is no identifiable quantity to measure without the construction.

## 4.2 The boundary is gate-shaped, not line-shaped

The move that splits `SC` is not special to `SC`: every bucket defined by a behavioural pattern is silent on the reason, and the reason is always the same causal-vs-shortcut axis. Applying it once more:

- **`U` splits two ways.** For an unstable item, apply `do(class-3)` (sever the shortcut). If the instability disappears (`U → stable`), the flipping was **shortcut-sourced**; if it persists with the `class-2` control also U-neutral, it is **not-shortcut-explained**. Because the presentations are answer-preserving and `M` is held fixed, a flip cannot be "`M` changed" — this two-way cut is well-posed and decidable with the existing instrument.
- **`SW` splits.** `SW-shortcut`: the model rides a `class-3` cue that points to the wrong answer here — decidable by the same `do(class-3)` test (severing changes the answer). `SW-misrule`: a stable but wrong *structural* rule — not shortcut-explained; positively characterizing it needs the **class-1 arm** (does the error transform in a structured way under interventions on causally-relevant variables?), a *different* licensed instrument. `SW-degenerate`: a near-constant predictor — a trivial output-distribution check, off the causal axis.

There is therefore not one cut and one gate but a small set of *licensed instruments* — the `do(class-3)` + `class-2` negative control, and the `class-1` correctly-variant arm — and the causal-vs-shortcut question recurses into every stratum, decidable exactly as far as some licensed instrument reaches and **non-identifiable past it**. The severing-instrument cuts (`SC → {grounded, spurious}`, the two-way `U` cut, `SW-shortcut`) share the construction-and-auditability gate of §3.2; the `class-1` arm opens one further gate (`SW-misrule`). Deeper cuts — representational fragility vs. boundary indeterminacy inside not-shortcut-explained `U`, and the positive content of a structural mis-rule — require instruments this paper does not license (representation probes; seed-multiplicity / underspecification analysis, in the sense of D'Amour et al., 2022), and inherit the same in-the-wild non-identifiability. The boundary is not a line; it is a sequence of gates, each opened only by a licensed instrument.

We deliberately do not multiply buckets beyond what a licensed instrument can decide. The contribution is not a lattice with more cells; it is that the causal-vs-shortcut question recurses into every behavioural stratum and is decidable exactly as far as a licensed instrument reaches.

## 4.3 One recursive consequence

One consequence of §4.2 is sharp enough to state on its own: `SC-spurious` and `SW-shortcut` are *the same mechanism* — a model riding the same `class-3` shortcut — partitioned only by whether the shortcut happens to agree with the gold label on the evaluation slice. The same model, on a different slice, moves between the "good" bucket and the "bad" bucket without changing at all. So the `SC`/`SW` boundary itself is, at the mechanism level, an artefact of the evaluation distribution: headline accuracy is not merely overstated (§3.1) — the very partition that would correct it is eval-contingent until the deepest cut is made. We note this as a corollary of the boundary structure, not as a separate claim; it inherits the same construction-and-auditability requirement and is decidable only where the deepest cut is.

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

This section provides controlled evidence for the main claim: **the hierarchy accuracy ⊋ stability ⊋ grounding has every containment non-empty, and the depth at which it is observable is a property of construction and auditability**. §6.1–6.3 demonstrate strict containment in the licensed regime; §6.4 demonstrates the boundary on a real benchmark. All numbers are from the existing experiment logs (see the evidence map); this section reorganizes the controlled results around the hierarchy and adds no claims beyond non-emptiness and the boundary.

## 6.1 Stability ⊋ grounding: matched accuracy, opposite grounding

A synthetic task ("can X fly?") with licensing structure `M` = animal type, a training-correlated irrelevant variable `color` (|r|≈0.81 → class-3) and an independent `name` (|r|≈0.005 → class-2 control), threshold 0.10. Two regimes: **A** = `M` available; **B** = `M` suppressed so only the shortcut carries label information.

| regime | headline acc | `SC` (stable-correct) | `do(class-3)` drop | `do(class-2)` control | `SC` resolves to |
|---|---|---|---|---|---|
| A — `M` available | 1.000 | high | Δ +.000 | Δ +.000 | **`SC-grounded`** |
| B — `M` suppressed | 0.912 | high | **Δ +.408** | Δ +.021 | **`SC-spurious`** |

Headline accuracy and stability are *comparable* across A and B; the grounding verdict is **opposite**. Level two cannot tell A from B; level three can. Regime B is the 0.912-accurate, fully stable, at-chance-off-slice model of the §1 case: this is the strict-containment witness for stability ⊋ grounding.

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

This is the **boundary-bites demonstration**: on a real benchmark, level two is computable and informative, while level three is undecidable here by construction of the setting — and that empty cell is itself the result. On a balanced 400-item BoolQ sample under four answer-preserving presentations, three zero-shot models — opaque pretraining, no stipulated graph, so §3.2(a) and (b) both fail:

| quantity | Qwen2.5-1.5B | DeepSeek V4 Flash | Claude Haiku |
|---|---|---|---|
| headline accuracy | 0.763 [0.723, 0.803] | 0.883 [0.850, 0.913] | 0.905 [0.877, 0.933] |
| `SC` fraction | 0.610 [0.565, 0.660] | 0.853 [0.818, 0.888] | 0.863 [0.827, 0.895] |
| **overstatement gap** | **+0.153 [0.118, 0.188]** | **+0.030 [0.015, 0.048]** | **+0.043 [0.023, 0.065]** |
| `SC → {grounded, spurious}` | **undecidable here** | **undecidable here** | **undecidable here** |

We compute levels one and two and *stop*. The empty deepest cell is the point: the gap excludes zero for all three and shrinks but does not vanish with capability, yet *no* assumption-free procedure can say how much of each `SC` is `SC-spurious` without the construction. Reporting a grounded fraction here would not be a stronger result — it would be the underdetermined one §4 describes.

---

# 7. Limits and What We Do *Not* Claim

- Level one (accuracy vs. stability) is **not ours**; it is established (§2) and used as the floor.
- The do-intervention and the negative-control logic are **borrowed** (Pearl; Hewitt & Liang; Lipsitch). The contribution is the nested-insufficiency structure and the decidability boundary, not these instruments.
- The in-the-wild **non-identifiability of level three** is here *argued and demonstrated by the empty BoolQ cell*, not formally proved; the formal treatment (and the stronger "ill-posed" framing) is the companion paper's job (§8). We claim non-identifiability from the standard observational instruments, not an impossibility theorem, in this paper.
- The by-construction demonstration uses template-generated items and a constructed shortcut-competition regime; it shows the levels are non-empty and the boundary bites, **not** that any particular deployed model is `SC-spurious` in the wild (that would itself require the undecidable level-three measurement).
- Every primary-claim model is fine-tuned on the distribution it is evaluated on; a portability check on a non-fine-tuned model strengthens §6 and is scripted but not yet run.

These are scope statements. Each marks a boundary the companion paper is designed to push.

---

# 8. Conclusion and the Two-Paper Split

The main claim is a strict hierarchy — **accuracy ⊋ stability ⊋ grounding** — with every containment non-empty, whose *depth of observability* is itself a structural property rather than a separate result: the hierarchy is observable to level two on any benchmark and to level three only where the graph is stipulated and the training auditable; deeper in the wild it is non-identifiable from standard observational instruments — ill-posed under the companion paper's stronger formal treatment, not merely hard. Level one is credited as prior work; the causal instrument is credited as borrowed; what is ours is the reflexive structure, the observability-depth property, and the controlled evidence that the levels separate (a model can be matched on accuracy *and* stability yet opposite in grounding).

This deliberately splits into two papers:

- **This paper (paper1-v2)** — the empirical hierarchy: define the levels, demonstrate non-emptiness and the boundary in the licensed regime, and decline level three on a real benchmark *as a result*. Modest, defensible, measurement-first.
- **Companion (thesis / position paper)** — the formal argument that level three is undecidable without construction-and-auditability, and the consequence that "causal-reasoning evaluation" claims on natural benchmarks are not well-posed past level two. Two deeper cuts are explicitly deferred to it: splitting not-shortcut-explained `U` into representational fragility vs. boundary indeterminacy, and the positive characterization of `SW-misrule` — both need instruments beyond the licensed set (representation probes; seed-multiplicity / underspecification analysis) and inherit the same in-the-wild undecidability. This is where the conceptual ambition belongs; it does not need a workshop methods frame to stand.

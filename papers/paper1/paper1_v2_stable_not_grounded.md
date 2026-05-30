# Stable Is Not Grounded
## Non-Circular Detection of Shortcut Reliance, by Separating Stipulated Irrelevance from Measured Entanglement

---

> **Thesis.** A stable, correct answer can still be riding a shortcut. To call it spurious you must decide that some variable is *irrelevant* — and on a natural benchmark that decision needs the very causal knowledge whose use you are testing, so it is circular. We break the circle by splitting "irrelevant" into two facts of different epistemic status: irrelevance fixed **by construction** (we author the item, so its graph is stipulated, not inferred from the model) and training entanglement fixed **by measurement** (a label correlation on an auditable corpus). A `do`-intervention that severs a variable satisfying both, paired with a mandatory pure-noise control, decides `SC → {grounded, spurious}` without circularity — and that procedure is the contribution. It is framed by a strict hierarchy (accuracy ⊋ stability ⊋ grounding, every containment non-empty) and bounded by a scope condition: the cut is licensed only where both sides are open, which §4 makes precise. The perturbation floor and the causal instrument are prior work, credited in §2.

---

# Abstract

Headline accuracy on a reasoning benchmark merges three behaviours that come apart under answer-preserving perturbation — unstable (the answer flips), stable-wrong (consistent and confidently incorrect), stable-correct — and the standard fix of checking stability separates only the first. The residual ambiguity is reflexive: a stable-correct answer can hold because the model tracks the structure that licenses the conclusion, or because it stably rides a shortcut also correlated with the answer at evaluation. Stability is necessary but not sufficient for grounding. The obstacle to settling which is which is **circularity** — calling a variable "irrelevant" normally requires the very causal knowledge whose use by the model is in question.

We break the circle by splitting "irrelevant" into two parts of different epistemic status: irrelevance fixed **by construction** (the item is authored, so its ground-truth graph is stipulated, not inferred from the model) and training entanglement fixed **by measurement** (a thresholded label correlation on an auditable corpus). A `do`-intervention that severs a variable satisfying both — paired with a mandatory negative ("pure-noise") control, without which any drop is uninterpretable — decides `SC → {grounded, spurious}` without reintroducing the circularity. **That non-circular decision procedure is the contribution.** It is framed by a strict hierarchy (accuracy ⊋ stability ⊋ grounding, every containment non-empty) and bounded by a scope condition: the cut is licensed only where both sides are open, so on a real benchmark one decides unstable / stable-wrong / stable-correct and stops. The same cut recurses into the unstable and stable-wrong strata; one consequence is sharp — a stably-correct-but-spurious model and a stably-wrong-by-shortcut model are the *same mechanism*, split only by evaluation luck, so even the stable-correct / stable-wrong boundary is eval-contingent until the deepest cut is made.

The accuracy-vs-stability floor and the causal instrument are prior work (§2); the non-circular split, its scope condition, and the controlled separation are ours. On two model classes the same headline accuracy and near-identical stability decompose into opposite grounding verdicts (and the verdict reproduces on a third architecture, Qwen-LoRA, on a two-premise task), and it ports to a never-fine-tuned reasoner that ignores even a perfect in-context shortcut (§7); on a real benchmark across three zero-shot models, the decomposition is computed exactly as deep as it is licensed, and no deeper. The formal treatment of the in-the-wild case is deferred to a companion paper (§8).

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

> **A 91%-accurate model that has learned nothing — keep this case in mind.** In the controlled setting of §6.1 a classifier reaches **0.912** accuracy on "can *X* fly?" and is **stable** under every answer-preserving presentation; by current practice it clears both the accuracy bar and the robustness check, and would ship. It was never reading the animal — it was reading an irrelevant `color` feature the training set happened to correlate with the label (|r| ≈ 0.81). Severing that one correlation with a `do`-intervention drops accuracy by **.408** (by **.429** for a full-fine-tuned DistilBERT, §6.2), while a matched pure-noise control moves only **.000–.021** — so the collapse is the shortcut, not fragility to any change. On a deployment slice where `color` no longer tracks the label, this model is at chance. Accuracy hid this; the stability check *endorsed* it. This is the `SC-spurious` bucket — and the rest of the paper is about why no benchmark without a stipulated graph can say how much of its headline number sits there.

**This paper's single contribution is a non-circular procedure for the deepest cut, `SC → {grounded, spurious}`.** Separating `U`/`SW`/`SC` (levels one and two) needs only gold labels and answer-preserving perturbations, so it runs on any benchmark; this level is prior work, credited in the next section. The deepest cut is different. Deciding it means calling some variable "irrelevant" and asking whether the model leans on it — but on a natural benchmark that judgement needs the causal structure whose use is in question, so it is circular. The procedure breaks the circle by fixing irrelevance and entanglement on two different footings:

- **Benchmark-side:** the item's ground-truth causal graph must be **stipulated by construction**, or else the choice of which variable is "irrelevant" smuggles in the very causal knowledge whose use is in question.
- **Model-side:** the candidate shortcut's **training correlation must be auditable**, or else the entanglement that defines `SC-spurious` cannot be measured at all.

When both sides are open the procedure runs and the cut is decided; when either goes dark the cut is **underdetermined** — the quantity that would separate grounded from spurious is not a function of what the setting observes. This is a scope condition on the procedure, not a tuning knob, and §4 states where it holds. The demonstration shows both halves: in the licensed regime the cut splits two models matched on accuracy *and* stability into opposite grounding verdicts (§6.1–6.3), it fires `SC-spurious` on a naturalistic benchmark whose graph is given (HANS; §6.4), and the verdict ports to a competent never-fine-tuned reasoner that ignores even a perfect in-context shortcut (§7); on a natural benchmark where both sides are dark, the procedure declines, and we report that as a result (§6.5).

The same cut recurses into the other strata, decidable only as far as a licensed instrument reaches (§4); §7 states the scope.

---

# 2. Related Work

That meaning-preserving perturbation exposes brittleness, and that consistency- or stability-aware decompositions are more informative than accuracy alone, is by now established: paraphrase consistency (Elazar et al., 2021), heuristic-driven "right for the wrong reasons" behaviour (McCoy et al., 2019), the large prompt-induced accuracy swings documented across recent evaluation studies (ProSA, EMNLP 2024; "What Did I Do Wrong?", NAACL 2025; "Flaw or Artifact?", 2025), and underspecification, where a pipeline yields many equally-accurate predictors whose shortcut reliance varies with seed (D'Amour et al., 2022). Our `U`/`SW`/`SC` split is a clean restatement of this line, and we take it as the floor on which the rest is built rather than as a contribution.

Our causal instrument is borrowed, and we use it as such. The do/see distinction and intervention semantics are Pearl's (2009; Pearl & Mackenzie, 2018); the requirement that a cause be non-redundant comes from the INUS tradition (Mackie, 1965; Halpern & Pearl, 2005). Equally, the idea that an intervention diagnostic is vacuous without a paired control is not ours: it is the control-task **selectivity** of probing (Hewitt & Liang, 2019) and the **negative-control** tradition of causal inference (Lipsitch et al., 2010; proximal causal inference). Our "pure-noise" control is exactly such a negative control.

The closest existing work ties "what counts as irrelevant" to causal structure, but for other ends, and recent shortcut-detection surveys catalogue detection and mitigation without confronting the circularity in *deciding* irrelevance. We take up the nearest of these — counterfactual-invariance training and CEBaB — where it bites, in stating what is new.

What is new here is therefore neither the perturbation floor nor the causal instrument, but the **non-circular decision procedure** for "is this variable a shortcut the model relies on?" — built by splitting *irrelevance* (stipulated by construction) from *entanglement* (measured on an auditable corpus), two facts of different epistemic status that prior shortcut work runs together. This is what separates it from its closest neighbours. Counterfactual-invariance objectives (Veitch et al., 2021; Eisenstein, 2022) use the irrelevance–causal link to *regularize*, not to return a per-item verdict. CEBaB (2022) does intervene by construction, but with human-written counterfactuals on natural data and to estimate effect *magnitudes* — so what counts as irrelevant is still inferred from the data, not stipulated, and the circularity is not addressed. Two further claims are ours and secondary to the procedure: the reflexive hierarchy accuracy ⊋ stability ⊋ grounding, which frames why the cut is needed at all, and the scope condition — the cut is licensed only where the graph is stipulated and the training auditable, and the verdict is underdetermined otherwise (the formal treatment is the companion paper's).

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

To make `SC-spurious` an investigator-independent label and not a judgement call, the candidate variable must be (a) **causally irrelevant by construction** (benchmark-side — we author the item; its ground-truth graph is stipulated, so this is not a causal fact discovered from the model) and (b) **training-correlated by measurement** (model-side — a label correlation on the auditable training set, fixed by a threshold). A variable satisfying (a)+(b) is the legitimate target of the severing intervention; a variable satisfying (a) but training-*independent* is the **mandatory negative control** (a "pure-noise" variable) — without it, a drop under the target intervention is uninterpretable (it is consistent with "any change breaks the model"). This control logic is borrowed (Hewitt & Liang, 2019; Lipsitch et al., 2010); the use here is to make the `SC` cut non-vacuous.

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

1. **Report a triple, not a scalar.** Replace headline accuracy with **(accuracy, `SC`-fraction, overstatement-gap)**. All three need only gold labels and answer-preserving presentations (§3.1), so any benchmark can compute them now. The gap is the share of the headline number that is not even stable — let alone grounded — and it is not small in practice (BoolQ: **+0.153** for Qwen2.5-1.5B, +0.030–0.043 for stronger models; §6.5).
2. **Do not report a "grounded" or "genuine-reasoning" fraction off-construction.** On a benchmark with no stipulated graph that quantity is non-identifiable (§4); a number printed for it is underdetermined, not conservative. The honest object in the wild is the triple above plus an *explicit declination* of level three. Shown a grounded fraction on a natural benchmark, a reader should ask to see the construction that licenses it.
3. **A leaderboard gap can be an artefact of the evaluation slice.** Because `SC-spurious` and `SW-shortcut` are one mechanism split only by which slice the shortcut agrees with (§4.3), two models tied on accuracy *and* stability can differ in grounding, and their *ranking* can invert under a slice change with neither model touched. A reported delta between two models is not safe to read as a delta in capability until level three is decided — which, off-construction, it cannot be.

---

# 4. When the Procedure Is Licensed

This section states where the cut can be made and where it cannot — the scope of the method. The formal version is the companion paper's.

## 4.1 Two-sided observability: where the cut can be made

**Level two is decidable anywhere.** `U`/`SW`/`SC` need only answer-preserving presentations and gold labels. Any benchmark qualifies.

**Level three is decidable only when both sides are observable** — the principle stated in §1. The `SC → {grounded, spurious}` cut requires conditions (a) and (b) of §3.2, one per side, and the two sides fail in different ways:

```text
(a) benchmark-side : causal irrelevance is fixed BY CONSTRUCTION.
                     We do not infer a causal fact from the model;
                     we authored the item. Non-circular by kind,
                     not by cleverness.
(b) model-side     : training correlation is fixed BY MEASUREMENT
                     on an auditable training set. No causal
                     assumption.
```

Remove either and the verdict is **underdetermined**: the quantity that would distinguish `SC-grounded` from `SC-spurious` is not a function of what the setting observes — the standard sense in which a causal quantity fails to be identified (Pearl, 2009; Pearl & Bareinboim, 2014), with the formal statement left to the companion paper. Without (a), the choice of which variable is "irrelevant" requires the very causal structure whose use by the model is in question — the original circularity returns. Without (b) — a model with unauditable pretraining — the correlation that defines `SC-spurious` cannot be measured and can only be estimated under assumptions the setting was meant to avoid. Therefore:

> **The depth to which the hierarchy is observable is exactly what two-sided observability buys.** On a real benchmark with an opaque-pretrained model, the honest decomposition *stops at* `U`/`SW`/`SC`. Reporting a "causally grounded" fraction there is not a stronger result; it is an underdetermined one — and the formal "ill-posed" treatment of this regime is the companion paper's job.

This reframes what is usually written as a tool's limitation into a scope condition on the method. It also explains a recurring pattern in evaluation debates: claims about "genuine reasoning" on natural benchmarks are contested not because the measurement is noisy but because, past level two, there is no well-defined quantity to measure without the construction.

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
5. **Boundary check.** On any non-constructed benchmark, run step 3 only — steps 1–2 and the sever-intervention (4) are not licensed there; report levels 1–2 and *explicitly decline* level 3 as undecidable (this is a result, not an omission).

Semantic-preservation validation gates step 3 (quantifier/modal/causal-direction shifts excluded). A consistency-only score is rejected: it rewards `SW`.

---

# 6. Demonstration: The Levels Are Non-Empty and the Boundary Bites

This section provides controlled evidence for the contribution: **the non-circular procedure decides `SC → {grounded, spurious}`, the levels of the hierarchy are non-empty, and the cut is licensed exactly where both sides are open**. §6.1–6.3 are the **both-sides-open** half of the §1 scope condition on synthetic items — graph stipulated, training auditable — where the procedure runs and every containment is strictly non-empty; §6.4 carries that same licensed cut onto **naturalistic** data (HANS), where the graph is given and the flag fires; §6.5 is the **both-sides-dark** half, where the graph is *not* given, the procedure declines, and we report that as a result. All numbers are from the existing experiment logs (see the evidence map); this section reorganizes the controlled results and adds no claims beyond non-emptiness and the licensing boundary.

## 6.1 Stability ⊋ grounding: matched accuracy, opposite grounding

A synthetic task ("can X fly?") with licensing structure `M` = animal type, a training-correlated irrelevant variable `color` (|r|≈0.81 → class-3) and an independent `name` (|r|≈0.005 → class-2 control), threshold 0.10. Two regimes: **A** = `M` available; **B** = `M` suppressed so only the shortcut carries label information.

| regime | headline acc | `SC` (stable-correct) | `do(class-3)` drop | `do(class-2)` control | `SC` resolves to |
|---|---|---|---|---|---|
| A — `M` available | 1.000 | high | Δ +.000 | Δ +.000 | **`SC-grounded`** |
| B — `M` suppressed | 0.912 | high | **Δ +.408** | Δ +.021 | **`SC-spurious`** |

Headline accuracy and stability are *comparable* across A and B; the grounding verdict is **opposite**. Level two cannot tell A from B; level three can. Regime B is the 0.912-accurate, fully stable, at-chance-off-slice model of the §1 case: this is the strict-containment witness for stability ⊋ grounding.

## 6.2 Not an architecture artifact

Identical protocol, two model classes (data-derived class assignment, identical across both):

| model | regime A `do(class-3)` | A verdict | regime B `do(class-3)` | B verdict |
|---|---|---|---|---|
| TF-IDF + LR | Δ +.000 | `SC-grounded` | Δ +.408 | `SC-spurious` |
| DistilBERT (66M, full FT) | Δ +.000 | `SC-grounded` | Δ +.429 | `SC-spurious` |

The negative control is intact in every cell (Δ ≈ +.00), ruling out "any intervention breaks it". The verdict reproduces on a third architecture (Qwen2.5-1.5B + LoRA) on the two-premise task of §6.3.

## 6.3 Non-redundancy and a correctly-variant arm

A two-premise transitive structure makes non-redundancy a real tested property; bootstrap 95% CIs, no hard thresholds. When the chain is required, ablating either premise drives withholding well above chance and a class-1 (causally relevant) break flips the answer 1.00 [1.00, 1.00]; when a strictly-more-reliable shortcut is available, ablation-withholding decouples (~chance), `do(class-3)` collapses correctness ≈0.50 [0.48, 0.53] with the negative control intact, and the class-1 flip rate collapses to 0.00. Same model, three corroborating signals: the chain-tracker is `SC-grounded` and correctly variant; the shortcut-rider is `SC-spurious`.

*Multi-seed (DistilBERT).* The sensitivity arm is byte-identical across five training seeds — expected, not a bug: at shortcut strength `s = 1.0` the shortcut is a perfectly separating feature, so the classifier converges to the same decision rule under every initialisation/batch order; with eval fixed, every metric is identical. The invariance measures how completely the shortcut dominates. The specificity arm honestly surfaces a P1/P2 premise-use asymmetry rather than a clean pass — the instrument exposing structure, not hiding it.

*Multi-seed (Qwen2.5-1.5B + LoRA).* The same two-premise verdict reproduces on a second architecture across three training seeds. Chain-required arm: `do(class-3)` drop 0.00, class-1 flip 1.00 [1.00, 1.00], non-redundancy respected in 3/3 seeds — `SC-grounded` and correctly variant. Shortcut-available arm: `do(class-3)` collapses correctness 0.517 [0.491, 0.542] with the flag firing in 3/3 seeds, class-1 flip 0.00, negative control intact — `SC-spurious`. The same P1/P2 ablation asymmetry as DistilBERT surfaces (the sensitivity arm not validated across every seed) — again the instrument exposing structure, not a clean pass.

## 6.4 A naturalistic instance, where the graph is given

The construction supplies the two-sided observability of §3.2, but it need not be *ours*: any benchmark that stipulates an irrelevance *and* exposes an auditable training set licenses the cut. HANS (McCoy et al., 2019) is exactly such a naturalistic instance. Its non-entailment cases carry high lexical overlap yet gold non-entailment — overlap-irrelevance fixed **by construction** (§3.2a) — and we fine-tune our own DistilBERT on an MNLI subset, so the training set, and the entanglement on it, are auditable (§3.2b: measured corr(overlap, entailment) = 0.36). The HANS non-entailment slice is the `do(class-3)` analog: the shortcut severed from the label.

| quantity | DistilBERT (50k MNLI, 3 seeds) |
|---|---|
| MNLI matched accuracy | 0.726 – 0.744 |
| HANS entailment (heuristic agrees) | 0.973 – 0.978 |
| HANS non-entailment (heuristic severed) | 0.026 – 0.031 |
| **shortcut drop (entailment − non-entailment)** | **0.947** |
| `SC → {grounded, spurious}` | **`SC-spurious`** |

A competent in-distribution reasoner (MNLI 0.73), `SC-grounded`-looking on the heuristic-agrees slice (0.98), collapses to 0.03 once overlap is severed from the label. The verdict is `SC-spurious` on real, independently-authored NLI text: the procedure is **not an artifact of our templates** — it fires wherever the graph is given, naturalistic or not. That is the setup for the contrast in §6.5, where the graph is *not* given and the same procedure must decline.

---

## 6.5 The boundary, demonstrated

This is the **boundary-bites demonstration**: on a real benchmark, level two is computable and informative, while level three is undecidable here by construction of the setting — and that empty cell is itself the result. On a balanced 400-item BoolQ sample under four answer-preserving presentations, three zero-shot models — opaque pretraining, no stipulated graph, so §3.2(a) and (b) both fail:

| quantity | Qwen2.5-1.5B | DeepSeek V4 Flash | Claude Haiku |
|---|---|---|---|
| headline accuracy | 0.763 [0.723, 0.803] | 0.883 [0.850, 0.913] | 0.905 [0.877, 0.933] |
| `SC` fraction | 0.610 [0.565, 0.660] | 0.853 [0.818, 0.888] | 0.863 [0.827, 0.895] |
| **overstatement gap** | **+0.153 [0.118, 0.188]** | **+0.030 [0.015, 0.048]** | **+0.043 [0.023, 0.065]** |
| `SC → {grounded, spurious}` | **undecidable here** | **undecidable here** | **undecidable here** |

We compute levels one and two and *stop*. The empty deepest cell is the point: the gap excludes zero for all three and shrinks but does not vanish with capability, yet *no* assumption-free procedure can say how much of each `SC` is `SC-spurious` without the construction. Reporting a grounded fraction here would not be a stronger result — it would be the underdetermined one §4 describes. This is the other half of two-sided observability: with both sides dark, the deepest cut does not get *hard*, it gets *undecidable*, and the empty cell is the confirmation — not a gap in the method.

---

# 7. Limits and What We Do *Not* Claim

- Level one (accuracy vs. stability) is **not ours**; it is established (§2) and used as the floor.
- The do-intervention and the negative-control logic are **borrowed** (Pearl; Hewitt & Liang; Lipsitch). The contribution is the nested-insufficiency structure and the decidability boundary, not these instruments.
- The in-the-wild **non-identifiability of level three** is here *argued and demonstrated by the empty BoolQ cell*, not formally proved; the formal treatment (and the stronger "ill-posed" framing) is the companion paper's job (§8). We claim non-identifiability from the standard observational instruments, not an impossibility theorem, in this paper.
- The by-construction *synthetic* demonstration uses template-generated items and a constructed shortcut-competition regime; it shows the levels are non-empty and the boundary bites, **not** that any particular deployed model is `SC-spurious` in the wild (that would itself require the undecidable level-three measurement). The naturalistic case (§6.4, HANS) lifts the cut off our own templates onto independently-authored NLI text, but only in the `SC-spurious` direction and on a single architecture; a naturalistic `SC-grounded` witness — a model high on *both* HANS slices — remains open.
- Every primary-claim model in §6.1–6.3 is fine-tuned on the distribution it is evaluated on. We reran the identical §5.4 ladder on two never-fine-tuned models, importing the class assignment from the construction's own statistics rather than re-measuring §3.2(b) on opaque pretraining; both are given `color` as a *perfect* in-context predictor (r = 1.0), the maximal shortcut temptation.
  - **(i) Weak model — zero-shot Qwen2.5-1.5B.** `do(class-3)` moves accuracy +0.048 [−0.027, 0.122] (CI crosses zero, no flag), but the model is a degenerate yes-sayer (baseline ≈ 0.65, fails non-redundancy and class-1 sensitivity), so its silence is only a **specificity lower-bound**, not a resistance demonstration.
  - **(ii) Competent model — zero-shot DeepSeek-V4-Flash (n = 200).** Baseline accuracy **1.000 / 0.995** and class-1 flip **0.995–1.000** (`correctly_variant`): a genuine, structure-tracking reasoner that flips its answer when the chain's middle term is broken. Yet with the same perfect color cue, `do(class-3)` moves accuracy by **−0.005 [−0.015, 0.0]** (no flag), class-2 control intact — versus **+0.517** for the same-ladder fine-tuned Qwen-LoRA (≈0.50 for DistilBERT, §6.3). A model that demonstrably respects the explicit chain still does **not** ride a perfect in-context shortcut: the `SC-spurious` flag indexes **trained-in** entanglement, not the mere presence of a predictive feature. This is the central non-circularity claim, now backed by a competent witness rather than a lower bound.

  *Caveat / scope.* DeepSeek withholds under premise ablation only ≈ 0.35–0.43, because the chains use **real categories** whose missing links it back-fills from world knowledge (it still flips on an *explicitly wrong* premise, class-1 = 1.0 — it trusts stated premises over priors but fills absent ones). This prior-contamination does not bear on the shortcut result, and is exactly the confound that synthetic/fictional concepts remove. The portability of the `SC-grounded` verdict on a by-construction grounded item remains the one open thread.

These are scope statements. Each marks a boundary the companion paper is designed to push.

---

# 8. Conclusion and the Two-Paper Split

The contribution is a non-circular procedure for the deepest cut — `SC → {grounded, spurious}` — decided by splitting stipulated irrelevance from measured entanglement, framed by a strict hierarchy (**accuracy ⊋ stability ⊋ grounding**, every containment non-empty) and bounded by a scope condition: the cut is licensed to level two on any benchmark and to level three only where the graph is stipulated and the training auditable; deeper in the wild the verdict is underdetermined — ill-posed under the companion paper's stronger treatment, not merely hard. The perturbation floor is credited as prior work and the causal instrument as borrowed; what is ours is the non-circular split, its scope condition, and the controlled evidence that the levels separate (a model can be matched on accuracy *and* stability yet opposite in grounding, and the verdict survives on a never-fine-tuned reasoner).

This deliberately splits into two papers:

- **This paper (paper1-v2)** — the empirical hierarchy: define the levels, demonstrate non-emptiness and the boundary in the licensed regime, and decline level three on a real benchmark *as a result*. Modest, defensible, measurement-first.
- **Companion (thesis / position paper)** — the formal argument that level three is undecidable without construction-and-auditability, and the consequence that "causal-reasoning evaluation" claims on natural benchmarks are not well-posed past level two. Two deeper cuts are explicitly deferred to it: splitting not-shortcut-explained `U` into representational fragility vs. boundary indeterminacy, and the positive characterization of `SW-misrule` — both need instruments beyond the licensed set (representation probes; seed-multiplicity / underspecification analysis) and inherit the same in-the-wild undecidability. This is where the conceptual ambition belongs; it does not need a workshop methods frame to stand.

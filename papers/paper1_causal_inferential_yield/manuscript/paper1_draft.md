# Non-Circular Detection of Shortcut Reliance
## Separating Stipulated Causal Irrelevance from Measured Training Entanglement

---

> **Positioning (read first).** This paper contributes one thing: a **non-circular procedure for deciding what counts as an "irrelevant" variable** when testing whether a model's correct answers are grounded in task structure or in a training-time shortcut. The procedure is a *disciplined package* — a class taxonomy, a mandatory control, and a boundary discipline — validated by a controlled demonstration that it is specific, sensitive, and non-vacuous. It is an evaluation-methodology contribution; the causal-minimality notions it uses are borrowed and cited in §2. §3.4 states precisely what is and is not new; §7 states the scope.

---

# Abstract

A recurring question in evaluating learned reasoning systems is whether a correct answer reflects the structure that licenses the conclusion or a correlation that happens to track it. Perturbation- and robustness-based evaluations probe this only indirectly and share an unstated circularity: deciding which perturbations *should* leave the answer unchanged presupposes knowledge of the causal structure being tested.

We remove this circularity by observing that the circular question — "is variable `Z` irrelevant to conclusion `C`?" — conflates two questions of *different epistemic status*: whether `Z` is **causally irrelevant** to `C` (a fact about the task), and whether `Z` is **entangled with `C` in the model's training distribution** (a fact about the data). The first is fixed non-circularly *by construction* (synthetic items whose ground-truth graph we stipulate); the second is fixed non-circularly *by measurement* (a training-set correlation requiring no causal assumption). A legitimate shortcut-detection target is exactly a variable that is causally irrelevant by construction **and** training-correlated by measurement.

The contribution is the disciplined package (neither half is individually new — §3.4 states the deflation in full): the **class-1/2/3 taxonomy** the split forces, a **mandatory class-2 control** that rules out the trivial reading "any intervention breaks the model," and a **boundary discipline** that confines the causal machinery to settings where the graph is stipulated and the training data auditable. On three model classes spanning the main families of learned sequence models — TF-IDF + logistic regression, fine-tuned DistilBERT (66M), and LoRA-fine-tuned Qwen2.5-1.5B — the procedure is **specific** (it does not flag a model that genuinely uses the stipulated structure), **sensitive** (it flags a shortcut-rider), and **non-vacuous** (the class-2 control remains intact); the verdict additionally holds on a two-premise structure where non-redundancy and a correctly-variant arm are real, CI-tested properties. Under training-seed variance the sensitivity arm is fully stable (5/5), while the multi-premise specificity arm does not return a clean pass: it surfaces a premise-use asymmetry (the model leans on one premise more than requiring both) — a diagnostic the procedure is designed to expose, not a result we smooth over (§5.4).

A complementary measurement-only section illustrates, on BoolQ across three zero-shot models, that headline accuracy can substantially overstate the fraction of *stably*-correct items.

---

# 1. Introduction

Learned systems frequently produce correct answers on reasoning benchmarks while relying on lexical shortcuts, dataset artifacts, or spurious correlations rather than the structure that actually licenses the conclusion. The standard diagnostic response is perturbation- or robustness-based evaluation: change the wording, insert distractors, shift the distribution, and check whether the answer survives.

This family of methods has a structural problem that is rarely stated:

```text
To decide which perturbations SHOULD leave the answer unchanged,
one must already know which parts of the input are causally
relevant to the conclusion — i.e., the very thing being tested.
```

If "irrelevant" is defined via the causal graph, the evaluation presupposes the causal competence it claims to measure. If it is defined informally ("this sentence is obviously off-topic"), the test is not reproducible and silently conflates different failure modes. (This circularity is acknowledged in prior work, not discovered here — §2.)

**The key observation.** The circular question — "is `Z` irrelevant to `C`?" — is in fact two questions that have been conflated, and they have *different epistemic status*:

```text
(a) Is Z causally irrelevant to C?     — a fact about the TASK
(b) Is Z entangled with C in the       — a fact about the
    model's training distribution?       model's DATA
```

Once separated, each can be answered without circularity: **(a) by construction** (synthetic items whose ground-truth graph we stipulate — we are not inferring a causal fact from the model, so there is nothing to be circular about), and **(b) by measurement** (a correlation on the training set, requiring no causal assumption). A legitimate test target is exactly a variable that is causally irrelevant by construction *and* training-correlated by measurement.

**This paper's single contribution is this non-circular operationalization, disciplined into a usable procedure.** The contribution is not the (a)/(b) split as an idea — neither half is individually novel, and §3.4 states that deflation in full — but the *package*: (i) the class-1/2/3 taxonomy the split forces; (ii) a **mandatory class-2 control** without which an observed degradation is uninterpretable; and (iii) a **boundary discipline** confining the causal machinery to settings where (a) is by construction and (b) is auditable. §5 demonstrates, in a controlled synthetic setting, that the procedure is specific, sensitive, and non-vacuous — with one honestly-reported wrinkle: under seed variance the multi-premise specificity arm reveals a premise-use asymmetry rather than a clean pass (§5.4), which we treat as a diagnostic the procedure is meant to expose. §6 adds a short, assumption-free decomposition lens as a complementary perspective.

---

# 2. Background and Borrowed Tools

We separate *what we borrow* from *what is new*, and *what is already known but unsolved* from *what we solve*; the novelty depends on these boundaries being explicit.

**Borrowed: causal-minimality notions.** The minimal-structure machinery is not new and is cited as such: Mackie's INUS condition (Mackie, 1965) — a cause as an insufficient but non-redundant part of an unnecessary but sufficient condition; the but-for / minimality clause of structural-model *actual causation* (Halpern & Pearl, 2005); and prime implicants / minimal models in propositional logic. Our non-redundancy clause (§3.1) *is* INUS non-redundancy; we claim no novelty for it.

**Borrowed: the causal hierarchy.** The observation–intervention distinction — conditioning vs. the interventional rung of Pearl's ladder (Pearl, 2009; Pearl & Mackenzie, 2018) — is used to separate rung-1 robustness (distractors) from the rung-2 property that makes a structure causal. Borrowed, not ours.

**The problem is already articulated.** We do *not* claim modern causal benchmarks are causally shallow — they are not. CLadder (Jin et al., 2023) spans all three rungs, including do-calculus, backdoor adjustment, collider bias, and counterfactuals; Corr2Cause (Jin et al., 2023), CRASS, CaLM, and e-CARE reach rung 2–3. What these benchmarks evaluate is *answer correctness on causal questions* (often with the graph verbalized in the prompt). They do not supply a procedure to test whether a model's sufficiency for an *arbitrary* conclusion is causally grounded versus parasitic on a training-time correlation. That this gap is real and unsolved is not our claim alone: a recent critical review of causal-reasoning benchmarks (arXiv:2407.08029) independently argues that *no existing benchmark adequately isolates genuine causal reasoning from shortcut exploitation*, and proposes "non-retrievable" construction as a desideratum. We therefore position our contribution as a *solution* to a problem the literature already states — **not** as the discoverer of that problem.

**Closest adjacent work, differentiated.** Fu et al. (2025, arXiv:2509.17380) also analyse "correlation vs. causation" in LLM reasoning, but the unit of analysis differs: they model the *reasoning pipeline* as a structural causal model over abstract components (instruction, thinking, CoT, answer) and intervene on those; their variables, by their own statement, "do not represent any question-specific target." We operate on *task-content* variables and ask whether a model's sufficiency is grounded in the task's structure or a training-correlated shortcut; they do not face the circularity of deciding which content variable counts as "irrelevant," which is precisely what our operationalization removes. Separately, shortcut-robustness work (irrelevant-condition perturbation suites; robustness-to-spurious-correlation evaluation, arXiv:2505.05704; causality-aware post-training, arXiv:2506.09433) adds or perturbs distractors *ad hoc*; none supply the non-circular by-construction-plus-measured-correlation decision procedure, and — critically — none pair the test with the mandatory class-2 control of §3.4.

> *Bibliographic note: inline author–year and arXiv identifiers are given for verifiability; final reference formatting is venue-dependent and not yet applied. Author lists for arXiv:2407.08029, :2505.05704, :2506.09433 to be filled from source on final pass.*

---

# 3. The Non-Circular Operationalization

## 3.1 Why stability needs a causal anchor

Pure invariance is insufficient in both directions: a model can be **stably wrong** (invariant yet systematically incorrect) or **correctly variant** (a perturbation that genuinely alters the licensing content *should* change the answer). Stability is meaningful only relative to what must be held fixed. We therefore define stable inference as the invariance of a conclusion under transformations that preserve its minimal causal structure, together with the appropriate variance under transformations that alter it.

For a target conclusion `C`, a **Minimal Causal Structure** `MCS(C)` is a premise set `M` such that:

```text
(i)   Sufficiency:      under the stipulated causal model, M is jointly
                        sufficient to license C.
(ii)  Non-redundancy:   every premise in M is counterfactually necessary.
(iii) Causal robustness: the sufficiency in (i) is invariant under
                        interventions on variables outside M.
```

Clauses (i)–(ii) are deductive minimality with a known lineage (§2); by themselves they do **not** distinguish a correlational from a causal structure. Two caveats are stated explicitly: minimality is **not unique** (we quantify over the set of minimal structures, never "the" structure) and is **background-relative** (the background is fixed by construction; §3.2). Clause (iii) is what closes the correlation/causation gap, and it requires the see/do distinction:

```text
conditioning / see :  P(C | M, Z = z)       — Z observed at z
intervention / do  :  P(C | M, do(Z = z))   — Z forced to z, severed
                                               from its own causes
```

A manipulation that merely adds observed information tests rung-1 robustness; only one that *sets a variable while breaking its natural correlations* tests the rung-2 property that makes a structure causal. The open question is which `Z` is a legitimate target for clause (iii) — answering that without circularity is §3.2.

## 3.2 The circular question and its two-part decomposition *(the central move)*

The circular question is "is `Z` causally irrelevant to `C`?" It is circular only because "irrelevant" is being decided by the same causal understanding we want to measure. It dissolves once split along epistemic status:

```text
(a) Is Z causally irrelevant to C?
    Answered BY CONSTRUCTION — synthetic items whose ground-truth
    graph we stipulate. We test whether the MODEL recovers a
    structure we defined; we do not discover a causal fact about
    the world from the model. There is nothing to be circular about.

(b) Is Z entangled with C in the model's training data?
    Answered EMPIRICALLY — a correlation measured on the training
    set, requiring no causal assumption.
```

A legitimate clause-(iii) intervention target is exactly a variable that is causally irrelevant by construction *(a)* **and** training-correlated *(b)*.

## 3.3 The three-class taxonomy the split forces

Separating (a) from (b) partitions every non-target variable into three classes — **defined relative to the model's training distribution, not the world's graph alone**:

| Class | Definition | What intervening tests |
|---|---|---|
| 1. Relevant | Ancestor / mediator / effect-modifier of `C` in the stipulated graph | Correct *variance*: the answer *should* change (§3.1) |
| 2. Pure noise | Causally irrelevant **and** independent of `C` in training data | Parsing / attention robustness (rung-1) |
| 3. Shortcut-correlated | Causally irrelevant **but** spuriously correlated with `C` in training data | Whether sufficiency is genuinely `M`-grounded (rung-2) — the clause-(iii) target |

Conflating class 2 and class 3 — which prior shortcut work does whenever "irrelevant" is left informal — lets a shortcut-reliant model pass a robustness test it should fail. The taxonomy is the operational payoff of the split, not a separate theoretical claim.

## 3.4 What makes the procedure non-vacuous

We state the deflation before a reviewer does. Taken individually, **neither half is new**: (a) is ordinary synthetic evaluation with a known generative process (the bAbI / CLEVR / gSCAN tradition stipulates ground truth by construction); (b) is ordinary shortcut analysis (annotation-artifact, HANS-style heuristic, and spurious-correlation work, including the controlled toy-feature datasets, all measure a candidate feature's label correlation). The conjunction, stated baldly, is "build a synthetic shortcut benchmark," which exists.

The contribution is therefore *not* the split as an idea. It is three things that only the disciplined package provides, and that ad-hoc shortcut suites typically lack:

1. **The class taxonomy the split forces (§3.3).** Prior work leaves "irrelevant" informal and conflates class-2 with class-3; making (a) and (b) separate, explicit answers is what yields a *measured*, investigator-independent class assignment.
2. **A mandatory class-2 control.** A non-circular shortcut verdict requires showing the class-3 intervention degrades the model *while a class-2 intervention does not*. Without the paired class-2 condition, a degradation under the class-3 intervention is uninterpretable — it is consistent with a model that breaks under *any* input change. This control is what rules out the trivial reading and is the part with experimental teeth (§5.2).
3. **A boundary discipline.** The decomposition is valid only where (a) is by construction *and* (b) is auditable. For models we train, (b) is a measurement. For frontier LLMs with unobservable pretraining, (b) degrades from measurement to estimation and the guarantee weakens; on benchmarks with no stipulated graph, importing the causal machinery would reintroduce the very circularity removed. The discipline is to confine the causal procedure to the by-construction setting and use only assumption-free decomposition elsewhere (§6). Respecting that boundary is itself part of the contribution.

A reviewer may still suspect this is ad-hoc shortcut practice in causal dress. The rebuttal is a concrete, falsifiable procedural test, not a philosophical claim: take any existing shortcut-robustness suite and check whether (i) *every* non-target variable carries a threshold-fixed, measured class label rather than an investigator's informal "this is obviously irrelevant", and (ii) each class-3 intervention is reported *paired* with a class-2 intervention on the same model, so a degradation cannot be read as "any change breaks it". To our knowledge none satisfy both. That conjunction — not the causal vocabulary, which is load-bearing only for clause-(iii) and is deliberately *dropped* in §6 — is the operational delta. The causal language earns its place only where a stipulated graph and an auditable training set are both available; outside that setting we do not invoke it, precisely so the contribution cannot be deflated to "we said 'do-operator' over a standard shortcut test".

The strength of §3.2 is thus entirely parasitic on the demonstration in §5 that the packaged procedure is specific, sensitive, and non-vacuous. Without that demonstration the split is a definitional move; with it, it is a validated discipline.

---

# 4. Instantiation Protocol

A reproducible recipe:

1. **Construct items with a stipulated DAG.** Each item carries a known ground-truth causal structure, so §3.2(a) holds by construction.
2. **Label every non-target variable by class** (§3.3) using the stipulated graph (for causal relevance) plus a training-correlation measurement (for entanglement), by a *fixed threshold*, not investigator discretion.
3. **Build perturbation groups** realizing the three classes: an `M`-exact group; a sub-premise group (withholding part of `M`); a **class-2 intervention** (distractor) group; a **class-3 intervention** (shortcut-reversal) group; and, for relevance, a **class-1 intervention** group (a causally relevant change whose answer *should* flip).
4. **Validate semantic preservation:** the perturbation must not silently alter `M` (quantifier / modal / causal-direction shifts are excluded).
5. **Score** with an accuracy-based dual metric; report the stable-correct / stable-wrong / unstable distribution, not only mean accuracy. A consistency-only score is rejected because it rewards stable-wrong behaviour.

---

# 5. Demonstration: A Single Controlled Ladder

This section demonstrates the central claim (§3.2–3.4): that a model's *correlational* sufficiency can be detected **non-circularly**, **specifically**, and **non-vacuously**. The claim is about the *procedure*, not model capability; scope is in §7.

## 5.1 The construction

We construct a synthetic task where the answer to "can X fly?" is causally determined by animal type (`M` = the minimal causal structure), with two non-`M` variables: `color` (made spuriously label-correlated in training) and `name` (assigned independently of the label). Per §3.2, the irrelevance of `color` and `name` is fixed **by construction** (3.2a — we author the items, so this is not a discovered causal fact); their entanglement is **measured** on the training set (3.2b): `|r(color, label)| = 0.81` → class-3; `|r(name, label)| = 0.005` → class-2, by a fixed threshold (0.10), not investigator discretion. We then compare `do(color)` (the class-3 intervention: sever the training correlation, `M` intact) against `do(name)` (the mandatory class-2 control), in two regimes — **A**: `M` present in the input; **B**: `M` suppressed so only the shortcut carries label information.

## 5.2 The result: specific, sensitive, non-vacuous

| regime | baseline | do(color) [class-3] | do(name) [class-2 control] | flagged? |
|---|---|---|---|---|
| A — `M` available | 1.000 | 1.000 (Δ +.000) | 1.000 (Δ +.000) | **No** |
| B — `M` suppressed | 0.912 | 0.504 (**Δ +.408**) | 0.891 (Δ +.021) | **Yes** |

The procedure is **specific** — it does not false-alarm when the model genuinely grounds in `M` (regime A is not flagged). This is the in-hand answer to the "you built the failure in" objection, and it is stronger than a disclaimer: regimes A and B use the *same constructed benchmark and the same data-derived class assignment* — only the model's training differs. The construction therefore does not fix the verdict; the verdict tracks whether the model actually rode the shortcut. A procedure that merely re-detected its own scaffolding would flag *both* regimes; this one flags only B. It is **sensitive** — it catches the shortcut-rider (regime B flagged). And it is **non-vacuous** — the class-2 control `do(name)` stays robust in *both* regimes (Δ +.000, +.021), ruling out "any intervention breaks it": only the class-3 intervention breaks sufficiency, and only when the model actually rode the shortcut. Detection is non-circular because (3.2a) is by construction and (3.2b) is measured. Regime B alone is the classical shortcut-reversal collapse — but now properly instrumented: with the *measured* class assignment and the class-2 control, it is an explicit clause-(iii) test, distinguished from its rung-1 distractor cousin, rather than an unexplained accuracy drop.

## 5.3 Across model classes

To check the verdict is not an architecture artifact, the identical protocol was rerun on two further model classes — a fine-tuned DistilBERT (encoder-only, bidirectional, 66M, full fine-tune) and a LoRA-fine-tuned Qwen2.5-1.5B (decoder-only, causal, 1.5B, 0.07% trainable). In all cases the training set is self-controlled, so §3.2(b) is valid; the measured class assignment is data-derived, not model-dependent, and identical across all three.

| model | regime A: do(color) drop | regime A: flagged | regime B: do(color) drop | regime B: flagged |
|---|---|---|---|---|
| TF-IDF + LR | Δ +.000 | No | Δ +.408 | **Yes** |
| DistilBERT (66M, full FT) | Δ +.000 | No | Δ +.429 | **Yes** |
| Qwen2.5-1.5B (LoRA) | Δ +.000 | No | Δ +.501 | **Yes** |

All three return the same verdict by design: no flag when `M` is available (regime A), a flag for the shortcut-rider (regime B), with the class-2 control intact (Δ +.000 in every case). The verdict is stable across a linear bag-of-words model, a 66M encoder, and a 1.5B decoder — a span covering the main families of learned sequence models.

The construction so far uses a single-premise `M`, which exercises the non-redundancy clause (§3.1-ii) only trivially. §5.4 removes that limitation.

## 5.4 Multi-premise non-redundancy and the correctly-variant arm

To make non-redundancy a *real* tested property, we use a two-premise transitive structure (P1: all A are B; P2: all B are C; "x is an A"; "is x a C?"), where YES vs. NO differs only in whether the middle term binds — there is no single-token cue, so a bag-of-words model cannot solve it; a fine-tuned DistilBERT can. We add the relative-learnability lever the framework predicts is necessary: in the shortcut regime the class-3 feature is made *strictly more reliable than the chain* (it tracks the label at `s = 1.0` while ~12% of training labels are corrupted off the binding). Verdicts are bootstrap 95% CIs, not hard thresholds.

**Class-1 arm.** §3.1 requires not only stability under class-2/3 interventions but *correct variance* under class-1 interventions — the answer should change when a causally relevant variable changes. We take would-be-YES items and apply `do(P2_content)`: replacing the middle term `b` in P2 with `b′ ≠ b`, so P1 (A→B) and P2 (B′→C) no longer bind (gold flips to NO). Color is left pointing toward YES throughout, so in the shortcut-available regime the class-3 temptation is fully active. The *class-1 flip rate* — fraction of broken-chain items the model answers NO — measures correctly-variant response and completes the three-way taxonomy.

| regime | P1-ablate withhold (95% CI) | P2-ablate withhold (95% CI) | do(color) drop (95% CI) | do(name) ctrl | class-1 flip rate (95% CI) |
|---|---|---|---|---|---|
| chain-required (specificity) | 0.80 [0.78, 0.82] | 0.83 [0.82, 0.85] | 0.00 [0.00, 0.00] | 0.00 | **1.00 [1.00, 1.00]** |
| shortcut-available (sensitivity) | 0.49 [0.47, 0.52] | 0.52 [0.49, 0.54] | **0.50 [0.48, 0.53]** | 0.00 | 0.00 [0.00, 0.00] |

All three arms show the expected CI pattern. **Specificity:** when the model genuinely tracks the two-premise chain, ablating *either* premise drives withholding significantly above chance (CI lower bounds 0.78, 0.82 > 0.5) — each premise is shown counterfactually necessary *to the model* (§3.1-ii made measurable) — the procedure does not false-alarm on class-3 (do(color) drop CI = [0,0]), and the model correctly varies under the class-1 chain-break (flip rate 1.00, CI [1.00, 1.00]). **Sensitivity:** when the shortcut is made strictly more reliable, the model rides it; ablation-withholding decouples from the premises (CIs straddle chance), corroborated by the class-3 flag (do(color) collapses ≈50 points, CI [0.48, 0.53]) with the class-2 control intact ([0,0]), and the class-1 flip rate collapses to 0.00 [0.00, 0.00] — the model ignores the chain-break, answering YES by color regardless. Three independent signals point to the same model: the chain-tracker is correctly variant; the shortcut-rider is not. The three-way taxonomy is now fully exercised.

**Multi-seed robustness (DistilBERT).** A 5-seed training-variance check (seeds 42–46, eval seed fixed so variance is over training stochasticity only) was run for the §5.4 regimes. The sensitivity arm is fully robust: all five seeds flag the correlational shortcut (5/5) and detect premise-decoupling (5/5). The specificity arm shows a nuanced, honestly-reported pattern: the class-1 correctly-variant test passes 5/5 and no shortcut is flagged (0/5), but `P2_withhold` is consistently low (mean 0.033 ± 0.073) while `P1_withhold` is higher (0.501 ± 0.091), indicating DistilBERT in this regime relies more on P1 presence than on requiring *both* premises. This is a genuine finding, not an instrumentation artifact: the diagnostic value of the procedure is precisely that it surfaces the P1/P2 asymmetry rather than returning a single pass/fail verdict. In the sensitivity regime every seed returns *byte-identical* metrics; this is expected, not a bug. With the shortcut at `s = 1.0`, `color` is a perfectly separating feature, so the classifier converges to the *same* colour-decision rule under every weight initialisation and batch order; with the evaluation sets fixed, the predictions — and hence all metrics — are identical. The invariance is itself a measure of how completely the shortcut dominates the learned solution: training stochasticity has no purchase once a perfect separator is present.

This completes the controlled demonstration; residual limits are stated in §7.

---

# 6. A Complementary Lens: Three-State Decomposition

The procedure above is causal and confined to the by-construction setting. This section adds one assumption-free companion observation — it needs only gold labels and answer-preserving perturbations, makes **no causal claim**, and so is admissible on a real benchmark without violating the §3.4 boundary (BoolQ carries no stipulated graph, so the clause-(iii) procedure deliberately does *not* apply here).

Decompose each item's behaviour under conservative answer-preserving presentations into **stable-correct (SC)**, **stable-wrong (SW)**, and **unstable (U)**. On a balanced 400-item BoolQ sample under four answer-preserving presentations, three zero-shot models:

| quantity | Qwen2.5-1.5B | DeepSeek V4 Flash | Claude Haiku |
|---|---|---|---|
| headline accuracy | 0.763 [0.723, 0.803] | 0.883 [0.850, 0.913] | 0.905 [0.877, 0.933] |
| stable-correct fraction | 0.610 [0.565, 0.660] | 0.853 [0.818, 0.888] | 0.863 [0.827, 0.895] |
| **overstatement gap** | **+0.153 [0.118, 0.188]** | **+0.030 [0.015, 0.048]** | **+0.043 [0.023, 0.065]** |
| group states (of 400) | SC 244 / SW 51 / U 105 | SC 341 / SW 29 / U 30 | SC 345 / SW 25 / U 30 |

All three gaps exclude zero; the gap shrinks monotonically with capability yet stays non-trivial for the strongest model. The single point: headline accuracy merges two distinct behaviours — fragility (U) and confident error (SW) — that the decomposition separates.

---

# 7. Limits and What We Do *Not* Claim

- We do **not** claim a novel theory of causation; the causal-minimality core is borrowed and cited (§2). The contribution is the non-circular *operationalization disciplined into a procedure*, and explicitly **not** the bare (a)/(b) split, nor the *problem* of shortcut-vs-causal evaluation (already articulated in the benchmark-critique literature — §2).
- We do **not** claim validation on capable models. Baselines are weak by design; §5 demonstrates the *procedure*, not model quality. Every primary-claim model is fine-tuned on the same synthetic distribution it is evaluated on — a procedure-portability check on a model not trained on the construction is the most direct strengthening (§8) and is not claimed here.
- We do **not** claim coverage of natural reasoning: the benchmark is template-generated and perturbations are more regular than human-written ones. The sensitivity arm uses a *constructed* shortcut-competition regime (the class-3 feature is deliberately made strictly more reliable than the chain) — legitimate by construction, but not a *naturalistic* shortcut.
- We do **not** claim the operationalization extends to systems with unauditable training data; for frontier LLMs, §3.2(b) is an open estimation problem (§3.4). The §6 decomposition is a perspective, not a validated benchmark instrument.
- The main ladder uses a single seed-set; only the §5.4 DistilBERT regimes have a multi-seed check, which itself surfaced (not hid) the P1/P2 asymmetry. That check varies weight initialisation and batch ordering over a *fixed* training corpus (regenerated with a fixed data seed), not a fresh data resample per seed; resample-per-seed is the stricter design and is not claimed here.

*Instrumentation note.* During development a CoT-vs-stability pilot appeared to halve stable-correctness until an answer-extraction artifact (a first-match parser locking onto a premature `Answer:` token) was traced and fixed, after which the effect was at noise level — a reminder that binary verdicts flip on instrumentation noise and effect sizes should be reported with variance, which the CI discipline of §5.4 enforces.

These are scope statements, not hedges: each marks a concrete boundary §8 is designed to push.

---

# 8. Conclusion and Next Step

The contribution is a single non-circular operationalization, disciplined into a usable procedure: split the circular "is `Z` irrelevant?" into a question fixed *by construction* and one fixed *by measurement*; force the class-1/2/3 taxonomy that split entails; require the class-2 control without which a degradation is uninterpretable; and confine the causal machinery to the by-construction setting. The validated discipline — demonstrated in §5 to be specific, sensitive, and non-vacuous, with non-redundancy and a correctly-variant arm made real, CI-tested properties in §5.4 — is the claim (§3.4 states what is not new). §6 adds, as a perspective only, an assumption-free decomposition showing headline accuracy can overstate stable correctness.

The single highest-leverage next step, which most directly answers the principal objection (a self-constructed shortcut detected on a self-trained model), is a **procedure-portability check**: rerun the *same* §5 ladder on a model not fine-tuned on the construction (e.g., a zero-shot instruction-tuned model) and verify the verdict structure — no flag in regime A, a flag in regime B, the class-2 control intact — survives. One subtlety must be stated to keep this consistent with the boundary discipline of §3.4: for a model with unauditable pretraining, §3.2(b) is *not* re-measured on that model. The class label is imported from the construction's own controlled data statistics — which we own — and the check tests only whether the verdict *structure* transfers to a model whose parameters were never shaped by the construction. It is therefore a portability check *under an imported class assignment*: explicitly not a re-measured (b), and not a capability claim. This is the intended line of subsequent work.

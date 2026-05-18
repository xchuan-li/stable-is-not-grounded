# Inferential Yield
## A Causally Grounded Evaluation Methodology for Stable Reasoning under Information Constraints

---

> **Positioning (read first).** This is an **evaluation-methodology** paper, not a new causal-theory paper. The causal concepts it uses (minimal sufficient causes, actual-causation minimality, the interventional rung of the causal hierarchy) are *borrowed tools*, cited as such. The contribution is a *measurement procedure for learned systems*: a non-circular way to decide what counts as an "irrelevant" variable, and a single quantity — Causal Inferential Yield (CIY) — that unifies several otherwise ad hoc reasoning-stability probes. Claims are scoped accordingly in Section 7.

---

# Abstract

A recurring question in evaluating learned reasoning systems is whether a model's correct answers reflect the causal structure that licenses the conclusion, or a correlation that happens to track it. Existing perturbation- and robustness-based evaluations probe this only indirectly, and they share an unstated circularity: deciding which perturbations *should* leave the answer unchanged presupposes knowledge of the causal structure being tested.

We present an evaluation methodology that removes this circularity. We anchor "stable inference" to a **Minimal Causal Structure (MCS)** — a minimal, non-redundant premise set whose sufficiency is required to be robust under interventions on variables outside it — and we define **Causal Inferential Yield (CIY)** as a model's ability to recover a conclusion *in units of* its MCS: correctly, minimally, and stably under intervention. The key methodological move is to split the circular question into two non-circular ones: whether a variable is causally irrelevant is fixed *by construction* (synthetic items whose ground-truth graph we stipulate), while whether it is entangled with the target is measured *empirically* on the model's training data.

We then show, on three model classes — TF-IDF + Logistic Regression, fine-tuned DistilBERT (66M), and LoRA fine-tuned Qwen2.5-1.5B (1.5B) — that the framework's distinctions are **not vacuous**. In a controlled synthetic setting, the procedure is specific (it does not flag a model that can use the stipulated MCS), sensitive (it flags a shortcut-rider), and non-vacuous (a class-2 control remains intact). On a trusted benchmark, the measurement-only half of the framework shows that headline accuracy can substantially overstate stable correctness: on a 400-item BoolQ sample, this holds across three zero-shot models spanning different capability levels and architecture families — Qwen2.5-1.5B (gap +0.153 [0.118, 0.188]), DeepSeek V4 Flash (gap +0.030 [0.015, 0.048]), and Claude Haiku (gap +0.043 [0.023, 0.065]) — with all gaps excluding zero. The framework is sensitive to capability differences (the gap shrinks monotonically with model strength) while remaining non-trivial for all three, and DeepSeek and Claude Haiku independently converge on nearly identical estimates, constituting a cross-architecture replication at the stronger-model tier. The empirical point is modest but important: ignoring the framework's distinctions yields different conclusions from inspecting correctness alone.

We are explicit about what we do not claim (Section 7) and frame the open program — validation on strong models, intervention-level prompting, and frontier-LLM training-correlation estimation — as the intended line of subsequent work.

---

# 1. Introduction

Learned systems frequently produce correct answers on reasoning benchmarks while relying on lexical shortcuts, dataset artifacts, or spurious correlations rather than the structure that actually licenses the conclusion. The standard diagnostic response is perturbation- or robustness-based evaluation: change the wording, insert distractors, shift the distribution, and check whether the answer survives.

This family of methods has a structural problem that is rarely stated:

```text
To decide which perturbations SHOULD leave the answer unchanged,
one must already know which parts of the input are causally
relevant to the conclusion — i.e., the very thing being tested.
```

If "irrelevant" is defined via the causal graph, the evaluation presupposes the causal competence it claims to measure. If it is defined informally ("this sentence is obviously off-topic"), the test is not reproducible and silently conflates different failure modes.

**This paper's contribution is methodological, not causal-theoretic.** We do not propose a new theory of causation; we use existing notions (Section 2) to build a measurement procedure with three properties:

1. **A causal anchor for stability.** Stability is defined relative to a Minimal Causal Structure (MCS), so that "stably wrong" and "correctly variant" are not mistaken for stability or instability (Section 3.1–3.3).
2. **A non-circular operationalization of "irrelevant."** The circular question is decomposed into a construction-time causal stipulation and an empirically measurable training correlation (Section 3.5). This is the central methodological result.
3. **A unifying measure.** Causal Inferential Yield (CIY) subsumes shortcut-reversal, distractor robustness, sub-premise withholding, and the Stable Inference Score as special cases (Section 3.6), turning a set of ad hoc probes into instances of one quantity.

We then provide a deliberately modest empirical demonstration whose only purpose is to show the distinctions are non-vacuous: ignoring them produces measurably wrong conclusions (Section 5).

---

# 2. Related Work and Borrowed Tools

We separate *what we borrow* from *what is new*, and — critically — *what is already known but unsolved* from *what we solve*. The novelty of the contribution depends on these boundaries being explicit, so we draw them sharply rather than overclaim.

**Borrowed: causal-minimality notions.** The minimal-structure machinery is not new and is cited as such: Mackie's INUS condition (Mackie, 1965) — a cause as an insufficient but non-redundant part of an unnecessary but sufficient condition; the but-for / minimality clause of structural-model *actual causation* (Halpern & Pearl, 2005); and prime implicants / minimal models in propositional logic. Our non-redundancy clause (3.2-ii) *is* INUS non-redundancy; we claim no novelty for it.

**Borrowed: the causal hierarchy.** The observation–intervention distinction — conditioning vs. the interventional rung of Pearl's ladder (Pearl, 2009; Pearl & Mackenzie, 2018) — is used to separate rung-1 robustness (distractors) from the rung-2 property that makes a structure causal. Again borrowed, not ours.

**Situated against: causal-reasoning benchmarks.** We are careful *not* to claim modern causal benchmarks are causally shallow — they are not. CLadder (Jin et al., 2023) explicitly spans all three rungs of Pearl's ladder, including do-calculus, backdoor adjustment, collider bias, and counterfactuals; Corr2Cause (Jin et al., 2023), CRASS, CaLM, and e-CARE reach rung 2–3. What these benchmarks evaluate is *answer correctness on causal questions* (often with the causal graph verbalized in the prompt). They do not provide a procedure to test whether a model's sufficiency for an *arbitrary* conclusion is causally grounded versus parasitic on a training-time correlation. That this gap is real and unsolved is not our claim alone: a recent critical review of causal-reasoning benchmarks (arXiv:2407.08029) independently argues that *no existing benchmark adequately isolates genuine causal reasoning from shortcut exploitation*, and proposes "non-retrievable" construction as a desideratum. We therefore position our contribution as a *solution* to a problem the literature already articulates — **not** as the discoverer of that problem.

**Closest adjacent work, explicitly differentiated.** Fu et al. (2025, arXiv:2509.17380) also analyse "correlation vs. causation" in LLM reasoning, but the unit of analysis is different: they model the *reasoning pipeline* as a structural causal model over abstract components — instruction, thinking, CoT, answer — and intervene on those components to assess CoT faithfulness and the effect of training paradigms (notably RLVR). Their variables, by their own statement, "do not represent any question-specific target." We instead operate on *task-content* variables and ask whether a model's sufficiency is grounded in the task's minimal causal structure or a class-3 shortcut; crucially, they do not face — and so do not address — the circularity of deciding which content variable counts as "irrelevant," which is precisely the problem our operationalization removes. Separately, shortcut-robustness work (e.g. irrelevant-condition perturbation suites in the spirit of Math500-Noop; robustness-to-spurious-correlation evaluation, arXiv:2505.05704; causality-aware post-training, arXiv:2506.09433) adds or perturbs distractors *ad hoc*; none supply the non-circular by-construction-plus-measured-correlation decision procedure.

**What is new (scoped).** (a) The non-circular operationalization of "irrelevant" via construction-time stipulation (3.5a) plus empirical training-correlation (3.5b) — the central contribution; the *problem* is acknowledged in prior work, the *non-circular solution* is ours. (b) The recasting of disparate stability probes (distractor, shortcut-reversal, withholding, SIS) as instances of one CIY measure (Section 3.6). (c) The boundary discipline itself: the causal operationalization is applied only where the DAG is fixed by construction, and only measurement-only metrics are used on real benchmarks (Section 6). We claim novelty for **none** of the causal concepts, and **not** for the problem of shortcut-vs-causal evaluation — only for the non-circular procedure and its disciplined application.

> *Bibliographic note: inline author–year and arXiv identifiers are given for verifiability; final reference formatting is venue-dependent and not yet applied. Author lists for arXiv:2407.08029, :2505.05704, :2506.09433 to be filled from source on final pass.*

---

# 3. The Framework

*(This section is the relocated and methodology-reframed theoretical core. It is reproduced here as the spine of this paper; in the empirical companion paper it appears only as a brief recap with a citation to this paper.)*

## 3.1 Why "Stable Inference" Needs a Causal Anchor

Pure invariance is insufficient in both directions: a model can be **stably wrong** (invariant yet systematically incorrect) or **correctly variant** (a perturbation that genuinely alters causal content *should* change the answer). Stability is meaningful only relative to what must be held fixed — the causal structure that licenses the conclusion. We therefore define stable inference as:

```text
the invariance of a conclusion under transformations that
preserve its minimal causal structure,
together with the appropriate variance under
transformations that alter it.
```

## 3.2 Minimal Causal Structure (MCS)

For a target conclusion `C`, a **Minimal Causal Structure** `MCS(C)` is a premise set `M` such that:

```text
(i)   Sufficiency:     under the stipulated causal model, M is jointly
                       sufficient to license C.
(ii)  Non-redundancy:  every premise in M is counterfactually necessary.
(iii) Causal robustness: the sufficiency in (i) is invariant under
                       interventions on variables outside M (3.3–3.5).
```

Clauses (i)–(ii) are deductive minimality with a known lineage (Section 2); they do **not** by themselves distinguish a correlational from a causal structure. Two standard caveats are stated explicitly: minimality is **not unique** (we quantify over the set of minimal structures, never "the" structure), and it is **background-relative** (the background is fixed by construction; Section 3.5). Clause (iii) is what closes the correlation/causation gap.

## 3.3 The Robustness Clause: Conditioning vs. Intervention

```text
conditioning / see :  P(C | M, Z = z)       — Z observed at z
intervention / do  :  P(C | M, do(Z = z))   — Z forced to z, severed
                                               from its own causes
```

A genuine `MCS(C)` requires: for every variable `Z` outside `M` and every admissible `do(Z=z)`, with `M` held fixed and present, the licensing of `C` from `M` is unchanged. A manipulation that merely adds observed information tests rung-1 robustness; only one that *sets a variable while breaking its natural correlations* tests the rung-2 property that makes a structure causal.

## 3.4 A Taxonomy of Non-Structural Variables

Class membership is defined **relative to the model's training distribution**, not the world's graph alone.

| Class | Definition | Intervening tests |
|---|---|---|
| 1. Relevant | Ancestor/mediator/effect-modifier of `C` in the stipulated graph | Nothing about shortcutting; excluded from clause (iii) |
| 2. Pure noise | Causally irrelevant **and** independent of `C` in training data | Parsing/attention robustness (rung-1) |
| 3. Shortcut-correlated | Causally irrelevant **but** spuriously correlated with `C` in training data | Whether sufficiency is genuinely `M`-grounded (rung-2) — **the clause (iii) target** |

Conflating class 2 and class 3 lets a shortcut-reliant model pass a robustness test it should fail.

## 3.5 A Non-Circular Operationalization of "Irrelevant" *(central methodological result)*

```text
(a) Is Z causally irrelevant to C?
    Answered BY CONSTRUCTION — synthetic items whose ground-truth
    graph we stipulate. We test whether the MODEL recovers a
    structure we defined; we do not discover a causal fact. No circularity.

(b) Is Z entangled with C in the model's training data?
    Answered EMPIRICALLY — a correlation measured on the training
    set, requiring no causal assumption.
```

A legitimate clause-(iii) intervention target is exactly a variable that is causally irrelevant by construction *(a)* **and** training-correlated *(b)* — i.e., class 3.

**Honest boundary.** Step (b) needs an auditable training distribution. This holds for models we train (TF-IDF, fine-tuned DistilBERT). For frontier LLMs with unobservable pretraining, (b) degrades from measurement to estimation; the framework's guarantees weaken accordingly. We state this as a principled limitation, not a solved problem.

## 3.6 Causal Inferential Yield (CIY)

A raw count of correct conclusions is ill-posed (deductive closure makes derivable truths unbounded — the logical-omniscience/relevance problem). Yield is well-defined only *in units of MCS*. For `C` with `M = MCS(C)`, CIY is the model's ability to:

```text
1. recover C given exactly M;
2. WITHHOLD C given a strict subset of M;
3. still recover C under do(Z) for class-2 or class-3 Z, M held fixed.
```

These are one quantity: correct, minimal, intervention-stable recovery. It subsumes existing probes:

| Existing construct | Recast as |
|---|---|
| Distractor perturbation | CIY-3 with a **class-2** variable (rung-1) |
| Shortcut-reversal | CIY-3 with a **class-3** variable (rung-2; the canonical causal test) |
| Sub-premise / negative control | CIY-2 (withholding) |
| Stable Inference Score | A scalar lower-bound proxy for CIY-1 ∧ CIY-3 |

### 3.6.1 The Stable Inference Score as a special case

```text
group_score(g) = correct_predictions(g) / total_samples(g)
SIS_v1         = mean over groups of group_score(g)
```

SIS is **accuracy-based**, not consistency-based (a consistency-only score rewards stable-wrong behavior). It captures CIY-1 and CIY-3 but not CIY-2, and does not separate label bias from genuine stability. We therefore report a dual metric (accuracy + intervention-consistency + a stable-correct / stable-wrong / unstable group taxonomy). SIS_v1 is a lower-bound proxy for CIY, not its definition.

---

# 4. Instantiation Protocol

A reproducible recipe for applying the framework:

1. **Construct items with a stipulated DAG.** Each item carries a known ground-truth causal structure (so 3.5(a) holds by construction).
2. **Label variables by class** (Table 3.4) using the stipulated graph + a training-correlation measurement (3.5(b)).
3. **Build perturbation groups** realizing CIY-1/2/3: MCS-exact, sub-premise, class-2 intervention (distractor), class-3 intervention (shortcut-reversal), plus semantics-preserving paraphrase/lexical/reasoning-path variants.
4. **Validate semantic preservation** (the perturbation must not silently change the MCS — quantifier/modal/causal-direction shifts are excluded).
5. **Score** with the dual metric; report the stable-correct / stable-wrong / unstable distribution, not only mean accuracy.

---

# 5. Demonstrating the Operationalization

This section demonstrates the paper's central methodological claim (Section 3.5): that a model's *correlational* sufficiency can be detected **non-circularly**, **specifically**, and **non-vacuously**. The empirical scope is intentionally controlled — the claim is about the *procedure*, not about model capability — and the honest limits are stated in Section 7.

## 5.1 The construction

We construct a synthetic task where the answer to "can X fly?" is causally determined by animal type (M = the minimal causal structure), with two non-M variables: `color` (made spuriously label-correlated in training) and `name` (assigned independently of the label). Per Section 3.5, the irrelevance of `color` and `name` is fixed **by construction** (3.5a — we author the items, so this is not a discovered causal fact); their entanglement is **measured** on the training set (3.5b): |r(color, label)| = 0.81 → class-3, |r(name, label)| = 0.005 → class-2, by a fixed threshold (0.10), not investigator discretion. We then compare `do(color)` (the class-3 intervention: sever the training correlation, M intact) against `do(name)` (the mandatory class-2 control), in two regimes — A: M present in the input; B: M suppressed so only the shortcut carries label information.

## 5.2 The result: specific and sensitive

| regime | baseline | do(color) [class-3] | do(name) [class-2 control] | flagged? |
|---|---|---|---|---|
| A — M available | 1.000 | 1.000 (Δ +.000) | 1.000 (Δ +.000) | **No** |
| B — M suppressed | 0.912 | 0.504 (**Δ +.408**) | 0.891 (Δ +.021) | **Yes** |

The operationalization is **specific** — it does not false-alarm when the model genuinely grounds in M (regime A is not flagged). This directly answers the "you built the failure in" objection: in A the failure is *not* built in and the procedure correctly stays silent. It is **sensitive** — it catches the shortcut-rider (regime B flagged). And it is **non-vacuous** — the class-2 control `do(name)` stays robust in *both* regimes (Δ +.000, +.021), ruling out the trivial reading "any intervention breaks it": only the class-3 intervention breaks sufficiency, and only when the model actually rode the shortcut. Detection is non-circular because (3.5a) is by construction and (3.5b) is measured. (Regime B, viewed alone, is the classical shortcut-reversal collapse — but now properly instrumented: with the measured class assignment and the class-2 control, it is an explicit clause-(iii) test, distinguished from its rung-1 distractor cousin, rather than an unexplained accuracy drop.)

## 5.3 Model-class robustness

To check the verdict is not an artifact of a specific architecture, the identical protocol was rerun on two additional model classes — a fine-tuned DistilBERT (encoder-only, bidirectional, 66M parameters, full fine-tune) and a LoRA fine-tuned Qwen2.5-1.5B (decoder-only, causal, 1.5B parameters, 0.07% trainable). In both cases the training set remains self-controlled, so §3.5(b) is valid. The measured class assignment is data-derived, not model-dependent, and is identical across all three models.

| model | regime A: do(color) drop | regime A: flagged | regime B: do(color) drop | regime B: flagged |
|---|---|---|---|---|
| TF-IDF + LR | Δ +.000 | No | Δ +.408 | **Yes** |
| DistilBERT (66M, full FT) | Δ +.000 | No | Δ +.429 | **Yes** |
| Qwen2.5-1.5B (LoRA) | Δ +.000 | No | Δ +.501 | **Yes** |

All three return the same verdict by design: the procedure does not flag when M is available (regime A), and flags the shortcut-rider (regime B) with the class-2 control intact (Δ +.000 in all cases). The class-2 control and the verdict are stable across a linear bag-of-words model, a 66M encoder, and a 1.5B decoder — an architectural span that covers the main families of learned sequence models. The measured class assignment is, as expected, the same across all three — it is data-derived, not model-dependent.

The construction so far uses a single-premise M, which exercises the non-redundancy clause (3.2-ii) only trivially. Section 5.4 removes that limitation.

## 5.4 Multi-premise non-redundancy

To make non-redundancy (3.2-ii) a *real* tested property, we use a two-premise transitive structure (P1: all A are B; P2: all B are C; "x is an A"; "is x a C?"), where YES vs. NO differs only in whether the middle term binds (B vs. B2) — there is no single-token cue, so a bag-of-words model cannot solve it; a fine-tuned DistilBERT can. We add the relative-learnability lever the framework predicts is necessary: in the shortcut regime the class-3 feature is made *strictly more reliable than the chain* (it tracks the label at s = 1.0 while ~12% of training labels are corrupted off the binding). Verdicts are bootstrap 95% CIs, not hard thresholds — exactly the discipline Section 7's instrumentation note argues for.

**Class-1 arm.** Section 3.1 requires not only that a model be *stable* under class-2/3 interventions, but that it be *correctly variant* under class-1 interventions — i.e., its answer should change when a causally relevant variable changes. We operationalise this by taking would-be-YES items and applying `do(P2_content)`: replacing the middle term b in P2 with b′ ≠ b, so P1 (A→B) and P2 (B′→C) no longer bind (gold flips to NO). Color is left pointing toward YES throughout, so in the shortcut-available regime the class-3 temptation is fully active. The *class-1 flip rate* — fraction of broken-chain items the model answers NO — measures correctly-variant response; it completes the three-way taxonomy (classes 1, 2, 3 all tested).

| regime | P1-ablate withhold (95% CI) | P2-ablate withhold (95% CI) | do(color) drop (95% CI) | do(name) ctrl | class-1 flip rate (95% CI) |
|---|---|---|---|---|---|
| chain-required (specificity) | 0.80 [0.78, 0.82] | 0.83 [0.82, 0.85] | 0.00 [0.00, 0.00] | 0.00 | **1.00 [1.00, 1.00]** |
| shortcut-available (sensitivity) | 0.49 [0.47, 0.52] | 0.52 [0.49, 0.54] | **0.50 [0.48, 0.53]** | 0.00 | 0.00 [0.00, 0.00] |

All three arms show the expected CI pattern. **Specificity:** when the model genuinely tracks the two-premise chain, ablating *either* premise drives withholding significantly above chance (CI lower bounds 0.78, 0.82 > 0.5) — each premise is shown counterfactually necessary *to the model* (3.2-ii made measurable, CIY-2 instrumented) — the procedure does not false-alarm on class-3 (do(color) drop CI = [0,0]), and the model correctly varies under the class-1 chain-break (flip rate 1.00, CI [1.00, 1.00]). **Sensitivity:** when the shortcut is made strictly more reliable, the model rides it; ablation-withholding decouples from the premises (CIs straddle chance), independently corroborated by the class-3 flag (do(color) collapses ≈50 points, CI [0.48, 0.53]) with the class-2 control intact ([0,0]), and the class-1 flip rate collapses to 0.00 [0.00, 0.00] — the model ignores the chain-break entirely, answering YES by color regardless. Three independent signals all point to the same model: the chain-tracker is correctly variant; the shortcut-rider is not. The three-way taxonomy (Table 3.4) is now fully exercised.

This completes the controlled demonstration. It makes no claim about real models or benchmarks (Section 6); residual limits (single seed-set; the shortcut-competition regime is constructed, not naturalistic) are stated in Section 7.

---

# 6. Application to a Trusted Benchmark

The operationalization above is controlled and synthetic. We now show the *measurement-only* half of the framework has bite on a benchmark the community already trusts: **BoolQ**.

This use is deliberately **non-causal**. BoolQ items carry no stipulated ground-truth graph, so applying the full clause-(iii) operationalization to them would reintroduce the very circularity the framework removes. The causal operationalization is therefore confined to the by-construction synthetic setting (Section 5); what is legitimate on a real benchmark is only the dual-metric, three-state decomposition, which needs nothing but gold labels and answer-preserving perturbations. Respecting that boundary is itself part of the contribution.

The same balanced **400-item** BoolQ sample was evaluated under four conservative answer-preserving presentations (original; an irrelevant appended sentence; two task-framing rephrasings with the question verbatim) using three zero-shot models spanning different capability levels and architecture families. All quantities are reported with bootstrap 95% CIs.

| quantity | Qwen2.5-1.5B | DeepSeek V4 Flash | Claude Haiku |
|---|---|---|---|
| headline accuracy | 0.763 [0.723, 0.803] | 0.883 [0.850, 0.913] | 0.905 [0.877, 0.933] |
| stable-correct fraction | 0.610 [0.565, 0.660] | 0.853 [0.818, 0.888] | 0.863 [0.827, 0.895] |
| **overstatement gap** | **+0.153 [0.118, 0.188]** | **+0.030 [0.015, 0.048]** | **+0.043 [0.023, 0.065]** |
| group states (of 400) | SC 244 / SW 51 / U 105 | SC 341 / SW 29 / U 30 | SC 345 / SW 25 / U 30 |
| SIS_consistency | 0.926 | 0.971 | 0.973 |

*SC = stable-correct, SW = stable-wrong, U = unstable.*

**Cross-model pattern.** All three models show a positive overstatement gap whose 95% CI excludes zero, confirming that the decomposition detects the same structural phenomenon across a capability range spanning roughly 30 headline-accuracy points. The gaps differ monotonically with capability: the weakest model (Qwen, headline 0.76) shows a gap of +0.153, while both stronger models (DeepSeek and Claude Haiku, headline ~0.88–0.91) converge to gaps of +0.030–0.043. This ordering is interpretively coherent: a more capable model is less susceptible to conservative answer-preserving edits, so its stable-correct fraction tracks closer to its headline figure. Critically, the framework does not collapse — it assigns a non-trivial gap to the stronger models too, rather than certifying them as fully stable. The 7.5% of items that remain unstable for both DeepSeek and Claude Haiku (30/400 each) are genuine answer flips under perturbations that, by construction, preserve the correct answer.

The three-model result provides two consistency checks on the instrument. First, it does not return the same verdict regardless of capability (it would be uninformative if it always returned a large gap). Second, it does not prematurely certify stronger models as stable (it would be uninformative in the other direction). The convergence of DeepSeek and Claude Haiku — two models from different organisations and architecture families — on nearly identical gap estimates (+0.030 vs +0.043) and identical unstable-item counts (30/400 each) also constitutes an independent replication of the measurement at the stronger-model tier.

The headline number (0.76) for Qwen conceals that only 61% of items are *stably* correct: 26% flip their answer under conservative answer-preserving edits, while a separate 13% are *stably wrong*. The decomposition separates two failure modes — fragility vs. confident error — that a single accuracy figure merges. An instrumentation check found 4/1600 unknown parses in the Qwen run, all from one item; excluding that item leaves results essentially unchanged. The unstable items are genuine answer flips rather than parser artifacts.

Scope, stated not hedged: single runs per model, deliberately conservative answer-preserving perturbations, measurement-only use (no causal claim). The three-model replication removes both the single-model and single-architecture objections; validated paraphrase/lexical perturbations with the semantic-preservation layer described in the instantiation protocol (Section 4) remain the publication-version extension, not claimed here.

---

# 7. Limits and What We Do *Not* Claim

- We do **not** claim a novel theory of causation; the causal-minimality core is borrowed and cited (Section 2). The contribution is the non-circular *operationalization*, not the causal concepts, and not the *problem* of shortcut-vs-causal evaluation (which the benchmark-critique literature already articulates — Section 2).
- We do **not** claim validation on capable models. Baselines are weak by design; Section 5 demonstrates the *procedure*, not model quality.
- We do **not** claim coverage of natural reasoning: the synthetic benchmark is template-generated; perturbations are more regular than human-written ones.
- We do **not** claim the operationalization extends to systems with unauditable training data; for frontier LLMs, 3.5(b) is an open estimation problem (stated in 3.5).

**Residual limits of the controlled demonstration.** Non-redundancy and the full three-way taxonomy are now demonstrated (Section 5.4), but one boundary remains: the sensitivity arm uses a *constructed* shortcut-competition regime (the class-3 feature is deliberately made strictly more reliable than the chain) — this is legitimate by-construction, but it is not a *naturalistic* shortcut, and we do not claim it is.

**Multi-seed variance (DistilBERT).** A 5-seed training variance check (seeds 42–46, eval seeds fixed at base 100 so variance is over training stochasticity only) was run for the DistilBERT §5.4 regimes. Results: the sensitivity arm (*shortcut_available*) is fully robust — all five seeds flag the correlational shortcut (*flagged_correlational* 5/5, *all_agree* = True) and correctly detect premise-decoupling failure (*nonredundancy_fails_decoupled* 5/5). The specificity arm (*chain_required*) shows a nuanced pattern: *c1_correctly_variant* passes 5/5 and no shortcut is flagged (0/5), but *P2_withhold* is consistently low (mean 0.033 ± 0.073), indicating that in this regime DistilBERT primarily relies on P1 presence rather than requiring both premises. This is a genuine finding, not an instrumentation artifact: the multi-seed check reveals that the P2 non-redundancy verdict has lower stability than the P1 verdict (P1_withhold 0.501 ± 0.091 vs. P2_withhold 0.033 ± 0.073). The overall *validated_across_all_seeds* flag is False on the strict criterion; the sensitivity half validates cleanly, the specificity half reveals that P2's gating role is weaker than P1's across training seeds. This is reported as-is — the framework's diagnostic value is precisely that it surfaces this asymmetry rather than returning a single pass/fail verdict.

**An instrumentation-discipline note.** A CoT-vs-stability pilot first appeared to show chain-of-thought nearly halving stable reasoning (stable-correct 75% → 57.5%). An instrumentation check traced ~85–90% of that to an answer-extraction artifact (a first-match parser locking onto a premature `Answer:` token); after fixing it the effect was at noise level. The lesson is methodological and on-thesis: the group-state decomposition plus an explicit instrumentation check is what caught a false headline, and binary "supported/not" verdicts flip on noise — argue for effect-size-with-variance reporting, not for the CoT claim (which we do not make).

These are scope statements, not hedges: each marks a concrete boundary that Section 8 is designed to push.

---

# 8. Future Work (Research Program)

This paper is the framework leg of a two-paper program; the companion empirical paper validates it. Planned extensions, ordered by what most strengthens the contribution:

1. **Harden the demonstration.** The three-way taxonomy is now fully exercised (Section 5.4); the remaining gaps are multi-seed training variance and a *naturalistic* (non-constructed) shortcut-competition regime.
2. **Strong-model validation** of CIY on instruction-tuned models, within the auditable-training-set boundary.
3. **Scaling the benchmark** beyond templated synthesis, including human-written perturbations with inter-annotator validation of semantic preservation.
4. **Frontier-LLM training-correlation estimation** to push 3.5(b) from measurement toward principled estimation where pretraining is unobservable.
5. **Sharper differentiation** from adjacent shortcut-vs-causal-reasoning work, treating the non-circular by-construction/measured split as the distinguishing contribution.

---

# 9. Conclusion

The contribution is a measurement methodology: a causal anchor for stability, a non-circular procedure for deciding what counts as an irrelevant variable, and a single quantity (CIY) that unifies several previously ad hoc reasoning-stability probes. Section 5 demonstrates the central claim in a controlled setting — correlational sufficiency is detected non-circularly, specifically (no false alarm when the model genuinely uses M), and non-vacuously (the class-2 control rules out "any intervention breaks it"), with the verdict stable across model classes and holding on a multi-premise structure where non-redundancy is a real, CI-tested property (5.4). Section 6 shows the measurement-only half has bite on a trusted public benchmark, while respecting the non-circularity boundary that forbids importing the causal machinery onto data without a stipulated graph. We do not claim strong models reason causally or fail to, nor that the framework is validated at scale; those are the program this paper sets up.

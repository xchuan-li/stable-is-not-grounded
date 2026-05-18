# Inferential Yield
## A Causally Grounded Evaluation Methodology for Stable Reasoning under Information Constraints

---

> **Positioning (read first).** This is an **evaluation-methodology** paper, not a new causal-theory paper. The causal concepts it uses (minimal sufficient causes, actual-causation minimality, the interventional rung of the causal hierarchy) are *borrowed tools*, cited as such. The contribution is a *measurement procedure for learned systems*: a non-circular way to decide what counts as an "irrelevant" variable, and a single quantity — Causal Inferential Yield (CIY) — that unifies several otherwise ad hoc reasoning-stability probes. Claims are scoped accordingly in Section 6.

---

# Abstract

A recurring question in evaluating learned reasoning systems is whether a model's correct answers reflect the causal structure that licenses the conclusion, or a correlation that happens to track it. Existing perturbation- and robustness-based evaluations probe this only indirectly, and they share an unstated circularity: deciding which perturbations *should* leave the answer unchanged presupposes knowledge of the causal structure being tested.

We present an evaluation methodology that removes this circularity. We anchor "stable inference" to a **Minimal Causal Structure (MCS)** — a minimal, non-redundant premise set whose sufficiency is required to be robust under interventions on variables outside it — and we define **Causal Inferential Yield (CIY)** as a model's ability to recover a conclusion *in units of* its MCS: correctly, minimally, and stably under intervention. The key methodological move is to split the circular question into two non-circular ones: whether a variable is causally irrelevant is fixed *by construction* (synthetic items whose ground-truth graph we stipulate), while whether it is entangled with the target is measured *empirically* on the model's training data.

We then show, on TF-IDF + Logistic Regression and fine-tuned DistilBERT baselines, that the framework's distinctions are **not vacuous**: a single aggregate accuracy number conceals that 63% of evaluated reasoning groups are unstable under semantics-preserving perturbation, and a conclusion drawn from a small-sample evaluation (that reasoning-path reformulation is the most damaging perturbation) is *reversed* once the framework's group-level analysis is applied at scale. Ignoring the distinctions therefore yields incorrect conclusions, which is the bar a measurement framework must clear.

We are explicit about what we do not claim (Section 6) and frame the open program — validation on strong models, intervention-level prompting, and frontier-LLM training-correlation estimation — as the intended line of subsequent work.

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

We separate *what we borrow* from *what is new*, because the novelty of the contribution depends on this boundary being explicit.

**Causal-minimality notions (borrowed).**
- *Mackie's INUS condition* — a cause as an insufficient but non-redundant part of an unnecessary but sufficient condition. Our non-redundancy clause is INUS non-redundancy.
- *Halpern & Pearl actual causation* — the but-for/minimality clause ("remove the element and the effect no longer follows") is reused directly.
- *Prime implicants / minimal models* — the minimal-sufficient-set structure.

**The causal hierarchy (borrowed).** The observation–intervention distinction (conditioning vs. the interventional rung) is used to separate rung-1 robustness (distractors) from the rung-2 property that makes a structure causal.

**Shortcut learning and perturbation evaluation (situated against).** Prior work documents shortcut reliance and uses paraphrase/adversarial/distractor perturbations to expose it. We do not add a new perturbation type; we supply the missing decision procedure for *which* perturbations are diagnostic and *of what*, and we show that conflating the categories under-detects shortcutting.

**What is new (scoped).** (a) The non-circular operationalization of "irrelevant" via construction-time stipulation + empirical training-correlation (Section 3.5). (b) The recasting of disparate stability probes as instances of a single CIY measure (Section 3.6). (c) Empirical evidence that these distinctions are non-vacuous (Section 5). We do **not** claim novelty for the underlying causal concepts.

> *Citations to be filled: Mackie 1965; Halpern & Pearl 2005; Pearl causal hierarchy; Geirhos et al. shortcut learning; relevance logic / logical omniscience; perturbation/robustness evaluation literature.*

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

# 5. Demonstration: The Distinctions Are Non-Vacuous

The empirical content here is intentionally modest. Its sole purpose is to show that **ignoring the framework's distinctions leads to measurably wrong conclusions** — the minimum bar for a measurement methodology.

**5.1 A benign aggregate hides systematic instability.** On a balanced 100-group, five-category synthetic benchmark, a fine-tuned DistilBERT yields `SIS_acc = 0.586` and `SIS_consistency = 0.790`. Read alone, the consistency figure looks acceptable. The framework's group-state decomposition shows otherwise:

| Group state | Share |
|---|---|
| stable-correct | 27% |
| stable-wrong | 10% |
| **unstable** | **63%** |

63% of groups change their answer under semantics-preserving perturbation. A single aggregate number conceals this; the framework's distinction surfaces it.

**5.2 A small-sample conclusion is reversed.** On a 4-group taxonomic-only evaluation, `reasoning_path_shift` appeared to be the most damaging perturbation (accuracy 0.000) — a tempting headline finding. Applying the framework's protocol at scale (100 balanced groups) **reverses** this: `reasoning_path_shift` is the *least* damaging (0.660). The earlier conclusion was small-sample noise; the framework's scaling discipline catches it.

**5.3 Shortcut-reversal is a clause-(iii) test.** A TF-IDF + LR baseline reaches 0.910 IID accuracy and collapses monotonically to 0.000 as a class-3 variable's correlation is reversed. In the framework's language this is a direct measurement that the model's apparent sufficiency is not robust under a class-3 `do()` — its `M` was correlational, not causal. The experiment was already a clause-(iii) test; the framework makes that explicit and distinguishes it from the rung-1 distractor cousin.

**5.4 A headline-friendly effect that was mostly an artifact.** We probed whether chain-of-thought (CoT) prompting trades accuracy for stability on a balanced 40-group subset (Qwen2.5-1.5B, greedy decoding, DIRECT vs. CoT). A first run produced a striking result:

| metric | DIRECT | CoT (run 1) | CoT (run 2) |
|---|---|---|---|
| SIS_acc | 0.910 | 0.855 (−0.055) | 0.915 (+0.005) |
| stable-correct | 75.0% | 57.5% (**−17.5pt**) | 72.5% (−2.5pt) |
| unstable | 22.5% | 42.5% (**+20pt**) | 27.5% (+5pt) |

Run 1 appears to show CoT nearly halving stable reasoning — a publishable headline. Inspection of raw outputs showed the small model frequently emits a leading `Answer: X` and then reasons toward the opposite (often truncated before a final answer line); a first-match answer extractor locked onto the premature token, systematically and *directionally* mis-scoring the CoT condition. After fixing the extractor (take the model's final committed answer) and the prompt/length, run 2 shows the true picture: roughly **85–90% of the apparent CoT destabilization was an answer-extraction artifact**; the residual effect is at noise level (Δ`SIS_acc` = +0.005 ≈ one sample; ±1–2 groups at n = 40).

Two methodological points follow, both on-thesis for an evaluation methodology paper. First, the framework's group-state decomposition plus an explicit instrumentation check is what exposed the artifact — naive aggregate scoring would have reported the run-1 headline. Second, the experiment's binary "supported / not supported" verdict flips (NOT supported → SUPPORTED) on noise-level differences, which argues against binary verdicts and for effect-size-with-variance reporting — exactly what the group-state metric, run with seeds, supports. (Caveats: single small model, single run, synthetic benchmark, prompt-induced CoT; this is a cautionary pilot, not a CoT result.)

**5.5 The instrument, turned on a trusted benchmark, finds hidden instability.** The preceding subsections use synthetic or own-task data. We now turn the *measurement-only* half of the framework on BoolQ, a widely used public benchmark. This deliberately uses **no causal structure**: BoolQ items carry no stipulated ground-truth graph, so applying the full clause-(iii) operationalization here would reintroduce the very circularity the framework removes. The causal operationalization is therefore confined to the by-construction synthetic setting; what is legitimate on a real benchmark is the dual-metric, three-state decomposition, which needs only gold labels and answer-preserving perturbations. On a balanced 40-item BoolQ sample, zero-shot Qwen2.5-1.5B was evaluated under four presentations that *cannot* change the correct answer by construction (original; an irrelevant appended sentence; two task-framing rephrasings with the question verbatim):

| quantity | value |
|---|---|
| headline accuracy (original prompt only) | 0.800 |
| stable-correct fraction (all 4 variants correct) | 0.675 |
| **headline overstatement of stable correctness** | **+0.125** |
| group states (of 40) | stable-correct 27 / stable-wrong 3 / **unstable 10** |
| per-variant accuracy | original .800 / distractor .825 / frame_a .800 / frame_b .775 |

The standard headline number (0.800) conceals that only 67.5% of items are *stably* correct: 25% flip their yes/no answer under edits that provably preserve the answer, while a separate 3 items are *stably wrong* (confidently, invariantly incorrect). The decomposition thus separates two failure modes — fragility vs. confident error — that a single accuracy figure merges. An instrumentation check confirmed the effect is genuine, not an extraction artifact: 0/160 parse failures, and all 10 unstable items are true yes↔no flips. Scope (stated, not hedged): single small model, single run, n = 40, and intentionally *conservative* perturbations — so 0.125 is a **lower bound** on instability, not an effect-size estimate; validated paraphrase/lexical perturbations with the semantic-preservation layer (Section 3.7) are the publication-version extension, not claimed here.

Together: the distinctions are not idle bookkeeping — discarding them yields a falsely reassuring number (5.1), a wrong empirical conclusion (5.2), a causal test mislabeled as a robustness test (5.3), a near-fully artifactual headline absent instrumentation discipline (5.4), and — on a benchmark the community trusts — a headline accuracy that overstates stable correctness while masking the fragile-vs-confidently-wrong distinction (5.5).

---

# 6. What We Do *Not* Claim

- We do **not** claim a novel theory of causation; the causal-minimality core is borrowed and cited (Section 2).
- We do **not** claim the empirical results validate the framework on capable models. The baselines are weak by design; Section 5 demonstrates non-vacuity, not model quality.
- We do **not** claim coverage of natural reasoning: the benchmark is synthetic and template-generated; perturbations are more regular than human-written ones.
- We do **not** claim the operationalization extends to systems with unauditable training data; for frontier LLMs, 3.5(b) is an open estimation problem (stated in 3.5).
- We do **not** claim CIY is fully formalized; CIY-2 (withholding) is specified but not yet empirically instrumented here.

These are scope statements, not hedges: each marks a concrete boundary that the subsequent program is designed to push.

---

# 7. Future Work (Research Program)

This paper is the framework leg of a two-paper program; the companion empirical paper validates it. Planned extensions:

1. **Strong-model validation.** Apply CIY to instruction-tuned and frontier models; the present demonstration only establishes non-vacuity on weak baselines.
2. **Intervention-level prompting and the CoT question.** Test whether chain-of-thought raises accuracy while lowering clause-(iii) stability (the "CoT improves correctness but degrades causal robustness" hypothesis) — a clean A/B with the reasoning mode as the only varied factor.
3. **Scaling the benchmark** beyond templated synthesis, including human-written perturbations, with inter-annotator validation of semantic preservation.
4. **Frontier-LLM training-correlation estimation** to push 3.5(b) from measurement toward principled estimation where pretraining is unobservable.
5. **CIY-2 instrumentation** (withholding under partial premises) as a first-class metric.

---

# 8. Conclusion

The contribution is a measurement methodology: a causal anchor for stability, a non-circular procedure for deciding what counts as an irrelevant variable, and a single quantity (CIY) that unifies several previously ad hoc reasoning-stability probes. The empirical section does not claim strong models reason causally or fail to; it shows only — but conclusively at this scale — that the framework's distinctions are non-vacuous, since discarding them produces a falsely reassuring aggregate and a reversed empirical conclusion. The validation of the framework on capable systems is deliberately left as the program this paper sets up.

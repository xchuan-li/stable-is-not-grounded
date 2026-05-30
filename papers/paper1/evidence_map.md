# Paper 1 Evidence Map

| Draft claim | Draft §  | Experiment | Code | Result |
|---|---:|---|---|---|
| Class-3 intervention detects shortcut-riding without circularly defining irrelevance | 6.1 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| Mandatory class-2 control stays robust, ruling out "any intervention breaks it" | 6.1 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| Verdict is not an architecture artifact (TF-IDF / DistilBERT / Qwen-LoRA) | 6.2 | `experiments/paper1/01_mcs_operationalization` | `run_distilbert.py` | `results/distilbert_summary.json` |
| Multi-premise non-redundancy is a real, CI-tested property via premise ablations | 6.3 | `experiments/paper1/02_multipremise_nonredundancy` | `run_distilbert.py` | `results/summary.json` |
| Class-1 correctly-variant arm: model flips on a causally relevant chain-break | 6.3 | `experiments/paper1/02_multipremise_nonredundancy` | `run_distilbert.py` | `results/summary.json` |
| Verdict is robust over training seeds (sensitivity 5/5; P1/P2 asymmetry surfaced, not hidden) | 6.3 | `experiments/paper1/06_multiseed_variance` | `run_distilbert_multiseed.py` | `results_distilbert_multiseed/multiseed_summary.json` |
| Two-premise verdict reproduces on a 2nd architecture across 3 seeds (Qwen-LoRA: grounded arm drop 0.00 / c1-flip 1.00 / non-redundant 3/3; shortcut arm drop 0.517 [0.491,0.542], flagged 3/3, c1-flip 0.00). NOTE this is the two-premise (§6.3) setup, |r|=1.0 — **not** the single-premise §6.1 regime-A/B; the old §6.1 "+.501" Qwen cell was a mis-paste of this and has been removed | 6.3 | `experiments/paper1/06_multiseed_variance` | `run_qwen_multiseed_kaggle.py` | `results_qwen_multiseed/multiseed_summary.json` |
| Semantic preservation needs an explicit validation layer | 5 | `experiments/paper1/05_semantic_validation` | `validate_semantics.py` | `results/semantic_validation_results.csv` |
| (Secondary lens, measurement-only) Headline accuracy overstates stable-correct fraction | 6.5 | `experiments/paper1/03_boolq_measurement` | `run_qwen_boolq.py`, `run_deepseek_boolq.py`, `run_claude_boolq.py` | `results/summary_400.json`, `results/summary_deepseek_400.json`, `results/summary_claude_*.json` |
| (Portability lower-bound, §7) `do(class-3)` flag does **not** fire on a weak never-fine-tuned model (Qwen2.5-1.5B: drop +0.048 [−0.027, 0.122], CI crosses 0) — but model is degenerate (yes-biased, fails non-redundancy + class-1) → specificity lower-bound only | 7 | `experiments/paper1/07_portability_check` | `run_zeroshot_portability.py` | `results/zeroshot_portability_summary.json` |
| (Portability, competent witness, §7) `do(class-3)` flag does **not** fire on a competent never-fine-tuned reasoner (DeepSeek-V4-Flash, n=200: baseline 1.0/0.995, class-1 flip ≈1.0 correctly-variant; do(color) drop −0.005 [−0.015, 0.0]) vs +0.517 same-ladder fine-tuned Qwen-LoRA (≈0.50 DistilBERT) → flag indexes **trained-in** entanglement, not feature presence. **Caveat:** real-category chains let it back-fill absent premises from world knowledge (withhold ≈0.35–0.43) — orthogonal to shortcut result | 7 | `experiments/paper1/07_portability_check` | `run_api_portability.py` | `results/api_portability_deepseek_summary.json` |
| (Naturalistic licensed witness, §6.4) The `SC → {grounded,spurious}` cut ports off our own templates: on HANS (overlap-irrelevance stipulated by construction §3.2a; own DistilBERT fine-tuned on MNLI → auditable corr(overlap,entailment)=0.36 §3.2b), the `do(class-3)` analog (HANS non-entailment slice) fires **`SC-spurious`** — MNLI matched 0.73, HANS entailment 0.98, non-entailment 0.03, shortcut drop **0.947** (3 seeds, 50k/1ep). Only the spurious direction, single architecture | 6.4 | `experiments/paper1/08_mnli_hans` | `run_mnli_hans.py` | `results/mnli_hans_summary.json` |

## Scope discipline (per §3.4)

- The causal procedure (§6.1–6.4) runs only in the licensed setting (graph stipulated + training auditable) — synthetic in §6.1–6.3, naturalistic HANS in §6.4; the §6.5 BoolQ decomposition is assumption-free measurement only, no causal claim.
- CIY is removed from this paper; there is no unifying-quantity claim to evidence.

## Current limits (mirror of §7)

- The primary ladder is synthetic / by-construction (the §6.3 sensitivity regime is a *constructed* shortcut-competition). The §6.4 HANS run now adds **one naturalistic licensed witness** — independently-authored NLI text, graph stipulated by HANS itself — but only in the `SC-spurious` direction and on a single architecture; a naturalistic `SC-grounded` witness remains open.
- Every primary-claim model is fine-tuned on the distribution it is evaluated on. Procedure-portability run (07) now covers both ends: weak model (Qwen-1.5B) = specificity lower-bound; **competent model (DeepSeek-V4-Flash) = competent witness** — baseline 1.0, class-1-variant, yet do(class-3) ≈ 0 under a perfect in-context shortcut. Still open: portability of the `SC-grounded` verdict on a by-construction grounded item.
- Main ladder is single seed-set; only §6.3 DistilBERT regimes have a multi-seed check.
- §3.2(b) requires an auditable training set; for frontier LLMs it degrades to estimation.

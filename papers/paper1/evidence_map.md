# Paper 1 Evidence Map

| Draft claim | Draft §  | Experiment | Code | Result |
|---|---:|---|---|---|
| Class-3 intervention detects shortcut-riding without circularly defining irrelevance | 5.1–5.2 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| Mandatory class-2 control stays robust, ruling out "any intervention breaks it" | 5.2 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| Verdict is not an architecture artifact (TF-IDF / DistilBERT / Qwen-LoRA) | 5.3 | `experiments/paper1/01_mcs_operationalization` | `run_distilbert.py` | `results/distilbert_summary.json` |
| Multi-premise non-redundancy is a real, CI-tested property via premise ablations | 5.4 | `experiments/paper1/02_multipremise_nonredundancy` | `run_distilbert.py` | `results/summary.json` |
| Class-1 correctly-variant arm: model flips on a causally relevant chain-break | 5.4 | `experiments/paper1/02_multipremise_nonredundancy` | `run_distilbert.py` | `results/summary.json` |
| Verdict is robust over training seeds (sensitivity 5/5; P1/P2 asymmetry surfaced, not hidden) | 5.4 | `experiments/paper1/06_multiseed_variance` | `run_distilbert_multiseed.py` | `results_distilbert_multiseed/multiseed_summary.json` |
| Semantic preservation needs an explicit validation layer | 4 | `experiments/paper1/05_semantic_validation` | `validate_semantics.py` | `results/semantic_validation_results.csv` |
| (Secondary lens, measurement-only) Headline accuracy overstates stable-correct fraction | 6 | `experiments/paper1/03_boolq_measurement` | `run_qwen_boolq.py`, `run_deepseek_boolq.py`, `run_claude_boolq.py` | `results/summary_400.json`, `results/summary_deepseek_400.json`, `results/summary_claude_*.json` |

## Scope discipline (per §3.4)

- The causal procedure (§5) is confined to the by-construction synthetic setting; the §6 BoolQ decomposition is assumption-free measurement only, no causal claim.
- CIY is removed from this paper; there is no unifying-quantity claim to evidence.

## Current limits (mirror of §7)

- Synthetic, by-construction, not naturalistic; the §5.4 sensitivity regime is a *constructed* shortcut-competition.
- Every primary-claim model is fine-tuned on the distribution it is evaluated on — procedure-portability on a non-fine-tuned model is the open next step (§8), not claimed.
- Main ladder is single seed-set; only §5.4 DistilBERT regimes have a multi-seed check.
- §3.2(b) requires an auditable training set; for frontier LLMs it degrades to estimation.

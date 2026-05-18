# Paper 1 Evidence Map

| Draft claim | Draft section | Experiment | Code | Result |
|---|---:|---|---|---|
| Class-3 intervention detects shortcut-riding without circularly defining irrelevance | 5.1-5.2 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| Class-2 control remains robust, ruling out "any intervention breaks it" | 5.1-5.2 | `experiments/paper1/01_mcs_operationalization` | `run_tfidf.py` | `results/tfidf_summary.json` |
| The operationalization is not only a TF-IDF artifact | 5.3 | `experiments/paper1/01_mcs_operationalization` | `run_distilbert.py` | `results/distilbert_summary.json` |
| Multi-premise non-redundancy can be tested with premise ablations | 5.4 | `experiments/paper1/02_multipremise_nonredundancy` | `run_distilbert.py` | `results/summary.json` |
| BoolQ headline accuracy overstates stable correctness | 6 | `experiments/paper1/03_boolq_measurement` | `run_qwen_boolq.py` | `results/summary_400.json`, `results/raw_400.csv` |
| Group-state decomposition separates stable-correct, stable-wrong, and unstable behavior | 6 | `experiments/paper1/03_boolq_measurement` and `04_dual_metric_synthetic` | `run_qwen_boolq.py`, `run_distilbert.py` | `results/summary_400.json`, `results/dual_metric_summary.json` |
| Semantic preservation needs an explicit validation layer | 4, 6 | `experiments/paper1/05_semantic_validation` | `validate_semantics.py` | `results/semantic_validation_results.csv` |

## Current Limits

- Synthetic causal operationalization is by construction, not naturalistic.
- DistilBERT runs are single-seed unless rerun with additional seeds.
- Class-1 relevant-variable intervention arm is not yet implemented.
- BoolQ is measurement-only; it does not support a full causal claim.


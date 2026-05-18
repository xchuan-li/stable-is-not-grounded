# Experiment 03: BoolQ Measurement

Paper section:
- 6

Claim supported:
- Headline original-prompt accuracy can overstate stable correctness.
- The dual metric separates stable-correct, stable-wrong, and unstable groups.

Run:

```bash
python run_qwen_boolq.py --sample data/boolq_sample_400.jsonl
```

Main outputs:

```text
results/summary_400.json
results/raw_400.csv
```

Note:
- The script writes rerun outputs using its internal names, e.g. `boolq_targeting_summary_400.json`.
- This experiment is measurement-only. BoolQ has no stipulated DAG, so it does not support a full causal operationalization claim.


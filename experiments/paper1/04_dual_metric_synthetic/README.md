# Experiment 04: Dual Metric Synthetic Benchmark

Paper section:
- 6, supporting comparison

Claim supported:
- SIS accuracy alone hides stable-wrong and unstable behavior.
- The group-state decomposition exposes these modes on a 100-group synthetic perturbation benchmark.

Generate benchmark:

```bash
python generate_benchmark.py
```

Run:

```bash
python run_distilbert.py
```

Outputs:

```text
results/dual_metric_summary.json
results/dual_metric_raw.csv
```

Known limits:
- Template-generated perturbations.
- The saved DistilBERT model is included for reproducibility, but reruns may differ if retrained.


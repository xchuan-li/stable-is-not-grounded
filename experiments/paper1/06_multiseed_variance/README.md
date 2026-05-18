# 06 — Multi-seed Variance

Removes the residual limitation in §7: *"single seed-set fine-tune per regime — multi-seed variance is not yet reported."*

## Design

Variance is measured over **training stochasticity** (model init + shuffle seed), not eval-set sampling.  Eval generation seeds are fixed (`EVAL_SEED_BASE=100`) so the same evaluation items are used across all training seeds.

N = 5 training seeds: [42, 43, 44, 45, 46]

## Files

| File | Where to run | Notes |
|---|---|---|
| `run_distilbert_multiseed.py` | Mac M4 (MPS) | ~30–50 min total |
| `run_qwen_multiseed_kaggle.py` | Kaggle T4 GPU | ~4–5 hours total |

## Kaggle setup

1. New Notebook → Settings → Accelerator → **GPU T4 x1**
2. Add Qwen2.5-1.5B from Kaggle Models
3. First cell: `!pip install peft -q`
4. Upload `run_qwen_multiseed_kaggle.py` and run

## Output

Each script saves:
- Per-seed checkpoint: `results_*/chain_required_seed{N}.json`
- Aggregated summary: `results_*/multiseed_summary.json`

The summary reports `mean ± std` across seeds for every metric and a `validated_across_all_seeds` boolean.

## What to report in paper

Replace the single-seed table in §5.4 with:

```
metric              chain_required       shortcut_available
P1_withhold         mean ± std           mean ± std
P2_withhold         mean ± std           mean ± std
drop_color          mean ± std           mean ± std
drop_name           mean ± std           mean ± std
c1_flip_rate        mean ± std           mean ± std
verdict (n/5 seeds) 5/5                  5/5
```

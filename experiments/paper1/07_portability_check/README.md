# 07 — §8 procedure-portability check (zero-shot, no fine-tune)

Addresses the principal objection to §5: a self-constructed shortcut detected
on a self-trained model. Reruns the **same §5.4 two-premise ladder** on a
model that was never fine-tuned on the construction.

**Boundary discipline (reframed §8 / §3.4).** For a model with unauditable
pretraining, §3.2(b) is *not* re-measured on the model. The class label
(`color` = class-3, `name` = class-2) is **imported** from the construction's
own controlled data statistics — which we own. The check tests only whether
the verdict *structure* transfers; it is not a re-measured (b) and not a
capability claim.

**No outcome is assumed.** A flag that does *not* reproduce in
`shortcut_available` is itself informative: it would show the §5 positive
verdict is specific to models that actually trained on the entanglement.
Report the verdict structure as-is.

## Run (local, no API key)

```bash
python run_zeroshot_portability.py                       # default Qwen2.5-1.5B-Instruct
python run_zeroshot_portability.py --n 300               # faster smoke run
python run_zeroshot_portability.py --model <hf-instruct-model>
```

Output: `results/zeroshot_portability_summary.json` (same schema as
`02_multipremise_nonredundancy/results/summary.json`, with `imported_*`
keys making the boundary explicit). Status: **script ready, not yet run** —
results must not be written into the manuscript until this is executed.

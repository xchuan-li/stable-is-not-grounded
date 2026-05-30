# Experiment 08: MNLI→HANS — Naturalistic Licensed SC-spurious Witness

Paper section:
- 6 (a naturalistic instance where the cut is *licensed* and fires), and §7 limits
  (relaxes the "synthetic / by-construction only" caveat).

Claim supported:
- The non-circular `SC → {grounded, spurious}` procedure is **not confined to our own
  synthetic templates**. On a pre-existing naturalistic NLI benchmark (HANS, McCoy et
  al. 2019) it fires `SC-spurious` — because HANS supplies exactly the two-sided
  observability the cut requires:
  - **§3.2(a) by construction** — HANS *stipulates* overlap-irrelevance: its
    non-entailment cases have HIGH lexical overlap but gold = non-entailment.
  - **§3.2(b) by measurement (auditable)** — we fine-tune our *own* DistilBERT on an
    MNLI subset, so the training set is fully auditable; measured
    corr(overlap, entailment) on MNLI-train = **0.356**.
  - **do(class-3) analog** — the HANS non-entailment slice (overlap severed from label).

Framework mapping:

| framework slot | naturalistic instantiation |
|---|---|
| shortcut `S` | lexical overlap (hypothesis tokens contained in premise) |
| training entanglement (§3.2b) | corr(overlap, entailment), MNLI-train = 0.356 |
| by-construction irrelevance (§3.2a) | HANS non-entailment = high overlap, gold non-entailment |
| `do(class-3)` analog | HANS non-entailment slice |

Result (canonical run: DistilBERT, 50k MNLI subset, 1 epoch, 3 seeds; ranges = min/max):
- MNLI matched accuracy: **0.726–0.744** (competent baseline)
- HANS entailment (heuristic agrees): **0.973–0.978**
- HANS non-entailment (heuristic severed): **0.026–0.031**
- shortcut drop (entailment − non-entailment): **0.947**
- **Verdict: `SC-spurious` (rides lexical overlap).**

Run:

```bash
caffeinate -is python run_mnli_hans.py --train_n 50000 --epochs 1 --batch 32
```

Outputs:

```text
results/mnli_hans_summary.json          # canonical (latest = 50k / 1ep)
results/mnli_hans_summary_50k_1ep.json   # archived copy of the 50k / 1ep run
results/model/                           # fine-tuned DistilBERT checkpoint
```

Known limits:
- Single architecture (DistilBERT, 66M); the verdict is not yet cross-architecture here.
- Canonical run is 50k MNLI / 1 epoch. A larger 100k / 2-epoch run was started
  (`run_100k.log`, `run_100k_v2.log`) but **did not complete** (stopped mid epoch 1,
  no summary written) — not used.
- Only the `SC-spurious` direction is demonstrated naturalistically. A naturalistic
  `SC-grounded` witness (a model high on *both* HANS subsets) remains open — the same
  open thread noted in §7.
- HANS is itself a curated diagnostic, not a deployment distribution; "naturalistic"
  here means real, independently-authored NLI text, not our templates.

# Experiment 09: Cross-Lingual Subject–Verb Agreement — the natural `SC-grounded` witness

**Status:** scaffold + pipeline smoke-tested (gpt2 on the demo set runs end-to-end).
**Real suites + runs are PENDING** — this fills the *results-forthcoming* table in
paper §6.4 (`\label{sec:agreement}`).

Paper section:
- §6.4 — lifts the `SC → {grounded, spurious}` cut onto **linguistic structure**;
  supplies the natural `SC-grounded` witness that HANS (§6.5, single-direction
  `SC-spurious`) cannot.
- §4.2 — the *eval-contingency* corollary, made concrete: a proximity-rider's
  ranking inverts **across languages**.
- §7 — relaxes the "naturalistic `SC-grounded` witness remains open" limit.

## Claim supported

Grammar is a **naturally-occurring stipulated graph**: the verb agrees with its
syntactic *subject* (the head), not with a linearly closer noun. So "linear
proximity is irrelevant" is fixed **by construction** (§3.2a), non-circular by
kind. A model high on **both** the aligned and the attractor slice *tracks the
subject* (`SC-grounded`); one that holds on aligned but collapses on attractor
*rides proximity* (`SC-spurious`). Typology severs proximity from the subject at
different rates across languages, so the same model can flip verdict between
(e.g.) English and German with neither model nor task touched.

## Framework mapping

| framework slot | agreement instantiation |
|---|---|
| shortcut `S` | "agree with the linearly nearest noun" (proximity cue) |
| by-construction irrelevance (§3.2a) | attractor item: intervening noun of the **opposite** number; grammar stipulates proximity irrelevant |
| `do(class-3)` analog | the **attractor** (number-mismatch) slice — proximity severed from the label |
| `class-2` control | the number-**match** intervener slice — present but no opposing cue; must stay intact |
| aligned | simple agreement, no intervener (high-baseline slice) |
| `SC-grounded` | high on **both** aligned and attractor |
| `SC-spurious` | high on aligned, collapses on attractor, control intact |

## Two regimes (note on §3.2b observability)

- **Zero-shot pretrained LM = PORTABILITY check (§7).** §3.2(a) is open (grammar),
  but §3.2(b) — auditable `corr(proximity, agreement)` in the *training* corpus —
  is **dark** under opaque pretraining. So a zero-shot verdict is portability, not
  the fully-licensed cut (same status as the BoolQ/DeepSeek runs).
- **Licensed = train our own small LM on an auditable corpus** and *measure*
  `corr(linear-nearest-noun-number, agreement)` there (mirrors how 08_mnli_hans
  fine-tunes our own DistilBERT). This is the both-sides-open cut. **TODO** — this
  script is already the eval engine for it; the training half is the next step.

`run_crosslingual_agreement.py` is the shared eval engine: pass `--tag zeroshot`
or `--tag licensed`.

## Data

Build the normalized minimal-pair JSONL with `prepare_clams.py`:
- **CLAMS** (Mueller et al. 2020, github.com/aaronmueller/clams) — en/de/fr/ru/he
  attractor suites. Primary claim = **en + de**.
- **Marvin & Linzen 2018** (en) — targeted syntactic evaluation.

Normalized line: `{"lang","condition","slice","good","bad"}`, `good` = grammatical,
`bad` = its agreement-violating twin, `slice ∈ {aligned, attractor, control}`.
`prepare_clams.py` assigns the slice via `CONDITION_SLICE` — **verify those
construction names against your clone**, and make sure the suite separates
number-match (control) from number-mismatch (attractor), or the control slice is
empty and the verdict reads `mixed`.

## Run

```bash
# 0) validate metric/verdict logic (no model, no data)
python run_crosslingual_agreement.py --selftest

# 1) smoke-test the full scoring path on the hand-authored demo set
python prepare_clams.py --demo
python run_crosslingual_agreement.py --pairs data/en_demo.jsonl --model gpt2 --lang en --tag smoketest

# 2) real run: convert a suite, then score >=2 architectures per language
python prepare_clams.py --tsv path/to/clams_en.tsv --lang en --out data/en.jsonl
python run_crosslingual_agreement.py --pairs data/en.jsonl --model gpt2   --lang en --tag zeroshot
python run_crosslingual_agreement.py --pairs data/de.jsonl --model dbmdz/german-gpt2 --lang de --tag zeroshot

# cluster: submit the en+de x {>=2 models} grid
sbatch run_agreement.sbatch
```

Suggested models (≥2 architectures per language): EN `gpt2`, `distilgpt2`,
`EleutherAI/pythia-160m`; DE `dbmdz/german-gpt2`, `malteos/gpt2-wechsel-german`;
multilingual `facebook/xglm-564M`, `ai-forever/mGPT`.

## Outputs

```text
results/agreement_<lang>_<model>_<tag>.json   # per slice acc + CIs, do(class-3) drop,
                                              # class-2 control gap, per-condition, verdict
data/<lang>.jsonl                             # normalized minimal pairs
data/<lang>_demo.jsonl                        # hand-authored smoke-test set (NOT eval data)
```

## Known limits / open

- Scoring is causal-LM minimal-pair surprisal (BLiMP/Marvin-Linzen style). Masked
  LMs (BERT-family) would need pseudo-log-likelihood — not yet implemented.
- The verdict thresholds (`--grounded_thr 0.80 --drop_thr 0.20 --control_eps 0.10`)
  are heuristics for the **auto-label** only; the paper reports the slice numbers
  and CIs, not the threshold.
- Strict §3.2(b) auditability needs the *licensed* (train-our-own) variant; the
  zero-shot grid is a portability check until then.
- Demo German pairs are illustrative; the real DE evaluation comes from CLAMS.

# Experiment 09: Cross-Lingual Subject–Verb Agreement — the natural `SC-grounded` witness

**Status:** EN + DE done (zero-shot / portability regime). Licensed
(train-our-own) variant still pending. Combined numbers in
`results/summary_crosslingual.json`.

Real CLAMS runs, capped & balanced (2000/slice, seed 42). EN uses prep_anim +
obj_rel_across_anim; DE uses obj_rel_across_anim (number read from the agreeing
verb, since German nouns are number-invariant).

| lang | model | aligned | attractor | control | ctl−att | ctl gap | control intact | verdict |
|---|---|---|---|---|---|---|---|---|
| en | gpt2 | 0.904 | 0.813 | 0.867 | 0.054 | 0.038 | yes | SC-grounded (borderline; attractor CI dips < .80) |
| en | distilgpt2 | 0.905 | 0.686 | 0.879 | 0.193 | 0.026 | yes | **SC-spurious** (rides proximity) |
| en | xglm-564M | 0.975 | 0.692 | 0.898 | 0.207 | 0.076 | yes | **SC-spurious** |
| en | pythia-160m | 0.911 | 0.532 | 0.750 | 0.218 | 0.161 | **no** | mixed (class-2 control vetoes) |
| de | german-gpt2 | 0.985 | 0.821 | 0.943 | 0.122 | 0.042 | yes | SC-grounded |
| de | xglm-564M | 0.987 | 0.889 | 0.986 | 0.097 | 0.001 | yes | SC-grounded |

**The cross-lingual flip (the §4.3 witness):** the *same* model, `xglm-564M`,
is **SC-spurious on English** but **SC-grounded on German** — with neither model
nor task touched, only the language. Construction-matched (obj_rel_across_anim in
both languages, the clean comparison): **en ctl−att 0.262 vs de ctl−att 0.097**
(the pooled-en 0.207 above is diluted by the easier prep_anim; matched is
stronger). German's richer agreement morphology severs proximity from the subject
more saliently, so the cut's verdict is eval-contingent across typology. This is
the naturalistic `SC-grounded` witness §7 listed as open.

Two more findings worth keeping: the class-2 control actively **vetoes**
pythia-160m (collapses even on same-number interveners → "any intervener hurts",
not a clean proximity rider); and the verdict varies across English models
(gpt2 grounded-ish, distilgpt2/xglm spurious).

**Caveats:** zero-shot = PORTABILITY (§3.2b dark — read labels in that sense).
The cleaner shortcut contrast is **ctl−att** (control and attractor share
structure; aligned does not), not the engine's current aligned−att `drop`.

> Data prep: `python prepare_clams.py --clams_dir /path/to/clams --lang en|de --cap 2000 --seed 42 --out data/<lang>.jsonl`
> (clone github.com/aaronmueller/clams; the `--clams_dir` mode slices by subject/intervener number — EN by noun morphology, DE by the agreeing verb).

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

"""
run_crosslingual_agreement.py — linguistic-structure instantiation of the
SC-grounded / SC-spurious cut on subject-verb agreement (paper §6.4,
\\label{sec:agreement}).

Why agreement: GRAMMAR is a naturally-occurring stipulated graph. The verb
agrees with its syntactic SUBJECT (the head), NOT with a linearly closer noun.
So "linear proximity is irrelevant" is fixed by the grammar of the language,
not inferred from the model -> §3.2(a) holds BY CONSTRUCTION, non-circular by
kind. Typology turns it into a natural experiment: the same dependency severs
proximity from the subject at different rates across languages.

Framework mapping (mirrors 08_mnli_hans):
  shortcut S          = "agree with the linearly nearest noun" (proximity cue)
  by-construction     = an ATTRACTOR item: an intervening noun of the OPPOSITE
   irrelevance §3.2a    number between subject and verb. Grammar says ignore it;
                        proximity says use it.
  do(class-3) analog  = the attractor (number-MISMATCH) slice — proximity cue
                        severed from the agreement label.
  class-2 control     = the number-MATCH intervener slice — an intervening noun
                        carrying NO opposing proximity cue. Must stay intact, or
                        the model is merely "any intervening noun breaks me".
  aligned             = simple agreement, no intervener (high-baseline slice).

Verdict (per language, per model):
  high on aligned AND attractor (both >= --grounded_thr)       -> SC-grounded
  high on aligned, COLLAPSES on attractor, control intact      -> SC-spurious
  otherwise                                                    -> mixed

Scoring is minimal-pair surprisal (BLiMP / Marvin-Linzen style): for each
(good, bad) pair we take an AutoModelForCausalLM, sum token log-probs of each
full sentence, and count the pair correct iff logprob(good) > logprob(bad).
Works zero-shot for any causal LM in any language; no fine-tuning needed.

NOTE on observability: zero-shot on an opaque-pretrained LM leaves §3.2(b)
(auditable training entanglement) DARK — so that run is a PORTABILITY check
(cf. §7), not the fully-licensed cut. The fully-licensed variant trains a small
LM on an auditable corpus and measures corr(proximity, agreement) there; this
script is the eval engine for both — pass --tag to label the regime.

Input: a normalized JSONL of minimal pairs (one object per line):
  {"lang":"en","condition":"obj_rel","slice":"attractor","good":"...","bad":"..."}
  slice in {aligned, attractor, control}.  Build it from CLAMS / Marvin-Linzen
  with prepare_clams.py (see README).

Usage:
  python run_crosslingual_agreement.py --pairs data/en.jsonl --model gpt2 --lang en
  python run_crosslingual_agreement.py --selftest      # validate metric logic, no model
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
BASE = Path(__file__).parent
RESULTS = BASE / "results"
SLICES = ("aligned", "attractor", "control")


def bootstrap_ci(correct, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(correct, float)
    if len(v) == 0:
        return (0.0, 0.0, 0.0)
    pt = float(v.mean())
    boots = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(pt, 4), round(float(lo), 4), round(float(hi), 4)


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            if o.get("slice") not in SLICES:
                raise ValueError(f"bad slice {o.get('slice')!r}; expected one of {SLICES}")
            for k in ("good", "bad"):
                if not o.get(k):
                    raise ValueError(f"pair missing {k}: {o}")
            pairs.append(o)
    return pairs


def score_minimal_pairs(pairs, model_name, device, batch=16, max_len=128):
    """Return a 0/1 correctness array aligned with `pairs` (good > bad logprob)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def logprobs(sents):
        out = []
        for k in range(0, len(sents), batch):
            chunk = sents[k:k + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=am).logits
            logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            tgt = ids[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = am[:, 1:].float()
            out.extend(((tok_lp * mask).sum(-1)).cpu().tolist())
        return out

    good = logprobs([p["good"] for p in pairs])
    bad = logprobs([p["bad"] for p in pairs])
    return np.array([1 if g > b else 0 for g, b in zip(good, bad)], dtype=int)


def summarize(pairs, correct, *, lang, model_name, tag, grounded_thr, drop_thr, control_eps):
    by_slice = defaultdict(list)
    by_cond = defaultdict(lambda: defaultdict(list))
    for p, c in zip(pairs, correct):
        by_slice[p["slice"]].append(int(c))
        by_cond[p["condition"]][p["slice"]].append(int(c))

    slice_acc = {s: bootstrap_ci(by_slice[s], seed=i) for i, s in enumerate(SLICES)}
    a = slice_acc["aligned"][0]
    t = slice_acc["attractor"][0]
    c = slice_acc["control"][0]
    drop = round(a - t, 4)                       # do(class-3): proximity severed
    control_gap = round(a - c, 4)                # class-2: must be ~0
    control_intact = abs(control_gap) <= control_eps

    if a >= grounded_thr and t >= grounded_thr:
        verdict = "SC-grounded (tracks the subject on both slices)"
    elif drop >= drop_thr and control_intact:
        verdict = "SC-spurious (rides linear proximity)"
    else:
        verdict = "mixed / inconclusive"

    return {
        "experiment": "cross-lingual subject-verb agreement: SC-grounded/SC-spurious cut",
        "lang": lang, "model": model_name, "tag": tag,
        "n_pairs": len(pairs),
        "n_by_slice": {s: len(by_slice[s]) for s in SLICES},
        "framework": {
            "shortcut": "agree with the linearly nearest noun (proximity cue)",
            "by_construction_(3.2a)": "attractor = intervening noun of opposite number; grammar stipulates proximity irrelevant",
            "do(class-3)_analog": "attractor (number-mismatch) slice",
            "class-2_control": "number-match intervener slice (no opposing cue)"},
        "accuracy_by_slice": slice_acc,
        "do(class-3)_drop_aligned_minus_attractor": drop,
        "class-2_control_gap_aligned_minus_control": control_gap,
        "control_intact": bool(control_intact),
        "thresholds": {"grounded_thr": grounded_thr, "drop_thr": drop_thr, "control_eps": control_eps},
        "accuracy_by_condition": {
            cond: {s: bootstrap_ci(by_cond[cond][s], seed=j) for j, s in enumerate(SLICES) if by_cond[cond][s]}
            for cond in sorted(by_cond)},
        "verdict": verdict}


# ---------------------------------------------------------------- self-test ---
def _selftest():
    """Validate slice bucketing + verdict logic WITHOUT downloading a model:
    feed synthetic correctness arrays and check the labels."""
    def fake(pairs, corr, **kw):
        return summarize(pairs, np.array(corr), lang="xx", model_name="fake", tag="selftest",
                         grounded_thr=0.8, drop_thr=0.2, control_eps=0.1, **kw)

    # proximity-rider: high aligned + high control, collapses on attractor
    pairs, corr = [], []
    for s, acc in (("aligned", 1.0), ("control", 0.95), ("attractor", 0.10)):
        for i in range(20):
            pairs.append({"condition": "c", "slice": s, "good": "g", "bad": "b"})
            corr.append(1 if i < round(acc * 20) else 0)
    s1 = fake(pairs, corr)
    assert s1["verdict"].startswith("SC-spurious"), s1["verdict"]
    assert s1["control_intact"], s1

    # subject-tracker: high on all three slices
    pairs, corr = [], []
    for s in SLICES:
        for i in range(20):
            pairs.append({"condition": "c", "slice": s, "good": "g", "bad": "b"})
            corr.append(1 if i < 19 else 0)   # 0.95
    s2 = fake(pairs, corr)
    assert s2["verdict"].startswith("SC-grounded"), s2["verdict"]

    print("selftest OK:")
    print(f"  rider   -> drop={s1['do(class-3)_drop_aligned_minus_attractor']} "
          f"control_gap={s1['class-2_control_gap_aligned_minus_control']} :: {s1['verdict']}")
    print(f"  tracker -> drop={s2['do(class-3)_drop_aligned_minus_attractor']} "
          f"control_gap={s2['class-2_control_gap_aligned_minus_control']} :: {s2['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="normalized minimal-pair JSONL (see prepare_clams.py)")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--tag", default="zeroshot",
                    help="regime label, e.g. zeroshot (portability) or licensed")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--grounded_thr", type=float, default=0.80)
    ap.add_argument("--drop_thr", type=float, default=0.20)
    ap.add_argument("--control_eps", type=float, default=0.10)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.pairs:
        ap.error("--pairs is required unless --selftest")

    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    pairs = load_pairs(args.pairs)
    n_by = {s: sum(p["slice"] == s for p in pairs) for s in SLICES}
    print(f"device={dev} | model={args.model} | lang={args.lang} | tag={args.tag}")
    print(f"  pairs={len(pairs)}  by_slice={n_by}")
    if min(n_by.values()) == 0:
        print(f"  WARNING: empty slice(s): {[s for s in SLICES if n_by[s] == 0]} "
              f"-> verdict may be 'mixed'. Check prepare_clams.py condition mapping.")

    t0 = time.time()
    correct = score_minimal_pairs(pairs, args.model, dev, args.batch, args.max_len)
    summary = summarize(pairs, correct, lang=args.lang, model_name=args.model, tag=args.tag,
                        grounded_thr=args.grounded_thr, drop_thr=args.drop_thr,
                        control_eps=args.control_eps)
    summary["seconds"] = round(time.time() - t0, 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    out = RESULTS / f"agreement_{args.lang}_{safe_model}_{args.tag}.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)

    sa = summary["accuracy_by_slice"]
    print("\n=== RESULT ===")
    print(f"  aligned   = {sa['aligned']}")
    print(f"  attractor = {sa['attractor']}   (do(class-3): proximity severed)")
    print(f"  control   = {sa['control']}     (class-2 negative control)")
    print(f"  do(class-3) drop = {summary['do(class-3)_drop_aligned_minus_attractor']}  "
          f"| control gap = {summary['class-2_control_gap_aligned_minus_control']}  "
          f"| control_intact={summary['control_intact']}")
    print(f"  VERDICT: {summary['verdict']}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()

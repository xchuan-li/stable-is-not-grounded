"""
run_mnli_hans.py — naturalistic, licensed instantiation of the SC-grounded /
SC-spurious cut on real NLI (the McCoy et al. 2019 HANS setting).

Framework mapping:
  shortcut S         = lexical overlap (hypothesis words contained in premise)
  training entangle. = corr(overlap, label=entailment) measured on MNLI  [§3.2(b)]
  by-construction    = HANS stipulates overlap-irrelevance: its non-entailment
                       cases have HIGH overlap but gold = non-entailment  [§3.2(a)]
  do(class-3) analog = the HANS non-entailment slice (overlap severed from label)

Verdict:
  high MNLI acc + high HANS-entailment + LOW HANS-non-entailment  -> SC-spurious
  high on BOTH HANS subsets                                       -> SC-grounded

We fine-tune our OWN DistilBERT on an MNLI subset, so the training set (and thus
the overlap correlation) is fully auditable. Eval + checkpoint AFTER EVERY EPOCH
(summary.json = latest; summary_epoch{N}.json kept) so an interrupted run keeps
the last finished epoch. Local, MPS/CPU. Run under `caffeinate` to avoid sleep.

Usage:
  caffeinate -is python run_mnli_hans.py --train_n 100000 --epochs 2 --batch 32
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(line_buffering=True)
BASE = Path(__file__).parent
RESULTS = BASE / "results"
_tok = re.compile(r"[a-z']+")
HANS_URL = ("https://raw.githubusercontent.com/tommccoy1/hans/master/"
            "heuristics_evaluation_set.txt")


def overlap_frac(prem, hyp):
    p = set(_tok.findall(prem.lower()))
    h = _tok.findall(hyp.lower())
    return sum(1 for w in h if w in p) / len(h) if h else 0.0


def load_hans():
    """HANS eval set from McCoy's TSV (datasets 4.x dropped the script dataset).
    label 0=entailment, 1=non-entailment."""
    import pandas as pd, urllib.request
    cache = BASE / "hans_eval.txt"
    if not cache.exists():
        urllib.request.urlretrieve(HANS_URL, cache)
    df = pd.read_csv(cache, sep="\t", quoting=csv.QUOTE_NONE)
    lab = (df["gold_label"].values != "entailment").astype(int)
    return {"premise": df["sentence1"].tolist(), "hypothesis": df["sentence2"].tolist(),
            "label": lab.tolist(), "heuristic": df["heuristic"].tolist()}


def bootstrap_ci(correct, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(correct, float)
    if len(v) == 0:
        return (0.0, 0.0, 0.0)
    pt = float(v.mean())
    boots = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(pt, 4), round(float(lo), 4), round(float(hi), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="distilbert-base-uncased")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} | model={args.model} | train_n={args.train_n} "
          f"epochs={args.epochs} batch={args.batch}")

    print("loading MNLI + HANS ...", flush=True)
    mnli = load_dataset("nyu-mll/multi_nli")
    mtrain = mnli["train"].filter(lambda r: r["label"] in (0, 1, 2))
    mtrain = mtrain.shuffle(seed=args.seed).select(range(min(args.train_n, len(mtrain))))
    mval = mnli["validation_matched"].filter(lambda r: r["label"] in (0, 1, 2))
    hval = load_hans()
    print(f"  MNLI train subset={len(mtrain)}  val_matched={len(mval)}  "
          f"HANS val={len(hval['label'])}")

    # §3.2(b): measured training entanglement (overlap <-> entailment)
    ov = np.array([overlap_frac(p, h) for p, h in zip(mtrain["premise"], mtrain["hypothesis"])])
    ent = (np.array(mtrain["label"]) == 0).astype(float)
    r_overlap = float(np.corrcoef(ov, ent)[0, 1])
    print(f"  measured corr(overlap, entailment) on MNLI train = {r_overlap:+.3f}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(prem, hyp):
        return tok(list(prem), list(hyp), truncation=True, max_length=args.max_len,
                   padding="max_length", return_tensors="pt")

    print("tokenizing train ...", flush=True)
    enc = encode(mtrain["premise"], mtrain["hypothesis"])
    y = torch.tensor(mtrain["label"])
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y),
                    batch_size=args.batch, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def preds3(prem, hyp, bs=128):
        out = []
        for k in range(0, len(prem), bs):
            e = encode(prem[k:k+bs], hyp[k:k+bs])
            logits = model(input_ids=e["input_ids"].to(dev),
                           attention_mask=e["attention_mask"].to(dev)).logits
            out.append(logits.argmax(-1).cpu().numpy())
        return np.concatenate(out)

    def eval_and_save(epoch):
        model.eval()
        p = preds3(mval["premise"], mval["hypothesis"])
        mnli_acc = bootstrap_ci((p == np.array(mval["label"])).astype(int), seed=1)
        hp2 = (preds3(hval["premise"], hval["hypothesis"]) != 0).astype(int)
        hlab = np.array(hval["label"]); heur = np.array(hval["heuristic"])
        hc = (hp2 == hlab).astype(int)
        ent_m, non_m = (hlab == 0), (hlab == 1)
        hans = {"overall": bootstrap_ci(hc, 2),
                "entailment (heuristic agrees)": bootstrap_ci(hc[ent_m], 3),
                "non_entailment (heuristic severed)": bootstrap_ci(hc[non_m], 4),
                "by_heuristic": {}}
        for hn in ["lexical_overlap", "subsequence", "constituent"]:
            hm = (heur == hn)
            hans["by_heuristic"][hn] = {
                "entailment": bootstrap_ci(hc[hm & ent_m], 5),
                "non_entailment": bootstrap_ci(hc[hm & non_m], 6)}
        ea = hans["entailment (heuristic agrees)"][0]
        na = hans["non_entailment (heuristic severed)"][0]
        verdict = ("SC-spurious (rides lexical overlap)" if na < 0.5 < ea
                   else "SC-grounded (robust on both HANS sides)" if na >= 0.5 else "mixed")
        summary = {
            "experiment": "MNLI->HANS naturalistic licensed SC-grounded/SC-spurious cut",
            "model": args.model, "train_n": len(mtrain), "epochs_done": epoch,
            "epochs_target": args.epochs,
            "framework": {
                "shortcut": "lexical overlap (hypothesis words in premise)",
                "measured_corr_overlap_entailment_MNLItrain_(3.2b)": round(r_overlap, 4),
                "by_construction_(3.2a)": "HANS non-entailment = high overlap, gold non-entailment",
                "do(class-3)_analog": "HANS non-entailment slice (overlap severed from label)"},
            "MNLI_matched_accuracy": mnli_acc,
            "HANS": hans,
            "shortcut_drop_HANS_ent_minus_nonent": round(ea - na, 4),
            "verdict": verdict}
        RESULTS.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(RESULTS / "mnli_hans_summary.json", "w"), indent=2)
        json.dump(summary, open(RESULTS / f"mnli_hans_summary_epoch{epoch}.json", "w"), indent=2)
        model.save_pretrained(RESULTS / "model"); tok.save_pretrained(RESULTS / "model")
        print(f"  [epoch {epoch}] MNLI={mnli_acc[0]} HANS_ent={ea} HANS_nonent={na} "
              f"drop={round(ea-na,4)} -> {verdict}  (saved)")
        model.train()
        return summary

    t0 = time.time()
    model.train()
    last = None
    for ep in range(args.epochs):
        tot = 0.0
        for i, (ids, am, yb) in enumerate(dl):
            ids, am, yb = ids.to(dev), am.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=am, labels=yb)
            out.loss.backward(); opt.step(); tot += out.loss.item()
            if i % 200 == 0:
                print(f"  ep{ep} step{i}/{len(dl)} loss={out.loss.item():.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  epoch {ep} train done: mean loss {tot/len(dl):.3f} [{time.time()-t0:.0f}s]; evaluating...")
        last = eval_and_save(ep + 1)   # checkpoint after each epoch

    print("\n=== FINAL ===")
    print(f"  corr(overlap,entailment) = {last['framework']['measured_corr_overlap_entailment_MNLItrain_(3.2b)']}")
    print(f"  MNLI matched acc        = {last['MNLI_matched_accuracy']}")
    print(f"  HANS entailment  acc    = {last['HANS']['entailment (heuristic agrees)']}")
    print(f"  HANS non-entailment acc = {last['HANS']['non_entailment (heuristic severed)']}")
    print(f"  lexical_overlap non-ent = {last['HANS']['by_heuristic']['lexical_overlap']['non_entailment']}")
    print(f"  VERDICT: {last['verdict']}")
    print(f"  Saved: {RESULTS/'mnli_hans_summary.json'}")


if __name__ == "__main__":
    main()

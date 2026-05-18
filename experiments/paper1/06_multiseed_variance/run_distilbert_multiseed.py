"""
run_distilbert_multiseed.py

Multi-seed variance extension of the §5.4 DistilBERT experiment.
Run locally on MacBook M4 (MPS) — DistilBERT 66M is fast enough.

Same design as run_qwen_multiseed_kaggle.py:
- N_SEEDS independent training seeds
- Eval seeds fixed so variance = training stochasticity only
- Reports mean ± std across seeds for all metrics

Runtime estimate: ~5–10 min per seed on M4 MPS → ~30–50 min total for 5 seeds
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SEEDS     = 5
TRAIN_SEEDS = [42, 43, 44, 45, 46]
EVAL_SEED_BASE = 100

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 64
EPOCHS     = 3
BATCH      = 16
LR         = 2e-5
THRESH     = 0.10

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_distilbert_multiseed"

# ---------------------------------------------------------------------------
# Data generation  (identical to run_distilbert.py)
# ---------------------------------------------------------------------------
CHAINS = [
    ("sparrow", "bird",   "animal"),
    ("salmon",  "fish",   "animal"),
    ("oak",     "tree",   "plant"),
    ("rose",    "flower", "plant"),
    ("ant",     "insect", "creature"),
    ("trout",   "fish",   "creature"),
]
ALL_B         = sorted({b for _, b, _ in CHAINS})
NAMES         = ["Tweety", "Robin", "Alex", "Milo", "Luna", "Charlie"]
HELDOUT_NAMES = ["Zara",   "Quinn", "Bode", "Iris", "Nor",  "Vex"]
COLORS        = ["red", "blue"]


def make_text(p1, p2, a, b, b2, c, name, color):
    parts = []
    if p1:
        parts.append(f"All {a}s are {b}s.")
    if p2:
        parts.append(f"All {b2}s are {c}s.")
    parts.append(f"{name} is a {color} {a}.")
    parts.append(f"Is {name} a {c}?")
    return " ".join(parts)


def gen(n, s, seed, names, color_mode, ablate=None, noise=0.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c  = rng.choice(CHAINS)
        name     = rng.choice(names)
        licensed = rng.choice([True, False])
        b2       = b if licensed else rng.choice([x for x in ALL_B if x != b])
        true_y   = 1 if licensed else 0

        if ablate is None:
            label = true_y
            if noise > 0.0 and rng.random() < noise:
                label = 1 - true_y
            color_target = label
        else:
            label        = 0
            color_target = true_y

        if color_mode == "correlated":
            color = (("red"  if color_target == 1 else "blue")
                     if rng.random() < s else
                     ("blue" if color_target == 1 else "red"))
        else:
            color = rng.choice(COLORS)

        rows.append({
            "text":  make_text(ablate != "P1", ablate != "P2",
                               a, b, b2, c, name, color),
            "label": label, "color": color, "name": name,
        })
    return pd.DataFrame(rows)


def gen_class1_intervention(n, s, seed, names, color_mode):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(CHAINS)
        name    = rng.choice(names)
        b_prime = rng.choice([x for x in ALL_B if x != b])
        label   = 0
        if color_mode == "correlated":
            color = "red" if rng.random() < s else "blue"
        else:
            color = rng.choice(COLORS)
        rows.append({
            "text":  make_text(True, True, a, b, b_prime, c, name, color),
            "label": label, "color": color, "name": name,
        })
    return pd.DataFrame(rows)


def corr_with_label(df, col):
    codes = df[col].astype("category").cat.codes
    if codes.nunique() < 2:
        return 0.0
    return float(abs(codes.corr(df["label"])))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DS(Dataset):
    def __init__(self, df, tok):
        self.t   = df["text"].tolist()
        self.y   = df["label"].tolist()
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(df, tok, device, train_seed):
    set_seed(train_seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    dl  = DataLoader(DS(df, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for enc, y in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            y   = y.to(device)
            opt.zero_grad()
            out = model(**enc, labels=y)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        print(f"      epoch {ep+1}/{EPOCHS}  loss={tot/len(dl):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def preds_of(df, tok, model, device):
    out = []
    for i in range(0, len(df), 64):
        b   = df.iloc[i:i+64]
        enc = tok(b["text"].tolist(), truncation=True, padding=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        out.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
    return np.array(out)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(values, stat=np.mean, n_boot=1000, seed=0, alpha=0.05):
    rng   = np.random.default_rng(seed)
    v     = np.asarray(values)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


# ---------------------------------------------------------------------------
# Single-seed regime run
# ---------------------------------------------------------------------------
def run_one_seed(regime, color_mode, s, noise, train_seed, tok, device):
    print(f"    [seed={train_seed}] training...", flush=True)

    train = gen(5000, s, seed=42, names=NAMES, color_mode=color_mode,
                ablate=None, noise=noise)
    measured = {
        "color": corr_with_label(train, "color"),
        "name":  corr_with_label(train, "name"),
    }
    klass = {z: ("class-3" if c > THRESH else "class-2")
             for z, c in measured.items()}

    model = train_model(train, tok, device, train_seed)

    iid      = gen(1500, s, EVAL_SEED_BASE+0, NAMES,         color_mode, ablate=None, noise=0.0)
    p1       = gen(1500, s, EVAL_SEED_BASE+1, NAMES,         color_mode, ablate="P1", noise=0.0)
    p2       = gen(1500, s, EVAL_SEED_BASE+2, NAMES,         color_mode, ablate="P2", noise=0.0)
    do_color = gen(1500, s, EVAL_SEED_BASE+3, NAMES,         "severed",  ablate=None, noise=0.0)
    do_name  = gen(1500, s, EVAL_SEED_BASE+4, HELDOUT_NAMES, color_mode, ablate=None, noise=0.0)
    c1_int   = gen_class1_intervention(1500, s, EVAL_SEED_BASE+5, NAMES, color_mode)

    def corr(df):  return (preds_of(df, tok, model, device) == df["label"].values).astype(int)
    def with_(df): return (preds_of(df, tok, model, device) == 0).astype(int)

    base              = bootstrap_ci(corr(iid),       seed=1)
    p1w               = bootstrap_ci(with_(p1),        seed=2)
    p2w               = bootstrap_ci(with_(p2),        seed=3)
    dc_pt,dc_lo,dc_hi = bootstrap_ci(corr(do_color),  seed=4)
    dn_pt,dn_lo,dn_hi = bootstrap_ci(corr(do_name),   seed=5)
    c1_flip           = bootstrap_ci(corr(c1_int),     seed=6)

    drop_color = (round(base[0]-dc_pt,4), round(base[1]-dc_hi,4), round(base[2]-dc_lo,4))
    drop_name  = (round(base[0]-dn_pt,4), round(base[1]-dn_hi,4), round(base[2]-dn_lo,4))

    respects   = p1w[1] > 0.5 and p2w[1] > 0.5
    fails_nred = p1w[1] <= 0.5 and p2w[1] <= 0.5
    flagged    = drop_color[1] > 0.0 and abs(drop_name[0]) < 0.05
    c1_variant = c1_flip[1] > 0.5
    c1_insens  = c1_flip[2] < 0.5

    return {
        "train_seed":                    train_seed,
        "measured_|r|":                  {k: round(v,4) for k,v in measured.items()},
        "class_assignment":              klass,
        "baseline_acc":                  base[0],
        "P1_withhold":                   p1w[0],
        "P2_withhold":                   p2w[0],
        "drop_color":                    drop_color[0],
        "drop_name":                     drop_name[0],
        "c1_flip_rate":                  c1_flip[0],
        "baseline_acc_ci":               base,
        "P1_withhold_ci":                p1w,
        "P2_withhold_ci":                p2w,
        "drop_color_ci":                 drop_color,
        "drop_name_ci":                  drop_name,
        "c1_flip_rate_ci":               c1_flip,
        "nonredundancy_respected":       bool(respects),
        "nonredundancy_fails_decoupled": bool(fails_nred),
        "flagged_correlational":         bool(flagged),
        "c1_correctly_variant":          bool(c1_variant),
        "c1_insensitive_to_chain":       bool(c1_insens),
    }


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------
def aggregate(seed_results):
    keys = ["baseline_acc", "P1_withhold", "P2_withhold",
            "drop_color", "drop_name", "c1_flip_rate"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in seed_results]
        agg[k] = {
            "mean":     round(float(np.mean(vals)), 4),
            "std":      round(float(np.std(vals, ddof=1)), 4),
            "per_seed": vals,
        }
    bool_keys = ["nonredundancy_respected", "nonredundancy_fails_decoupled",
                 "flagged_correlational", "c1_correctly_variant",
                 "c1_insensitive_to_chain"]
    for k in bool_keys:
        vals = [r[k] for r in seed_results]
        agg[k] = {"count_true": sum(vals), "n_seeds": len(vals),
                  "all_agree": len(set(vals)) == 1}
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Device: {device} | model: {MODEL_NAME}")
    print(f"Running {N_SEEDS} training seeds: {TRAIN_SEEDS}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_results = {}

    for regime, color_mode, s, noise in [
        ("chain_required",     "severed",    1.0, 0.0),
        ("shortcut_available", "correlated", 1.0, 0.12),
    ]:
        print(f"\n=== Regime: {regime} ===")
        seed_results = []
        for ts in TRAIN_SEEDS:
            r = run_one_seed(regime, color_mode, s, noise, ts, tok, device)
            seed_results.append(r)
            chk = RESULTS_DIR / f"{regime}_seed{ts}.json"
            json.dump(r, open(chk, "w"), indent=2)
            print(f"    → saved {chk}", flush=True)

        agg = aggregate(seed_results)
        all_results[regime] = {"per_seed": seed_results, "aggregate": agg}

        print(f"\n  [{regime}] cross-seed summary:")
        for k in ["baseline_acc", "P1_withhold", "P2_withhold",
                  "drop_color", "drop_name", "c1_flip_rate"]:
            print(f"    {k:20s}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        for k in ["nonredundancy_respected", "flagged_correlational",
                  "c1_correctly_variant"]:
            print(f"    {k:35s}: {agg[k]}")

    cr = all_results["chain_required"]["aggregate"]
    sa = all_results["shortcut_available"]["aggregate"]

    spec_ok = (cr["nonredundancy_respected"]["count_true"]       == N_SEEDS and
               cr["flagged_correlational"]["count_true"]         == 0       and
               cr["c1_correctly_variant"]["count_true"]          == N_SEEDS)
    sens_ok = (sa["nonredundancy_fails_decoupled"]["count_true"] == N_SEEDS and
               sa["flagged_correlational"]["count_true"]         == N_SEEDS and
               sa["c1_insensitive_to_chain"]["count_true"]       == N_SEEDS)
    validated = spec_ok and sens_ok

    summary = {
        "experiment":  "§5.4 multi-seed variance — DistilBERT",
        "model":       MODEL_NAME,
        "n_seeds":     N_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "note": (
            "Eval seeds fixed (EVAL_SEED_BASE=100); variance is over "
            "training stochasticity only."
        ),
        "results":    all_results,
        "validated_across_all_seeds": bool(validated),
        "spec_ok":    bool(spec_ok),
        "sens_ok":    bool(sens_ok),
    }

    out = RESULTS_DIR / "multiseed_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n=== Final validation: {validated} (spec={spec_ok}, sens={sens_ok}) ===")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

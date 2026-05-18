"""
evaluate_mcs_operationalization_bert.py

Robustness rung for paper5 Section 5.6: re-run the SAME Section-3.5
operationalization (stipulated DAG + measured training correlation +
class-2 control + A/B regimes) but with a fine-tuned DistilBERT instead
of TF-IDF + logistic regression.

Why this rung exists
--------------------
The TF-IDF demonstration validates the operationalization's *internal*
properties (non-circular, specific, sensitive) on a transparent model
whose reliance is controllable. TF-IDF behaves cleanly/binary, which a
reviewer can dismiss as too easy. This rung tests *external* validity:
does the procedure still discriminate a shortcut-rider from a genuine
M-user when the model has distributed, entangled representations?

Hard ceiling (by the framework's own boundary): Section 3.5(b) requires
an auditable training set, so this can only use a model whose training
data we control — a DistilBERT we fine-tune ourselves, NOT a frontier
LLM. "Use a bigger model" is bounded at "the largest model whose
training set you still control."

The measured class assignment is a property of the DATA, not the model
(|r(color,label)|, |r(name,label)| on the training split), so it is
identical to the TF-IDF run — reinforcing that 3.5(b) is non-circular.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_mcs"
SUMMARY_FILE = RESULTS_DIR / "mcs_operationalization_bert_summary.json"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
EPOCHS = 3
BATCH = 16
LR = 2e-5
SEED = 42

TRAIN_NAMES = ["Tweety", "Robin", "Alex", "Milo", "Luna", "Charlie"]
HELDOUT_NAMES = ["Zara", "Quinn", "Bode", "Iris", "Nor", "Vex"]
ANIMALS = ["bird", "penguin"]
COLORS = ["red", "blue"]
RULE = "Birds can fly. Penguins cannot fly."


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_text(name, color, animal, regime):
    if regime == "M_available":
        return f"{RULE} {name} is a {color} {animal}. Can {name} fly?"
    return f"{name} is a {color} animal. Can {name} fly?"


def gen(n, s, seed, names, regime, color_mode="correlated"):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        animal = rng.choice(ANIMALS)
        y = 1 if animal == "bird" else 0
        name = rng.choice(names)
        if color_mode == "correlated":
            color = (("red" if y == 1 else "blue")
                     if rng.random() < s else
                     ("blue" if y == 1 else "red"))
        else:
            color = rng.choice(COLORS)
        rows.append({"text": make_text(name, color, animal, regime),
                     "label": y, "color": color, "name": name})
    return pd.DataFrame(rows)


def corr_with_label(df, col):
    codes = df[col].astype("category").cat.codes
    if codes.nunique() < 2:
        return 0.0
    return float(abs(codes.corr(df["label"])))


class DS(Dataset):
    def __init__(self, df, tok):
        self.t = df["text"].tolist()
        self.y = df["label"].tolist()
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, padding="max_length",
                     max_length=MAX_LEN, return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in e.items()},
                torch.tensor(self.y[i]))


def train_model(df, tok, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    dl = DataLoader(DS(df, tok), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for ep in range(EPOCHS):
        tot = 0.0
        for enc, y in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)
            opt.zero_grad()
            out = model(**enc, labels=y)
            out.loss.backward()
            opt.step()
            tot += out.loss.item()
        print(f"    epoch {ep+1}/{EPOCHS} loss={tot/len(dl):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def acc_of(df, tok, model, device):
    correct = 0
    for i in range(0, len(df), 64):
        batch = df.iloc[i:i+64]
        enc = tok(batch["text"].tolist(), truncation=True,
                  padding=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        pred = model(**enc).logits.argmax(-1).cpu().numpy()
        correct += int((pred == batch["label"].values).sum())
    return correct / len(df)


def run_regime(regime, s, thresh, device):
    print(f"  [{regime}] generating + fine-tuning DistilBERT...", flush=True)
    train = gen(1500, s, 42, TRAIN_NAMES, regime, "correlated")
    measured = {"color": corr_with_label(train, "color"),
                "name": corr_with_label(train, "name")}
    klass = {z: ("class-3" if c > thresh else "class-2")
             for z, c in measured.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    set_seed(SEED)
    model = train_model(train, tok, device)

    iid = gen(800, s, 43, TRAIN_NAMES, regime, "correlated")
    do_color = gen(800, s, 44, TRAIN_NAMES, regime, "severed")
    do_name = gen(800, s, 45, HELDOUT_NAMES, regime, "correlated")

    acc = {
        "baseline_iid": acc_of(iid, tok, model, device),
        "do(color)_class3": acc_of(do_color, tok, model, device),
        "do(name)_class2_control": acc_of(do_name, tok, model, device),
    }
    d3 = acc["baseline_iid"] - acc["do(color)_class3"]
    d2 = acc["baseline_iid"] - acc["do(name)_class2_control"]
    flagged = d3 > 0.20 and d2 < 0.05
    return {
        "measured_|r|": {k: round(v, 4) for k, v in measured.items()},
        "class_assignment": klass,
        "accuracy": {k: round(v, 4) for k, v in acc.items()},
        "drop_do(color)_class3": round(d3, 4),
        "drop_do(name)_class2_control": round(d2, 4),
        "flagged_correlational": bool(flagged),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {MODEL_NAME} (fine-tuned, "
          f"auditable training set)")
    s, thresh = 0.9, 0.10
    A = run_regime("M_available", s, thresh, device)
    B = run_regime("M_suppressed", s, thresh, device)
    validated = (A["flagged_correlational"] is False
                 and B["flagged_correlational"] is True)

    summary = {
        "model": f"{MODEL_NAME} fine-tuned (distributed representations; "
                 f"training set auditable -> 3.5(b) valid)",
        "shortcut_strength_s": s,
        "class_assignment_threshold": thresh,
        "regime_A_M_available": A,
        "regime_B_M_suppressed": B,
        "operationalization_validated": bool(validated),
        "note": ("Class assignment is data-derived, identical to the TF-IDF "
                 "run (3.5b non-circular). Single fine-tune per regime, "
                 "seed-set; seed variance is the documented extension."),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SUMMARY_FILE, "w"), indent=2)

    def show(n, R):
        print(f"\n--- {n} ---")
        print(f"  measured |r|: color={R['measured_|r|']['color']:.4f}"
              f"({R['class_assignment']['color']}) "
              f"name={R['measured_|r|']['name']:.4f}"
              f"({R['class_assignment']['name']})")
        for k, v in R["accuracy"].items():
            print(f"  {k:28s} {v:.3f}")
        print(f"  drop do(color)[c3]={R['drop_do(color)_class3']:+.3f} "
              f"drop do(name)[c2]={R['drop_do(name)_class2_control']:+.3f} "
              f"flagged={R['flagged_correlational']}")

    print("\n=== Section 3.5 operationalization — DistilBERT rung ===")
    show("Regime A : M available", A)
    show("Regime B : M suppressed", B)
    print(f"\nSPECIFIC (A not flagged) AND SENSITIVE (B flagged) "
          f"-> validated on distributed-representation model: {validated}")
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    main()

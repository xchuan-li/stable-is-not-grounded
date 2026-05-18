"""
evaluate_mcs_nonredundancy_v2.py

Redesign of the non-redundancy / CIY-2 enrichment after v1 revealed a
self-inflicted target-vocabulary shortcut (negatives were made by swapping
the asked category, so a model could answer from the target token alone and
ignore the premises -> "non-redundancy violated" was partly a design
artifact).

v2 removes every single-token cue:
  YES : "All A are B. All B are C. <name> is a <color> A. Is <name> a C?"
  NO  : "All A are B. All B2 are C. <name> is a <color> A. Is <name> a C?"
        (B2 != B: A->B is known and B2->C is known, but B->C is NOT, so the
         C-conclusion is not licensed.)
Surface form is near-identical between YES and NO; the ONLY discriminating
signal is whether the middle term binds (B vs B2). Bag-of-words cannot
represent this binding, so a fine-tuned DistilBERT is used out of necessity
(the structure requires at least a relational model to have a genuine
chain-tracker arm at all). Training set is self-controlled, so Section
3.5(b) remains valid (the framework's boundary forbids a frontier model).

Two regimes (mirrors the 5.6 specific/sensitive A-B design):
  chain_required    : color assigned independently of the label. The only
                      signal is the chain binding. A model that solves the
                      task must track the premises -> ablation should force
                      withholding (non-redundancy RESPECTED; specific: not
                      flagged because there is no class-3 variable here).
  shortcut_available: color spuriously label-correlated (class-3). The model
                      can ride color instead of the chain -> ablation leaves
                      color intact, model keeps asserting (non-redundancy
                      VIOLATED), do(color) flags it, class-2 control clean
                      (sensitive + non-vacuous).

Validated iff: chain_required -> non-redundancy respected AND not flagged;
shortcut_available -> non-redundancy violated AND flagged with class-2
control robust.
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

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results_mcs"
SUMMARY_FILE = RESULTS_DIR / "mcs_nonredundancy_v2_summary.json"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
EPOCHS = 3
BATCH = 16
LR = 2e-5
SEED = 42

# (A, B, C) chains; B terms are drawn across chains to form B2 (mismatched
# middle term) so "B2 present" is never a clean dataset-wide cue.
CHAINS = [
    ("sparrow", "bird", "animal"),
    ("salmon", "fish", "animal"),
    ("oak", "tree", "plant"),
    ("rose", "flower", "plant"),
    ("ant", "insect", "creature"),
    ("trout", "fish", "creature"),
]
ALL_B = sorted({b for _, b, _ in CHAINS})
NAMES = ["Tweety", "Robin", "Alex", "Milo", "Luna", "Charlie"]
HELDOUT_NAMES = ["Zara", "Quinn", "Bode", "Iris", "Nor", "Vex"]
COLORS = ["red", "blue"]


def make_text(p1, p2, a, b, b2, c, name, color):
    parts = []
    if p1:
        parts.append(f"All {a}s are {b}s.")
    if p2:
        parts.append(f"All {b2}s are {c}s.")  # b2 == b for YES; b2 != b for NO
    parts.append(f"{name} is a {color} {a}.")
    parts.append(f"Is {name} a {c}?")
    return " ".join(parts)


def gen(n, s, seed, names, regime, color_mode, ablate=None):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(CHAINS)
        name = rng.choice(names)
        licensed = rng.choice([True, False])
        if licensed:
            b2 = b                      # chain binds -> licensed
        else:
            others = [x for x in ALL_B if x != b]
            b2 = rng.choice(others)     # chain broken -> not licensed

        if ablate is None:
            y = 1 if licensed else 0
        else:
            # a necessary premise removed -> C-conclusion not licensed for
            # anyone; correct answer is 0 (withhold) on every ablated item.
            y = 0

        if color_mode == "correlated":
            color = (("red" if y == 1 else "blue")
                     if rng.random() < s else
                     ("blue" if y == 1 else "red"))
        else:
            color = rng.choice(COLORS)

        rows.append({"text": make_text(ablate != "P1", ablate != "P2",
                                       a, b, b2, c, name, color),
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
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i])


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


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
def preds_of(df, tok, model, device):
    out = []
    for i in range(0, len(df), 64):
        b = df.iloc[i:i+64]
        enc = tok(b["text"].tolist(), truncation=True, padding=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        out.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
    return np.array(out)


def run_regime(regime, color_mode, s, thresh, device):
    print(f"  [{regime}] gen + fine-tune...", flush=True)
    train = gen(5000, s, 42, NAMES, regime, color_mode, ablate=None)
    measured = {"color": corr_with_label(train, "color"),
                "name": corr_with_label(train, "name")}
    klass = {z: ("class-3" if c > thresh else "class-2")
             for z, c in measured.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    set_seed(SEED)
    model = train_model(train, tok, device)

    def acc(df):
        return float((preds_of(df, tok, model, device) == df["label"].values).mean())

    iid = gen(1500, s, 43, NAMES, regime, color_mode, ablate=None)
    p1 = gen(1500, s, 44, NAMES, regime, color_mode, ablate="P1")
    p2 = gen(1500, s, 45, NAMES, regime, color_mode, ablate="P2")
    do_color = gen(1500, s, 46, NAMES, regime, "severed", ablate=None)
    do_name = gen(1500, s, 47, HELDOUT_NAMES, regime, color_mode, ablate=None)

    base = acc(iid)
    p1_withhold = float((preds_of(p1, tok, model, device) == 0).mean())
    p2_withhold = float((preds_of(p2, tok, model, device) == 0).mean())
    d3 = base - acc(do_color)
    d2 = base - acc(do_name)

    nonredundancy_respected = p1_withhold > 0.80 and p2_withhold > 0.80
    flagged = d3 > 0.20 and d2 < 0.05
    return {
        "measured_|r|": {k: round(v, 4) for k, v in measured.items()},
        "class_assignment": klass,
        "baseline_fullM_acc": round(base, 4),
        "P1_ablated_withhold_rate": round(p1_withhold, 4),
        "P2_ablated_withhold_rate": round(p2_withhold, 4),
        "nonredundancy_respected": bool(nonredundancy_respected),
        "drop_do(color)_class3": round(d3, 4),
        "drop_do(name)_class2_control": round(d2, 4),
        "flagged_correlational": bool(flagged),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {MODEL_NAME} (fine-tuned)")
    s, thresh = 0.9, 0.10
    CR = run_regime("chain_required", "severed", s, thresh, device)
    SA = run_regime("shortcut_available", "correlated", s, thresh, device)

    # specificity: chain_required -> non-redundancy respected & not flagged
    # sensitivity: shortcut_available -> non-redundancy violated & flagged
    validated = (CR["nonredundancy_respected"] and not CR["flagged_correlational"]
                 and (not SA["nonredundancy_respected"]) and SA["flagged_correlational"])

    summary = {
        "design": "two-premise transitive M; YES vs NO differ only in middle-"
                  "term binding (B vs B2); no single-token cue. DistilBERT "
                  "(relational model required; training set self-controlled "
                  "so 3.5b valid).",
        "shortcut_strength_s": s,
        "regime_chain_required": CR,
        "regime_shortcut_available": SA,
        "validated_specific_and_sensitive": bool(validated),
        "scope": "synthetic, single seed-set fine-tune per regime; seed "
                 "variance and graded chain depth are documented extensions.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SUMMARY_FILE, "w"), indent=2)

    def show(n, R):
        print(f"\n--- {n} ---")
        print(f"  |r| color={R['measured_|r|']['color']:.3f}"
              f"({R['class_assignment']['color']}) "
              f"name={R['measured_|r|']['name']:.3f}"
              f"({R['class_assignment']['name']})")
        print(f"  baseline_fullM={R['baseline_fullM_acc']:.3f}")
        print(f"  P1-ablated withhold={R['P1_ablated_withhold_rate']:.3f}  "
              f"P2-ablated withhold={R['P2_ablated_withhold_rate']:.3f}  "
              f"-> non-redundancy respected={R['nonredundancy_respected']}")
        print(f"  drop do(color)[c3]={R['drop_do(color)_class3']:+.3f}  "
              f"drop do(name)[c2]={R['drop_do(name)_class2_control']:+.3f}  "
              f"flagged={R['flagged_correlational']}")

    print("\n=== MCS non-redundancy v2 (binding-based, DistilBERT) ===")
    show("Regime chain_required (no shortcut; must track chain)", CR)
    show("Regime shortcut_available (class-3 color present)", SA)
    print(f"\nvalidated (specific: CR respects & not flagged ; "
          f"sensitive: SA violates & flagged): {validated}")
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

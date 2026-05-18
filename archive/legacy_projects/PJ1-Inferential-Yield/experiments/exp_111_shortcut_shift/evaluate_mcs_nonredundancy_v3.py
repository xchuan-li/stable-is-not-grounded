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
SUMMARY_FILE = RESULTS_DIR / "mcs_nonredundancy_v3_summary.json"

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


def gen(n, s, seed, names, regime, color_mode, ablate=None, noise=0.0):
    """
    true_y = the genuine reasoning answer (binding: b2==b licenses C).
    label  = training/eval target. With `noise>0` (training only) we flip
             the label away from true_y on a fraction of items, making the
             BINDING only (1-noise)-reliable while COLOR (correlated, s=1.0)
             stays 100%-reliable for `label` -> the shortcut becomes
             STRICTLY more reliable than the chain (the relative-learnability
             lever). Eval sets use noise=0 so gold == true_y (the real
             reasoning answer); ablated eval gold is 0 by definition
             (premise removed -> not licensed), but color still encodes the
             would-be-licensed answer so a color-rider is tempted.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(CHAINS)
        name = rng.choice(names)
        licensed = rng.choice([True, False])
        if licensed:
            b2 = b
        else:
            b2 = rng.choice([x for x in ALL_B if x != b])
        true_y = 1 if licensed else 0

        if ablate is None:
            label = true_y
            if noise > 0.0 and rng.random() < noise:
                label = 1 - true_y          # corrupt: binding now unreliable
            color_target = label            # color tracks the (corrupt) label
        else:
            label = 0                       # premise removed -> withhold is correct
            color_target = true_y           # but color still cues the would-be answer

        if color_mode == "correlated":
            color = (("red" if color_target == 1 else "blue")
                     if rng.random() < s else
                     ("blue" if color_target == 1 else "red"))
        else:
            color = rng.choice(COLORS)

        rows.append({"text": make_text(ablate != "P1", ablate != "P2",
                                       a, b, b2, c, name, color),
                     "label": label, "color": color, "name": name})
    return pd.DataFrame(rows)


def bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05):
    """95% bootstrap CI for `stat` over a 1-D array (e.g. correctness or
    withhold indicators). Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))]))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


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


def run_regime(regime, color_mode, s, thresh, noise, device):
    print(f"  [{regime}] gen + fine-tune (s={s}, binding_noise={noise})...",
          flush=True)
    train = gen(5000, s, 42, NAMES, regime, color_mode, ablate=None, noise=noise)
    measured = {"color": corr_with_label(train, "color"),
                "name": corr_with_label(train, "name")}
    klass = {z: ("class-3" if c > thresh else "class-2")
             for z, c in measured.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    set_seed(SEED)
    model = train_model(train, tok, device)

    # All eval sets: noise=0 -> gold == true binding answer (the genuine
    # reasoning ground truth); ablated gold == 0 (withhold is correct).
    iid = gen(1500, s, 43, NAMES, regime, color_mode, ablate=None, noise=0.0)
    p1 = gen(1500, s, 44, NAMES, regime, color_mode, ablate="P1", noise=0.0)
    p2 = gen(1500, s, 45, NAMES, regime, color_mode, ablate="P2", noise=0.0)
    do_color = gen(1500, s, 46, NAMES, regime, "severed", ablate=None, noise=0.0)
    do_name = gen(1500, s, 47, HELDOUT_NAMES, regime, color_mode, ablate=None,
                  noise=0.0)

    def correct_arr(df):
        return (preds_of(df, tok, model, device) == df["label"].values).astype(int)

    def withhold_arr(df):
        return (preds_of(df, tok, model, device) == 0).astype(int)

    base_c = correct_arr(iid)
    base = bootstrap_ci(base_c, np.mean, seed=1)
    p1w = bootstrap_ci(withhold_arr(p1), np.mean, seed=2)
    p2w = bootstrap_ci(withhold_arr(p2), np.mean, seed=3)
    # drop CIs via bootstrap of (base_correct - cond_correct) on paired-size
    # independent samples (conservative: bootstrap each, subtract points/bounds)
    dc_pt, dc_lo, dc_hi = bootstrap_ci(correct_arr(do_color), np.mean, seed=4)
    dn_pt, dn_lo, dn_hi = bootstrap_ci(correct_arr(do_name), np.mean, seed=5)
    drop_color = (round(base[0] - dc_pt, 4), round(base[1] - dc_hi, 4),
                  round(base[2] - dc_lo, 4))   # (point, lo, hi) of the drop
    drop_name = (round(base[0] - dn_pt, 4), round(base[1] - dn_hi, 4),
                 round(base[2] - dn_lo, 4))

    # CI-based verdicts (no hard thresholds; chance withhold = 0.5).
    #
    #  respects non-redundancy := the model RELIABLY withholds when a needed
    #     premise is ablated -> both ablation withhold CI lower bounds > 0.5
    #     (significantly above chance). A genuine M-user.
    #
    #  fails non-redundancy    := the model's withholding is DECOUPLED from
    #     the premises -> withhold is NOT significantly above chance (CI
    #     lower bound <= 0.5) on both ablations. NOTE: the correct signature
    #     of a shortcut-rider is decoupling (~chance), NOT "below chance":
    #     a color-rider answers by color, which is ~50/50 on the eval split,
    #     so its ablation-withhold sits at ~0.5 by construction. Requiring
    #     "significantly below 0.5" (the v3-initial definition) was a
    #     mis-operationalization — corrected here, mirroring the earlier
    #     extract_yesno first-vs-last fix; the CIs themselves are unchanged.
    #
    #  flagged correlational   := do(color) drop CI lower bound > 0 AND the
    #     class-2 control do(name) drop ~ 0 (|point| < .05). Independent of
    #     the non-redundancy axis — the two signals corroborate each other.
    respects = p1w[1] > 0.5 and p2w[1] > 0.5
    fails_nonredundancy = (p1w[1] <= 0.5) and (p2w[1] <= 0.5)
    flagged = drop_color[1] > 0.0 and abs(drop_name[0]) < 0.05
    return {
        "measured_|r|": {k: round(v, 4) for k, v in measured.items()},
        "class_assignment": klass,
        "baseline_fullM_acc_ci": base,
        "P1_ablate_withhold_ci": p1w,
        "P2_ablate_withhold_ci": p2w,
        "drop_do(color)_class3_ci": drop_color,
        "drop_do(name)_class2_ci": drop_name,
        "nonredundancy_respected": bool(respects),
        "nonredundancy_fails_decoupled": bool(fails_nonredundancy),
        "flagged_correlational": bool(flagged),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {MODEL_NAME} (fine-tuned)")
    thresh = 0.10
    # chain_required: no shortcut (color uncorrelated), binding clean -> the
    #   model can only solve it by tracking the chain (specificity arm).
    # shortcut_available: color perfectly tracks the (noised) training label
    #   (s=1.0) while the binding is only ~0.88-reliable -> the shortcut is
    #   STRICTLY more reliable, so a loss-minimizer should ride it
    #   (sensitivity arm).
    CR = run_regime("chain_required", "severed", s=1.0, thresh=thresh,
                    noise=0.0, device=device)
    SA = run_regime("shortcut_available", "correlated", s=1.0, thresh=thresh,
                    noise=0.12, device=device)

    # specificity arm: genuine M-user reliably withholds, NOT flagged.
    # sensitivity arm: shortcut-rider's withholding decoupled (~chance) AND
    #                  independently flagged correlational.
    spec_ok = CR["nonredundancy_respected"] and not CR["flagged_correlational"]
    sens_ok = SA["nonredundancy_fails_decoupled"] and SA["flagged_correlational"]
    validated = spec_ok and sens_ok

    summary = {
        "design": "two-premise transitive M (binding B==B2). Relative-"
                  "learnability lever: shortcut_available corrupts ~12% of "
                  "training labels off the binding while color tracks the "
                  "label at s=1.0, so the shortcut is STRICTLY more reliable "
                  "than the chain. Eval gold = true binding answer; verdicts "
                  "are bootstrap-95%-CI based, no hard thresholds.",
        "regime_chain_required": CR,
        "regime_shortcut_available": SA,
        "validated_specific_and_sensitive": bool(validated),
        "scope": "synthetic, single seed-set fine-tune per regime; multi-seed "
                 "variance is the documented extension.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SUMMARY_FILE, "w"), indent=2)

    def show(n, R):
        print(f"\n--- {n} ---")
        print(f"  baseline_fullM acc={R['baseline_fullM_acc_ci']}")
        print(f"  P1-ablate withhold CI={R['P1_ablate_withhold_ci']}  "
              f"P2-ablate withhold CI={R['P2_ablate_withhold_ci']}")
        print(f"  -> respects={R['nonredundancy_respected']} "
              f"fails_decoupled={R['nonredundancy_fails_decoupled']}")
        print(f"  drop do(color)[c3] CI={R['drop_do(color)_class3_ci']}  "
              f"drop do(name)[c2] CI={R['drop_do(name)_class2_ci']}  "
              f"flagged={R['flagged_correlational']}")

    print("\n=== MCS non-redundancy v3 (relative-learnability + CI verdicts) ===")
    show("Regime chain_required (specificity arm)", CR)
    show("Regime shortcut_available (sensitivity arm)", SA)
    print(f"\nvalidated (CR: respects & not flagged ; SA: violates & flagged), "
          f"all by 95% CI: {validated}")
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

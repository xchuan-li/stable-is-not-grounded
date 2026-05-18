"""
run_qwen_kaggle.py

Same multi-premise non-redundancy experiment as run_distilbert.py (§5.4),
rerun with Qwen2.5-1.5B + LoRA to extend the model-class robustness claim
in §5.3 to a decoder-only generative architecture.

--- Kaggle setup (do this before running) ---
1. New Notebook > Settings > Accelerator > GPU T4 x1
2. In the first code cell run:
       !pip install peft -q
3. Paste or upload this file and run.

Architecture difference from DistilBERT:
  DistilBERT : encoder-only, bidirectional, 66M params, full fine-tune
  Qwen2.5-1.5B: decoder-only, causal, 1.5B params, LoRA fine-tune
Both are trained only on the self-controlled synthetic set, so §3.5(b)
(auditable training distribution) remains valid — the framework boundary
that forbids frontier zero-shot models is respected.

Expected verdicts (mirrors DistilBERT):
  chain_required    : nonredundancy_respected=True, flagged=False,
                      class1_correctly_variant=True
  shortcut_available: nonredundancy_fails_decoupled=True, flagged=True,
                      class1_insensitive_to_chain=True
"""

# ---------------------------------------------------------------------------
# 0. Imports and install check
# ---------------------------------------------------------------------------
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    raise SystemExit("peft not found — run:  !pip install peft -q")


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
MAX_LEN     = 96        # inputs are short; 96 is ample
EPOCHS      = 5         # LoRA needs a few more epochs than full fine-tune
BATCH       = 8         # conservative for T4 + 1.5B
LR          = 1e-4      # standard LoRA learning rate
LORA_R      = 8
LORA_ALPHA  = 16
LORA_DROPOUT = 0.05

RESULTS_DIR  = Path("results_qwen")
SUMMARY_FILE = RESULTS_DIR / "summary_qwen.json"

# ---------------------------------------------------------------------------
# 2. Data generation  (identical to run_distilbert.py)
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


def gen(n, s, seed, names, regime, color_mode, ablate=None, noise=0.0):
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
    """
    Class-1 intervention: break the chain at P2 (replace middle term b with
    b_prime != b); gold = 0 (NO). Color kept pointing at YES to maintain
    class-3 temptation in shortcut_available.
    flip_rate = fraction model answers NO = correctly-variant.
    """
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
        text = make_text(True, True, a, b, b_prime, c, name, color)
        rows.append({"text": text, "label": label, "color": color, "name": name})
    return pd.DataFrame(rows)


def corr_with_label(df, col):
    codes = df[col].astype("category").cat.codes
    if codes.nunique() < 2:
        return 0.0
    return float(abs(codes.corr(df["label"])))


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class DS(Dataset):
    def __init__(self, df, tok):
        self.t = df["text"].tolist()
        self.y = df["label"].tolist()
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        e = self.tok(
            self.t[i], truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(self.y[i])


# ---------------------------------------------------------------------------
# 4. Model helpers
# ---------------------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(device):
    """Load Qwen2.5-1.5B for sequence classification, wrap with LoRA."""
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type      = TaskType.SEQ_CLS,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = ["q_proj", "v_proj"],  # standard for Qwen2.5
        bias           = "none",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model.to(device)


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Decoder models often lack a pad token — reuse eos_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # standard for decoder models
    return tok


def train_model(df, tok, device):
    model = build_model(device)
    dl    = DataLoader(DS(df, tok), batch_size=BATCH, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
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
        print(f"    epoch {ep+1}/{EPOCHS}  loss={tot/len(dl):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def preds_of(df, tok, model, device):
    out = []
    for i in range(0, len(df), 32):   # smaller batch for inference on large model
        b   = df.iloc[i:i+32]
        enc = tok(
            b["text"].tolist(), truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        out.extend(model(**enc).logits.argmax(-1).cpu().numpy().tolist())
    return np.array(out)


# ---------------------------------------------------------------------------
# 5. Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(values, stat, n_boot=1000, seed=0, alpha=0.05):
    rng   = np.random.default_rng(seed)
    v     = np.asarray(values)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


# ---------------------------------------------------------------------------
# 6. Regime runner  (same logic as run_distilbert.py)
# ---------------------------------------------------------------------------
def run_regime(regime, color_mode, s, thresh, noise, device):
    print(f"  [{regime}] gen + LoRA fine-tune (s={s}, noise={noise})...", flush=True)

    train = gen(5000, s, 42, NAMES, regime, color_mode, ablate=None, noise=noise)
    measured = {
        "color": corr_with_label(train, "color"),
        "name":  corr_with_label(train, "name"),
    }
    klass = {z: ("class-3" if c > thresh else "class-2")
             for z, c in measured.items()}
    print(f"    measured |r|: {measured}  →  {klass}", flush=True)

    tok = get_tokenizer()
    set_seed(42)
    model = train_model(train, tok, device)

    iid      = gen(1500, s, 43, NAMES,         regime, color_mode, ablate=None, noise=0.0)
    p1       = gen(1500, s, 44, NAMES,         regime, color_mode, ablate="P1", noise=0.0)
    p2       = gen(1500, s, 45, NAMES,         regime, color_mode, ablate="P2", noise=0.0)
    do_color = gen(1500, s, 46, NAMES,         regime, "severed",  ablate=None, noise=0.0)
    do_name  = gen(1500, s, 47, HELDOUT_NAMES, regime, color_mode, ablate=None, noise=0.0)
    c1_int   = gen_class1_intervention(1500, s, 48, NAMES, color_mode)

    def correct_arr(df):
        return (preds_of(df, tok, model, device) == df["label"].values).astype(int)

    def withhold_arr(df):
        return (preds_of(df, tok, model, device) == 0).astype(int)

    base            = bootstrap_ci(correct_arr(iid),        np.mean, seed=1)
    p1w             = bootstrap_ci(withhold_arr(p1),        np.mean, seed=2)
    p2w             = bootstrap_ci(withhold_arr(p2),        np.mean, seed=3)
    dc_pt,dc_lo,dc_hi = bootstrap_ci(correct_arr(do_color), np.mean, seed=4)
    dn_pt,dn_lo,dn_hi = bootstrap_ci(correct_arr(do_name),  np.mean, seed=5)
    c1_flip         = bootstrap_ci(correct_arr(c1_int),     np.mean, seed=6)

    drop_color = (round(base[0]-dc_pt,4), round(base[1]-dc_hi,4), round(base[2]-dc_lo,4))
    drop_name  = (round(base[0]-dn_pt,4), round(base[1]-dn_hi,4), round(base[2]-dn_lo,4))

    respects            = p1w[1] > 0.5 and p2w[1] > 0.5
    fails_nonredundancy = p1w[1] <= 0.5 and p2w[1] <= 0.5
    flagged             = drop_color[1] > 0.0 and abs(drop_name[0]) < 0.05
    c1_correctly_variant = c1_flip[1] > 0.5
    c1_insensitive       = c1_flip[2] < 0.5

    return {
        "model":                      MODEL_NAME,
        "measured_|r|":               {k: round(v,4) for k,v in measured.items()},
        "class_assignment":           klass,
        "baseline_fullM_acc_ci":      base,
        "P1_ablate_withhold_ci":      p1w,
        "P2_ablate_withhold_ci":      p2w,
        "drop_do(color)_class3_ci":   drop_color,
        "drop_do(name)_class2_ci":    drop_name,
        "class1_flip_rate_ci":        c1_flip,
        "nonredundancy_respected":    bool(respects),
        "nonredundancy_fails_decoupled": bool(fails_nonredundancy),
        "flagged_correlational":      bool(flagged),
        "class1_correctly_variant":   bool(c1_correctly_variant),
        "class1_insensitive_to_chain": bool(c1_insensitive),
    }


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {MODEL_NAME} (LoRA fine-tune)")
    if device.type != "cuda":
        print("WARNING: no GPU detected — this will be very slow on CPU.")

    thresh = 0.10
    CR = run_regime("chain_required",    "severed",    s=1.0, thresh=thresh,
                    noise=0.0,  device=device)
    SA = run_regime("shortcut_available","correlated", s=1.0, thresh=thresh,
                    noise=0.12, device=device)

    spec_ok = (CR["nonredundancy_respected"]
               and not CR["flagged_correlational"]
               and CR["class1_correctly_variant"])
    sens_ok = (SA["nonredundancy_fails_decoupled"]
               and SA["flagged_correlational"]
               and SA["class1_insensitive_to_chain"])
    validated = spec_ok and sens_ok

    summary = {
        "design": (
            "Same two-premise transitive M as run_distilbert.py (§5.4), "
            f"rerun with {MODEL_NAME} + LoRA to extend §5.3 model-class "
            "robustness claim to a decoder-only architecture. "
            "Class-1 arm included (P2 middle-term broken, color tempts YES). "
            "Verdicts are bootstrap-95%-CI based, no hard thresholds."
        ),
        "regime_chain_required":    CR,
        "regime_shortcut_available": SA,
        "validated_specific_and_sensitive": bool(validated),
        "scope": (
            "synthetic, single seed-set LoRA fine-tune per regime; "
            "multi-seed variance is the documented extension."
        ),
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
        print(f"  class-1 flip_rate CI={R['class1_flip_rate_ci']}  "
              f"correctly_variant={R['class1_correctly_variant']}  "
              f"insensitive={R['class1_insensitive_to_chain']}")

    print(f"\n=== MCS non-redundancy — {MODEL_NAME} (LoRA) ===")
    show("Regime chain_required (specificity arm)", CR)
    show("Regime shortcut_available (sensitivity arm)", SA)
    print(f"\nvalidated (CR: respects & not flagged & class1_variant ; "
          f"SA: violates & flagged & class1_insensitive), "
          f"all by 95% CI: {validated}")
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

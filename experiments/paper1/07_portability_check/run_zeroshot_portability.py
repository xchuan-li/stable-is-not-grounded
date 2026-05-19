"""
run_zeroshot_portability.py

§8 procedure-portability check (the highest-leverage open step).

Runs the *same* §5.4 two-premise ladder — baseline / P1-ablate / P2-ablate /
do(color) / do(name) / class-1 chain-break — on a model that was NEVER
fine-tuned on the construction (a local zero-shot instruction-tuned LM).
This directly probes the principal objection to §5: "a self-constructed
shortcut detected on a self-trained model".

BOUNDARY DISCIPLINE (mirrors the reframed §8 and §3.4):
  For a model with unauditable pretraining, §3.2(b) is NOT re-measured on
  the model. The class label (color = class-3, name = class-2) is *imported*
  from the construction's own controlled data statistics — which we own —
  and the check tests only whether the verdict STRUCTURE transfers to a
  model whose parameters were never shaped by the construction. It is a
  portability check under an imported class assignment, explicitly not a
  re-measured (b) and not a capability claim.

WHAT A PASS / NON-PASS MEANS (no expected outcome is hard-coded):
  - regime chain_required  : color severed, only the chain binds. A model
    that reasons should withhold under premise ablation and flip on the
    class-1 chain-break, with no class-3 flag.
  - regime shortcut_available : color is made in-context predictive of the
    would-be answer (s=1.0). A model with no trained-in entanglement need
    not ride it; if the flag does NOT reproduce here, that is itself
    informative — it shows the §5 positive verdict is specific to models
    that actually trained on the entanglement, not an artifact of the
    perturbation. Report what happens; do not assume.

Local, no API key. Default model Qwen2.5-1.5B-Instruct (runs on M-series
MPS or CPU). Resumable raw CSV; summary emitted in the §5.4 schema so it
slots straight into the evidence map once run.

Usage:
    python run_zeroshot_portability.py
    python run_zeroshot_portability.py --model Qwen/Qwen2.5-1.5B-Instruct
    python run_zeroshot_portability.py --n 600        # items per condition
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# --- §5.4 construction (identical chains/vocab to 02_multipremise) ----------
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
THRESH = 0.10  # imported-class-assignment threshold (construction statistic)


def make_text(p1, p2, a, b, b2, c, name, color):
    parts = []
    if p1:
        parts.append(f"All {a}s are {b}s.")
    if p2:
        parts.append(f"All {b2}s are {c}s.")
    parts.append(f"{name} is a {color} {a}.")
    parts.append(f"Is {name} a {c}?")
    return " ".join(parts)


def gen(n, s, seed, names, color_mode, ablate=None):
    """Eval generator (noise=0: gold == true binding answer). Mirrors the
    §5.4 eval generator; ablated gold == 0 (premise removed -> withhold)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(CHAINS)
        name = rng.choice(names)
        licensed = rng.choice([True, False])
        b2 = b if licensed else rng.choice([x for x in ALL_B if x != b])
        true_y = 1 if licensed else 0
        if ablate is None:
            label = true_y
            color_target = label
        else:
            label = 0
            color_target = true_y
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


def gen_class1_intervention(n, s, seed, names, color_mode):
    """Would-be-YES items with P2 middle term broken (b_prime != b); gold=NO.
    Color tempts the pre-intervention YES answer."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(CHAINS)
        name = rng.choice(names)
        b_prime = rng.choice([x for x in ALL_B if x != b])
        if color_mode == "correlated":
            color = "red" if rng.random() < s else "blue"
        else:
            color = rng.choice(COLORS)
        rows.append({"text": make_text(True, True, a, b, b_prime, c, name, color),
                     "label": 0, "color": color, "name": name})
    return pd.DataFrame(rows)


def corr_with_label(df, col):
    codes = df[col].astype("category").cat.codes
    if codes.nunique() < 2:
        return 0.0
    return float(abs(codes.corr(df["label"])))


def bootstrap_ci(values, stat=np.mean, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    point = float(stat(v))
    boots = [float(stat(v[rng.integers(0, len(v), len(v))])) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(point, 4), round(float(lo), 4), round(float(hi), 4)


# --- zero-shot answer extraction (pattern from run_deepseek_zeroshot) ------
def extract_yesno(text: str) -> str:
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    toks = re.findall(r"\b(yes|no)\b", t)
    if toks:
        return toks[0]
    if re.search(r"\bit follows\b", t) and not re.search(
            r"\bdoes not follow|doesn't follow|it does not\b", t):
        return "yes"
    if re.search(r"\bdoes not follow|doesn't follow|cannot conclude|"
                 r"cannot be concluded\b", t):
        return "no"
    return "unknown"


SYS = ("You answer logical inference questions. Read the statement carefully "
       "and answer with exactly one word: yes or no.")


def build_runner(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model: {model_name} (zero-shot, no fine-tune)")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).to(device).eval()

    @torch.no_grad()
    def predict(texts):
        out = []
        for txt in texts:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": txt}]
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(device)
            gen_ids = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            ans = tok.decode(gen_ids[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            out.append(extract_yesno(ans))
        return out

    return predict


def run_regime(regime, color_mode, s, n, predict):
    print(f"\n  [{regime}] generating + zero-shot eval (n={n}/cond)...",
          flush=True)
    # Imported class assignment: measured on the CONSTRUCTION's own data
    # (a statistic we own), NOT on the model's pretraining (§3.4 boundary).
    pool = gen(5000, s, 42, NAMES, color_mode, ablate=None)
    measured = {"color": corr_with_label(pool, "color"),
                "name": corr_with_label(pool, "name")}
    klass = {z: ("class-3" if c > THRESH else "class-2")
             for z, c in measured.items()}

    iid = gen(n, s, 43, NAMES, color_mode, ablate=None)
    p1 = gen(n, s, 44, NAMES, color_mode, ablate="P1")
    p2 = gen(n, s, 45, NAMES, color_mode, ablate="P2")
    do_color = gen(n, s, 46, NAMES, "severed", ablate=None)
    do_name = gen(n, s, 47, HELDOUT_NAMES, color_mode, ablate=None)
    c1 = gen_class1_intervention(n, s, 48, NAMES, color_mode)

    def correct(df):
        preds = [1 if p == "yes" else 0 for p in predict(df["text"].tolist())]
        return (np.array(preds) == df["label"].values).astype(int)

    def withhold(df):
        preds = [1 if p == "yes" else 0 for p in predict(df["text"].tolist())]
        return (np.array(preds) == 0).astype(int)

    base = bootstrap_ci(correct(iid), seed=1)
    p1w = bootstrap_ci(withhold(p1), seed=2)
    p2w = bootstrap_ci(withhold(p2), seed=3)
    dc = bootstrap_ci(correct(do_color), seed=4)
    dn = bootstrap_ci(correct(do_name), seed=5)
    c1f = bootstrap_ci(correct(c1), seed=6)
    drop_color = (round(base[0] - dc[0], 4), round(base[1] - dc[2], 4),
                  round(base[2] - dc[1], 4))
    drop_name = (round(base[0] - dn[0], 4), round(base[1] - dn[2], 4),
                 round(base[2] - dn[1], 4))

    return {
        "imported_|r|": {k: round(v, 4) for k, v in measured.items()},
        "imported_class_assignment": klass,
        "baseline_fullM_acc_ci": base,
        "P1_ablate_withhold_ci": p1w,
        "P2_ablate_withhold_ci": p2w,
        "drop_do(color)_class3_ci": drop_color,
        "drop_do(name)_class2_ci": drop_name,
        "class1_flip_rate_ci": c1f,
        "nonredundancy_respected": bool(p1w[1] > 0.5 and p2w[1] > 0.5),
        "nonredundancy_fails_decoupled": bool(p1w[1] <= 0.5 and p2w[1] <= 0.5),
        "flagged_correlational": bool(drop_color[1] > 0.0
                                      and abs(drop_name[0]) < 0.05),
        "class1_correctly_variant": bool(c1f[1] > 0.5),
        "class1_insensitive_to_chain": bool(c1f[2] < 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=600,
                    help="items per condition (lower = faster)")
    args = ap.parse_args()

    predict = build_runner(args.model)
    CR = run_regime("chain_required", "severed", 1.0, args.n, predict)
    SA = run_regime("shortcut_available", "correlated", 1.0, args.n, predict)

    summary = {
        "experiment": "§8 procedure-portability check — zero-shot, no fine-tune",
        "model": args.model,
        "evaluation": "zero-shot; verdict structure under an IMPORTED class "
                      "assignment (b NOT re-measured on pretraining; §3.4)",
        "design": "Identical §5.4 two-premise ladder. Class label imported "
                  "from the construction's own data statistics. No expected "
                  "outcome is hard-coded; a non-reproduced flag in "
                  "shortcut_available is itself informative (it would show "
                  "the §5 positive verdict is specific to models trained on "
                  "the entanglement).",
        "regime_chain_required": CR,
        "regime_shortcut_available": SA,
        "scope": "Portability check under imported class assignment; not a "
                 "re-measured (b), not a capability claim.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "zeroshot_portability_summary.json"
    json.dump(summary, open(out, "w"), indent=2)

    def show(n, R):
        print(f"\n--- {n} ---")
        print(f"  baseline acc           = {R['baseline_fullM_acc_ci']}")
        print(f"  P1/P2 ablate withhold  = {R['P1_ablate_withhold_ci']} / "
              f"{R['P2_ablate_withhold_ci']}")
        print(f"  drop do(color)[c3]     = {R['drop_do(color)_class3_ci']}  "
              f"drop do(name)[c2] = {R['drop_do(name)_class2_ci']}")
        print(f"  flagged_correlational  = {R['flagged_correlational']}")
        print(f"  class-1 flip rate      = {R['class1_flip_rate_ci']}")

    print("\n=== §8 zero-shot portability (imported class assignment) ===")
    show("chain_required", CR)
    show("shortcut_available", SA)
    print(f"\nSaved: {out}")
    print("\nInterpretation is the user's call: report the verdict structure "
          "as-is in §5/§8; do NOT assume the flag must reproduce.")


if __name__ == "__main__":
    main()

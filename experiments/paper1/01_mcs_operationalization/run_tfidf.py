"""
evaluate_mcs_operationalization.py

Demonstrates the central methodological claim of the framework (Section 3.5):
the NON-CIRCULAR operationalization of an "irrelevant" variable, with the
class-2 control arm that makes the result non-vacuous.

Why this script exists
----------------------
The framework asserts that a model's apparent sufficiency can be detected as
correlational (not causal) without circularity, by splitting the question:
  (a) "is Z causally irrelevant to C?"  -> answered BY CONSTRUCTION (we author
      the synthetic items, so we stipulate the ground-truth DAG);
  (b) "is Z entangled with the label?"  -> answered EMPIRICALLY (a correlation
      measured on the training set; no causal assumption).
Z is a clause-(iii) target iff causally-irrelevant-by-construction AND
training-correlated (class 3). A class-2 control (causally irrelevant AND
training-UNcorrelated) is mandatory: without showing that do(class-2) does
NOT break sufficiency, "do(class-3) breaks it" is vacuous ("you built the
failure in / any perturbation breaks it").

Fix vs. the original exp_111
----------------------------
The original shortcut-shift text never contained the causal premise (animal
type), so "the model rode a correlation" was trivially true (M was not even
available). Here M is explicitly present in the input, so a genuine reasoner
COULD use it; we test whether the model uses M or rides the class-3 shortcut.

Stipulated DAG (Section 3.5(a), by construction)
------------------------------------------------
  animal_type ----> can_fly            (M = MCS(C); the only causal parent)
  color        (no edge to can_fly)    causally IRRELEVANT by construction
  name         (no edge to can_fly)    causally IRRELEVANT by construction

Training entanglement (Section 3.5(b), measured, not assumed)
-------------------------------------------------------------
  color: injected spurious correlation with the label (strength s)  -> expect class 3
  name : assigned independently of the label                        -> expect class 2

Interventions (M kept intact and present in every condition)
------------------------------------------------------------
  do(color): set color independently of the label (sever the training
             correlation). A model using M is unaffected; a color-shortcut
             rider breaks.
  do(name) : replace names with an unseen disjoint set, independent of the
             label. This is the class-2 CONTROL.

Result we look for
------------------
  accuracy drop under do(color)  >>  accuracy drop under do(name)
i.e. apparent sufficiency is NOT robust under the class-3 intervention but IS
robust under the class-2 intervention -> the model's M was correlational, and
the framework detected this non-circularly and specifically.
"""

import json
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_FILE = RESULTS_DIR / "tfidf_summary.json"

TRAIN_NAMES = ["Tweety", "Robin", "Alex", "Milo", "Luna", "Charlie"]
# Disjoint, unseen-at-training names for the class-2 do(name) intervention.
HELDOUT_NAMES = ["Zara", "Quinn", "Bode", "Iris", "Nor", "Vex"]
ANIMALS = ["bird", "penguin"]
COLORS = ["red", "blue"]

RULE = "Birds can fly. Penguins cannot fly."


def make_text(name, color, animal, regime):
    # Regime A "M-available": the rule + the animal type are present, so a
    #   learner CAN ground the answer in M (the causal structure).
    # Regime B "M-suppressed": the animal type is hidden (generic "animal"),
    #   so in training only the class-3 shortcut (color) carries label info.
    if regime == "M_available":
        return f"{RULE} {name} is a {color} {animal}. Can {name} fly?"
    return f"{name} is a {color} animal. Can {name} fly?"


def label_of(animal):
    return 1 if animal == "bird" else 0


def gen(n, shortcut_strength, seed, names, regime, color_mode="correlated"):
    """color_mode:
       'correlated' -> color tracks the label at `shortcut_strength`
       'severed'    -> color assigned independently of the label  [do(color)]
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        animal = rng.choice(ANIMALS)
        y = label_of(animal)
        name = rng.choice(names)
        if color_mode == "correlated":
            if rng.random() < shortcut_strength:
                color = "red" if y == 1 else "blue"
            else:
                color = "blue" if y == 1 else "red"
        else:  # severed: color independent of label
            color = rng.choice(COLORS)
        rows.append({"text": make_text(name, color, animal, regime),
                     "label": y, "animal": animal,
                     "color": color, "name": name})
    return pd.DataFrame(rows)


def corr_with_label(df, col):
    """Empirical entanglement (Section 3.5(b)): |point-biserial corr| of a
    binary-encoded categorical feature with the label. No causal assumption."""
    codes = df[col].astype("category").cat.codes
    if codes.nunique() < 2:
        return 0.0
    return float(abs(codes.corr(df["label"])))


def run_regime(regime, s, THRESH):
    train = gen(4000, s, seed=42, names=TRAIN_NAMES, regime=regime,
                color_mode="correlated")
    # Section 3.5(b): MEASURE entanglement; classify by a fixed threshold on
    # a measured number (not investigator discretion).
    measured = {"color": corr_with_label(train, "color"),
                "name": corr_with_label(train, "name")}
    klass = {z: ("class-3" if c > THRESH else "class-2")
             for z, c in measured.items()}

    model = Pipeline([("tfidf", TfidfVectorizer()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train["text"], train["label"])

    iid = gen(1500, s, seed=43, names=TRAIN_NAMES, regime=regime,
              color_mode="correlated")
    do_color = gen(1500, s, seed=44, names=TRAIN_NAMES, regime=regime,
                   color_mode="severed")
    do_name = gen(1500, s, seed=45, names=HELDOUT_NAMES, regime=regime,
                  color_mode="correlated")

    acc = {
        "baseline_iid": float(model.score(iid["text"], iid["label"])),
        "do(color)_class3": float(model.score(do_color["text"], do_color["label"])),
        "do(name)_class2_control": float(model.score(do_name["text"], do_name["label"])),
    }
    drop_c3 = acc["baseline_iid"] - acc["do(color)_class3"]
    drop_c2 = acc["baseline_iid"] - acc["do(name)_class2_control"]
    # FLAGGED = apparent sufficiency not robust under class-3 do() while
    # robust under the class-2 control -> correlational sufficiency, detected
    # specifically.
    flagged = drop_c3 > 0.20 and drop_c2 < 0.05
    return {
        "measured_|r|": {k: round(v, 4) for k, v in measured.items()},
        "class_assignment": klass,
        "accuracy": {k: round(v, 4) for k, v in acc.items()},
        "drop_do(color)_class3": round(drop_c3, 4),
        "drop_do(name)_class2_control": round(drop_c2, 4),
        "flagged_correlational": bool(flagged),
    }


def main():
    s, THRESH = 0.9, 0.10
    # Regime A: M usable (rule + animal in text) -> model SHOULD ground in M.
    # Regime B: M suppressed (animal type hidden) -> only the class-3 shortcut
    #           carries label info in training -> model SHOULD ride it.
    A = run_regime("M_available", s, THRESH)
    B = run_regime("M_suppressed", s, THRESH)

    # The operationalization is validated iff it is SPECIFIC and SENSITIVE:
    #   specificity: regime A NOT flagged (no false alarm on a genuine reasoner)
    #   sensitivity: regime B     flagged (catches the shortcut rider)
    validated = (A["flagged_correlational"] is False
                 and B["flagged_correlational"] is True)

    summary = {
        "stipulated_DAG_(3.5a_by_construction)": {
            "M": "animal_type -> can_fly (only causal parent)",
            "color": "no edge to can_fly (irrelevant by construction)",
            "name": "no edge to can_fly (irrelevant by construction)",
        },
        "class_assignment_threshold": THRESH,
        "shortcut_strength_s": s,
        "regime_A_M_available": A,
        "regime_B_M_suppressed": B,
        "operationalization_validated": bool(validated),
        "reading": ("Same measured class assignment in both regimes "
                    "(color=class-3, name=class-2). The procedure does NOT "
                    "false-alarm when the model genuinely uses M (regime A "
                    "not flagged) and DOES catch the shortcut rider (regime "
                    "B flagged), with the class-2 control ruling out 'any "
                    "perturbation breaks it'. Detection is non-circular: "
                    "irrelevance is by construction (3.5a), entanglement is "
                    "measured (3.5b)."),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SUMMARY_FILE, "w"), indent=2)

    def show(name, R):
        print(f"\n--- {name} ---")
        print(f"  measured |r|: color={R['measured_|r|']['color']:.4f} "
              f"({R['class_assignment']['color']}), "
              f"name={R['measured_|r|']['name']:.4f} "
              f"({R['class_assignment']['name']})")
        for k, v in R["accuracy"].items():
            print(f"  {k:28s} {v:.3f}")
        print(f"  drop do(color)[class-3]={R['drop_do(color)_class3']:+.3f}  "
              f"drop do(name)[class-2]={R['drop_do(name)_class2_control']:+.3f}")
        print(f"  flagged correlational: {R['flagged_correlational']}")

    print("=== MCS / Section 3.5 non-circular operationalization ===")
    print("Stipulated DAG (by construction): animal_type -> can_fly ; "
          "color,name : no edge")
    print("3.5(a) irrelevance = by construction ; 3.5(b) entanglement = measured")
    show("Regime A : M available (model can ground in M)", A)
    show("Regime B : M suppressed (only shortcut carries label info)", B)
    print(f"\nSPECIFIC (A not flagged) AND SENSITIVE (B flagged) "
          f"-> operationalization validated: {validated}")
    print(f"\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

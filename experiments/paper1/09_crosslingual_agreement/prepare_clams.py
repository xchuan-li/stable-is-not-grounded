"""
prepare_clams.py — build the normalized minimal-pair JSONL that
run_crosslingual_agreement.py consumes.

Two entry points:

  1) --demo
     Writes a SMALL, hand-authored smoke-test set (data/en.jsonl, data/de.jsonl)
     covering all three slices, so the scoring engine can be validated
     end-to-end on a real model without downloading any suite. These are NOT the
     evaluation data — they exist only to exercise the pipeline.

  2) --tsv FILE --lang en --out data/en.jsonl
     Convert a paired TSV (header: condition<TAB>good<TAB>bad) into normalized
     JSONL, assigning each row a slice via CONDITION_SLICE[lang]. This is the
     format to massage CLAMS (Mueller et al. 2020, github.com/aaronmueller/clams)
     and Marvin & Linzen (2018) suites into — one row per minimal pair, `good`
     = grammatical sentence, `bad` = its agreement-violating twin.

Slice semantics (paper §6.4):
  aligned    : simple agreement, no intervening noun (high-baseline slice)
  attractor  : intervening noun of the OPPOSITE number  -> do(class-3) sever
  control    : intervening noun of the SAME number      -> class-2 neg. control

>>> VERIFY THE MAPPING <<< CONDITION_SLICE below encodes the standard
CLAMS/Marvin-Linzen construction names. Confirm the names match YOUR clone's
output (they differ slightly by version), and confirm that your suite separates
attractor number-MATCH vs number-MISMATCH — if a construction lumps both, split
it upstream, or the control slice will be empty and the verdict will read 'mixed'.
"""
import argparse, json
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"

# Standard construction -> slice. Adjust to your suite's exact condition names.
CONDITION_SLICE = {
    "en": {
        "simple_agrmt": "aligned",
        "vp_coord": "aligned",
        "long_vp_coord": "aligned",
        "prep_match": "control",
        "prep_mismatch": "attractor",
        "subj_rel_match": "control",
        "subj_rel_mismatch": "attractor",
        "obj_rel_match": "control",
        "obj_rel_mismatch": "attractor",
        "obj_rel_across_match": "control",
        "obj_rel_across_mismatch": "attractor",
    },
    "de": {
        "simple_agrmt": "aligned",
        "vp_coord": "aligned",
        "prep_match": "control",
        "prep_mismatch": "attractor",
        "subj_rel_match": "control",
        "subj_rel_mismatch": "attractor",
        "obj_rel_match": "control",
        "obj_rel_mismatch": "attractor",
    },
}

# --- hand-authored smoke-test pairs (NOT the evaluation suite) ----------------
# good = grammatical, bad = agreement-violating twin (verb number flipped).
DEMO = {
    "en": [
        # aligned: subject adjacent to verb, no intervener
        ("simple_agrmt", "The author laughs .", "The author laugh ."),
        ("simple_agrmt", "The authors laugh .", "The authors laughs ."),
        # control: intervening noun of the SAME number as the subject (no cue conflict)
        ("obj_rel_match", "The author that the guard likes laughs .",
                          "The author that the guard likes laugh ."),
        ("obj_rel_match", "The authors that the guards like laugh .",
                          "The authors that the guards like laughs ."),
        # attractor: intervening noun of the OPPOSITE number (proximity vs structure)
        ("obj_rel_mismatch", "The author that the guards like laughs .",
                            "The author that the guards like laugh ."),
        ("obj_rel_mismatch", "The authors that the guard likes laugh .",
                            "The authors that the guard likes laughs ."),
    ],
    "de": [
        ("simple_agrmt", "Der Autor lacht .", "Der Autor lachen ."),
        ("simple_agrmt", "Die Autoren lachen .", "Die Autoren lacht ."),
        ("obj_rel_match", "Der Autor, den der Wächter mag, lacht .",
                          "Der Autor, den der Wächter mag, lachen ."),
        ("obj_rel_match", "Die Autoren, die die Wächter mögen, lachen .",
                          "Die Autoren, die die Wächter mögen, lacht ."),
        ("obj_rel_mismatch", "Der Autor, den die Wächter mögen, lacht .",
                            "Der Autor, den die Wächter mögen, lachen ."),
        ("obj_rel_mismatch", "Die Autoren, die der Wächter mag, lachen .",
                            "Die Autoren, die der Wächter mag, lacht ."),
    ],
}


def slice_for(lang, condition):
    m = CONDITION_SLICE.get(lang, {})
    if condition not in m:
        raise KeyError(f"condition {condition!r} not in CONDITION_SLICE[{lang!r}] — "
                       f"add it (slice in {{aligned,attractor,control}})")
    return m[condition]


def write_jsonl(rows, lang, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {"aligned": 0, "attractor": 0, "control": 0}
    with open(out, "w") as f:
        for condition, good, bad in rows:
            sl = slice_for(lang, condition)
            n[sl] += 1
            f.write(json.dumps({"lang": lang, "condition": condition, "slice": sl,
                                "good": good, "bad": bad}, ensure_ascii=False) + "\n")
        f.flush()
    print(f"wrote {sum(n.values())} pairs -> {out}  by_slice={n}")
    if min(n.values()) == 0:
        empty = [s for s, c in n.items() if c == 0]
        print(f"  WARNING: empty slice(s) {empty}; the verdict needs all three.")


def read_tsv(path):
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for need in ("condition", "good", "bad"):
            if need not in idx:
                raise ValueError(f"TSV must have columns condition/good/bad; got {header}")
        for ln in f:
            if not ln.strip():
                continue
            c = ln.rstrip("\n").split("\t")
            rows.append((c[idx["condition"]], c[idx["good"]], c[idx["bad"]]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="write hand-authored smoke-test sets")
    ap.add_argument("--tsv", help="paired TSV (condition\\tgood\\tbad) to convert")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", help="output JSONL (default data/<lang>.jsonl)")
    args = ap.parse_args()

    if args.demo:
        for lang, rows in DEMO.items():
            write_jsonl(rows, lang, DATA / f"{lang}_demo.jsonl")
        return
    if not args.tsv:
        ap.error("pass --demo or --tsv FILE")
    out = args.out or (DATA / f"{args.lang}.jsonl")
    write_jsonl(read_tsv(args.tsv), args.lang, out)


if __name__ == "__main__":
    main()

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
import argparse, json, re, random
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


# --- CLAMS evalset ingestion (number-based slicing) --------------------------
# A CLAMS condition file is alternating True (grammatical) / False (violation)
# lines. Aligned conditions have no intervener -> slice=aligned. Intervener
# conditions are split into attractor (subject/intervener number MISMATCH) vs
# control (number MATCH) by parsing the two NPs from the GRAMMATICAL sentence.
# This is the §3.2 cut realized on real CLAMS data: attractor = do(class-3)
# (proximity severed), control = class-2 (intervener present, no opposing cue).
CLAMS_ALIGNED = {
    "en": ["simple_agrmt", "vp_coord", "long_vp_coord"],
    "de": ["simple_agrmt", "vp_coord", "long_vp_coord"],
}
# Intervener conditions (per language). EN uses noun morphology; DE uses verb
# agreement (German nouns are often number-invariant: der/die Schriftsteller).
# DE is restricted to obj_rel_across_anim, where both the subject (main verb) and
# the intervener (relative-clause verb) carry a readable agreeing verb — prep_anim
# is skipped for DE because the dative intervener has no agreeing verb and its
# number would need fragile case morphology (vom / den ...-n).
CLAMS_INTERVENER = {
    "en": ["prep_anim", "obj_rel_across_anim"],
    "de": ["obj_rel_across_anim"],
}


# --- English: number on the noun head -----------------------------------------
def _np_extract_en(sent):
    # Head noun of each `the ...` NP. Number is marked on the HEAD, so for the one
    # multiword noun in CLAMS-en's animate vocab — "taxi driver(s)" — the head is
    # the 2nd token ("the taxi drivers" is PLURAL), not "taxi". (Inanimate "the
    # movie the ..." compounds live in *_inanim files, which we do not use.)
    toks = sent.split()
    heads = []
    i = 0
    while i < len(toks):
        if toks[i] == "the" and i + 1 < len(toks):
            head = toks[i + 1]
            if head == "taxi" and i + 2 < len(toks) and toks[i + 2].startswith("driver"):
                head = toks[i + 2]
            heads.append(head)
            i += 2
        else:
            i += 1
    return (heads[0], heads[-1]) if len(heads) >= 2 else None


def _num_en(noun):
    # CLAMS/Marvin-Linzen use curated regular count nouns; trailing -s = plural.
    return "pl" if noun.endswith("s") else "sg"


def _slice_en(cond, good):
    nps = _np_extract_en(good)
    if not nps:
        return None
    return "control" if _num_en(nps[0]) == _num_en(nps[1]) else "attractor"


# --- German: number on the agreeing verb --------------------------------------
# 3sg present = -t, 3pl = -en; irregulars: sein (ist/sind), mögen (mag/mögen).
_DE_VERB_NUM = {"ist": "sg", "sind": "pl", "mag": "sg", "mögen": "pl"}


def _num_de_verb(v):
    if v in _DE_VERB_NUM:
        return _DE_VERB_NUM[v]
    if v.endswith("en"):
        return "pl"
    if v.endswith("t"):
        return "sg"
    return None


def _slice_de(cond, good):
    # obj_rel_across_anim: "DET SUBJ , RELPRON DET INTERV RCVERB , MAINVERB ..."
    # subject number = main verb; intervener number = relative-clause verb.
    parts = good.split(" , ")
    if len(parts) < 3:
        return None
    main_v = parts[-1].split()[0]
    rc = parts[1].split()
    if len(rc) < 2:
        return None
    sn, inn = _num_de_verb(main_v), _num_de_verb(rc[-1])
    if sn is None or inn is None:
        return None
    return "control" if sn == inn else "attractor"


# condition-aware slicer per language: (condition, good_sentence) -> slice | None
SLICER = {"en": _slice_en, "de": _slice_de}


def read_clams_pairs(path):
    """Pair alternating True/False lines -> [(good, bad), ...]."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(path) if ln.strip()]
    pairs = []
    for i in range(0, len(rows) - 1, 2):
        (t1, s1), (t2, s2) = rows[i], rows[i + 1]
        if t1 == "True" and t2 == "False":
            pairs.append((s1, s2))
    return pairs


def build_from_clams(clams_dir, lang, out, cap=None, seed=0):
    """Read a CLAMS <lang>_evalset/ dir, slice by number, write normalized JSONL."""
    rng = random.Random(seed)
    evdir = Path(clams_dir) / f"{lang}_evalset"
    if not evdir.is_dir():
        raise FileNotFoundError(f"{evdir} not found (clone github.com/aaronmueller/clams)")
    if lang not in SLICER:
        raise KeyError(f"no CLAMS slicer for lang={lang!r}; add it to SLICER/CLAMS_*")
    slicer = SLICER[lang]
    buckets = {"aligned": [], "attractor": [], "control": []}

    for cond in CLAMS_ALIGNED[lang]:
        p = evdir / f"{cond}.txt"
        if not p.exists():
            print(f"  skip aligned {cond} (missing)"); continue
        for good, bad in read_clams_pairs(p):
            buckets["aligned"].append((cond, good, bad))

    skipped = 0
    for cond in CLAMS_INTERVENER[lang]:
        p = evdir / f"{cond}.txt"
        if not p.exists():
            print(f"  skip intervener {cond} (missing)"); continue
        for good, bad in read_clams_pairs(p):
            sl = slicer(cond, good)
            if sl is None:
                skipped += 1; continue
            suffix = "match" if sl == "control" else "mismatch"
            buckets[sl].append((f"{cond}_{suffix}", good, bad))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {s: 0 for s in buckets}
    with open(out, "w") as f:
        for sl, items in buckets.items():
            if cap and len(items) > cap:
                items = rng.sample(items, cap)
            for cond, good, bad in items:
                n[sl] += 1
                f.write(json.dumps({"lang": lang, "condition": cond, "slice": sl,
                                    "good": good, "bad": bad}, ensure_ascii=False) + "\n")
    print(f"wrote {sum(n.values())} pairs -> {out}  by_slice={n}"
          + (f"  (capped {cap}/slice)" if cap else "")
          + (f"  [{skipped} unparsed intervener lines]" if skipped else ""))
    if min(n.values()) == 0:
        print(f"  WARNING: empty slice(s) {[s for s, c in n.items() if c == 0]}")


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
    ap.add_argument("--clams_dir", help="path to a CLAMS clone; reads <lang>_evalset/, slices by number")
    ap.add_argument("--tsv", help="paired TSV (condition\\tgood\\tbad) to convert")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--cap", type=int, default=None, help="max pairs per slice (balanced subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output JSONL (default data/<lang>.jsonl)")
    args = ap.parse_args()

    if args.demo:
        for lang, rows in DEMO.items():
            write_jsonl(rows, lang, DATA / f"{lang}_demo.jsonl")
        return
    if args.clams_dir:
        out = args.out or (DATA / f"{args.lang}.jsonl")
        build_from_clams(args.clams_dir, args.lang, out, cap=args.cap, seed=args.seed)
        return
    if not args.tsv:
        ap.error("pass --demo, --clams_dir DIR, or --tsv FILE")
    out = args.out or (DATA / f"{args.lang}.jsonl")
    write_jsonl(read_tsv(args.tsv), args.lang, out)


if __name__ == "__main__":
    main()

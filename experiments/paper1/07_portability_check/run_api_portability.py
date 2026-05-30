"""
run_api_portability.py

§8 procedure-portability check on a COMPETENT, non-fine-tuned model.

Same intent as run_zeroshot_portability.py, but instead of a weak local LM it
drives a frontier API model (DeepSeek or Gemini, OpenAI-compatible). The point
is the open item left by the local run: zero-shot Qwen2.5-1.5B did not fire the
do(class-3) flag, but it is a degenerate yes-biased reasoner — so its silence is
a specificity LOWER-BOUND, not a demonstration that a *capable* model resists a
perfect in-context shortcut. This run supplies the competent witness:
  - high baseline + premise-respecting  => actually a reasoner
  - color made a perfect in-context cue (r=1.0) => maximal shortcut temptation
  - if do(color) still does NOT collapse it, that is the strong portability
    result; if it DOES, that is equally informative (a capable model rides the
    in-context shortcut, and the flag generalizes beyond trained-in entanglement).

The by-construction ladder (gen / regimes / conditions / bootstrap) is IMPORTED
verbatim from run_zeroshot_portability.py so this is the identical §5.4 ladder —
only the model call differs. run_regime() already takes `predict` as a param.

BOUNDARY DISCIPLINE is unchanged: class assignment is IMPORTED from the
construction's own data statistics, NOT re-measured on the model's pretraining
(§3.4). This is a portability check under an imported class assignment, not a
re-measured (b) and not a capability claim.

Usage:
    export DEEPSEEK_API_KEY=sk-...        # or GEMINI_API_KEY=...
    python run_api_portability.py --provider deepseek
    python run_api_portability.py --provider gemini --n 200 --workers 8
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

# --- import the construction verbatim (single source of truth) --------------
_spec = importlib.util.spec_from_file_location(
    "rzp", BASE_DIR / "run_zeroshot_portability.py")
rzp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rzp)

PROVIDERS = {
    "deepseek": {"model": "deepseek-v4-flash",
                 "base_url": "https://api.deepseek.com",
                 "key_env": "DEEPSEEK_API_KEY"},
    "gemini":   {"model": "gemini-2.5-flash",
                 "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "key_env": "GEMINI_API_KEY"},
}


def build_api_runner(client, model_name, workers):
    """Return predict(texts)->list[str] mirroring run_deepseek_zeroshot.call_api
    (retries + reasoning_content fallback), parallelized across `workers`."""

    def call_one(prompt, max_retries=6):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name, max_tokens=512,
                    messages=[{"role": "system", "content": rzp.SYS},
                              {"role": "user", "content": prompt}])
                msg = resp.choices[0].message
                text = msg.content or ""
                if not text or rzp.extract_yesno(text) == "unknown":
                    rc = getattr(msg, "reasoning_content", "") or ""
                    if rc:
                        text = rc
                return rzp.extract_yesno(text)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  [give-up after retries: {e}]", flush=True)
                    return "unknown"
        return "unknown"

    def predict(texts):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(call_one, texts))

    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    ap.add_argument("--n", type=int, default=200, help="items per condition")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = PROVIDERS[args.provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        sys.exit(f"Error: {cfg['key_env']} not set. "
                 f"export {cfg['key_env']}=... and re-run.")
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    model_name = cfg["model"]
    predict = build_api_runner(client, model_name, args.workers)
    print(f"Provider: {args.provider} | model: {model_name} | "
          f"n={args.n}/cond | workers={args.workers}", flush=True)

    CR = rzp.run_regime("chain_required", "severed", 1.0, args.n, predict)
    SA = rzp.run_regime("shortcut_available", "correlated", 1.0, args.n, predict)

    summary = {
        "experiment": "§8 procedure-portability check — competent non-fine-tuned model (API)",
        "model": model_name,
        "provider": args.provider,
        "evaluation": "zero-shot frontier model; verdict structure under an "
                      "IMPORTED class assignment (b NOT re-measured on "
                      "pretraining; §3.4)",
        "design": "Identical §5.4 two-premise ladder (imported verbatim from "
                  "run_zeroshot_portability.py). Class label imported from the "
                  "construction's own statistics. Competent witness for the open "
                  "item left by the weak local run: does a capable, premise-"
                  "respecting model ride a perfect (r=1.0) in-context color cue? "
                  "No expected outcome is hard-coded — both directions inform.",
        "n_per_condition": args.n,
        "regime_chain_required": CR,
        "regime_shortcut_available": SA,
        "scope": "Portability check under imported class assignment; not a "
                 "re-measured (b), not a capability claim.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"api_portability_{args.provider}_summary.json"
    json.dump(summary, open(out, "w"), indent=2)

    def show(name, R):
        print(f"\n--- {name} ---")
        print(f"  imported class        = {R['imported_class_assignment']}")
        print(f"  baseline acc          = {R['baseline_fullM_acc_ci']}")
        print(f"  P1/P2 ablate withhold = {R['P1_ablate_withhold_ci']} / "
              f"{R['P2_ablate_withhold_ci']}")
        print(f"  drop do(color)[c3]    = {R['drop_do(color)_class3_ci']}  "
              f"drop do(name)[c2] = {R['drop_do(name)_class2_ci']}")
        print(f"  flagged_correlational = {R['flagged_correlational']}")
        print(f"  class-1 flip rate     = {R['class1_flip_rate_ci']}  "
              f"(correctly_variant={R['class1_correctly_variant']})")

    print(f"\n=== §8 portability — {model_name} (imported class assignment) ===")
    show("chain_required", CR)
    show("shortcut_available", SA)
    print(f"\nSaved: {out}")
    print("\nReading: baseline + premise-respecting => competent witness; then "
          "do(color) flag in shortcut_available is the decisive test.")


if __name__ == "__main__":
    main()

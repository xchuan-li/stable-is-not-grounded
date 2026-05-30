"""
make_figures.py — paper1 figures, generated live from the result JSONs.

Design: a restrained, cohesive palette (slate / muted blue / teal / rose),
hairline spines, generous whitespace. Sized for a SINGLE COLUMN (~3.3in) so the
floats place cleanly in the two-column layout without overlapping text.

Outputs PDF (manuscript) + PNG (preview):
  fig_hierarchy   — accuracy ⊋ stability ⊋ grounding (conceptual)
  fig_portability — §7 non-circularity (sever a perfect in-context shortcut)
  fig_boolq_gap   — §6.4 overstatement gap; SC split undecidable in the wild

Run:  ../../../.venv/bin/python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).parent
EXP = FIG_DIR.parents[2] / "experiments" / "paper1"

# --- cohesive palette -------------------------------------------------------
INK    = "#26323a"   # near-black slate: text, axes
ROSE   = "#c44e5a"   # the sever / "flag" colour
TEAL   = "#2a9d8f"   # grounded / good
BLUE   = "#37618e"   # accuracy / primary measure
SAND   = "#e6b450"   # secondary accent (gap)
GREY   = "#d3d7db"   # neutral control
FAINT  = "#8a949c"   # de-emphasised text

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.major.size": 0, "ytick.major.size": 2.5,
    "figure.dpi": 200,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}")


def ci_err(ci):
    p, lo, hi = ci
    return [[max(0., p - lo)], [max(0., hi - p)]]


# ---------------------------------------------------------------------------
def fig_portability():
    ft = json.load(open(EXP / "06_multiseed_variance/results_qwen_multiseed/"
          "multiseed_summary.json"))["results"]["shortcut_available"]["aggregate"]
    zs = json.load(open(EXP / "07_portability_check/results/"
          "zeroshot_portability_summary.json"))["regime_shortcut_available"]
    ds = json.load(open(EXP / "07_portability_check/results/"
          "api_portability_deepseek_summary.json"))["regime_shortcut_available"]

    labels = ["Qwen-LoRA\nfine-tuned", "Qwen-1.5B\nzero-shot", "DeepSeek\nzero-shot"]
    c3 = [ft["drop_color"]["mean"], zs["drop_do(color)_class3_ci"][0],
          ds["drop_do(color)_class3_ci"][0]]
    c2 = [ft["drop_name"]["mean"], zs["drop_do(name)_class2_ci"][0],
          ds["drop_do(name)_class2_ci"][0]]
    ez, ed = ci_err(zs["drop_do(color)_class3_ci"]), ci_err(ds["drop_do(color)_class3_ci"])
    c3e = [[0., ez[0][0], ed[0][0]], [0., ez[1][0], ed[1][0]]]

    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    x, w = range(3), 0.36
    ax.bar([i - w/2 for i in x], c3, w, yerr=c3e, capsize=2.5,
           error_kw=dict(lw=0.8), color=ROSE, label="do(class-3): sever shortcut", zorder=3)
    ax.bar([i + w/2 for i in x], c2, w, color=GREY,
           label="do(class-2): noise control", zorder=3)
    ax.axhline(0, color=INK, lw=0.7)
    ax.grid(axis="y", color=GREY, lw=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("accuracy drop  $\\Delta$")
    ax.set_ylim(-0.07, 0.60)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.3)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right",
              handlelength=1.1, borderaxespad=0.2)
    for i, v in enumerate(c3):
        ax.text(i - w/2, v + (0.02 if v >= 0 else -0.035), f"{v:+.2f}",
                ha="center", fontsize=7, fontweight="bold", color=ROSE)
    ax.annotate("trained on the shortcut", xy=(0, 0.52), xytext=(1.05, 0.50),
                fontsize=6.6, style="italic", color=FAINT, va="center",
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.6))
    ax.annotate("never trained on it → no flag", xy=(2, 0.02), xytext=(1.55, 0.24),
                fontsize=6.6, style="italic", color=FAINT, ha="center",
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.6))
    save(fig, "fig_portability")
    print("  c3:", [round(v, 3) for v in c3])


# ---------------------------------------------------------------------------
def fig_boolq_gap():
    rows = [("Qwen-1.5B", "summary_400"), ("DeepSeek", "summary_deepseek_400"),
            ("Claude", "summary_claude_400")]
    acc, accE, sc, scE, gap = [], [[], []], [], [[], []], []
    for _, f in rows:
        d = json.load(open(EXP / f"03_boolq_measurement/results/{f}.json"))
        acc.append(d["headline_accuracy_original_prompt"]); sc.append(d["stable_correct_pct"])
        gap.append(d["headline_overstatement_gap"])
        ea, es = ci_err(d["headline_accuracy_ci"]), ci_err(d["stable_correct_ci"])
        accE[0].append(ea[0][0]); accE[1].append(ea[1][0])
        scE[0].append(es[0][0]); scE[1].append(es[1][0])

    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    x, w = range(3), 0.36
    ax.bar([i - w/2 for i in x], acc, w, yerr=accE, capsize=2.5, error_kw=dict(lw=0.8),
           color=BLUE, label="headline accuracy", zorder=3)
    ax.bar([i + w/2 for i in x], sc, w, yerr=scE, capsize=2.5, error_kw=dict(lw=0.8),
           color=TEAL, label="stable-correct (SC)", zorder=3)
    ax.grid(axis="y", color=GREY, lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    for i in x:
        ax.annotate("", xy=(i + w/2, sc[i]), xytext=(i + w/2, acc[i]),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
        ax.text(i + w/2 + 0.05, (acc[i] + sc[i]) / 2, f"{gap[i]:+.2f}",
                fontsize=6.6, va="center", color=INK)
    ax.set_ylabel("fraction of items")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], fontsize=7.3)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right",
              handlelength=1.1, borderaxespad=0.2)
    ax.text(0.02, 1.0, "gap = accuracy $-$ SC; within SC,\n"
            "grounded vs spurious is undecidable here",
            transform=ax.transAxes, fontsize=6.4, style="italic", color=FAINT, va="top")
    save(fig, "fig_boolq_gap")
    print("  gap:", [round(g, 3) for g in gap])


# ---------------------------------------------------------------------------
def fig_hierarchy():
    fig, ax = plt.subplots(figsize=(3.35, 2.95))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")

    def band(x, y, w, h, fc, ec, label, lc):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.015,rounding_size=0.18",
                     fc=fc, ec=ec, lw=1.1, zorder=1))
        ax.text(x + 0.28, y + h - 0.42, label, fontsize=8, color=lc,
                fontweight="bold", zorder=2)

    band(0.4, 0.6, 9.2, 7.4, "#eef2f6", BLUE,  "accuracy", BLUE)
    band(1.25, 1.15, 7.5, 5.5, "#e3ecf3", BLUE, "stability  (SC)", BLUE)
    band(2.2, 1.7, 5.6, 3.6, "#e1f1ee", TEAL,  "grounding", TEAL)

    ax.text(5.0, 3.0, "SC-grounded", ha="center", fontsize=8.5,
            color=TEAL, fontweight="bold")
    ax.text(5.0, 2.45, "survives do(class-3)", ha="center", fontsize=6.6,
            color=FAINT, style="italic")
    ax.text(5.0, 1.36, "SC-spurious  ·  collapses under do(class-3)",
            ha="center", fontsize=6.6, color=FAINT, style="italic")
    ax.text(5.0, 0.82, "U  ·  unstable under perturbation",
            ha="center", fontsize=6.6, color=FAINT, style="italic")
    ax.text(5.0, 8.45, "SW  ·  stable-wrong  (outside accuracy)",
            ha="center", fontsize=6.8, color=FAINT, style="italic")
    ax.text(9.45, 4.2, "each containment\nstrict, non-empty", ha="right",
            va="center", fontsize=6.2, color=FAINT)
    save(fig, "fig_hierarchy")


if __name__ == "__main__":
    fig_hierarchy(); fig_portability(); fig_boolq_gap()

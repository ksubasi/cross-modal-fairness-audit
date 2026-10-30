"""Figure: effect of the token cap on each language subgroup.

(a) fused-text length per clip against the cap (violin, log scale);
(b) median word pieces per character with bootstrap 95% intervals.

Usage:
    python plot_truncation.py --csv path/to/tikharm_vi_en_complete.csv [--out ../paper/figures/truncation.pdf]
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

import config

# Muted palette that stays legible in greyscale.
PALETTE = ["#6BAF92", "#E08A5F", "#9E9E9E"]
ORDER = ["en", "vi", "unknown"]
LABEL = {"en": "English", "vi": "Viet.", "unknown": "undet."}
def recolour(ax):
    """Apply PALETTE by position."""
    bodies = [c for c in ax.collections if hasattr(c, "set_facecolor")]
    for body, colour in zip(bodies, PALETTE):
        body.set_facecolor(colour)
        body.set_edgecolor("#333333")


MIN_CHARS = 20   # same floor the audit uses for the segmentation-rate comparison


def build(csv_path: str, split: str) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].copy()
    df["combined_text"] = df["combined_text"].fillna("")
    df = df[df["combined_text"].str.strip().astype(bool)]
    df["n_tokens"] = [len(tok.encode(t, add_special_tokens=True)) for t in df["combined_text"]]
    df["n_chars"] = df["combined_text"].str.len()
    df["tokens_per_char"] = df["n_tokens"] / df["n_chars"].clip(lower=1)
    df["lang"] = pd.Categorical(df["language"], categories=ORDER, ordered=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="../paper/figures/truncation.pdf")
    args = ap.parse_args()

    df = build(args.csv, args.split)
    cap = config.MAX_TEXT_LENGTH

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    # single-column width for the ACL template
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.15, 2.35), gridspec_kw={"wspace": 0.45})

    sns.violinplot(data=df, x="lang", y="n_tokens", order=ORDER, inner="quartile",
                   linewidth=0.6, cut=0, ax=ax1)
    recolour(ax1)
    ax1.axhline(cap, color="#333333", linestyle="--", linewidth=0.9)
    ax1.text(2.42, cap * 1.12, f"cap = {cap}", ha="right", va="bottom", fontsize=6.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("fused-text length (word pieces)", fontsize=7.5)
    ax1.set_xlabel("")
    for i, g in enumerate(ORDER):
        keep = (cap / df[df.lang == g].n_tokens.clip(lower=1)).clip(upper=1).median()
        ax1.text(i / (len(ORDER) - 1) * 0.86 + 0.07, 1.015, f"{100*keep:.0f}%",
                 transform=ax1.transAxes, ha="center", va="bottom", fontsize=6.5)
    ax1.set_title("(a) what the cap cuts\n" + r"$\it{median\ share\ kept}$",
                  fontsize=8, pad=12)

    # Panel (b): point estimates with intervals.
    rate = df[df["n_chars"] > MIN_CHARS]
    rng = np.random.default_rng(0)
    for i, g in enumerate(ORDER):
        v = rate[rate.lang == g].tokens_per_char.values
        boot = [np.median(rng.choice(v, len(v))) for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax2.errorbar(i, np.median(v), yerr=[[np.median(v) - lo], [hi - np.median(v)]],
                     fmt="o", ms=3.5, color=PALETTE[i], ecolor="#333333",
                     elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
    ax2.set_ylabel("word pieces per character", fontsize=7.5)
    ax2.set_xlabel("")
    ax2.set_xlim(-0.5, len(ORDER) - 0.5)
    ax2.set_title("(b) why: median $\\pm$ 95% CI", fontsize=8, pad=4)

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[g] for g in ORDER], fontsize=7)
        ax.margins(x=0.06)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)
    for g in ORDER:
        s = df[df.lang == g]
        r = rate[rate.lang == g]
        print(f"  {LABEL[g]:11s} n={len(s):4d}  median tokens={s.n_tokens.median():6.0f}  "
              f"%over cap={100*(s.n_tokens > cap).mean():5.1f}  "
              f"median pieces/char={r.tokens_per_char.median():.4f}")


if __name__ == "__main__":
    main()

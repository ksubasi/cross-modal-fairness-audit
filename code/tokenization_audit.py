"""Does the fixed token cap remove the same share of text for each language?

The classifier reads one fused string truncated at config.MAX_TEXT_LENGTH word
pieces. Four Vietnamese-vs-English comparisons are tested and Holm-corrected.

Usage:
    python tokenization_audit.py --csv path/to/tikharm_vi_en_complete.csv [--split test] [--out ../outputs/tokenization_audit.json]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from scipy import stats
from transformers import AutoTokenizer

import config

# Tokens per character is unstable for very short texts, so that comparison
# uses only rows above this length. The truncation measures use every row.
MIN_CHARS_FOR_RATE = 20


def holm_bonferroni(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for k, (name, p) in enumerate(items):
        running = max(running, (m - k) * p)   # step-down: adjusted p is monotone
        out[name] = min(1.0, running)
    return out


def measure(csv_path: str, split: str) -> dict:
    tok = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    cap = config.MAX_TEXT_LENGTH

    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].copy()
    df["combined_text"] = df["combined_text"].fillna("")
    df["n_tokens"] = [len(tok.encode(t, add_special_tokens=True)) for t in df["combined_text"]]
    df["n_chars"] = df["combined_text"].str.len()
    # share of the clip's own text that survives the cap, capped at 1 for
    # clips already shorter than the limit
    df["frac_kept"] = (cap / df["n_tokens"].clip(lower=1)).clip(upper=1)
    df["tokens_per_char"] = df["n_tokens"] / df["n_chars"].clip(lower=1)

    vi, en = df[df.language == "vi"], df[df.language == "en"]
    rate = df[df["n_chars"] > MIN_CHARS_FOR_RATE]
    vi_r, en_r = rate[rate.language == "vi"], rate[rate.language == "en"]

    # 1. truncated at all: Fisher's exact
    table = [
        [int((vi.n_tokens > cap).sum()), int((vi.n_tokens <= cap).sum())],
        [int((en.n_tokens > cap).sum()), int((en.n_tokens <= cap).sum())],
    ]
    _, p_over = stats.fisher_exact(table)

    # 2. share retained: Mann-Whitney (skewed distributions)
    _, p_kept = stats.mannwhitneyu(vi.frac_kept, en.frac_kept, alternative="two-sided")

    # 3-4. word pieces per character, and text length
    _, p_rate = stats.mannwhitneyu(vi_r.tokens_per_char, en_r.tokens_per_char, alternative="two-sided")
    _, p_vol = stats.mannwhitneyu(vi_r.n_chars, en_r.n_chars, alternative="two-sided")

    out = {
        "split": split,
        "cap_tokens": cap,
        "tokenizer": config.TEXT_MODEL_NAME,
        "n": {g: int(len(s)) for g, s in df.groupby("language")},
        "pct_over_cap": {g: round(100 * float((s.n_tokens > cap).mean()), 1) for g, s in df.groupby("language")},
        "mean_frac_kept": {g: round(float(s.frac_kept.mean()), 3) for g, s in df.groupby("language")},
        "median_frac_kept": {g: round(float(s.frac_kept.median()), 3) for g, s in df.groupby("language")},
        "median_tokens_per_char": {g: round(float(s.tokens_per_char.median()), 4) for g, s in rate.groupby("language")},
        "median_chars": {g: int(s.n_chars.median()) for g, s in rate.groupby("language")},
        "tests_vi_vs_en": {
            "pct_over_cap": {"gap_pp": round(100 * float((vi.n_tokens > cap).mean() - (en.n_tokens > cap).mean()), 1),
                             "fisher_p": round(float(p_over), 4)},
            "frac_kept": {"mannwhitney_p": round(float(p_kept), 4)},
            "tokens_per_char": {"vi_over_en_pct": round(100 * (float(vi_r.tokens_per_char.median()) /
                                                              float(en_r.tokens_per_char.median()) - 1), 1),
                                "mannwhitney_p": float(f"{p_rate:.3g}")},
            "n_chars": {"mannwhitney_p": round(float(p_vol), 4)},
        },
    }
    raw = {
        "pct_over_cap": float(p_over),
        "frac_kept": float(p_kept),
        "tokens_per_char": float(p_rate),
        "n_chars": float(p_vol),
    }
    adj = holm_bonferroni(raw)
    out["holm_adjusted_p"] = {k: float(f"{v:.3g}") for k, v in adj.items()}
    out["significant_after_holm"] = sorted(k for k, v in adj.items() if v < 0.05)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="processed combined_text CSV from preprocess.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="../outputs/tokenization_audit.json")
    args = ap.parse_args()

    res = measure(args.csv, args.split)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

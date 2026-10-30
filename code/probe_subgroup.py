"""Subgroup probe on pre-fusion features.

A linear probe predicts source-language subgroup from each branch's features
(5-fold macro-F1). Each branch is compared against its own permutation null, the
same probe trained on that branch's features with shuffled labels.

Inputs (written by audit.py): full_text_features.npy, full_video_features.npy,
audit_per_item.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe_macro_f1(X, y, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, solver="lbfgs"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv, scoring="f1_macro").mean()


def evaluate_branch(name, X, y, n_perm, rng):
    observed = probe_macro_f1(X, y, seed=0)
    null = np.array([probe_macro_f1(X, rng.permutation(y), seed=i) for i in range(n_perm)])
    # +1 smoothing so a permutation test can never report exactly p=0.
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    n_sd = (observed - null.mean()) / null.std()
    print(
        f"{name:12s} observed macro-F1={observed:.4f} | "
        f"chance (shuffled) mean={null.mean():.4f} std={null.std():.4f} "
        f"range[{null.min():.3f},{null.max():.3f}] | "
        f"{n_sd:.1f} SD above chance, permutation p={p:.4f}"
    )
    return {"observed": observed, "null_mean": null.mean(), "null_std": null.std(), "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="outputs",
                    help="dir holding full_{text,video}_features.npy and audit_per_item.csv")
    ap.add_argument("--n-perm", type=int, default=50, help="permutations per branch's null")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    d = Path(args.features_dir)
    text = np.load(d / "full_text_features.npy")
    video = np.load(d / "full_video_features.npy")
    df = pd.read_csv(d / "audit_per_item.csv")
    full = df[df["config"] == "full"].reset_index(drop=True)
    y = full["subgroup"].astype(str).to_numpy()
    assert len(y) == text.shape[0] == video.shape[0], "feature/label count mismatch"

    rng = np.random.default_rng(args.seed)
    evaluate_branch("Text branch", text, y, args.n_perm, rng)
    evaluate_branch("Video branch", video, y, args.n_perm, rng)


if __name__ == "__main__":
    main()

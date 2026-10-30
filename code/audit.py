"""Subgroup-disaggregated FPR/FNR for the audited pipeline, plus the
leave-one-modality-out comparison.

Usage:
    python audit.py --checkpoint outputs/checkpoints/final --csv data/processed/tikharm_vi_en_complete.csv
"""

from __future__ import annotations

import os

# Inference runs on a single GPU; set before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from adapters import MTikGuardAdapter, is_flagged
from adapters.base import CLASS_NAMES
from dataset import MultimodalDataset, TextField, collate_fn
from model import MultimodalFairnessModel

# (label, text_field, null_video): "full" is the unablated pipeline;
# each of the others removes one modality.
MODALITY_CONFIGS: list[tuple[str, TextField, bool]] = [
    ("full", "combined", False),
    ("- video", "combined", True),
    ("- audio", "ocr_only", False),
    ("- text", "audio_only", False),
]


@torch.no_grad()
def run_predictions(
    model, tokenizer, df: pd.DataFrame, text_field: TextField, null_video: bool, device,
    batch_size: int = 8, return_features: bool = False,
):
    """Return predictions, and the pre-fusion per-branch features when
    return_features=True (used for the subgroup probe).
    """
    ds = MultimodalDataset(df, tokenizer, text_field=text_field, null_video=null_video)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    all_preds = []
    all_text_feats = [] if return_features else None
    all_video_feats = [] if return_features else None
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_features=return_features)
        all_preds.append(torch.argmax(outputs["logits"], dim=1).cpu().numpy())
        if return_features:
            all_text_feats.append(outputs["text_features"].cpu().numpy())
            all_video_feats.append(outputs["video_features"].cpu().numpy())
    preds = np.concatenate(all_preds)
    if return_features:
        return preds, np.concatenate(all_text_feats), np.concatenate(all_video_feats)
    return preds


def binary_fpr_fnr(y_true_flagged: np.ndarray, y_pred_flagged: np.ndarray) -> dict:
    """y_*_flagged: 1 = flagged (non-Safe), 0 = passed (Safe)."""
    actual_negative = y_true_flagged == 0  # ground-truth Safe
    actual_positive = y_true_flagged == 1  # ground-truth flagged

    fp = int(np.sum(actual_negative & (y_pred_flagged == 1)))
    tn = int(np.sum(actual_negative & (y_pred_flagged == 0)))
    fn = int(np.sum(actual_positive & (y_pred_flagged == 0)))
    tp = int(np.sum(actual_positive & (y_pred_flagged == 1)))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    accuracy = (tp + tn) / len(y_true_flagged) if len(y_true_flagged) > 0 else float("nan")
    return {"fpr": fpr, "fnr": fnr, "accuracy": accuracy, "n": int(len(y_true_flagged))}


def max_pairwise_gap(per_subgroup: dict[str, dict], metric: str) -> float:
    values = [v[metric] for v in per_subgroup.values() if not np.isnan(v[metric])]
    if len(values) < 2:
        return float("nan")
    return max(values) - min(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="e.g. outputs/checkpoints/final, from train.py")
    parser.add_argument("--csv", default=str(config.PROCESSED_CSV))
    parser.add_argument("--output", default=str(config.OUTPUTS_DIR / "audit_results.json"))
    args = parser.parse_args()

    adapter = MTikGuardAdapter(args.csv)
    splits = adapter.load_splits()
    test_df = splits.test
    subgroup_col = adapter.subgroup_columns()[0]  # "language" for MTikGuard
    print(f"[audit] test set: {len(test_df)} items, subgroup axis: {subgroup_col}")
    print(f"[audit] subgroup distribution:\n{test_df[subgroup_col].value_counts()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    print(f"[audit] loading checkpoint from {args.checkpoint}...")
    t0 = time.time()
    # Sub-encoders are built empty in __init__; this call loads all weights,
    # including the fine-tuned encoders.
    model = MultimodalFairnessModel.from_pretrained(args.checkpoint).to(device)
    print(f"[audit] checkpoint loaded in {time.time() - t0:.1f}s")

    y_true_class = test_df["class_name"].values
    y_true_flagged = np.array([1 if is_flagged(c) else 0 for c in y_true_class])

    # The DataLoader does not shuffle, so per-item rows stay aligned with test_df.
    test_df_ordered = test_df.reset_index(drop=True)
    per_item_rows: list[dict] = []

    results: dict[str, dict] = {}
    for config_name, text_field, null_video in MODALITY_CONFIGS:
        print(f"[audit] running configuration: {config_name}")
        # Features are saved for the unablated pipeline only.
        if config_name == "full":
            preds, text_feats, video_feats = run_predictions(
                model, tokenizer, test_df, text_field, null_video, device, return_features=True
            )
            np.save(Path(args.output).parent / "full_text_features.npy", text_feats)
            np.save(Path(args.output).parent / "full_video_features.npy", video_feats)
            print(f"[audit] saved full-config feature vectors: text {text_feats.shape}, video {video_feats.shape}")
        else:
            preds = run_predictions(model, tokenizer, test_df, text_field, null_video, device)
        pred_class = [CLASS_NAMES[p] for p in preds]
        y_pred_flagged = np.array([1 if is_flagged(c) else 0 for c in pred_class])

        for i in range(len(test_df_ordered)):
            per_item_rows.append({
                "config": config_name,
                "video_path": test_df_ordered.loc[i, "video_path"],
                "subgroup": test_df_ordered.loc[i, subgroup_col],
                "true_class": test_df_ordered.loc[i, "class_name"],
                "pred_class": pred_class[i],
            })

        per_subgroup = {}
        for group_value in test_df[subgroup_col].unique():
            mask = (test_df[subgroup_col] == group_value).values
            per_subgroup[str(group_value)] = binary_fpr_fnr(y_true_flagged[mask], y_pred_flagged[mask])

        aggregate = binary_fpr_fnr(y_true_flagged, y_pred_flagged)

        results[config_name] = {
            "per_subgroup": per_subgroup,
            "aggregate": aggregate,
            "fpr_gap_pp": round(max_pairwise_gap(per_subgroup, "fpr") * 100, 2),
            "fnr_gap_pp": round(max_pairwise_gap(per_subgroup, "fnr") * 100, 2),
        }
        print(f"[audit]   {config_name}: aggregate={aggregate}, per_subgroup={per_subgroup}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[audit] wrote results to {output_path}")

    # Per-item predictions for all four configurations.
    per_item_path = output_path.parent / "audit_per_item.csv"
    pd.DataFrame(per_item_rows).to_csv(per_item_path, index=False)
    print(f"[audit] wrote {len(per_item_rows)} per-item rows to {per_item_path}")

    print("\n[audit] per-subgroup error rates, full model:")
    for group, stats in results["full"]["per_subgroup"].items():
        print(f"  {group}: FPR={stats['fpr']:.3f} FNR={stats['fnr']:.3f} N={stats['n']}")
    print(f"  Aggregate: FPR={results['full']['aggregate']['fpr']:.3f} FNR={results['full']['aggregate']['fnr']:.3f} N={results['full']['aggregate']['n']}")

    print("\n[audit] FPR/FNR gap (pp) per configuration:")
    for config_name, _, _ in MODALITY_CONFIGS:
        r = results[config_name]
        print(f"  {config_name}: FPR gap={r['fpr_gap_pp']} pp, FNR gap={r['fnr_gap_pp']} pp")


if __name__ == "__main__":
    main()

"""Adapter for MTikGuard/TikHarm. Reads the CSV produced by preprocess.py.

The CSV follows the schema of MTikGuard's training notebook, with one added
column, `language`, which is the audit's subgroup axis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import DatasetAdapter, Splits

REQUIRED_COLUMNS = [
    "video_path",
    "class_name",
    "combined_text",
    "audio_text",
    "ocr_text",
    "language",
    "split",
]


class MTikGuardAdapter(DatasetAdapter):
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    def load_splits(self) -> Splits:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"{self.csv_path} not found. Run code/preprocess.py first "
                "to build it from the raw TikHarm videos (see README's "
                "'Getting the data' section)."
            )
        df = pd.read_csv(self.csv_path)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required column(s) {missing}. "
                "Was this CSV built by code/preprocess.py?"
            )

        # Fail early on missing or unreadable video files.
        exists = df["video_path"].apply(lambda p: Path(p).exists())
        if not exists.all():
            n_missing = (~exists).sum()
            df = df[exists].reset_index(drop=True)
            print(f"[mtikguard_adapter] dropped {n_missing} rows with a missing video_path")

        df["combined_text"] = df["combined_text"].fillna("")
        df["audio_text"] = df["audio_text"].fillna("")
        df["ocr_text"] = df["ocr_text"].fillna("")

        unknown_classes = set(df["class_name"].unique()) - set(self.class_names())
        if unknown_classes:
            raise ValueError(
                f"Unexpected class_name value(s) in {self.csv_path}: {unknown_classes}. "
                f"Expected one of {self.class_names()}."
            )

        return Splits(
            train=df[df["split"] == "train"].reset_index(drop=True),
            val=df[df["split"] == "val"].reset_index(drop=True),
            test=df[df["split"] == "test"].reset_index(drop=True),
        )

    def subgroup_columns(self) -> list[str]:
        return ["language"]

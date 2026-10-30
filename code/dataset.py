"""torch Dataset over an adapter dataframe, shared by train.py and audit.py."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

from adapters.base import CLASS_NAMES
from model import VideoLoader

TextField = Literal["combined", "ocr_only", "audio_only"]


def build_text(row: pd.Series, field: TextField) -> str:
    """Build the text input for a modality-ablation condition: 'combined' is the
    full pipeline; 'ocr_only' and 'audio_only' drop the other segment.
    """
    if field == "combined":
        return str(row.get("combined_text", "") or "")
    if field == "ocr_only":
        ocr = str(row.get("ocr_text", "") or "")
        return f"OCR: {ocr}" if ocr else ""
    if field == "audio_only":
        audio = str(row.get("audio_text", "") or "")
        return f"AUDIO: {audio}" if audio else ""
    raise ValueError(f"unknown text_field: {field}")


class MultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_field: TextField = "combined",
        null_video: bool = False,
        max_text_length: int = 64,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.null_video = null_video
        self.max_text_length = max_text_length
        self.video_loader = VideoLoader()
        self.class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        text = build_text(row, self.text_field) or "[EMPTY]"
        text_inputs = self.tokenizer(
            text, max_length=self.max_text_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        pixel_values = (
            self.video_loader.blank_video() if self.null_video else self.video_loader.load(row["video_path"])
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.class_to_id[row["class_name"]], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

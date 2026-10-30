"""Dataset adapter interface used by train.py and audit.py."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd

# MTikGuard is a four-way classifier; "Safe" is the only non-flagged class.
# FPR and FNR are computed on the binary flagged-vs-passed collapse.
CLASS_NAMES = ["Adult Content", "Harmful Content", "Safe", "Suicide"]
FLAGGED_CLASSES = {"Adult Content", "Harmful Content", "Suicide"}


def is_flagged(class_name: str) -> bool:
    return class_name in FLAGGED_CLASSES


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class DatasetAdapter(abc.ABC):
    """One adapter per dataset. Every dataframe returned by load_splits()
    must have at least these columns:

        video_path     -- path to the .mp4 file
        class_name     -- one of CLASS_NAMES
        combined_text  -- fused transcript/OCR text fed to the text encoder
        audio_text     -- transcript only (for modality ablation)
        ocr_text       -- on-screen text only (for modality ablation)

    plus the columns named by subgroup_columns().
    """

    @abc.abstractmethod
    def load_splits(self) -> Splits:
        ...

    @abc.abstractmethod
    def subgroup_columns(self) -> list[str]:
        """Column name(s) to disaggregate the audit by, e.g. ["language"]."""
        ...

    def class_names(self) -> list[str]:
        return CLASS_NAMES

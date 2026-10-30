"""Paths and constants shared across the scripts. Most can be overridden by CLI flags."""

from pathlib import Path

# Raw TikHarm videos from the public Kaggle dataset anhoangvo/tikharm-dataset,
# laid out as Dataset/{train,val,test}/<class_name>/*.mp4.
KAGGLE_TIKHARM_DATASET = "anhoangvo/tikharm-dataset"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw" / "Dataset"  # {train,val,test}/<class_name>/*.mp4
PROCESSED_CSV = DATA_DIR / "processed" / "tikharm_vi_en_complete.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"

TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"
VIDEO_MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
WHISPER_MODEL_NAME = "openai/whisper-small"

TARGET_LANGUAGES = ["vi", "en"]  # MTikGuard's language scope

MAX_FRAMES = 8
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 64

# Hyperparameters follow MTikGuard's training notebook. Per-device batch size
# is 4 with 4 accumulation steps (effective batch 16, as published) to fit a T4.
LEARNING_RATE = 1e-4
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch size 16
NUM_EPOCHS = 6

"""Build the combined_text CSV from raw TikHarm videos.

Follows MTikGuard's extraction: Whisper ASR (Vietnamese/English) for speech and
EasyOCR for on-screen text, concatenated into one combined_text field. The
detected language is also written to a `language` column, the audit's subgroup.

Usage:
    python preprocess.py --data-dir data/raw/Dataset --output data/processed/tikharm_vi_en_complete.csv
    python preprocess.py --smoke-test   # 3 videos/class/split

Expects Dataset/{train,val,test}/<class_name>/*.mp4.
"""

from __future__ import annotations

import os

# Run on a single GPU; set before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import subprocess
import tempfile
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from adapters.base import CLASS_NAMES

warnings.filterwarnings("ignore")


def extract_audio_wav(video_path: Path, target_sr: int = 16000) -> Path:
    """Extract a mono wav to the system temp directory. Raises on failure."""
    out_path = Path(tempfile.gettempdir()) / f"preprocess_{uuid.uuid4().hex}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-loglevel", "error", str(out_path),
        ],
        check=True,
        timeout=60,
    )
    if not out_path.exists():
        raise RuntimeError(f"ffmpeg reported success but {out_path} was not created")
    return out_path


WHISPER_CHUNK_LENGTH_S = 30.0  # Whisper's native fixed context window


def _chunk_audio(audio: np.ndarray, sr: int = 16000, chunk_length_s: float = WHISPER_CHUNK_LENGTH_S) -> list[np.ndarray]:
    """Split audio into non-overlapping chunks that fit Whisper's 30 s window."""
    chunk_samples = int(chunk_length_s * sr)
    if len(audio) <= chunk_samples:
        return [audio]
    return [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples)]


class SpeechTranscriber:
    """Whisper transcription called directly (not via pipeline()), with manual
    chunking for clips longer than 30 s.
    """

    def __init__(self, model_name: str = config.WHISPER_MODEL_NAME):
        import time

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        print(f"[preprocess] loading Whisper ({model_name})...")
        t0 = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[preprocess] Whisper loaded in {time.time() - t0:.1f}s (device={self.device})")

    def _transcribe_chunk(self, chunk: np.ndarray, lang: str) -> str:
        import torch

        input_features = self.processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(input_features, language=lang, task="transcribe")
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def transcribe(self, video_path: Path) -> str:
        """Transcribe with each target language forced and keep the longer transcript,
        as MTikGuard does.
        """
        wav_path = extract_audio_wav(video_path)
        try:
            import librosa

            audio, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            duration_s = len(audio) / 16000
            peak_amplitude = float(np.abs(audio).max()) if len(audio) else 0.0
            if duration_s < 0.1 or peak_amplitude < 1e-4:
                # silent or missing audio track
                return ""

            chunks = _chunk_audio(audio)

            best = ""
            any_chunk_succeeded = False
            last_error: Exception | None = None
            for lang in config.TARGET_LANGUAGES:
                chunk_texts = []
                for chunk in chunks:
                    try:
                        chunk_text = self._transcribe_chunk(chunk, lang)
                        any_chunk_succeeded = True
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                    except Exception as e:
                        # skip a failed chunk and keep the rest
                        last_error = e
                        continue
                text = " ".join(chunk_texts)
                if len(text) > len(best):
                    best = text

            if not best and not any_chunk_succeeded and last_error is not None:
                # every chunk failed in both languages
                raise RuntimeError(f"Whisper generation failed for every chunk in both {config.TARGET_LANGUAGES}: {last_error}") from last_error
            return best
        finally:
            wav_path.unlink(missing_ok=True)


class OnScreenTextExtractor:
    def __init__(self):
        import time

        import easyocr

        print("[preprocess] loading EasyOCR (en+vi detection/recognition models)...")
        t0 = time.time()
        self.reader = easyocr.Reader(["en", "vi"], gpu=self._has_gpu())
        print(f"[preprocess] EasyOCR loaded in {time.time() - t0:.1f}s")

    @staticmethod
    def _has_gpu() -> bool:
        import torch

        return torch.cuda.is_available()

    def extract(self, video_path: Path, num_frames: int = 5) -> str:
        cap = cv2.VideoCapture(str(video_path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return ""
            indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
            texts: list[str] = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                results = self.reader.readtext(frame, detail=0)
                texts.extend(r.strip() for r in results if r.strip())
            # de-duplicate; on-screen text often repeats across frames
            seen, deduped = set(), []
            for t in texts:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    deduped.append(t)
            return " ".join(deduped)
        finally:
            cap.release()


def detect_language(text: str) -> str:
    """Return 'vi' or 'en', or 'unknown' for empty or ambiguous text."""
    if not text or not text.strip():
        return "unknown"
    try:
        from langdetect import detect

        code = detect(text)
        return code if code in config.TARGET_LANGUAGES else "unknown"
    except Exception:
        return "unknown"


def discover_videos(raw_dir: Path, limit_per_class: int | None = None) -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            print(f"[preprocess] warning: {split_dir} not found, skipping")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in CLASS_NAMES:
                print(f"[preprocess] warning: unrecognized class folder '{class_name}', skipping")
                continue
            videos = sorted(class_dir.glob("*.mp4"))
            if limit_per_class is not None:
                videos = videos[:limit_per_class]
            for video_path in videos:
                records.append(
                    {
                        "video_path": str(video_path),
                        "video_name": video_path.name,
                        "class_name": class_name,
                        "original_dir": split,
                        "split": split,
                    }
                )
    return records


OUTPUT_FIELDNAMES = [
    "video_path", "video_name", "class_name", "original_dir", "split",
    "ocr_text", "audio_text", "combined_text", "language",
    "processing_status", "error_message",
]


def process_one(record: dict, transcriber: "SpeechTranscriber", ocr: "OnScreenTextExtractor") -> dict:
    video_path = Path(record["video_path"])
    status, error = "ok", ""
    try:
        audio_text = transcriber.transcribe(video_path)
    except Exception as e:
        audio_text, status, error = "", "error", f"ASR error: {e}"[:200]
    try:
        ocr_text = ocr.extract(video_path)
    except Exception as e:
        ocr_text, status, error = "", "error", (error + f" OCR error: {e}")[:200]

    parts = []
    if audio_text:
        parts.append(f"AUDIO: {audio_text}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    combined_text = " | ".join(parts)

    # detect language from the transcript, falling back to on-screen text
    language = detect_language(audio_text) if audio_text else detect_language(ocr_text)

    record.update(
        ocr_text=ocr_text,
        audio_text=audio_text,
        combined_text=combined_text,
        language=language,
        processing_status=status,
        error_message=error,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(config.RAW_VIDEO_DIR), help="Dataset/{train,val,test}/<class>/*.mp4 root")
    parser.add_argument("--output", default=str(config.PROCESSED_CSV))
    parser.add_argument("--smoke-test", action="store_true", help="process only 3 videos/class/split, for a fast sanity check")
    parser.add_argument("--limit-per-class", type=int, default=None, help="cap videos per class/split (overridden by --smoke-test)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="start fresh instead of skipping videos already in --output",
    )
    args = parser.parse_args()

    limit = 3 if args.smoke_test else args.limit_per_class

    raw_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = discover_videos(raw_dir, limit_per_class=limit)
    if not records:
        raise SystemExit(
            f"No videos found under {raw_dir}. Expected Dataset/{{train,val,test}}/<class_name>/*.mp4 "
            "-- see README's 'Getting the data' section for the Kaggle download command."
        )
    print(f"[preprocess] discovered {len(records)} videos across train/val/test")

    already_done: set[str] = set()
    write_header = True
    if output_path.exists() and not args.overwrite:
        existing = pd.read_csv(output_path)
        already_done = set(existing["video_path"])
        write_header = False
        print(f"[preprocess] resuming: {len(already_done)} videos already in {output_path}, skipping those")
    elif output_path.exists() and args.overwrite:
        output_path.unlink()

    todo = [r for r in records if r["video_path"] not in already_done]
    print(f"[preprocess] {len(todo)} videos left to process")
    if not todo:
        print("[preprocess] nothing to do")
        return

    transcriber = SpeechTranscriber()
    ocr = OnScreenTextExtractor()

    import csv
    import time

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, record in enumerate(todo):
            video_start = time.time()
            record = process_one(record, transcriber, ocr)
            video_elapsed = time.time() - video_start

            # write after every video so an interrupted run can resume
            writer.writerow({k: record.get(k, "") for k in OUTPUT_FIELDNAMES})
            f.flush()

            print(f"[preprocess] {i + 1}/{len(todo)} this run ({len(already_done) + i + 1}/{len(records)} total) "
                  f"-- {record['video_name']} took {video_elapsed:.1f}s, status={record['processing_status']}")

    df = pd.read_csv(output_path)
    print(f"[preprocess] {output_path} now has {len(df)} rows total")
    print(f"[preprocess] language distribution:\n{df['language'].value_counts()}")
    print(f"[preprocess] class distribution:\n{df['class_name'].value_counts()}")
    print(f"[preprocess] processing_status distribution:\n{df['processing_status'].value_counts()}")


if __name__ == "__main__":
    main()

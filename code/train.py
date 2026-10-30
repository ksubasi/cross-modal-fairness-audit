"""Fine-tune the audited pipeline on TikHarm.

Hyperparameters (config.py) follow MTikGuard: learning rate 1e-4, effective batch
size 16, 6 epochs, best checkpoint by validation macro-F1.

Usage:
    python train.py
    python train.py --csv path/to.csv --output-dir path/to/checkpoints
"""

from __future__ import annotations

import os

# Train on a single GPU; set before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

import config
from adapters import MTikGuardAdapter
from dataset import MultimodalDataset, collate_fn
from model import MultimodalFairnessConfig, MultimodalFairnessModel


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_score": f1_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(config.PROCESSED_CSV))
    parser.add_argument("--output-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="evaluate and save every step for one epoch, for use with the --smoke-test CSV",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="start from scratch instead of resuming from the latest checkpoint in --output-dir",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="resume from this checkpoint-N/ path instead of the latest one",
    )
    args = parser.parse_args()

    resume_from_checkpoint = None
    if args.checkpoint:
        resume_from_checkpoint = args.checkpoint
        print(f"[train] resuming from explicitly given checkpoint: {resume_from_checkpoint}")
    elif not args.fresh:
        from pathlib import Path

        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        ) if output_dir.exists() else []
        if checkpoints:
            resume_from_checkpoint = str(checkpoints[-1])
            print(f"[train] resuming from latest checkpoint found: {resume_from_checkpoint}")

    adapter = MTikGuardAdapter(args.csv)
    splits = adapter.load_splits()
    print(f"[train] train={len(splits.train)} val={len(splits.val)} test={len(splits.test)}")

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_ds = MultimodalDataset(splits.train, tokenizer, text_field="combined")
    val_ds = MultimodalDataset(splits.val, tokenizer, text_field="combined")

    model_config = MultimodalFairnessConfig(num_classes=len(adapter.class_names()))
    model = MultimodalFairnessModel(model_config)

    # Load base pretrained weights into the sub-encoders.
    print("[train] loading pretrained encoder weights (text + video)...")
    t0 = time.time()
    model.load_pretrained_encoders()
    print(f"[train] pretrained encoders loaded in {time.time() - t0:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type != "cuda":
        print("[train] WARNING: no GPU detected. This model is not trainable on CPU in "
              "reasonable time -- run this on a Kaggle GPU notebook or PLGrid, per README.")

    eval_save_steps = 1 if args.smoke_test else 50
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1 if args.smoke_test else config.NUM_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=1 if args.smoke_test else config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=0 if args.smoke_test else 100,
        logging_steps=eval_save_steps,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        dataloader_drop_last=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    val_results = trainer.evaluate()
    print(f"[train] validation results: {val_results}")

    with open(f"{args.output_dir}/train_results.json", "w") as f:
        json.dump({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in val_results.items()}, f, indent=2)

    print(f"[train] checkpoint saved to {args.output_dir}/final -- next: python audit.py --checkpoint {args.output_dir}/final")


if __name__ == "__main__":
    main()

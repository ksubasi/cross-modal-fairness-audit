"""MTikGuard's fusion architecture.

Two encoders: video (TimeSformer) and text (multilingual DistilBERT). Transcript
and on-screen text are concatenated into one string in preprocess.py, so there is
no separate audio encoder; removing audio in the ablation means dropping the
transcript segment before tokenization.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, TimesformerForVideoClassification

import config

warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultimodalFairnessConfig(PretrainedConfig):
    model_type = "multimodal_fairness"

    def __init__(
        self,
        num_classes: int = 4,
        text_model_name: str = config.TEXT_MODEL_NAME,
        video_model_name: str = config.VIDEO_MODEL_NAME,
        fusion_hidden_size: int = 256,
        classifier_hidden_size: int = 128,
        max_frames: int = config.MAX_FRAMES,
        image_size: int = config.IMAGE_SIZE,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.text_model_name = text_model_name
        self.video_model_name = video_model_name
        self.fusion_hidden_size = fusion_hidden_size
        self.classifier_hidden_size = classifier_hidden_size
        self.max_frames = max_frames
        self.image_size = image_size
        self.dropout = dropout


class MultimodalFairnessModel(PreTrainedModel):
    """Text encoder + video encoder -> projections -> fusion MLP -> linear classifier.
    Layer shapes match MTikGuard's SmartMultimodalModel.

    Sub-encoders are built empty in __init__ and filled by load_pretrained_encoders()
    for a fresh model. This keeps from_pretrained() on a fine-tuned checkpoint
    compatible with transformers' meta-device loading.
    """

    config_class = MultimodalFairnessConfig

    def __init__(self, cfg: MultimodalFairnessConfig):
        super().__init__(cfg)
        import time

        from transformers import AutoConfig, AutoModel, TimesformerConfig

        # Fetch the sub-encoder configs.
        t0 = time.time()
        text_config = AutoConfig.from_pretrained(cfg.text_model_name)
        print(f"[model] text config ({cfg.text_model_name}) fetched in {time.time() - t0:.1f}s")
        self.text_encoder = AutoModel.from_config(text_config)

        t0 = time.time()
        video_config = TimesformerConfig.from_pretrained(cfg.video_model_name)
        print(f"[model] video config ({cfg.video_model_name}) fetched in {time.time() - t0:.1f}s")
        self.video_encoder = TimesformerForVideoClassification(video_config)

        text_dim = self.text_encoder.config.hidden_size
        video_dim = self.video_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, cfg.fusion_hidden_size), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, cfg.fusion_hidden_size), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.fusion_hidden_size * 2, cfg.fusion_hidden_size),
            nn.LayerNorm(cfg.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_size, cfg.classifier_hidden_size),
            nn.LayerNorm(cfg.classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Linear(cfg.classifier_hidden_size, cfg.num_classes)

        # Required by PreTrainedModel for composite models.
        self.post_init()

    def load_pretrained_encoders(self):
        """Load base pretrained weights into the sub-encoders. Call once on a fresh model
        before training, never after loading a fine-tuned checkpoint.
        """
        from transformers import AutoModel, TimesformerForVideoClassification

        pretrained_text = AutoModel.from_pretrained(self.config.text_model_name)
        self.text_encoder.load_state_dict(pretrained_text.state_dict())

        pretrained_video = TimesformerForVideoClassification.from_pretrained(
            self.config.video_model_name, num_labels=self.config.num_classes, ignore_mismatched_sizes=True
        )
        # backbone only; the classification head is unused
        self.video_encoder.timesformer.load_state_dict(pretrained_video.timesformer.state_dict())

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, return_features=False, **kwargs):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)

        video_features = self.video_encoder.timesformer(pixel_values).last_hidden_state[:, 0]
        video_features = self.video_projection(video_features)

        fused = self.fusion(torch.cat([text_features, video_features], dim=1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
        output = {"loss": loss, "logits": logits}
        if return_features:
            # Pre-fusion branch features, returned for the subgroup probe.
            output["text_features"] = text_features
            output["video_features"] = video_features
        return output


class VideoLoader:
    """Uniformly samples max_frames frames with ImageNet normalization. Unreadable
    files return black frames, the same input used for the null-video ablation.
    """

    def __init__(self, max_frames: int = config.MAX_FRAMES, image_size: int = config.IMAGE_SIZE):
        self.max_frames = max_frames
        self.image_size = image_size
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def blank_video(self) -> torch.Tensor:
        black = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        black = (black - self.mean) / self.std
        frames = np.stack([black] * self.max_frames)
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    def load(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                warnings.warn(f"unreadable video, using blank fallback: {video_path}")
                return self.blank_video()

            indices = np.linspace(0, total - 1, self.max_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    frames.append(frames[-1] if frames else np.zeros_like(self.mean.repeat(self.image_size, 0).repeat(self.image_size, 1)))
                    continue
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame = (frame - self.mean) / self.std
                frames.append(frame)
            video = np.stack(frames[: self.max_frames])
            return torch.from_numpy(video).permute(0, 3, 1, 2).float()
        finally:
            cap.release()

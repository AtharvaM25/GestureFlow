"""Checkpoint loading and single-frame prediction. No camera, no UI."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gestureflow.core import preprocessing
from gestureflow.core.model import SkeletonCNN


class ConfigMismatch(RuntimeError):
    """The checkpoint was trained under different preprocessing settings."""


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: np.ndarray       # full softmax, one entry per label -- kept for decoding
    skeleton: np.ndarray    # (H, W, 3) uint8, exactly what the CNN saw
    embedding: np.ndarray   # 128-d pooled features, L2-normalized -- used by calibration


class GesturePredictor:
    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        # Catches the classic silent failure: preprocessing edited, model not
        # retrained, so the CNN sees images it was never trained on.
        live_cfg = preprocessing.config()
        if ckpt["handnorm_config"] != live_cfg:
            raise ConfigMismatch(
                "preprocessing changed since training.\n"
                f"  trained with: {ckpt['handnorm_config']}\n"
                f"  now:          {live_cfg}")

        self.labels = ckpt["labels"]
        self.model = SkeletonCNN(len(self.labels))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()              # BatchNorm uses running stats, dropout off

    @torch.no_grad()
    def predict(self, lm_list):
        """lm_list: the 21 landmarks from the hand detector, (x, y) or (x, y, z) each."""
        pts = preprocessing.normalize(np.asarray(lm_list, np.float32))
        img = preprocessing.render(pts)

        x = torch.from_numpy(preprocessing.to_input(img)).unsqueeze(0)
        logits, emb = self.model.forward_with_embedding(x)
        probs = F.softmax(logits, dim=1)[0].numpy()
        idx = int(probs.argmax())
        return Prediction(self.labels[idx], float(probs[idx]), probs, img,
                          F.normalize(emb, dim=1)[0].numpy())

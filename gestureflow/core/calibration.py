"""
Personal calibration: a few frames of each letter from one user, averaged into "prototypes"
in the model's 128-d feature space. A frame's prediction is then a blend of the model's own
probabilities and how close the frame is to each prototype:

    p = (1 - ALPHA) * softmax(model) + ALPHA * softmax(cosine(frame, prototype) / TAU)

The model is not retrained, so calibration takes seconds and is undone by deleting it.
Measured effect (docs/calibration_report.json): on a recording session the model never saw,
5 frames per letter raised accuracy from 92.7% to 97.3%, and the share of frames confident
enough to commit a letter from 25% to 79%, with no wrong letter confident enough to commit.
"""

import numpy as np

ALPHA = 0.75          # weight of the prototypes; see the sweep in docs/calibration_report.json
# Softmax temperature for cosine similarities. Every letter's features are fairly similar, so
# a soft temperature (0.1 was the first choice) spread the vote and capped confidence below the
# 0.9 commit threshold: calibrated users could never commit a letter. 0.03 keeps the accuracy
# gain and lets letters commit -- calibration_eval reports both, so this can't regress silently.
TAU = 0.03
MIN_SAMPLES = 3       # frames per letter
MAX_SAMPLES = 30


def build_prototypes(embeddings: dict[str, list[np.ndarray]], labels: list[str]) -> np.ndarray:
    """embeddings: label -> that user's L2-normalized embeddings. Returns (len(labels), 128),
    one unit-length mean per label, in the model's label order."""
    missing = [label for label in labels if len(embeddings.get(label, [])) < MIN_SAMPLES]
    if missing:
        raise ValueError(f"need at least {MIN_SAMPLES} samples for: {', '.join(missing)}")
    protos = np.stack([np.mean(embeddings[label], axis=0) for label in labels])
    return protos / np.linalg.norm(protos, axis=1, keepdims=True)


def _softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Calibration:
    def __init__(self, prototypes: np.ndarray, alpha: float = ALPHA, tau: float = TAU):
        self.prototypes = np.asarray(prototypes, dtype=np.float32)
        self.alpha = alpha
        self.tau = tau

    def apply(self, probs: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """Blend the model's probabilities with prototype similarity. Works on one frame
        (probs (C,), embedding (128,)) or a batch ((N, C), (N, 128))."""
        proto = _softmax(np.asarray(embedding) @ self.prototypes.T / self.tau)
        return (1 - self.alpha) * np.asarray(probs) + self.alpha * proto

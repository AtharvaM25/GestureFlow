"""
One recognition stream: landmarks in, a result per frame out.

The webcam app, the REST API and the WebSocket all drive recognition through this class, so
they share the same prediction, smoothing and commit behaviour. One instance per stream --
the commit state must not be shared between users.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from gestureflow.core.calibration import Calibration
from gestureflow.core.inference import GesturePredictor, Prediction
from gestureflow.core.temporal import LetterCommitter

TOP_K = 3


@dataclass
class FrameResult:
    hand: bool
    gesture: str | None = None          # smoothed label shown to the user
    raw: str | None = None              # this frame's argmax
    confidence: float | None = None     # this frame's top probability
    stability: float = 0.0              # progress of the current hold toward a commit, 0..1
    committed: str | None = None        # letter committed on this frame, if any
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0             # time spent in step(), measured
    calibrated: bool = False            # whether a personal calibration was applied
    prediction: Prediction | None = None


def top_k(predictor: GesturePredictor, probs: np.ndarray, k: int = TOP_K):
    idx = np.argsort(probs)[::-1][:k]
    return [(predictor.labels[i], float(probs[i])) for i in idx]


class RecognitionSession:
    def __init__(self, predictor: GesturePredictor, committer: LetterCommitter | None = None,
                 calibration: Calibration | None = None):
        self.predictor = predictor
        self.committer = committer or LetterCommitter()
        self.calibration = calibration

    def step(self, landmarks) -> FrameResult:
        """landmarks: 21 points from the hand detector, or None when no hand is visible."""
        t0 = time.perf_counter()
        if landmarks is None:
            self.committer.no_hand()
            return FrameResult(hand=False, latency_ms=(time.perf_counter() - t0) * 1000,
                               calibrated=self.calibration is not None)

        pred = self.predictor.predict(landmarks)
        probs = pred.probs
        if self.calibration is not None:
            probs = self.calibration.apply(probs, pred.embedding)
        idx = int(np.argmax(probs))
        label, confidence = self.predictor.labels[idx], float(probs[idx])

        step = self.committer.update(label, confidence)
        return FrameResult(
            hand=True,
            gesture=step.smoothed,
            raw=label,
            confidence=confidence,
            stability=self.committer.progress,
            committed=step.committed,
            alternatives=top_k(self.predictor, probs),
            latency_ms=(time.perf_counter() - t0) * 1000,
            calibrated=self.calibration is not None,
            prediction=pred,
        )

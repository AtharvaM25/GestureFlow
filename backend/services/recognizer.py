"""The one GesturePredictor the server process uses, loaded on first request."""

import hashlib
from functools import lru_cache
from pathlib import Path

from gestureflow.core.inference import GesturePredictor


class Recognizer:
    def __init__(self, model_path: str):
        path = Path(model_path)
        self.predictor = GesturePredictor(path)
        # identifies exactly which checkpoint produced a session's letters
        self.model_version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    @property
    def labels(self) -> list[str]:
        return self.predictor.labels


@lru_cache
def get_recognizer() -> Recognizer:
    from backend.config import get_settings
    return Recognizer(get_settings().model_path)

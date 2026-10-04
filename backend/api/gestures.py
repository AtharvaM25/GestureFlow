"""The letters the loaded model recognises, with how well each does on an unseen session."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends

from backend.config import Settings, get_settings
from backend.deps import current_user
from backend.schemas import GestureOut
from backend.services.recognizer import Recognizer, get_recognizer

router = APIRouter(prefix="/gestures", tags=["gestures"], dependencies=[Depends(current_user)])


@lru_cache
def f1_by_label(report_path: str) -> dict[str, float]:
    """Mean per-letter F1 across the evaluation report's held-out sessions."""
    p = Path(report_path)
    if not p.exists():
        return {}
    scores: dict[str, list[float]] = {}
    for fold in json.loads(p.read_text()).get("folds", {}).values():
        for label, f1 in fold.get("per_class_f1", {}).items():
            scores.setdefault(label, []).append(f1)
    return {label: round(sum(v) / len(v), 4) for label, v in scores.items()}


@router.get("", response_model=list[GestureOut])
def list_gestures(recognizer: Recognizer = Depends(get_recognizer),
                  settings: Settings = Depends(get_settings)):
    f1 = f1_by_label(settings.eval_report_path)
    return [GestureOut(label=label, f1_unseen_session=f1.get(label))
            for label in sorted(recognizer.labels)]

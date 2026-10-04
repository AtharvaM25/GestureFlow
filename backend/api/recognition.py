"""Classify a single frame over REST. Live streams use the WebSocket instead."""

import time

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.api.calibration import calibration_for
from backend.db import get_db
from backend.deps import current_user
from backend.models import User
from backend.schemas import Alternative, LandmarksIn, PredictionOut
from backend.services.recognizer import Recognizer, get_recognizer
from gestureflow.core.calibration import Calibration
from gestureflow.core.session import top_k

router = APIRouter(prefix="/recognition", tags=["recognition"])


def predict(recognizer: Recognizer, landmarks, calibration: Calibration | None) -> PredictionOut:
    t0 = time.perf_counter()
    pred = recognizer.predictor.predict(landmarks)
    probs = pred.probs if calibration is None else calibration.apply(pred.probs, pred.embedding)
    best = top_k(recognizer.predictor, probs)
    return PredictionOut(
        gesture=best[0][0],
        confidence=best[0][1],
        alternatives=[Alternative(gesture=g, confidence=c) for g, c in best],
        latency_ms=(time.perf_counter() - t0) * 1000,
        calibrated=calibration is not None,
    )


@router.post("/landmarks", response_model=PredictionOut)
async def recognize_landmarks(body: LandmarksIn, db: Session = Depends(get_db),
                              user: User = Depends(current_user),
                              recognizer: Recognizer = Depends(get_recognizer)):
    """Classify one frame of landmarks, with the user's calibration if they have one.
    Nothing is stored."""
    calibration = calibration_for(db, user.id, recognizer)
    return await run_in_threadpool(predict, recognizer, body.landmarks, calibration)

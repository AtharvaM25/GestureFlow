"""
Personal calibration: record a few frames of each letter once, and recognition adapts to
your hand (gestureflow/core/calibration.py). The frames are turned into per-letter average
features on arrival; the landmarks themselves are not stored.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.models import CalibrationProfile, User
from backend.schemas import CalibrationIn, CalibrationOut
from backend.services.recognizer import Recognizer, get_recognizer
from gestureflow.core.calibration import Calibration, build_prototypes

router = APIRouter(prefix="/calibration", tags=["calibration"])


def profile_for(db: Session, user_id: int) -> CalibrationProfile | None:
    return db.scalar(select(CalibrationProfile).where(CalibrationProfile.user_id == user_id))


def calibration_for(db: Session, user_id: int, recognizer: Recognizer) -> Calibration | None:
    """The user's calibration, if they have one for the model this server is running."""
    p = profile_for(db, user_id)
    if p is None or p.model_version != recognizer.model_version or p.labels != recognizer.labels:
        return None
    return Calibration(p.prototypes)


def status_of(p: CalibrationProfile | None, recognizer: Recognizer) -> CalibrationOut:
    current = p is not None and p.model_version == recognizer.model_version \
        and p.labels == recognizer.labels
    return CalibrationOut(calibrated=current, stale=p is not None and not current,
                          samples_per_label=p.samples_per_label if p else None,
                          created_at=p.created_at if p else None, labels=recognizer.labels)


@router.get("", response_model=CalibrationOut)
def get_calibration(db: Session = Depends(get_db), user: User = Depends(current_user),
                    recognizer: Recognizer = Depends(get_recognizer)):
    return status_of(profile_for(db, user.id), recognizer)


@router.post("", response_model=CalibrationOut)
async def save_calibration(body: CalibrationIn, db: Session = Depends(get_db),
                           user: User = Depends(current_user),
                           recognizer: Recognizer = Depends(get_recognizer)):
    """Replace the user's calibration. Every letter the model knows needs 3-30 frames."""
    unknown = sorted(set(body.samples) - set(recognizer.labels))
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            f"not letters this model knows: {', '.join(unknown)}")

    def embed():
        return {label: [recognizer.predictor.predict(f).embedding for f in frames]
                for label, frames in body.samples.items()}

    embeddings = await run_in_threadpool(embed)
    try:
        protos = build_prototypes(embeddings, recognizer.labels)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e

    p = profile_for(db, user.id)
    if p is None:
        p = CalibrationProfile(user_id=user.id)
        db.add(p)
    p.model_version = recognizer.model_version
    p.labels = list(recognizer.labels)
    p.prototypes = protos.round(6).tolist()
    p.samples_per_label = min(len(v) for v in embeddings.values())
    db.commit()
    db.refresh(p)
    return status_of(p, recognizer)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def delete_calibration(db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = profile_for(db, user.id)
    if p is not None:
        db.delete(p)
        db.commit()

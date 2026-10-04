"""Recognition sessions: one per live stream. Letters are added by the WebSocket."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import Settings, get_settings
from backend.db import get_db
from backend.deps import current_user, sentence_limit
from backend.models import RecognitionSession, User
from backend.schemas import SentenceOut, SessionDetailOut, SessionOut
from backend.services.recognizer import get_recognizer
from gestureflow.sentence import generate_sentence

router = APIRouter(prefix="/sessions", tags=["sessions"])


def owned_or_404(db: Session, session_id: int, user: User) -> RecognitionSession:
    s = db.get(RecognitionSession, session_id)
    if s is None or s.user_id != user.id:      # don't reveal other users' session ids
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    return s


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(db: Session = Depends(get_db), user: User = Depends(current_user),
                   recognizer=Depends(get_recognizer)):
    s = RecognitionSession(user_id=user.id, model_version=recognizer.model_version)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


@router.get("", response_model=list[SessionOut])
def list_sessions(limit: int = 50, db: Session = Depends(get_db),
                  user: User = Depends(current_user)):
    return db.scalars(select(RecognitionSession)
                      .where(RecognitionSession.user_id == user.id)
                      .order_by(RecognitionSession.id.desc())
                      .limit(min(max(limit, 1), 200))).all()


@router.get("/{session_id}", response_model=SessionDetailOut)
def get_session(session_id: int, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    return owned_or_404(db, session_id, user)


@router.post("/{session_id}/end", response_model=SessionOut)
def end_session(session_id: int, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    s = owned_or_404(db, session_id, user)
    if s.ended_at is None:
        s.ended_at = datetime.now(UTC)
        db.commit()
    return s


@router.post("/{session_id}/sentence", response_model=SentenceOut)
async def make_sentence(session_id: int, db: Session = Depends(get_db),
                        user: User = Depends(current_user),
                        settings: Settings = Depends(get_settings)):
    """Turn the session's letters into a sentence with the configured LLM (Ollama locally,
    Groq online). Runs once per request, never per frame; limited per user per hour."""
    s = owned_or_404(db, session_id, user)
    if settings.sentence_provider == "none":
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "sentence generation is turned off on this server")
    sentence_limit.hit(f"user:{user.id}")
    if settings.sentence_provider == "groq":
        args = (list(s.text), settings.groq_model, None, "groq", settings.groq_api_key)
    else:
        args = (list(s.text), settings.ollama_model, settings.ollama_url, "ollama", None)
    try:
        sentence = await run_in_threadpool(generate_sentence, *args)
    except Exception as e:  # connection refused, model not pulled, bad key, quota, ...
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"sentence model unavailable: {type(e).__name__}") from e
    s.sentence = sentence
    db.commit()
    return SentenceOut(sentence=sentence)

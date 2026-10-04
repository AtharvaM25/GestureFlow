"""Your own usage over a time window: totals, letters per day, and which letters you sign."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.deps import current_user
from backend.models import CommittedLetter, RecognitionSession, User
from backend.schemas import AnalyticsOut, DayCount, LetterStats

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)   # SQLite returns naive UTC


@router.get("", response_model=AnalyticsOut)
def analytics(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db),
              user: User = Depends(current_user)):
    today = datetime.now(UTC).date()
    first_day = today - timedelta(days=days - 1)
    since = datetime.combine(first_day, datetime.min.time(), tzinfo=UTC)

    sessions = db.scalars(select(RecognitionSession).where(
        RecognitionSession.user_id == user.id, RecognitionSession.started_at >= since)).all()
    letters = db.scalars(
        select(CommittedLetter).join(RecognitionSession)
        .where(RecognitionSession.user_id == user.id, CommittedLetter.created_at >= since)).all()

    per_day = {first_day + timedelta(days=i): 0 for i in range(days)}
    per_letter: dict[str, list[float]] = defaultdict(list)
    for letter in letters:
        day = _utc(letter.created_at).date()
        if day in per_day:
            per_day[day] += 1
        per_letter[letter.gesture].append(letter.confidence)

    n = len(letters)
    return AnalyticsOut(
        days=days,
        sessions=len(sessions),
        letters=n,
        frames=sum(s.frame_count for s in sessions),
        avg_confidence=round(sum(x.confidence for x in letters) / n, 4) if n else None,
        avg_latency_ms=round(sum(x.latency_ms for x in letters) / n, 2) if n else None,
        per_day=[DayCount(date=d, letters=c) for d, c in per_day.items()],
        per_letter=sorted(
            (LetterStats(gesture=g, count=len(c), avg_confidence=round(sum(c) / len(c), 4))
             for g, c in per_letter.items()),
            key=lambda s: (-s.count, s.gesture)),
    )

"""
Four tables: users, their recognition sessions, the letters committed in each, and each
user's calibration.

Privacy: nothing here stores camera frames, images or landmark coordinates -- only the
letters a user commits, and for calibration the per-letter average of the model's features.
"""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


def _now():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = _now()


class RecognitionSession(Base):
    __tablename__ = "recognition_sessions"
    __table_args__ = (
        CheckConstraint("frame_count >= 0", name="ck_sessions_frames"),
        Index("ix_sessions_user_started", "user_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         nullable=False)
    # first 12 hex characters of the checkpoint's SHA-256
    model_version: Mapped[str] = mapped_column(String(12), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    sentence: Mapped[str | None] = mapped_column(Text)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    started_at: Mapped[datetime] = _now()
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    letters: Mapped[list["CommittedLetter"]] = relationship(
        back_populates="session", order_by="CommittedLetter.id", cascade="all, delete-orphan")


class CommittedLetter(Base):
    """A letter the user held long enough to commit -- not every frame."""
    __tablename__ = "committed_letters"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_letters_confidence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("recognition_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    gesture: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    alternatives: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = _now()

    session: Mapped[RecognitionSession] = relationship(back_populates="letters")


class CalibrationProfile(Base):
    """One per user: per-letter prototypes in the model's feature space (see
    gestureflow/core/calibration.py). Only valid for the checkpoint it was built with."""
    __tablename__ = "calibration_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         nullable=False, unique=True)
    model_version: Mapped[str] = mapped_column(String(12), nullable=False)
    labels: Mapped[list] = mapped_column(JSON, nullable=False)
    prototypes: Mapped[list] = mapped_column(JSON, nullable=False)   # len(labels) x 128
    samples_per_label: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _now()

"""Request and response bodies. Every field the API accepts is validated here."""

import math
from datetime import UTC, datetime
from datetime import date as date_type
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field, field_validator


def _as_utc(v: datetime) -> datetime:
    # SQLite hands back timestamps without a timezone; they are UTC. Saying so explicitly
    # stops browsers from reading them as local time.
    return v if v.tzinfo else v.replace(tzinfo=UTC)


UTCDateTime = Annotated[datetime, AfterValidator(_as_utc)]


def validate_landmarks(v):
    if len(v) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(v)}")
    for p in v:
        if len(p) not in (2, 3):
            raise ValueError("each landmark must be [x, y] or [x, y, z]")
        if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in p):
            raise ValueError("landmark coordinates must be finite numbers")
    return v


# ------------------------------------------------------------- recognition
class LandmarksIn(BaseModel):
    landmarks: list[list[float]] = Field(
        description="21 hand landmarks as [x, y] or [x, y, z], in pixel coordinates of the "
                    "unmirrored camera frame (MediaPipe's normalized x * width, y * height).")

    @field_validator("landmarks")
    @classmethod
    def _shape(cls, v):
        return validate_landmarks(v)


class Alternative(BaseModel):
    gesture: str
    confidence: float


class PredictionOut(BaseModel):
    gesture: str
    confidence: float
    alternatives: list[Alternative]
    latency_ms: float = Field(description="Server time to classify the frame, measured.")
    calibrated: bool = Field(description="Whether the user's calibration was applied.")


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_classes: int
    database: Literal["ok", "unavailable"]
    sentence_provider: Literal["ollama", "groq", "none"]


# -------------------------------------------------------------------- auth
class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class PasswordConfirm(BaseModel):
    password: str = Field(max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    created_at: UTCDateTime


# ----------------------------------------------------------------- gestures
class GestureOut(BaseModel):
    label: str
    f1_unseen_session: float | None = Field(
        description="F1 on a recording session the model never saw, from the evaluation "
                    "report. None if the report has no entry for this letter.")


# ----------------------------------------------------------------- sessions
class CommittedLetterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    gesture: str
    confidence: float
    latency_ms: float
    alternatives: list[Alternative]
    created_at: UTCDateTime


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    text: str
    sentence: str | None
    frame_count: int
    started_at: UTCDateTime
    ended_at: UTCDateTime | None
    model_version: str = Field(description="First 12 hex chars of the checkpoint's SHA-256.")


class SessionDetailOut(SessionOut):
    letters: list[CommittedLetterOut]


class SentenceOut(BaseModel):
    sentence: str


# -------------------------------------------------------------- calibration
class CalibrationIn(BaseModel):
    samples: dict[str, list[list[list[float]]]] = Field(
        description="Letter -> 3 to 30 frames of landmarks (same format as /recognition), "
                    "one entry for every letter the model knows.")

    @field_validator("samples")
    @classmethod
    def _frames(cls, v):
        for frames in v.values():
            if len(frames) > 30:
                raise ValueError("at most 30 frames per letter")
            for f in frames:
                validate_landmarks(f)
        return v


class CalibrationOut(BaseModel):
    calibrated: bool = Field(description="A calibration exists and matches the loaded model.")
    stale: bool = Field(description="A calibration exists but was made with a different model.")
    samples_per_label: int | None = None
    created_at: UTCDateTime | None = None
    labels: list[str] = Field(description="Letters a calibration must include.")


# ---------------------------------------------------------------- analytics
class DayCount(BaseModel):
    date: date_type
    letters: int


class LetterStats(BaseModel):
    gesture: str
    count: int
    avg_confidence: float = Field(description="Mean confidence at the frame the letter committed.")


class AnalyticsOut(BaseModel):
    days: int
    sessions: int
    letters: int
    frames: int
    avg_confidence: float | None
    avg_latency_ms: float | None = Field(description="Mean server time on commit frames.")
    per_day: list[DayCount] = Field(description="Every day in the window, oldest first, UTC.")
    per_letter: list[LetterStats] = Field(description="Most committed first.")

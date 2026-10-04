"""
Live recognition over a WebSocket. The browser runs MediaPipe and sends only landmarks --
never video. Protocol (JSON text frames):

  client -> {"type": "auth", "token": "<access token>", "session_id": 12}     first message
  server -> {"type": "ready", "session_id": 12, "text": "...", "calibrated": false}
  client -> {"type": "frame", "landmarks": [[x, y], ...21] | null, "client_ts": 123.4}
  server -> {"type": "result", "hand": true, "gesture": "A", "confidence": 0.94,
             "stability": 0.4, "committed": null, "alternatives": [...],
             "latency_ms": 3.1, "client_ts": 123.4, "text": "HEL"}
  client -> {"type": "backspace"}            server -> {"type": "text", "text": "HE"}
  server -> {"type": "error", "message": "..."}   on a bad message (connection stays open)

latency_ms is measured server time for the frame. client_ts is echoed so the browser can
measure the full round trip itself.
"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from backend.api.calibration import calibration_for
from backend.db import get_sessionmaker
from backend.deps import user_from_token
from backend.models import CommittedLetter, RecognitionSession
from backend.schemas import validate_landmarks
from backend.services.recognizer import get_recognizer
from gestureflow.core.session import RecognitionSession as Stream

router = APIRouter(tags=["websocket"])

AUTH_TIMEOUT_S = 5
MAX_MESSAGE_BYTES = 8 * 1024
FLUSH_FRAMES_EVERY = 100

CLOSE_UNAUTHORIZED = 4401
CLOSE_NOT_FOUND = 4404
CLOSE_BAD_REQUEST = 4400


def _authorize(token, session_id):
    """(close code or None, session text, the user's calibration or None)"""
    with get_sessionmaker()() as db:
        user = user_from_token(token, db) if isinstance(token, str) else None
        if user is None:
            return CLOSE_UNAUTHORIZED, None, None
        s = db.get(RecognitionSession, session_id) if isinstance(session_id, int) else None
        if s is None or s.user_id != user.id or s.ended_at is not None:
            return CLOSE_NOT_FOUND, None, None
        return None, s.text, calibration_for(db, user.id, get_recognizer())


def _commit_letter(session_id, result):
    with get_sessionmaker()() as db:
        s = db.get(RecognitionSession, session_id)
        db.add(CommittedLetter(
            session_id=session_id, gesture=result.committed, confidence=result.confidence,
            latency_ms=result.latency_ms,
            alternatives=[{"gesture": g, "confidence": c} for g, c in result.alternatives]))
        s.text += result.committed
        db.commit()
        return s.text


def _backspace(session_id):
    with get_sessionmaker()() as db:
        s = db.get(RecognitionSession, session_id)
        last = db.scalar(select(CommittedLetter)
                         .where(CommittedLetter.session_id == session_id)
                         .order_by(CommittedLetter.id.desc()).limit(1))
        if last is not None:
            db.delete(last)
            s.text = s.text[:-1]
            db.commit()
        return s.text


def _add_frames(session_id, n):
    with get_sessionmaker()() as db:
        s = db.get(RecognitionSession, session_id)
        s.frame_count += n
        db.commit()


@router.websocket("/ws/recognition")
async def recognition_ws(ws: WebSocket):
    await ws.accept()
    try:
        first = json.loads(await asyncio.wait_for(ws.receive_text(), AUTH_TIMEOUT_S))
    except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        await _close(ws, CLOSE_UNAUTHORIZED, "send an auth message first")
        return
    if not isinstance(first, dict) or first.get("type") != "auth":
        await _close(ws, CLOSE_UNAUTHORIZED, "send an auth message first")
        return

    session_id = first.get("session_id")
    code, text, calibration = await run_in_threadpool(_authorize, first.get("token"), session_id)
    if code is not None:
        await _close(ws, code, "unauthorized" if code == CLOSE_UNAUTHORIZED
                     else "session not found or already ended")
        return

    stream = Stream(get_recognizer().predictor, calibration=calibration)
    await ws.send_json({"type": "ready", "session_id": session_id, "text": text,
                        "calibrated": calibration is not None})

    pending_frames = 0
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_MESSAGE_BYTES:
                await _close(ws, CLOSE_BAD_REQUEST, "message too large")
                return
            try:
                msg = json.loads(raw)
                kind = msg.get("type")
            except (json.JSONDecodeError, AttributeError):
                await ws.send_json({"type": "error", "message": "invalid JSON"})
                continue

            if kind == "frame":
                landmarks = msg.get("landmarks")
                if landmarks is not None:
                    try:
                        validate_landmarks(landmarks)
                    except (ValueError, TypeError) as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        continue
                result = await run_in_threadpool(stream.step, landmarks)
                if result.committed:
                    text = await run_in_threadpool(_commit_letter, session_id, result)
                pending_frames += 1
                if pending_frames >= FLUSH_FRAMES_EVERY:
                    await run_in_threadpool(_add_frames, session_id, pending_frames)
                    pending_frames = 0
                await ws.send_json({
                    "type": "result", "hand": result.hand, "gesture": result.gesture,
                    "raw": result.raw, "confidence": result.confidence,
                    "stability": result.stability, "committed": result.committed,
                    "alternatives": [{"gesture": g, "confidence": c}
                                     for g, c in result.alternatives],
                    "latency_ms": round(result.latency_ms, 3),
                    "calibrated": result.calibrated,
                    "client_ts": msg.get("client_ts"), "text": text,
                })
            elif kind == "backspace":
                text = await run_in_threadpool(_backspace, session_id)
                await ws.send_json({"type": "text", "text": text})
            elif kind == "ping":
                await ws.send_json({"type": "pong", "client_ts": msg.get("client_ts")})
            else:
                await ws.send_json({"type": "error", "message": f"unknown type {kind!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        if pending_frames:
            await run_in_threadpool(_add_frames, session_id, pending_frames)


async def _close(ws: WebSocket, code: int, reason: str):
    try:
        await ws.close(code=code, reason=reason)
    except RuntimeError:
        pass  # already closed by the client

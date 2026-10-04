# GestureFlow API

FastAPI service in `backend/`, versioned under `/api/v1`. Interactive docs are generated
from the code at `http://localhost:8000/docs`.

The server never receives camera frames. The browser runs MediaPipe and sends 21 hand
landmarks per frame.

## Running it

```bash
uv run alembic upgrade head
```

```bash
uv run uvicorn backend.main:app --reload
```

By default the database is a SQLite file, `data/gestureflow.db`, so there is nothing to
install. Docker Compose uses PostgreSQL instead. The models and migrations are the same for
both, and the test suite runs the migrations on both. Settings are `GESTUREFLOW_*`
environment variables or a `.env` file; see [`.env.example`](../.env.example).

## Endpoints

| Method | Path | Login | Purpose |
| --- | --- | --- | --- |
| GET | `/health` | — | Model loaded, database reachable, which sentence provider is on |
| GET | `/` | — | Redirects to these docs |
| POST | `/auth/register` | — | Create an account |
| POST | `/auth/login` | — | Returns a JWT, valid 24 hours |
| GET | `/auth/me` | ✓ | Current user |
| DELETE | `/auth/me` | ✓ | Delete the account and all its data; body `{"password": "…"}` |
| POST | `/recognition/landmarks` | ✓ | Classify one frame of landmarks; nothing stored |
| GET | `/gestures` | ✓ | The model's letters, with per-letter F1 on an unseen session |
| POST | `/sessions` | ✓ | Start a recognition session |
| GET | `/sessions`, `/sessions/{id}` | owner | List, or one session with its committed letters |
| POST | `/sessions/{id}/end` | owner | Close the session |
| POST | `/sessions/{id}/sentence` | owner | Letters → sentence via Ollama or Groq (`GESTUREFLOW_SENTENCE_PROVIDER`); 20/hour per user; 503 if unavailable or turned off |
| GET | `/calibration` | ✓ | Whether you're calibrated (or stale after a model change) |
| POST | `/calibration` | ✓ | Save a calibration: 3–30 frames of landmarks for every letter |
| DELETE | `/calibration` | ✓ | Remove it |
| GET | `/analytics?days=7\|30\|90` | ✓ | Your sessions, letters, letters per day, letters by count |
| WS | `/ws/recognition` | ✓ | Live recognition stream |

To sign out, the client deletes its token. There is no server-side token list to revoke
from. That is the usual trade-off of plain JWTs, and it's why tokens expire after a day.

Landmarks are `[x, y]` or `[x, y, z]` in **pixel coordinates of the unmirrored camera frame**,
which is what the training data uses. From MediaPipe in the browser, that means
`x * videoWidth, y * videoHeight`. Mirroring the preview is fine, but the coordinates sent must
not be mirrored, because a mirrored hand is a different hand shape.

## WebSocket protocol

Browsers can't put an `Authorization` header on a WebSocket. Putting the token in the URL
would leak it into server logs, so it goes in the first message instead.

```text
→ {"type": "auth", "token": "<jwt>", "session_id": 12}
← {"type": "ready", "session_id": 12, "text": "", "calibrated": false}
→ {"type": "frame", "landmarks": [[x, y], ... 21] | null, "client_ts": 1234.5}
← {"type": "result", "hand": true, "gesture": "A", "raw": "A", "confidence": 0.94,
   "stability": 0.4, "committed": null, "alternatives": [{"gesture": "A", "confidence": 0.94}, ...],
   "latency_ms": 29.1, "calibrated": false, "client_ts": 1234.5, "text": "HEL"}
→ {"type": "backspace"}          ← {"type": "text", "text": "HE"}
```

- `gesture` is smoothed over recent frames, and `raw` is this frame's top prediction.
- `stability` is the progress of the current hold toward a commit, from 0 to 1. A letter commits after 15 confident frames in a row that agree. On that frame, `committed` holds the letter and it is saved straight away.
- `latency_ms` is the server's measured time for the frame. `client_ts` is echoed back so the client can measure the round trip.
- A malformed message gets `{"type": "error"}` back and the connection stays open.

Close codes:
- `4401`: not logged in
- `4404`: the session isn't yours or has ended
- `4400`: the message was over 8 KB

## Measured latency

From `python -m backend.bench_ws --frames 1000` against a local server: real landmarks from
`data/landmarks.csv`, CPU only, Intel Core with 32 threads. 980 frames were measured after a
20-frame warm-up.

| | p50 | p95 | p99 |
| --- | --- | --- | --- |
| Server time per frame (normalize + render + CNN + smoothing) | 29.1 ms | 33.0 ms | 35.3 ms |
| Full WebSocket round trip | 31.8 ms | 36.3 ms | 40.8 ms |

This excludes hand detection, which runs in the browser. On another machine, re-run the
benchmark rather than quoting these numbers.

## Calibration

Sign each letter a few times once, and recognition adapts to your hand without retraining.
Each frame becomes the CNN's 128-number feature vector, and the vectors for a letter are
averaged into a "prototype". Every prediction then blends the model's probabilities with how
close the frame is to each prototype (`gestureflow/core/calibration.py`). Only the
prototypes are stored, not the landmarks. Calibration applies to REST and WebSocket
recognition in Server mode. If the model changes, the calibration is marked `stale` and
ignored.

Measured by `python -m gestureflow.calibration_eval` (`docs/calibration_report.json`), with a
model trained without session `day2`, where `day2` plays the new user, using 5 frames per
letter:

| | Accuracy | Macro F1 | K F1 | Frames confident enough to commit | Wrong and confident enough |
| --- | --- | --- | --- | --- | --- |
| Without calibration | 92.7% | 0.916 | 0.14 | 25% | 0.2% |
| With calibration | **97.3%** | **0.971** | **0.96** | **79%** | **0%** |

Both commit columns are tracked because a letter only commits above 90% confidence. The first
version used a softer similarity temperature (0.1): it was just as accurate but never reached
90%, so calibrated users couldn't commit any letter. The temperature is now 0.03, and a
regression test on real recorded frames guards it.

The calibration and test frames come from the same sitting. That's how calibration is used,
but it's the favourable case.

## Logs

Every request writes one JSON line to the `gestureflow.access` logger:
`{"method": "GET", "path": "/api/v1/health", "status": 200, "ms": 3.2}`.

## Database

Four tables, created by Alembic migrations `0001` and `0002`:

| Table | Holds |
| --- | --- |
| `users` | Email and Argon2 password hash |
| `recognition_sessions` | One per live stream: text so far, sentence, frame count, which model |
| `committed_letters` | Each committed letter with confidence, top-3 and server time |
| `calibration_profiles` | One per user: per-letter prototypes, tied to a model version |

No table holds images, video or landmark coordinates, and only committed letters are saved,
not every frame.

## Security, kept simple

- **Passwords:** hashed with Argon2. A login with an unknown email takes as long as one with a wrong password and returns the same message, so the response doesn't reveal which accounts exist.
- **Tokens:** JWTs signed with a secret from the environment. In production the server refuses to start without one.
- **Login rate limit:** 10 attempts per minute per IP. The count is kept in memory, which is fine for one server process.
- **Validation:** every input goes through Pydantic, and WebSocket messages are capped at 8 KB.
- **Privacy:** each session can only be read by its owner. Another user's session id returns 404, not 403.
- **CORS:** an explicit list of allowed origins.

"""Shared FastAPI dependencies: the logged-in user, and rate limits."""

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.db import get_db
from backend.models import User
from backend.security import user_id_from_token

_bearer = HTTPBearer(auto_error=False)


def user_from_token(token: str, db: Session) -> User | None:
    user_id = user_id_from_token(token)
    return db.get(User, user_id) if user_id is not None else None


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
                 db: Session = Depends(get_db)) -> User:
    user = user_from_token(creds.credentials, db) if creds else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "please log in again",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


class SlidingWindowLimit:
    """At most `limit` hits per key in the last `window_s` seconds, counted in memory.

    Good enough for one server process. With several, the counts would need a shared
    store such as Redis.
    """

    def __init__(self, window_s: float, limit_setting: str, message: str):
        self.window_s = window_s
        self.limit_setting = limit_setting
        self.message = message
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def hit(self, key: str) -> None:
        limit = getattr(get_settings(), self.limit_setting)
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, self.message)
            q.append(now)

    def reset(self) -> None:
        with self.lock:
            self.hits.clear()


_login_window = SlidingWindowLimit(60, "login_attempts_per_minute",
                                   "too many attempts, try again in a minute")
sentence_limit = SlidingWindowLimit(3600, "sentences_per_hour",
                                    "sentence limit reached, try again later")


def login_rate_limit(request: Request) -> None:
    """Per IP, for register/login."""
    _login_window.hit(request.client.host if request.client else "unknown")


login_rate_limit.reset = _login_window.reset  # tests clear it between runs

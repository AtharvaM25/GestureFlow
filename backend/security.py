"""Password hashing (Argon2) and login tokens (JWT)."""

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from backend.config import get_settings

ALGORITHM = "HS256"
_hasher = PasswordHasher()

# Checked against when the email doesn't exist, so a login for an unknown account takes as
# long as a wrong password -- the response time doesn't reveal which emails are registered.
_DUMMY_HASH = _hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, InvalidHashError):
        return False


def create_token(user_id: int) -> tuple[str, int]:
    settings = get_settings()
    ttl = settings.token_hours * 3600
    now = datetime.now(UTC)
    token = jwt.encode({"sub": str(user_id), "iat": now, "exp": now + timedelta(seconds=ttl)},
                       settings.resolved_jwt_secret(), algorithm=ALGORITHM)
    return token, ttl


def user_id_from_token(token: str) -> int | None:
    try:
        claims = jwt.decode(token, get_settings().resolved_jwt_secret(), algorithms=[ALGORITHM],
                            options={"require": ["sub", "exp"]})
        return int(claims["sub"])
    except (jwt.PyJWTError, ValueError):
        return None

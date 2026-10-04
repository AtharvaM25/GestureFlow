"""Settings, read from environment variables prefixed GESTUREFLOW_ (or a .env file)."""

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gestureflow.paths import EVAL_REPORT, MODEL_PATH, REPO


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GESTUREFLOW_", env_file=".env", extra="ignore")

    env: str = "development"                 # "production" makes a JWT secret mandatory

    # SQLite needs no setup, so it's the default for running locally. Online, point this at
    # PostgreSQL (Neon, Supabase, Docker Compose); the code and migrations are the same.
    database_url: str = f"sqlite:///{(REPO / 'data' / 'gestureflow.db').as_posix()}"

    # Signs login tokens. Unset in development -> a random one per process, so you are
    # logged out whenever the server restarts. Set it to stay logged in.
    jwt_secret: str = ""
    token_hours: int = 24

    cors_origins: list[str] = ["http://localhost:3000"]
    login_attempts_per_minute: int = 10

    model_path: str = str(MODEL_PATH)
    eval_report_path: str = str(EVAL_REPORT)

    # Sentence generation. "ollama" runs the model on this machine (no key needed);
    # "groq" calls Groq's hosted API with your key; "none" turns the feature off.
    sentence_provider: Literal["ollama", "groq", "none"] = "ollama"
    sentences_per_hour: int = 20             # per user, so one person can't use up the quota
    ollama_model: str = "gemma3:4b"
    ollama_url: str | None = None            # None -> Ollama's default, http://localhost:11434
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""

    @field_validator("database_url")
    @classmethod
    def _psycopg_driver(cls, url: str) -> str:
        # Hosts hand out postgres:// or postgresql:// URLs; SQLAlchemy needs to be told to
        # use the psycopg (v3) driver this project installs.
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    def resolved_jwt_secret(self) -> str:
        if self.jwt_secret:
            return self.jwt_secret
        if self.env == "production":
            raise RuntimeError("GESTUREFLOW_JWT_SECRET must be set in production")
        return _dev_secret()


@lru_cache
def _dev_secret() -> str:
    return secrets.token_urlsafe(32)


@lru_cache
def get_settings() -> Settings:
    return Settings()

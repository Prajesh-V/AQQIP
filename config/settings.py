"""
AQQIP Configuration — Centralized Settings Loader

Reads from environment variables / .env file.
All configuration is typed via Pydantic for validation.
No secrets are hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


# Project root is 2 levels up from config/settings.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Look for .env in project root
_ENV_FILE: Path = PROJECT_ROOT / ".env"


class MySQLSettings(BaseSettings):
    """MySQL connection configuration."""

    host: str = Field(default="localhost", alias="MYSQL_HOST")
    port: int = Field(default=3306, alias="MYSQL_PORT")
    database: str = Field(default="aqqip", alias="MYSQL_DATABASE")
    user: str = Field(default="aqqip_user", alias="MYSQL_USER")
    password: str = Field(default="aqqip_pass", alias="MYSQL_PASSWORD")
    root_password: str = Field(default="aqqip_root_pass", alias="MYSQL_ROOT_PASSWORD")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


class MongoSettings(BaseSettings):
    """MongoDB connection configuration."""

    host: str = Field(default="localhost", alias="MONGO_HOST")
    port: int = Field(default=27017, alias="MONGO_PORT")
    database: str = Field(default="aqqip_raw", alias="MONGO_INITDB_DATABASE")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}

    @property
    def uri(self) -> str:
        return f"mongodb://{self.host}:{self.port}"


class ChromaSettings(BaseSettings):
    """ChromaDB local persistence configuration."""

    persist_dir: str = Field(default="./chroma_data", alias="CHROMA_PERSIST_DIR")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}

    @property
    def persist_path(self) -> Path:
        path = Path(self.persist_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


class MLflowSettings(BaseSettings):
    """MLflow tracking configuration — local SQLite backend."""

    tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")
    backend_store_uri: str = Field(
        default="sqlite:///mlruns/mlflow.db", alias="MLFLOW_BACKEND_STORE_URI"
    )

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


class LLMSettings(BaseSettings):
    """LLM provider configuration — Ollama (local) + Gemini (free tier)."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    ollama_host: str = Field(
        default="http://localhost:11434", alias="OLLAMA_HOST"
    )
    ollama_model: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL")
    ollama_fallback_model: str = Field(
        default="phi4-mini", alias="OLLAMA_FALLBACK_MODEL"
    )

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}

    @property
    def gemini_available(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def ollama_available(self) -> bool:
        """Best-effort check; actual availability requires a network call."""
        return bool(self.ollama_host)


class EdgarSettings(BaseSettings):
    """SEC EDGAR access configuration."""

    user_agent: str = Field(
        default="AQQIP research-bot contact@example.com",
        alias="EDGAR_USER_AGENT",
    )

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError(
                "EDGAR_USER_AGENT must be a compliant User-Agent string "
                "(e.g., 'AppName contact@email.com')"
            )
        return v


class EmbeddingSettings(BaseSettings):
    """Sentence Transformer embedding configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    batch_size: int = Field(default=32)
    device: str = Field(default="cpu")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


class PredictionTargetSettings(BaseSettings):
    """Frozen prediction target definition per evaluation_protocol.md §4."""

    horizon_trading_days: int = Field(
        default=20, alias="PREDICTION_HORIZON_TRADING_DAYS"
    )
    sharp_drop_threshold_pct: float = Field(
        default=10.0, alias="SHARP_DROP_THRESHOLD_PCT"
    )
    price_field: str = Field(
        default="adj_close", alias="PRICE_FIELD_FOR_OUTCOME"
    )

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


class MARLSettings(BaseSettings):
    """MARL environment defaults."""

    episode_length: int = Field(default=20, alias="MARL_EPISODE_LENGTH")
    default_seed: int = Field(default=42, alias="MARL_DEFAULT_SEED")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="./logs", alias="LOG_DIR")

    model_config = {"env_file": str(_ENV_FILE), "extra": "ignore"}

    @property
    def log_path(self) -> Path:
        path = Path(self.log_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path


class Settings:
    """
    Top-level settings aggregator.

    Usage:
        from config.settings import get_settings
        settings = get_settings()
        settings.mysql.host
        settings.llm.gemini_api_key
    """

    def __init__(self) -> None:
        self.mysql = MySQLSettings()
        self.mongo = MongoSettings()
        self.chroma = ChromaSettings()
        self.mlflow = MLflowSettings()
        self.llm = LLMSettings()
        self.edgar = EdgarSettings()
        self.embedding = EmbeddingSettings()
        self.prediction_target = PredictionTargetSettings()
        self.marl = MARLSettings()
        self.logging = LoggingSettings()


# Module-level singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

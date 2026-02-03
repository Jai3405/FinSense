"""
SPIKE API Configuration
Handles all environment variables and settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- Application -----
    APP_NAME: str = "SPIKE API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ----- API -----
    API_V1_PREFIX: str = "/api/v1"
    API_SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # ----- Database -----
    DATABASE_URL: PostgresDsn = "postgresql+asyncpg://user:password@localhost:5432/spike"  # type: ignore
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # ----- Redis -----
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"  # type: ignore
    REDIS_CACHE_TTL: int = 300  # 5 minutes

    # ----- Authentication (Clerk) -----
    CLERK_SECRET_KEY: str = ""
    CLERK_PUBLISHABLE_KEY: str = ""
    CLERK_JWT_ISSUER: str = "https://clerk.spike.ai"

    # ----- AI Services -----
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    # ----- Market Data -----
    TRUEDATA_USER: str = ""
    TRUEDATA_PASSWORD: str = ""

    # ----- Rate Limiting -----
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # ----- FinScore -----
    FINSCORE_CACHE_TTL: int = 3600  # 1 hour
    FINSCORE_UPDATE_INTERVAL: int = 300  # 5 minutes

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

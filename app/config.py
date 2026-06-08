from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="secure-rag-azure", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")

    api_key: str = Field(default="", alias="API_KEY")
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_MODEL")
    llm_provider: Literal["gemini", "groq"] = Field(default="gemini", alias="LLM_PROVIDER")
    llm_max_output_tokens: int = Field(default=512, alias="LLM_MAX_OUTPUT_TOKENS")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    vector_store_path: Path = Field(default=Path(".vectorstore"), alias="VECTOR_STORE_PATH")
    docs_path: Path = Field(default=Path("./data/pdfs"), alias="DOCS_PATH")
    max_upload_mb: int = Field(default=20, alias="MAX_UPLOAD_MB")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")
    similarity_threshold: float = Field(default=0.35, alias="SIMILARITY_THRESHOLD")
    max_prompt_chars: int = Field(default=2000, alias="MAX_PROMPT_CHARS")
    max_prompt_tokens_approx: int = Field(default=500, alias="MAX_PROMPT_TOKENS_APPROX")
    max_query_results_chars: int = Field(default=4000, alias="MAX_QUERY_RESULTS_CHARS")

    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_storage_url: str = Field(default="memory://", alias="RATE_LIMIT_STORAGE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")

    azure_key_vault_url: str | None = Field(default=None, alias="AZURE_KEY_VAULT_URL")
    azure_managed_identity_client_id: str | None = Field(default=None, alias="AZURE_MANAGED_IDENTITY_CLIENT_ID")
    azure_storage_account_url: str | None = Field(default=None, alias="AZURE_STORAGE_ACCOUNT_URL")
    azure_blob_container: str | None = Field(default=None, alias="AZURE_BLOB_CONTAINER")
    azure_monitor_connection_string: str | None = Field(default=None, alias="AZURE_MONITOR_CONNECTION_STRING")
    allow_local_key_fallback: bool = Field(default=False, alias="ALLOW_LOCAL_KEY_FALLBACK")

    @field_validator("api_key", "gemini_api_key", mode="before")
    @classmethod
    def normalize_secrets(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("docs_path", "vector_store_path", mode="before")
    @classmethod
    def expand_paths(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
        return value

    @field_validator("max_upload_mb", "chunk_size", "chunk_overlap", "top_k", "max_prompt_chars", "max_prompt_tokens_approx", "max_query_results_chars", "rate_limit_per_minute", "request_timeout_seconds")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("numeric settings must be positive")
        return value

    def vector_store_dir(self) -> Path:
        return self.vector_store_path

    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

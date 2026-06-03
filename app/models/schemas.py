from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            raise ValueError("question cannot be empty")
        return cleaned


class Citation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None
    score: float | None = None
    excerpt: str | None = None


class AskResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    status: Literal["answered", "refused", "blocked"]
    model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    error: str
    detail: str | None = None
    status: int


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float


class PromptAssessment(BaseModel):
    is_suspicious: bool
    reasons: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)


class AuthContext(BaseModel):
    principal: str
    api_key_id: str | None = None


class AppLogEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str | UUID | None = None
    event: str
    status: str | None = None
    reason: str | None = None
    token_usage: dict[str, int] | None = None
    client_ip: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

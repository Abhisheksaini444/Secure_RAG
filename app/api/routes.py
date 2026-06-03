from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.services.logging_service import logging_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(app_name="secure-rag-azure", version="1.0.0")


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    # The next major section will connect this endpoint to the retriever and LLM.
    # For now it emits an explicit secure refusal rather than fake answers.
    request_id = getattr(request.state, "request_id", None)
    logging_service.log_event(
        "ask_request_stub",
        status="blocked",
        reason="RAG pipeline not yet wired",
        request_id=request_id,
        client_ip=getattr(request.state, "client_ip", None),
    )
    return AskResponse(answer="I cannot answer from the provided documents.", status="refused")

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.rag.retriever import get_default_retriever
from app.services.logging_service import logging_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(app_name="secure-rag-azure", version="1.0.0")


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    request_id = getattr(request.state, "request_id", None)
    retriever = get_default_retriever()
    response = retriever.answer(payload.question, payload.top_k)
    logging_service.log_response(
        "ask_request",
        status=response.status,
        request_id=request_id,
        client_ip=getattr(request.state, "client_ip", None),
    )
    return response

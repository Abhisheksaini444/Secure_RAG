from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import AskRequest, AskResponse, HealthResponse
from app.rag.retriever import get_default_retriever
from app.services.logging_service import logging_service
from app.services.llm_provider import get_llm_provider, LLMProviderError, REFUSAL_TEXT

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(app_name="secure-rag-azure", version="1.0.0")


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    request_id = getattr(request.state, "request_id", None)
    retriever = get_default_retriever()
    results = retriever.retrieve(payload.question, payload.top_k)
    if not results or results[0].score < retriever.settings.similarity_threshold:
        response = AskResponse(answer=REFUSAL_TEXT, status="refused", citations=[], token_usage={})
        logging_service.log_response(
            "ask_request",
            status=response.status,
            request_id=request_id,
            client_ip=getattr(request.state, "client_ip", None),
            token_usage=response.token_usage,
        )
        return response

    logging_service.log_response(
        "ask_request",
        status="retrieved",
        request_id=request_id,
        client_ip=getattr(request.state, "client_ip", None),
    )

    try:
        provider = get_llm_provider()
        llm_out = provider.generate_answer(
            payload.question,
            results,
            timeout_seconds=retriever.settings.request_timeout_seconds,
            max_tokens=retriever.settings.llm_max_output_tokens,
            temperature=retriever.settings.llm_temperature,
            request_id=str(request_id) if request_id is not None else None,
        )
    except LLMProviderError as exc:
        logging_service.log_rejection("llm_unavailable", str(exc), request_id=request_id)
        return AskResponse(answer=REFUSAL_TEXT, status="refused", citations=[], token_usage={})

    citations = retriever.citation_builder.build(results)
    status = "refused" if llm_out.text == REFUSAL_TEXT else "answered"
    response = AskResponse(
        answer=llm_out.text,
        status=status,
        citations=citations if status == "answered" else [],
        token_usage=llm_out.token_usage,
        model=llm_out.model,
    )
    logging_service.log_response(
        "ask_request",
        status=response.status,
        request_id=request_id,
        client_ip=getattr(request.state, "client_ip", None),
        token_usage=response.token_usage,
    )
    return response

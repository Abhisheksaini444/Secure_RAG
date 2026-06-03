from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings
from app.models.schemas import AskResponse
from app.rag.citation_builder import DEFAULT_CITATION_BUILDER, CitationBuilder
from app.rag.ingest import ingest_pdf_corpus
from app.rag.vector_store import ChunkEmbeddingStore, build_embeddings


@dataclass(slots=True)
class RetrievalDecision:
    answer: str
    status: str
    confidence: float
    citations: list
    token_usage: dict[str, int]


class SecureRAGRetriever:
    """High-confidence retrieval gate with strict refusal below threshold."""

    def __init__(
        self,
        store: ChunkEmbeddingStore,
        settings: Settings | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.citation_builder = citation_builder or DEFAULT_CITATION_BUILDER

    def answer(self, question: str, top_k: int | None = None) -> AskResponse:
        resolved_top_k = top_k or self.settings.top_k
        results = self.store.search(question, resolved_top_k)
        if not results:
            return AskResponse(
                answer="I cannot answer from the provided documents.",
                status="refused",
                citations=[],
                token_usage={},
            )

        top_score = results[0].score
        if top_score < self.settings.similarity_threshold:
            return AskResponse(
                answer="I cannot answer from the provided documents.",
                status="refused",
                citations=[],
                token_usage={},
            )

        citations = self.citation_builder.build(results)
        answer = self._format_grounded_answer(results)
        return AskResponse(
            answer=answer,
            status="answered",
            citations=citations,
            token_usage={},
            model="retrieval-only",
        )

    def _format_grounded_answer(self, results) -> str:
        primary = results[0].chunk
        supporting = "\n".join(
            f"- {item.chunk.text[:350]}" for item in results[: self.settings.top_k]
        )
        return (
            "Based on the retrieved document context, the most relevant evidence is:\n"
            f"{supporting}\n\n"
            f"Primary source: {Path(primary.source).name} page {primary.page}."
        )


@lru_cache(maxsize=1)
def get_default_retriever() -> SecureRAGRetriever:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    store_path = settings.vector_store_dir()
    try:
        store = ChunkEmbeddingStore.load(store_path, embeddings)
    except FileNotFoundError:
        from app.rag.ingest import ingest_pdf_corpus

        ingest_pdf_corpus(settings.docs_path, settings)
        store = ChunkEmbeddingStore.load(store_path, embeddings)
    return SecureRAGRetriever(store=store, settings=settings)

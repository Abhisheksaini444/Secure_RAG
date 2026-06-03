from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.models.schemas import DocumentChunk
from app.rag.citation_builder import CitationBuilder
from app.rag.retriever import SecureRAGRetriever
from app.rag.vector_store import ChunkEmbeddingStore


@dataclass
class ScopeEmbeddings:
    chunk_vector: np.ndarray
    query_vector: np.ndarray

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.chunk_vector for _ in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.query_vector, dtype=np.float32)


def test_out_of_scope_query_is_refused(tmp_path):
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source="/documents/owasp.pdf",
        page=2,
        text="This page discusses prompt injection defenses.",
        metadata={"source_name": "owasp.pdf", "page": 2, "chunk_index": 0},
    )
    embeddings = ScopeEmbeddings(
        chunk_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        query_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    store = ChunkEmbeddingStore.build([chunk], embeddings, tmp_path)
    settings = Settings.model_validate(
        {
            "API_KEY": "test-key",
            "DOCS_PATH": str(tmp_path),
            "VECTOR_STORE_PATH": str(tmp_path),
            "SIMILARITY_THRESHOLD": 0.9,
            "TOP_K": 4,
        }
    )
    retriever = SecureRAGRetriever(store=store, settings=settings, citation_builder=CitationBuilder())

    response = retriever.answer("What are the company payroll details?")

    assert response.status == "refused"
    assert response.answer == "I cannot answer from the provided documents."
    assert response.citations == []

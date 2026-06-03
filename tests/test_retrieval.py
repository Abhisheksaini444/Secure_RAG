from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.models.schemas import DocumentChunk
from app.rag.citation_builder import CitationBuilder
from app.rag.retriever import SecureRAGRetriever
from app.rag.vector_store import ChunkEmbeddingStore


@dataclass
class FixedEmbeddings:
    vectors: dict[str, np.ndarray]
    query_vector: np.ndarray

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.query_vector, dtype=np.float32)


def build_store(tmp_path, chunk_text: str, query_vector: np.ndarray, chunk_vector: np.ndarray) -> SecureRAGRetriever:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source="/documents/corpus.pdf",
        page=1,
        text=chunk_text,
        metadata={"source_name": "corpus.pdf", "page": 1, "chunk_index": 0},
    )
    embeddings = FixedEmbeddings(vectors={chunk_text: chunk_vector}, query_vector=query_vector)
    store = ChunkEmbeddingStore.build([chunk], embeddings, tmp_path)
    settings = Settings.model_validate(
        {
            "API_KEY": "test-key",
            "GEMINI_API_KEY": "",
            "DOCS_PATH": str(tmp_path),
            "VECTOR_STORE_PATH": str(tmp_path),
            "SIMILARITY_THRESHOLD": 0.35,
            "TOP_K": 4,
        }
    )
    return SecureRAGRetriever(store=store, settings=settings, citation_builder=CitationBuilder())


def test_retrieval_returns_citations(tmp_path):
    retriever = build_store(
        tmp_path,
        chunk_text="The OWASP document recommends restricting prompt instructions and hiding internal state.",
        query_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        chunk_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    response = retriever.answer("How should hidden state be protected?")

    assert response.status == "answered"
    assert response.answer.startswith("Based on the retrieved document context")
    assert response.citations
    assert response.citations[0].source == "corpus.pdf"
    assert response.citations[0].page == 1
    assert response.citations[0].chunk_id == "chunk-1"


def test_retrieval_below_threshold_refuses(tmp_path):
    retriever = build_store(
        tmp_path,
        chunk_text="This chunk is unrelated to the question.",
        query_vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        chunk_vector=np.array([0.1, 0.0, 0.0], dtype=np.float32),
    )

    response = retriever.answer("What is the secret deployment key?")

    assert response.status == "refused"
    assert response.answer == "I cannot answer from the provided documents."
    assert response.citations == []

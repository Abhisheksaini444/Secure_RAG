from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - exercised in constrained test environments
    faiss = None

from app.config import Settings, get_settings
from app.models.schemas import DocumentChunk, RetrievalResult


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbeddings:
    """Sentence-transformers embeddings with cosine-friendly normalization.

    We normalize vectors before indexing so retrieval scores are cosine
    similarities rather than raw inner products. This makes the similarity
    threshold easier to reason about and more stable across documents.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device or "cpu")
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


class _NumpyInnerProductIndex:
    """Lightweight inner-product index used when FAISS is unavailable.

    This keeps the retrieval tests runnable in constrained environments while
    preserving the same cosine-similarity semantics as the FAISS index.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match index dimension")
        self._vectors = np.concatenate([self._vectors, np.asarray(vectors, dtype=np.float32)], axis=0)

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors.size == 0:
            empty_scores = np.empty((1, 0), dtype=np.float32)
            empty_indices = np.empty((1, 0), dtype=np.int64)
            return empty_scores, empty_indices

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError("Query dimension does not match index dimension")

        scores = self._vectors @ query[0]
        order = np.argsort(scores)[::-1][:top_k]
        return scores[order].reshape(1, -1), order.astype(np.int64).reshape(1, -1)


def _create_index(dimension: int):
    if faiss is not None:
        return faiss.IndexFlatIP(dimension)
    return _NumpyInnerProductIndex(dimension)


@dataclass(slots=True)
class ChunkEmbeddingStore:
    """Minimal FAISS-backed chunk store with metadata sidecar persistence."""

    index: faiss.Index
    chunks: list[DocumentChunk]
    embeddings: EmbeddingBackend
    store_path: Path

    INDEX_FILENAME = "index.faiss"
    METADATA_FILENAME = "chunks.json"
    VECTORS_FILENAME = "vectors.npy"

    @classmethod
    def build(
        cls,
        chunks: Iterable[DocumentChunk],
        embeddings: EmbeddingBackend,
        store_path: Path,
    ) -> "ChunkEmbeddingStore":
        chunk_list = list(chunks)
        index = _create_index(embeddings.dimension)
        store = cls(index=index, chunks=[], embeddings=embeddings, store_path=store_path)
        if chunk_list:
            store.add_chunks(chunk_list)
        return store

    @classmethod
    def load(cls, store_path: Path, embeddings: EmbeddingBackend) -> "ChunkEmbeddingStore":
        index_path = store_path / cls.INDEX_FILENAME
        metadata_path = store_path / cls.METADATA_FILENAME
        vectors_path = store_path / cls.VECTORS_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError("Vector store has not been created yet")

        if index_path.exists() and faiss is not None:
            index = faiss.read_index(str(index_path))
        elif vectors_path.exists():
            index = _NumpyInnerProductIndex(embeddings.dimension)
            vectors = np.load(vectors_path)
            if vectors.size:
                index.add(np.asarray(vectors, dtype=np.float32))
        else:
            raise FileNotFoundError("Vector store has not been created yet")

        chunks = [DocumentChunk.model_validate(item) for item in json.loads(metadata_path.read_text(encoding="utf-8"))]
        return cls(index=index, chunks=chunks, embeddings=embeddings, store_path=store_path)

    def add_chunks(self, chunks: Iterable[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunk_list])
        if vectors.shape[1] != self.embeddings.dimension:
            raise ValueError("Embedding dimension does not match FAISS index dimension")
        self.index.add(np.asarray(vectors, dtype=np.float32))
        self.chunks.extend(chunk_list)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.index.ntotal == 0:
            return []
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype=np.float32)
        scores, indices = self.index.search(query_vector, top_k)
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        if faiss is not None and hasattr(faiss, "write_index") and not isinstance(self.index, _NumpyInnerProductIndex):
            faiss.write_index(self.index, str(self.store_path / self.INDEX_FILENAME))
        else:
            np.save(self.store_path / self.VECTORS_FILENAME, getattr(self.index, "_vectors", np.empty((0, self.embeddings.dimension), dtype=np.float32)))
        payload = [chunk.model_dump(mode="json") for chunk in self.chunks]
        (self.store_path / self.METADATA_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def best_score(self, query: str, top_k: int) -> float:
        results = self.search(query, top_k)
        return results[0].score if results else 0.0


def build_embeddings(settings: Settings | None = None) -> SentenceTransformerEmbeddings:
    resolved = settings or get_settings()
    return SentenceTransformerEmbeddings(resolved.embedding_model)

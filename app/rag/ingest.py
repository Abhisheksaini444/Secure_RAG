from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import Settings, get_settings
from app.models.schemas import DocumentChunk
from app.rag.vector_store import ChunkEmbeddingStore, build_embeddings


@dataclass(slots=True)
class IngestionResult:
    documents: int
    pages: int
    chunks: int
    store_path: Path


class PDFIngestor:
    """Ingest PDFs into chunked, metadata-rich embeddings.

    Security decision: pages are split before embedding so every chunk keeps
    page-level provenance. That makes citations auditable and limits the amount
    of text surfaced from any single retrieval hit.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_corpus(self, pdf_paths: Iterable[Path]) -> tuple[list[DocumentChunk], IngestionResult]:
        import fitz

        chunks: list[DocumentChunk] = []
        documents = 0
        pages = 0
        for pdf_path in pdf_paths:
            documents += 1
            chunks.extend(self.ingest_pdf(pdf_path))
            with fitz.open(pdf_path) as document:
                pages += document.page_count
        result = IngestionResult(documents=documents, pages=pages, chunks=len(chunks), store_path=self.settings.vector_store_dir())
        return chunks, result

    def ingest_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        import fitz

        pdf_path = pdf_path.expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        extracted_chunks: list[DocumentChunk] = []
        with fitz.open(pdf_path) as document:
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                page_text = self._normalize_text(page.get_text("text"))
                if not page_text:
                    continue
                split_texts = self.text_splitter.split_text(page_text)
                for chunk_index, chunk_text in enumerate(split_texts):
                    extracted_chunks.append(self._build_chunk(pdf_path, page_number + 1, chunk_index, chunk_text))
        return extracted_chunks

    def _build_chunk(self, pdf_path: Path, page_number: int, chunk_index: int, text: str) -> DocumentChunk:
        normalized_text = self._normalize_text(text)
        chunk_id = self._chunk_id(pdf_path, page_number, chunk_index, normalized_text)
        return DocumentChunk(
            chunk_id=chunk_id,
            source=str(pdf_path),
            page=page_number,
            text=normalized_text,
            metadata={
                "source_name": pdf_path.name,
                "page": page_number,
                "chunk_index": chunk_index,
                "sha256": self._file_sha256(pdf_path),
            },
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()

    @staticmethod
    def _file_sha256(pdf_path: Path) -> str:
        digest = hashlib.sha256()
        with pdf_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8192), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _chunk_id(pdf_path: Path, page_number: int, chunk_index: int, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(str(pdf_path).encode("utf-8"))
        digest.update(str(page_number).encode("utf-8"))
        digest.update(str(chunk_index).encode("utf-8"))
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()


def ingest_pdf_corpus(documents_path: Path | None = None, settings: Settings | None = None) -> IngestionResult:
    resolved_settings = settings or get_settings()
    corpus_path = documents_path or resolved_settings.docs_path
    pdf_paths = sorted(path for path in corpus_path.glob("*.pdf") if path.is_file())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF documents found in {corpus_path}")

    ingestor = PDFIngestor(resolved_settings)
    chunks, result = ingestor.ingest_corpus(pdf_paths)
    embeddings = build_embeddings(resolved_settings)
    store = ChunkEmbeddingStore.build(chunks, embeddings, resolved_settings.vector_store_dir())
    store.save()
    return result

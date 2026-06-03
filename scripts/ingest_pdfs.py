from __future__ import annotations

from app.rag.ingest import ingest_pdf_corpus


if __name__ == "__main__":
    result = ingest_pdf_corpus()
    print(
        f"Ingested {result.documents} documents, {result.pages} pages, {result.chunks} chunks into {result.store_path}"
    )

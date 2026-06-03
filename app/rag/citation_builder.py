from __future__ import annotations

from pathlib import Path

from app.models.schemas import Citation, RetrievalResult


class CitationBuilder:
    """Builds citations from retrieved chunks without exposing full documents."""

    def build(self, results: list[RetrievalResult], max_excerpt_chars: int = 280) -> list[Citation]:
        citations: list[Citation] = []
        for result in results:
            chunk = result.chunk
            excerpt = " ".join(chunk.text.split())[:max_excerpt_chars]
            citations.append(
                Citation(
                    source=self._display_source(chunk.source),
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    score=round(result.score, 4),
                    excerpt=excerpt,
                )
            )
        return citations

    @staticmethod
    def _display_source(source: str) -> str:
        return Path(source).name if source else source


DEFAULT_CITATION_BUILDER = CitationBuilder()

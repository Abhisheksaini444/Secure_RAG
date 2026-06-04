import asyncio
from types import SimpleNamespace

from app.models.schemas import AskRequest


class DummyRequest:
    def __init__(self):
        self.state = SimpleNamespace()
        self.state.request_id = None
        self.state.client_ip = "127.0.0.1"


def test_ask_route_invokes_llm(monkeypatch):
    # Stub retriever to return an answered response
    class StubRetriever:
        def __init__(self):
            self.settings = SimpleNamespace(similarity_threshold=0.35, request_timeout_seconds=5, llm_max_output_tokens=128, llm_temperature=0.0)
            self.citation_builder = SimpleNamespace(build=lambda results: [])

        def retrieve(self, question, top_k):
            from app.models.schemas import DocumentChunk, RetrievalResult

            chunk = DocumentChunk(
                chunk_id="chunk-1",
                source="/documents/corpus.pdf",
                page=1,
                text="The document says Gemini should answer from retrieved evidence only.",
                metadata={"source_name": "corpus.pdf", "page": 1},
            )
            return [RetrievalResult(chunk=chunk, score=0.99)]

    monkeypatch.setattr("app.api.routes.get_default_retriever", lambda: StubRetriever())

    # Stub LLM provider to return safe text
    class StubProvider:
        def generate_answer(self, question, retrieval_results, timeout_seconds=None, max_tokens=None, temperature=None, request_id=None):
            from app.services.llm_provider import LLMAnswer

            return LLMAnswer(text="safe answer", token_usage={"total_token_count": 3}, model="gemini-test")

    monkeypatch.setattr("app.api.routes.get_llm_provider", lambda: StubProvider())

    from app.api import routes

    req = DummyRequest()
    payload = AskRequest(question="What is the title of the document?")

    resp = asyncio.run(routes.ask(req, payload))
    assert resp.status == "answered"
    assert resp.answer == "safe answer"
    assert resp.model == "gemini-test"

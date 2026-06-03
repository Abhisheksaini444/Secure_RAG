"""Retrieval and ingestion primitives for secure RAG.

The package intentionally avoids eager imports of heavyweight ML dependencies so
tests and lightweight tooling can import the package without downloading the
full embedding stack.
"""

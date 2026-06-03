from __future__ import annotations

"""Compatibility shim for future LLM-backed answering.

The retriever is intentionally usable on its own so the service can refuse
unsupported questions before a model call is ever attempted.
"""

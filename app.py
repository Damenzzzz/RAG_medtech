"""Vercel entrypoint for the bundled FastAPI RAG service."""

from basic_rag.basic_rag_server import app


__all__ = ["app"]

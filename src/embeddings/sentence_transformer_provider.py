"""Adapt a SentenceTransformer model to the embedding provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerProvider:
    """Generate embeddings using an existing SentenceTransformer model."""

    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Embed document texts in input order."""
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str, batch_size: int) -> list[float]:
        """Embed a search query."""
        return self.model.encode(
            text, batch_size=batch_size, show_progress_bar=False
        ).tolist()

"""Interface for generating document and query embeddings."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Provide vectors for document texts and search queries."""

    def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Return one vector per text, in input order."""
        ...

    def embed_query(self, text: str, batch_size: int) -> list[float]:
        """Return a vector for a search query."""
        ...

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


def get_embedding_dimension(provider: EmbeddingProvider, batch_size: int) -> int:
    """
    Determine vector dimensions by embedding a short probe.

    The provider may contact a service or load a model when called. Equal
    dimensions do not imply that different models produce compatible vectors.

    Raises:
        ValueError: If batch_size is not positive or the vector is empty.
            Provider errors propagate.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    vector = provider.embed_query("Embedding dimension probe", batch_size)
    if not vector:
        raise ValueError("Embedding provider returned an empty vector")
    return len(vector)

"""Generate document and query embeddings through an Ollama server."""

import math

if "." in __package__:
    from ..providers.http_helper import post_json
else:
    from providers.http_helper import post_json


class OllamaEmbeddingProvider:
    """Implement the embedding provider interface using Ollama's embed API."""

    def __init__(
        self, model: str, base_url: str = "http://localhost:11434", timeout: float = 120
    ) -> None:
        base_url = base_url.rstrip("/")
        if not model.strip() or not base_url.strip():
            raise ValueError("Ollama model and base URL must not be empty")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Return document vectors in input order, using bounded requests."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        embeddings: list[list[float]] = []
        dimension: int | None = None
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = post_json(
                f"{self.base_url}/api/embed",
                {"model": self.model, "input": batch, "truncate": False},
                headers={},
                timeout=self.timeout,
            )
            vectors = response.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise ValueError("Ollama response must contain one embedding per input")
            for vector in vectors:
                if not isinstance(vector, list) or not vector:
                    raise ValueError("Ollama embeddings must be non-empty vectors")
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise ValueError("Ollama embeddings must have consistent dimensions")
                if not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError("Ollama embeddings must contain finite numbers")
                embeddings.append([float(value) for value in vector])
        return embeddings

    def embed_query(self, text: str, batch_size: int) -> list[float]:
        """Return a query vector using the same model as document embeddings."""
        return self.embed_documents([text], batch_size)[0]

"""Tests for the embedding provider interface."""

import unittest

from src.embeddings.embedding_provider import EmbeddingProvider


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str, batch_size: int) -> list[float]:
        return [float(len(text))]


class EmbeddingProviderTests(unittest.TestCase):
    def test_provider_can_supply_document_and_query_vectors(self) -> None:
        provider: EmbeddingProvider = FakeEmbeddingProvider()

        self.assertEqual(provider.embed_documents(["a", "abc"], 2), [[1.0], [3.0]])
        self.assertEqual(provider.embed_query("abc", 2), [3.0])


if __name__ == "__main__":
    unittest.main()

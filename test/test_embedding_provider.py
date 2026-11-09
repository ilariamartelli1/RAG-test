"""Tests for the embedding provider interface."""

import unittest
from unittest.mock import MagicMock, patch

from src.embeddings.embedding_provider import EmbeddingProvider, get_embedding_dimension
from src.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider


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

    def test_detects_dimensions_and_passes_batch_size(self) -> None:
        provider = MagicMock()
        for dimension in (1, 3, 768):
            with self.subTest(dimension=dimension):
                provider.embed_query.return_value = [0.0] * dimension
                self.assertEqual(get_embedding_dimension(provider, 8), dimension)
                provider.embed_query.assert_called_once_with("Embedding dimension probe", 8)
                provider.reset_mock()

    def test_rejects_empty_vector_and_invalid_batch_size(self) -> None:
        provider = MagicMock()
        provider.embed_query.return_value = []
        with self.assertRaisesRegex(ValueError, "empty vector"):
            get_embedding_dimension(provider, 1)
        provider.reset_mock()
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "batch_size"):
                get_embedding_dimension(provider, size)
        provider.embed_query.assert_not_called()

    def test_provider_errors_propagate(self) -> None:
        provider = MagicMock()
        provider.embed_query.side_effect = RuntimeError("model unavailable")
        with self.assertRaisesRegex(RuntimeError, "model unavailable"):
            get_embedding_dimension(provider, 1)

    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_detects_ollama_dimensions(self, post_json) -> None:
        post_json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        self.assertEqual(get_embedding_dimension(OllamaEmbeddingProvider("model"), 4), 3)
        post_json.assert_called_once_with(
            "http://localhost:11434/api/embed",
            {"model": "model", "input": ["Embedding dimension probe"], "truncate": False},
            headers={}, timeout=120,
        )

    def test_detects_sentence_transformer_dimensions(self) -> None:
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1, 0.2]
        self.assertEqual(get_embedding_dimension(SentenceTransformerProvider(model), 4), 2)
        model.encode.assert_called_once_with(
            "Embedding dimension probe", batch_size=4, show_progress_bar=False,
        )


if __name__ == "__main__":
    unittest.main()

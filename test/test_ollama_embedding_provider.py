"""Tests for the Ollama embedding provider."""

import unittest
from unittest.mock import call, patch
from urllib.error import URLError

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.embeddings.embedding_provider import EmbeddingProvider
from src.embeddings.ollama_embedding_provider import OllamaEmbeddingProvider
from src.models.chunk_models import ChunkPayload
from src.storage.chunk_service import retrieve_chunks, store_chunks


class OllamaEmbeddingProviderTests(unittest.TestCase):
    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_embeds_documents_in_batches_and_preserves_order(self, post_json) -> None:
        post_json.side_effect = [
            {"embeddings": [[1, 0], [0, 1]]},
            {"embeddings": [[0.5, 0.5]]},
        ]
        provider: EmbeddingProvider = OllamaEmbeddingProvider("embed-model", "http://localhost:11434/", 30)

        self.assertEqual(
            provider.embed_documents(["first", "second", "third"], 2),
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        )
        self.assertEqual(post_json.call_args_list, [
            call(
                "http://localhost:11434/api/embed",
                {"model": "embed-model", "input": ["first", "second"], "truncate": False},
                headers={}, timeout=30,
            ),
            call(
                "http://localhost:11434/api/embed",
                {"model": "embed-model", "input": ["third"], "truncate": False},
                headers={}, timeout=30,
            ),
        ])

    @patch("src.embeddings.ollama_embedding_provider.post_json", return_value={"embeddings": [[0.2, 0.4]]})
    def test_embeds_query(self, post_json) -> None:
        provider = OllamaEmbeddingProvider("embed-model")

        self.assertEqual(provider.embed_query("question", 8), [0.2, 0.4])
        post_json.assert_called_once_with(
            "http://localhost:11434/api/embed",
            {"model": "embed-model", "input": ["question"], "truncate": False},
            headers={}, timeout=120,
        )

    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_empty_input_and_invalid_batch_size_do_not_call_server(self, post_json) -> None:
        provider = OllamaEmbeddingProvider("embed-model")
        self.assertEqual(provider.embed_documents([], 2), [])

        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "batch_size"):
                provider.embed_documents([], size)
            with self.subTest(query_size=size), self.assertRaisesRegex(ValueError, "batch_size"):
                provider.embed_query("query", size)
        post_json.assert_not_called()

    def test_rejects_invalid_settings(self) -> None:
        for kwargs in (
            {"model": ""}, {"model": " \t"}, {"model": "m", "base_url": "/"},
            {"model": "m", "timeout": 0}, {"model": "m", "timeout": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaEmbeddingProvider(**kwargs)

    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_rejects_malformed_embeddings(self, post_json) -> None:
        responses = (
            {}, {"embeddings": "invalid"}, {"embeddings": []}, {"embeddings": [[]]},
            {"embeddings": [[True, 1]]}, {"embeddings": [["1", 2]]},
            {"embeddings": [[float("nan"), 1]]}, {"embeddings": [[float("inf"), 1]]},
        )
        provider = OllamaEmbeddingProvider("embed-model")
        for response in responses:
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "Ollama"):
                post_json.return_value = response
                provider.embed_documents(["text"], 1)

    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_rejects_inconsistent_dimensions_across_batches(self, post_json) -> None:
        post_json.side_effect = [{"embeddings": [[1.0, 0.0]]}, {"embeddings": [[1.0]]}]

        with self.assertRaisesRegex(ValueError, "consistent dimensions"):
            OllamaEmbeddingProvider("embed-model").embed_documents(["first", "second"], 1)

    @patch("src.embeddings.ollama_embedding_provider.post_json", side_effect=URLError("unavailable"))
    def test_server_errors_propagate(self, post_json) -> None:
        with self.assertRaises(URLError):
            OllamaEmbeddingProvider("embed-model").embed_query("question", 1)

    @patch("src.embeddings.ollama_embedding_provider.post_json")
    def test_stores_and_retrieves_with_qdrant(self, post_json) -> None:
        post_json.side_effect = [
            {"embeddings": [[1.0, 0.0]]},
            {"embeddings": [[1.0, 0.0]]},
        ]
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        qdrant.create_collection("documents", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
        provider = OllamaEmbeddingProvider("embed-model")
        payload = ChunkPayload(text="document text", metadata={"source": "notes.txt"})

        self.assertEqual(store_chunks(qdrant, provider, "documents", [payload], 8), 1)
        results = retrieve_chunks(qdrant, provider, "documents", "question", 8)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], payload)
        self.assertAlmostEqual(results[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()

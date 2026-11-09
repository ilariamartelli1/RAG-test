"""Tests for the SentenceTransformer embedding provider adapter."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.embeddings.embedding_provider import EmbeddingProvider
from src.embeddings.sentence_transformer_provider import SentenceTransformerProvider


class SentenceTransformerProviderTests(unittest.TestCase):
    def test_embeds_documents_and_query_as_lists(self) -> None:
        model = MagicMock()
        model.encode.side_effect = [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 0.0]),
        ]
        provider: EmbeddingProvider = SentenceTransformerProvider(model)

        self.assertEqual(provider.embed_documents(["first", "second"], 8), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(provider.embed_query("question", 8), [1.0, 0.0])
        self.assertEqual(model.encode.call_count, 2)
        model.encode.assert_any_call(["first", "second"], batch_size=8, show_progress_bar=False)
        model.encode.assert_any_call("question", batch_size=8, show_progress_bar=False)

    def test_model_errors_propagate(self) -> None:
        model = MagicMock()
        model.encode.side_effect = RuntimeError("embedding failed")
        provider = SentenceTransformerProvider(model)

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            provider.embed_documents(["text"], 1)
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            provider.embed_query("text", 1)


if __name__ == "__main__":
    unittest.main()

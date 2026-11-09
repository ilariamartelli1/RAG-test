"""Tests for Qdrant chunk storage and retrieval."""

import unittest
from unittest.mock import MagicMock

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.storage.chunk_service import retrieve_chunks, store_chunks
from src.chunkers.dataset_chunker import chunk_dataset_row
from src.models.chunk_models import ChunkPayload
from src.rag.prompt_utils import build_rag_prompt


class ChunkServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qdrant = QdrantClient(":memory:")
        self.addCleanup(self.qdrant.close)
        self.qdrant.create_collection(
            collection_name="test_chunks",
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        self.embedder = MagicMock()

    def test_dataset_chunk_can_be_stored_retrieved_and_used_in_prompt(self) -> None:
        chunk = chunk_dataset_row(pd.Series({"subject": "Astronomy"}), "row-1")[0]
        payload = ChunkPayload(
            text=chunk.text,
            metadata={
                "source": "dataset",
                "row_id": chunk.row_id,
                "chunk_index": chunk.chunk_index,
                "field_name": chunk.field_name,
            },
        )
        self.embedder.embed_documents.return_value = [[1.0, 0.0]]
        self.embedder.embed_query.return_value = [1.0, 0.0]

        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload], 8),
            1,
        )
        results = retrieve_chunks(self.qdrant, self.embedder, "test_chunks", "Query", 8)

        self.embedder.embed_documents.assert_called_once_with([payload.text], 8)
        self.embedder.embed_query.assert_called_once_with("Query", 8)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], payload)
        self.assertAlmostEqual(results[0][1], 1.0)
        prompt = build_rag_prompt("Query", [result.text for result, _ in results])
        self.assertIn("Record: row-1\nsubject: Astronomy", prompt[0]["content"])

    def test_accepts_embedding_provider_without_sentence_transformer(self) -> None:
        class FakeProvider:
            def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text: str, batch_size: int) -> list[float]:
                return [1.0, 0.0]

        payload = ChunkPayload(text="Local document", metadata={"source": "notes.txt"})
        provider = FakeProvider()

        self.assertEqual(store_chunks(self.qdrant, provider, "test_chunks", [payload], 8), 1)
        results = retrieve_chunks(self.qdrant, provider, "test_chunks", "Query", 8)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], payload)

    def test_embedding_errors_propagate(self) -> None:
        self.embedder.embed_documents.side_effect = RuntimeError("document embedding failed")
        self.embedder.embed_query.side_effect = RuntimeError("query embedding failed")

        with self.assertRaisesRegex(RuntimeError, "document embedding failed"):
            store_chunks(
                self.qdrant, self.embedder, "test_chunks",
                [ChunkPayload(text="text", metadata={})], 8,
            )
        with self.assertRaisesRegex(RuntimeError, "query embedding failed"):
            retrieve_chunks(self.qdrant, self.embedder, "test_chunks", "Query", 8)

        self.assertEqual(self.qdrant.count("test_chunks").count, 0)

    def test_repeated_chunks_are_not_embedded_again(self) -> None:
        payload = ChunkPayload(text="Example", metadata={"row_id": "1"})
        self.embedder.embed_documents.return_value = [[1.0, 0.0]]

        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload, payload], 8),
            1,
        )
        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload], 8),
            0,
        )
        self.embedder.embed_documents.assert_called_once()

    def test_empty_batch_does_not_call_embedder(self) -> None:
        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", [], 8), 0)
        self.embedder.embed_documents.assert_not_called()

    def test_qdrant_errors_are_not_ignored(self) -> None:
        payload = ChunkPayload(text="Example", metadata={})

        with self.assertRaises(ValueError):
            store_chunks(self.qdrant, self.embedder, "missing_collection", [payload], 8)
        self.embedder.embed_documents.assert_not_called()

    def test_same_text_from_different_rows_keeps_both_sources(self) -> None:
        chunks = [
            ChunkPayload(text="Same text", metadata={"row_id": "1"}),
            ChunkPayload(text="Same text", metadata={"row_id": "2"}),
        ]
        self.embedder.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]

        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", chunks, 8), 2)
        self.assertEqual(self.qdrant.count("test_chunks").count, 2)

    def test_retrieval_uses_default_and_custom_score_thresholds(self) -> None:
        chunks = [
            ChunkPayload(text="Matching", metadata={}),
            ChunkPayload(text="Above threshold", metadata={}),
            ChunkPayload(text="Below threshold", metadata={}),
        ]
        self.embedder.embed_documents.return_value = [
            [1.0, 0.0], [0.3, 0.9539392], [0.1, 0.9949874],
        ]
        self.embedder.embed_query.side_effect = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.8],
        ]
        store_chunks(self.qdrant, self.embedder, "test_chunks", chunks, 8)

        default_results = retrieve_chunks(self.qdrant, self.embedder, "test_chunks", "Query", 8)
        all_results = retrieve_chunks(
            self.qdrant, self.embedder, "test_chunks", "Query", 8, score_threshold=0.0
        )
        no_results = retrieve_chunks(
            self.qdrant, self.embedder, "test_chunks", "Query", 8, score_threshold=1.0
        )

        self.assertEqual(
            [chunk.text for chunk, _ in default_results], ["Matching", "Above threshold"]
        )
        self.assertEqual(
            [chunk.text for chunk, _ in all_results],
            ["Matching", "Above threshold", "Below threshold"],
        )
        self.assertEqual(no_results, [])


if __name__ == "__main__":
    unittest.main()

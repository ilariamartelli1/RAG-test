import unittest
from unittest.mock import MagicMock

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.chunk_service import retrieve_chunks, store_chunks
from src.dataset_utils import chunk_dataset_row
from src.chunk_models import ChunkPayload
from src.prompt_utils import build_rag_prompt


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
        self.embedder.encode.return_value.tolist.side_effect = [
            [[1.0, 0.0]],
            [1.0, 0.0],
        ]

        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload], 8),
            1,
        )
        results = retrieve_chunks(self.qdrant, self.embedder, "test_chunks", "Query", 8)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], payload)
        self.assertAlmostEqual(results[0][1], 1.0)
        prompt = build_rag_prompt("Query", [result.text for result, _ in results])
        self.assertIn("Record: row-1\nsubject: Astronomy", prompt[0]["content"])

    def test_repeated_chunks_are_not_embedded_again(self) -> None:
        payload = ChunkPayload(text="Example", metadata={"row_id": "1"})
        self.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]

        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload, payload], 8),
            1,
        )
        self.assertEqual(
            store_chunks(self.qdrant, self.embedder, "test_chunks", [payload], 8),
            0,
        )
        self.embedder.encode.assert_called_once()

    def test_empty_batch_does_not_call_embedder(self) -> None:
        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", [], 8), 0)
        self.embedder.encode.assert_not_called()

    def test_qdrant_errors_are_not_ignored(self) -> None:
        payload = ChunkPayload(text="Example", metadata={})

        with self.assertRaises(ValueError):
            store_chunks(self.qdrant, self.embedder, "missing_collection", [payload], 8)
        self.embedder.encode.assert_not_called()

    def test_same_text_from_different_rows_keeps_both_sources(self) -> None:
        chunks = [
            ChunkPayload(text="Same text", metadata={"row_id": "1"}),
            ChunkPayload(text="Same text", metadata={"row_id": "2"}),
        ]
        self.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0], [1.0, 0.0]]

        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", chunks, 8), 2)
        self.assertEqual(self.qdrant.count("test_chunks").count, 2)

    def test_retrieval_uses_default_and_custom_score_thresholds(self) -> None:
        chunks = [
            ChunkPayload(text="Matching", metadata={}),
            ChunkPayload(text="Above threshold", metadata={}),
            ChunkPayload(text="Below threshold", metadata={}),
        ]
        self.embedder.encode.return_value.tolist.side_effect = [
            [[1.0, 0.0], [0.3, 0.9539392], [0.1, 0.9949874]],
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

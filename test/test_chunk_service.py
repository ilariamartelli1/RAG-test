"""Test chunk writes and retrieval with local Qdrant and fake embeddings.

These tests do not establish behavior against a remote Qdrant server or under
concurrent collection changes.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, UpdateStatus, VectorParams

from src.storage.chunk_service import retrieve_chunks, retrieve_document_chunks, store_chunks
from src.chunkers.dataset_chunker import chunk_dataset_row
from src.ingestion.document_indexing import build_document_index
from src.models.chunk_models import ChunkPayload
from src.models.index_models import build_index_settings
from src.rag.prompt_utils import build_rag_prompt
from src.storage.collection_service import create_collection_with_settings


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
        """Store a dataset chunk, retrieve its payload and score, and build a prompt."""
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
        """Store and retrieve a chunk through a fake provider implementing the interface."""
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
        """Propagate document and query embedding failures without storing a point."""
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
        """Identical payloads in one batch and a later call are embedded only once."""
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
        """An empty batch stores zero chunks without invoking the embedder."""
        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", [], 8), 0)
        self.embedder.embed_documents.assert_not_called()

    def test_qdrant_errors_are_not_ignored(self) -> None:
        """A missing collection fails during lookup before document embedding."""
        payload = ChunkPayload(text="Example", metadata={})

        with self.assertRaises(ValueError):
            store_chunks(self.qdrant, self.embedder, "missing_collection", [payload], 8)
        self.embedder.embed_documents.assert_not_called()

    def test_incomplete_write_response_raises_error(self) -> None:
        """With wait=True, acknowledged or timed-out writes still raise an error."""
        payload = ChunkPayload(text="Example", metadata={})
        self.embedder.embed_documents.return_value = [[1.0, 0.0]]
        qdrant = MagicMock()
        qdrant.retrieve.return_value = []
        for status in (UpdateStatus.ACKNOWLEDGED, UpdateStatus.WAIT_TIMEOUT):
            qdrant.upsert.return_value.status = status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "did not complete"):
                store_chunks(qdrant, self.embedder, "test_chunks", [payload], 8)
            self.assertTrue(qdrant.upsert.call_args.kwargs["wait"])

    def test_expected_dimension_rejects_invalid_embeddings_before_write(self) -> None:
        """A wrong count, dimension or vector value never reaches Qdrant upsert."""
        payload = ChunkPayload(text="Example", metadata={})
        for vectors in (
            [], [[1.0, 0.0], [1.0, 0.0]], [[1.0]], [[1.0, float("inf")]],
            [[1.0, "bad"]], [[1.0, True]],
        ):
            with self.subTest(vectors=vectors):
                self.embedder.embed_documents.return_value = vectors
                with patch.object(self.qdrant, "upsert", wraps=self.qdrant.upsert) as upsert:
                    with self.assertRaises(ValueError):
                        store_chunks(
                            self.qdrant, self.embedder, "test_chunks", [payload], 8,
                            embedding_dimension=2,
                        )
                    upsert.assert_not_called()
        self.assertEqual(self.qdrant.count("test_chunks").count, 0)

    def test_nonpositive_expected_dimension_does_not_contact_qdrant(self) -> None:
        """Reject an invalid expected dimension before lookup or embedding."""
        qdrant = MagicMock()
        for dimension in (0, -1):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                store_chunks(
                    qdrant, self.embedder, "test_chunks", [], 8,
                    embedding_dimension=dimension,
                )
        qdrant.retrieve.assert_not_called()
        self.embedder.embed_documents.assert_not_called()

    def test_same_text_from_different_rows_keeps_both_sources(self) -> None:
        """Identical text with different row metadata is stored as two points."""
        chunks = [
            ChunkPayload(text="Same text", metadata={"row_id": "1"}),
            ChunkPayload(text="Same text", metadata={"row_id": "2"}),
        ]
        self.embedder.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]

        self.assertEqual(store_chunks(self.qdrant, self.embedder, "test_chunks", chunks, 8), 2)
        self.assertEqual(self.qdrant.count("test_chunks").count, 2)

    def test_retrieval_uses_default_and_custom_score_thresholds(self) -> None:
        """Check which results pass default, zero and strict similarity thresholds."""
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

    def test_document_retrieval_checks_readiness_settings_and_dimension(self) -> None:
        """Search a built index, then reject a changed model and vector dimension.

        The rejected requests must not send another query to Qdrant.
        """
        self.embedder.embed_query.return_value = [1.0, 0.0]
        self.embedder.embed_documents.side_effect = lambda texts, batch_size: [
            [1.0, 0.0] for _ in texts
        ]
        settings = build_index_settings(
            document_bytes=b"document content", document_filename="doc.txt",
            embedding_provider="fake", embedding_model="model", max_chunk_chars=100,
        )
        build_document_index(
            qdrant=self.qdrant, collection_name="document", data=b"document content",
            document_id="document-1", build_id="build-1",
            filename="doc.txt", embedder=self.embedder, embedding_provider="fake",
            embedding_model="model", embedding_batch_size=2, chunk_batch_size=2,
            max_chunk_chars=100,
        )
        with patch.object(self.qdrant, "query_points", wraps=self.qdrant.query_points) as query:
            results = retrieve_document_chunks(self.qdrant, self.embedder, "document", settings, "ask", 2)
            self.assertEqual(results[0][0].text, "document content")
            self.assertEqual(results[0][0].metadata["source"], "doc.txt")
            self.assertEqual(query.call_count, 1)

            changed = build_index_settings(
                document_bytes=b"document content", document_filename="doc.txt",
                embedding_provider="fake", embedding_model="other", max_chunk_chars=100,
            )
            with self.assertRaisesRegex(ValueError, "embedding_model"):
                retrieve_document_chunks(self.qdrant, self.embedder, "document", changed, "ask", 2)
            self.embedder.embed_query.return_value = [1.0, 0.0, 0.0]
            with self.assertRaisesRegex(ValueError, "expected size 3"):
                retrieve_document_chunks(self.qdrant, self.embedder, "document", settings, "ask", 2)
            self.assertEqual(query.call_count, 1)

    def test_document_retrieval_does_not_query_incomplete_or_legacy_collection(self) -> None:
        """Building, legacy and missing collections fail before Qdrant search."""
        self.embedder.embed_query.return_value = [1.0, 0.0]
        settings = build_index_settings(
            document_bytes=b"document", document_filename="doc.txt",
            embedding_provider="fake", embedding_model="model", max_chunk_chars=100,
        )
        create_collection_with_settings(self.qdrant, "building", 2, settings)
        with patch.object(self.qdrant, "query_points", wraps=self.qdrant.query_points) as query:
            for name in ("building", "test_chunks", "absent"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    retrieve_document_chunks(self.qdrant, self.embedder, name, settings, "ask", 2)
            query.assert_not_called()
        self.assertFalse(self.qdrant.collection_exists("absent"))

    def test_document_retrieval_rejects_empty_vectors_and_invalid_batch_size(self) -> None:
        """Reject an empty query vector and reject a zero batch size before embedding."""
        settings = build_index_settings(
            document_bytes=b"document", document_filename="doc.txt",
            embedding_provider="fake", embedding_model="model", max_chunk_chars=100,
        )
        self.embedder.embed_query.return_value = []
        with self.assertRaisesRegex(ValueError, "empty vector"):
            retrieve_document_chunks(self.qdrant, self.embedder, "test_chunks", settings, "ask", 2)
        self.embedder.embed_query.reset_mock()
        with self.assertRaisesRegex(ValueError, "batch_size"):
            retrieve_document_chunks(self.qdrant, self.embedder, "test_chunks", settings, "ask", 0)
        self.embedder.embed_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()

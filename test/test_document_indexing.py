"""Tests for building document indexes in local Qdrant."""

import unittest
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, UpdateStatus, VectorParams

from src.ingestion.document_indexing import build_document_index
from src.models.index_models import build_index_settings
from src.storage.collection_service import (
    ensure_collection_index_ready,
    get_collection_index_settings,
    get_collection_index_status,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.document_calls = 0
        self.fail_on_call: int | None = None

    def embed_query(self, text: str, batch_size: int) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str], batch_size: int) -> list[list[float]]:
        self.document_calls += 1
        if self.document_calls == self.fail_on_call:
            raise RuntimeError("embedding failed")
        return [[1.0, 0.0] for _ in texts]


class DocumentIndexingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qdrant = QdrantClient(":memory:")
        self.addCleanup(self.qdrant.close)
        self.embedder = FakeEmbedder()
        self.arguments = {
            "qdrant": self.qdrant,
            "collection_name": "documents",
            "data": b"abcdefgh",
            "filename": "notes.txt",
            "embedder": self.embedder,
            "embedding_provider": "fake",
            "embedding_model": "test-model",
            "embedding_batch_size": 2,
            "chunk_batch_size": 1,
            "max_chunk_chars": 4,
        }

    def test_builds_new_index_and_marks_it_ready(self) -> None:
        self.assertEqual(build_document_index(**self.arguments), 2)

        self.assertEqual(self.qdrant.count("documents").count, 2)
        self.assertIsNone(ensure_collection_index_ready(self.qdrant, "documents"))
        self.assertEqual(get_collection_index_settings(self.qdrant, "documents"), build_index_settings(
            document_bytes=b"abcdefgh",
            document_filename="notes.txt",
            embedding_provider="fake",
            embedding_model="test-model",
            max_chunk_chars=4,
        ))
        points, _ = self.qdrant.scroll("documents", with_payload=True)
        self.assertEqual({point.payload["text"] for point in points}, {"abcd", "efgh"})

    def test_failed_later_batch_leaves_partial_index_building(self) -> None:
        self.embedder.fail_on_call = 2
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            build_document_index(**self.arguments)

        self.assertEqual(self.qdrant.count("documents").count, 1)
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "building")
        with self.assertRaisesRegex(ValueError, "still building"):
            ensure_collection_index_ready(self.qdrant, "documents")

    def test_empty_or_invalid_documents_are_not_marked_ready(self) -> None:
        for name, data, filename in (
            ("empty", b"", "empty.txt"),
            ("blank", b" \n", "blank.txt"),
            ("unsupported", b"data", "file.docx"),
            ("undecodable", b"\xff", "file.txt"),
        ):
            arguments = self.arguments | {"collection_name": name, "data": data, "filename": filename}
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_document_index(**arguments)
            self.assertEqual(get_collection_index_status(self.qdrant, name), "building")

    def test_existing_collection_is_not_modified(self) -> None:
        self.qdrant.create_collection(
            "documents",
            vectors_config=VectorParams(size=2, distance=Distance.DOT),
            metadata={"original": "keep"},
        )
        self.qdrant.upsert("documents", points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "keep"})
        ])

        with self.assertRaises(ValueError):
            build_document_index(**self.arguments)
        self.assertEqual(self.qdrant.count("documents").count, 1)
        self.assertEqual(self.qdrant.get_collection("documents").config.metadata, {"original": "keep"})
        self.assertEqual(self.qdrant.get_collection("documents").config.params.vectors.distance, Distance.DOT)

    def test_invalid_settings_and_dimension_probe_fail_before_creation(self) -> None:
        for field, value in (
            ("chunk_batch_size", 0),
            ("embedding_batch_size", 0),
            ("max_chunk_chars", 0),
            ("embedding_model", " "),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_document_index(**(self.arguments | {field: value}))
            self.assertFalse(self.qdrant.collection_exists("documents"))

        with patch.object(self.embedder, "embed_query", side_effect=RuntimeError("probe failed")):
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                build_document_index(**self.arguments)
        self.assertFalse(self.qdrant.collection_exists("documents"))

    def test_unconfirmed_write_does_not_mark_index_ready(self) -> None:
        with patch.object(self.qdrant, "upsert", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                build_document_index(**self.arguments)
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "building")

    def test_acknowledged_but_incomplete_write_does_not_mark_index_ready(self) -> None:
        result = MagicMock(status=UpdateStatus.ACKNOWLEDGED)
        with patch.object(self.qdrant, "upsert", return_value=result) as upsert:
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                build_document_index(**self.arguments)
        self.assertTrue(upsert.call_args.kwargs["wait"])
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "building")

    def test_ready_update_errors_propagate(self) -> None:
        with patch(
            "src.ingestion.document_indexing.mark_collection_index_ready",
            side_effect=RuntimeError("status update failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "status update failed"):
                build_document_index(**self.arguments)
        self.assertEqual(self.qdrant.count("documents").count, 2)
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "building")


if __name__ == "__main__":
    unittest.main()

"""Tests for building document indexes in local Qdrant."""

import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, UpdateStatus, VectorParams

from src.ingestion.document_indexing import build_document_index
from src.models.index_models import IndexSettings, build_index_settings
from src.storage.chunk_service import retrieve_document_chunks
from src.storage.collection_service import (
    IndexSettingsMismatchError,
    ensure_collection_index_ready,
    get_collection_index_settings,
    get_collection_index_status,
    mark_collection_index_ready,
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

    def test_building_replacement_keeps_current_collection_ready(self) -> None:
        build_document_index(**self.arguments)
        old_settings = get_collection_index_settings(self.qdrant, "documents")
        arguments = self.arguments | {
            "collection_name": "replacement", "data": b"new data", "max_chunk_chars": 8,
        }
        self.assertEqual(build_document_index(**arguments), 1)
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "ready")
        self.assertEqual(get_collection_index_settings(self.qdrant, "documents"), old_settings)
        old_results = retrieve_document_chunks(
            self.qdrant, self.embedder, "documents", old_settings, "query", 2
        )
        self.assertEqual({chunk.text for chunk, _ in old_results}, {"abcd", "efgh"})
        new_settings = build_index_settings(
            document_bytes=b"new data", document_filename="notes.txt",
            embedding_provider="fake", embedding_model="test-model", max_chunk_chars=8,
        )
        new_results = retrieve_document_chunks(
            self.qdrant, self.embedder, "replacement", new_settings, "query", 2
        )
        self.assertEqual([chunk.text for chunk, _ in new_results], ["new data"])

    def test_failed_replacement_leaves_original_ready_and_replacement_building(self) -> None:
        build_document_index(**self.arguments)
        old_settings = get_collection_index_settings(self.qdrant, "documents")
        self.embedder.fail_on_call = self.embedder.document_calls + 2
        arguments = self.arguments | {"collection_name": "replacement"}

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            build_document_index(**arguments)

        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "ready")
        self.assertEqual(get_collection_index_settings(self.qdrant, "documents"), old_settings)
        self.assertEqual(self.qdrant.count("documents").count, 2)
        old_results = retrieve_document_chunks(
            self.qdrant, self.embedder, "documents", old_settings, "query", 2
        )
        self.assertEqual({chunk.text for chunk, _ in old_results}, {"abcd", "efgh"})
        self.assertEqual(get_collection_index_status(self.qdrant, "replacement"), "building")
        self.assertEqual(self.qdrant.count("replacement").count, 1)
        with self.assertRaisesRegex(ValueError, "still building"):
            ensure_collection_index_ready(self.qdrant, "replacement")

    def test_replacement_failure_after_status_update_leaves_both_collections(self) -> None:
        build_document_index(**self.arguments)
        arguments = self.arguments | {"collection_name": "replacement"}

        def update_then_fail(qdrant, collection_name, settings):
            mark_collection_index_ready(qdrant, collection_name, settings)
            raise RuntimeError("verification failed")

        with patch(
            "src.ingestion.document_indexing.mark_collection_index_ready",
            side_effect=update_then_fail,
        ):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                build_document_index(**arguments)
        self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "ready")
        self.assertEqual(self.qdrant.count("documents").count, 2)
        self.assertEqual(get_collection_index_status(self.qdrant, "replacement"), "ready")
        self.assertEqual(self.qdrant.count("replacement").count, 2)

    def test_building_new_collection_leaves_legacy_collection_untouched(self) -> None:
        self.qdrant.create_collection(
            "legacy", vectors_config=VectorParams(size=2, distance=Distance.DOT),
            metadata={"original": "keep"},
        )
        self.qdrant.upsert("legacy", points=[
            PointStruct(id=1, vector=[1.0, 0.0], payload={"text": "keep"})
        ])
        arguments = self.arguments | {"collection_name": "replacement"}
        self.assertEqual(build_document_index(**arguments), 2)
        self.assertEqual(self.qdrant.count("legacy").count, 1)
        info = self.qdrant.get_collection("legacy")
        self.assertEqual(info.config.metadata, {"original": "keep"})
        self.assertEqual(info.config.params.vectors.distance, Distance.DOT)

    def test_interrupted_replacement_is_not_queryable_after_reopening(self) -> None:
        old_settings = build_index_settings(
            document_bytes=b"abcdefgh", document_filename="notes.txt",
            embedding_provider="fake", embedding_model="test-model", max_chunk_chars=4,
        )
        new_settings = build_index_settings(
            document_bytes=b"ijklmnop", document_filename="notes.txt",
            embedding_provider="fake", embedding_model="test-model", max_chunk_chars=4,
        )
        with TemporaryDirectory() as path:
            with closing(QdrantClient(path=path)) as qdrant:
                arguments = self.arguments | {"qdrant": qdrant}
                build_document_index(**arguments)
                self.embedder.fail_on_call = self.embedder.document_calls + 2
                with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                    build_document_index(**(arguments | {
                        "collection_name": "interrupted", "data": b"ijklmnop",
                    }))

            with closing(QdrantClient(path=path)) as qdrant:
                self.assertEqual(get_collection_index_status(qdrant, "interrupted"), "building")
                self.assertEqual(qdrant.count("interrupted").count, 1)
                with patch.object(qdrant, "query_points", wraps=qdrant.query_points) as query:
                    with self.assertRaisesRegex(ValueError, "still building"):
                        retrieve_document_chunks(
                            qdrant, self.embedder, "interrupted", new_settings, "query", 2
                        )
                    query.assert_not_called()
                    old_results = retrieve_document_chunks(
                        qdrant, self.embedder, "documents", old_settings, "query", 2
                    )
                    self.assertEqual({chunk.text for chunk, _ in old_results}, {"abcd", "efgh"})

                self.embedder.fail_on_call = None
                self.assertEqual(build_document_index(**(self.arguments | {
                    "qdrant": qdrant, "collection_name": "retry", "data": b"ijklmnop",
                })), 2)
                retry_results = retrieve_document_chunks(
                    qdrant, self.embedder, "retry", new_settings, "query", 2
                )
                self.assertEqual({chunk.text for chunk, _ in retry_results}, {"ijkl", "mnop"})
                self.assertEqual(get_collection_index_status(qdrant, "interrupted"), "building")
                self.assertEqual(qdrant.count("interrupted").count, 1)
                self.assertEqual(qdrant.count("documents").count, 2)

    def test_incompatible_settings_do_not_query_or_modify_original_index(self) -> None:
        build_document_index(**self.arguments)
        old_settings = get_collection_index_settings(self.qdrant, "documents")
        for field, value in (
            ("document_fingerprint", "b" * 64),
            ("document_filename", "other.txt"),
            ("embedding_provider", "other-provider"),
            ("embedding_model", "other-model"),
            ("max_chunk_chars", 10),
        ):
            settings = IndexSettings.model_validate(old_settings.model_dump() | {field: value})
            with self.subTest(field=field):
                with patch.object(self.qdrant, "query_points", wraps=self.qdrant.query_points) as query:
                    with self.assertRaises(IndexSettingsMismatchError) as raised:
                        retrieve_document_chunks(
                            self.qdrant, self.embedder, "documents", settings, "query", 2
                        )
                    self.assertEqual(raised.exception.differences, (field,))
                    query.assert_not_called()
                self.assertEqual(get_collection_index_settings(self.qdrant, "documents"), old_settings)
                self.assertEqual(get_collection_index_status(self.qdrant, "documents"), "ready")
                self.assertEqual(self.qdrant.count("documents").count, 2)

        old_results = retrieve_document_chunks(
            self.qdrant, self.embedder, "documents", old_settings, "query", 2
        )
        self.assertEqual({chunk.text for chunk, _ in old_results}, {"abcd", "efgh"})


if __name__ == "__main__":
    unittest.main()

"""Tests for Qdrant collection setup.

QdrantClient(":memory:") runs locally without a server or disk storage, so its
collections disappear when the client closes. Persistence tests instead use
temporary directories to check metadata after reopening. Mocked cases simulate
Qdrant errors or configurations unavailable in local mode. These tests do not
exercise concurrent operations or a remote server.
"""

import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, MultiVectorComparator, MultiVectorConfig, PointStruct, VectorParams

from src.models.index_models import IndexSettings
from src.storage.collection_service import (
    INDEX_SETTINGS_METADATA_KEY,
    INDEX_STATUS_METADATA_KEY,
    IndexSettingsMetadataError,
    IndexSettingsMismatchError,
    compare_collection_index_settings,
    create_collection_with_settings,
    ensure_collection,
    ensure_collection_index_settings,
    ensure_collection_index_ready,
    ensure_document_index_compatible,
    get_collection_index_settings,
    get_collection_index_status,
    mark_collection_index_ready,
)


class CollectionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = IndexSettings(
            document_fingerprint="a" * 64,
            document_filename="example.txt",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            max_chunk_chars=1000,
        )

    def test_creates_collection_with_persistent_settings(self) -> None:
        """Create an index, reopen local storage, and verify vectors and metadata.

        The saved settings and initial building status must survive reopening.
        """
        with TemporaryDirectory() as path:
            with closing(QdrantClient(path=path)) as qdrant:
                create_collection_with_settings(qdrant, "documents", 3, self.settings)

            with closing(QdrantClient(path=path)) as qdrant:
                collection = qdrant.get_collection("documents")
                self.assertEqual(collection.config.params.vectors.size, 3)
                self.assertEqual(collection.config.params.vectors.distance, Distance.COSINE)
                self.assertEqual(
                    collection.config.metadata[INDEX_SETTINGS_METADATA_KEY],
                    self.settings.model_dump(),
                )
                self.assertEqual(collection.config.metadata[INDEX_STATUS_METADATA_KEY], "building")
                self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)

    def test_reads_collection_index_settings(self) -> None:
        """Settings read from a newly created collection match the supplied model."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)

        self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)

    def test_index_status_transition_preserves_metadata_and_points(self) -> None:
        """Reject building as ready, then mark it ready and reopen storage.

        Confirm settings, unrelated metadata and the point remain; a second
        ready transition is rejected.
        """
        with TemporaryDirectory() as path:
            with closing(QdrantClient(path=path)) as qdrant:
                create_collection_with_settings(qdrant, "documents", 3, self.settings)
                qdrant.update_collection("documents", metadata={"other": "keep"})
                qdrant.upsert("documents", points=[
                    PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
                ])
                self.assertEqual(get_collection_index_status(qdrant, "documents"), "building")
                with self.assertRaisesRegex(ValueError, "still building"):
                    ensure_collection_index_ready(qdrant, "documents")
                mark_collection_index_ready(qdrant, "documents", self.settings)

            with closing(QdrantClient(path=path)) as qdrant:
                self.assertEqual(get_collection_index_status(qdrant, "documents"), "ready")
                self.assertIsNone(ensure_collection_index_ready(qdrant, "documents"))
                self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)
                self.assertEqual(qdrant.get_collection("documents").config.metadata["other"], "keep")
                self.assertEqual(qdrant.count("documents").count, 1)
                with self.assertRaisesRegex(ValueError, "not building"):
                    mark_collection_index_ready(qdrant, "documents", self.settings)

    def test_missing_and_invalid_index_status_are_rejected(self) -> None:
        """Missing, unknown and null statuses cannot pass the status checks.

        Where settings are present, marking ready also fails; metadata remains.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        for name, metadata in (
            ("no_metadata", None),
            ("missing", {INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump()}),
            ("invalid", {
                INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump(),
                INDEX_STATUS_METADATA_KEY: "unknown",
            }),
            ("null", {
                INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump(),
                INDEX_STATUS_METADATA_KEY: None,
            }),
        ):
            qdrant.create_collection(
                name,
                vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                metadata=metadata,
            )
            for operation in (get_collection_index_status, ensure_collection_index_ready):
                with self.subTest(name=name, operation=operation.__name__), self.assertRaises(ValueError):
                    operation(qdrant, name)
            if metadata is not None:
                with self.subTest(name=name, operation="mark"), self.assertRaises(ValueError):
                    mark_collection_index_ready(qdrant, name, self.settings)
            self.assertEqual(qdrant.get_collection(name).config.metadata, metadata)

    def test_mark_ready_rejects_mismatched_or_missing_settings(self) -> None:
        """A different model or missing settings prevents a building-to-ready transition."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)
        current = IndexSettings(**(self.settings.model_dump() | {"embedding_model": "other"}))
        with self.assertRaises(IndexSettingsMismatchError):
            mark_collection_index_ready(qdrant, "documents", current)
        self.assertEqual(get_collection_index_status(qdrant, "documents"), "building")

        qdrant.create_collection(
            "legacy",
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
            metadata={INDEX_STATUS_METADATA_KEY: "building"},
        )
        with self.assertRaises(IndexSettingsMetadataError):
            mark_collection_index_ready(qdrant, "legacy", self.settings)
        self.assertEqual(get_collection_index_status(qdrant, "legacy"), "building")

    def test_mark_ready_reports_update_and_verification_failures(self) -> None:
        """Simulate update errors, false confirmation and failed status verification.

        Each failure is surfaced, and no collection deletion is requested.
        """
        qdrant = MagicMock()
        qdrant.get_collection.return_value.config.metadata = {
            INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump(),
            INDEX_STATUS_METADATA_KEY: "building",
        }
        qdrant.update_collection.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            mark_collection_index_ready(qdrant, "documents", self.settings)

        qdrant.update_collection.side_effect = None
        qdrant.update_collection.return_value = False
        with self.assertRaisesRegex(RuntimeError, "did not confirm"):
            mark_collection_index_ready(qdrant, "documents", self.settings)

        qdrant.update_collection.return_value = True
        with self.assertRaisesRegex(RuntimeError, "could not be verified"):
            mark_collection_index_ready(qdrant, "documents", self.settings)

        collection = qdrant.get_collection.return_value
        qdrant.get_collection.side_effect = [collection, collection, RuntimeError("read failed")]
        with self.assertRaisesRegex(RuntimeError, "read failed"):
            mark_collection_index_ready(qdrant, "documents", self.settings)
        qdrant.delete_collection.assert_not_called()

    def test_index_status_lookup_errors_propagate(self) -> None:
        """A failed status lookup propagates without attempting an update."""
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            get_collection_index_status(qdrant, "documents")
        qdrant.update_collection.assert_not_called()

    def test_compares_collection_index_settings_without_modification(self) -> None:
        """Compare matching and changed settings against stored metadata.

        Report the three changed fields and preserve the original settings and point.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.upsert("documents", points=[
            PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
        ])

        self.assertEqual(compare_collection_index_settings(qdrant, "documents", self.settings), [])
        current = IndexSettings(**(self.settings.model_dump() | {
            "document_fingerprint": "b" * 64,
            "embedding_model": "other-model",
            "max_chunk_chars": 500,
        }))
        self.assertEqual(
            compare_collection_index_settings(qdrant, "documents", current),
            ["document_fingerprint", "embedding_model", "max_chunk_chars"],
        )
        self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)
        self.assertEqual(qdrant.count("documents").count, 1)

    def test_comparison_propagates_missing_and_invalid_settings(self) -> None:
        """Comparison raises for missing or invalid settings and preserves metadata."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        for name, metadata in (
            ("missing", None),
            ("invalid", {INDEX_SETTINGS_METADATA_KEY: {}}),
        ):
            qdrant.create_collection(
                name,
                vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                metadata=metadata,
            )
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "index settings"):
                compare_collection_index_settings(qdrant, name, self.settings)
            self.assertEqual(qdrant.get_collection(name).config.metadata, metadata)

    def test_comparison_propagates_qdrant_errors(self) -> None:
        """A Qdrant lookup failure propagates without creating, updating or deleting."""
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            compare_collection_index_settings(qdrant, "documents", self.settings)
        qdrant.create_collection.assert_not_called()
        qdrant.update_collection.assert_not_called()
        qdrant.delete_collection.assert_not_called()

    def test_settings_check_accepts_matching_settings(self) -> None:
        """Matching stored and current settings pass the settings-only check."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)

        self.assertIsNone(ensure_collection_index_settings(qdrant, "documents", self.settings))

    def test_document_compatibility_checks_ready_settings_and_vectors(self) -> None:
        """Reject a building index, a changed model and a mismatched vector size.

        Matching ready settings pass; the stored point, settings and status remain.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.upsert("documents", points=[
            PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
        ])
        with self.assertRaisesRegex(ValueError, "still building"):
            ensure_document_index_compatible(qdrant, "documents", 3, self.settings)

        mark_collection_index_ready(qdrant, "documents", self.settings)
        self.assertIsNone(ensure_document_index_compatible(qdrant, "documents", 3, self.settings))
        changed = IndexSettings(**(self.settings.model_dump() | {"embedding_model": "other"}))
        with self.assertRaises(IndexSettingsMismatchError):
            ensure_document_index_compatible(qdrant, "documents", 3, changed)
        with self.assertRaisesRegex(ValueError, "expected size 4"):
            ensure_document_index_compatible(qdrant, "documents", 4, self.settings)

        self.assertEqual(qdrant.count("documents").count, 1)
        self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)
        self.assertEqual(get_collection_index_status(qdrant, "documents"), "ready")

    def test_document_compatibility_rejects_missing_or_incompatible_collections(self) -> None:
        """Reject missing metadata, dot distance, named vectors and an absent collection.

        Existing metadata remains unchanged and the absent collection is not created.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        for name, metadata, vectors in (
            ("missing_status", {INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump()},
             VectorParams(size=3, distance=Distance.COSINE)),
            ("missing_settings", {INDEX_STATUS_METADATA_KEY: "ready"},
             VectorParams(size=3, distance=Distance.COSINE)),
            ("dot", {INDEX_STATUS_METADATA_KEY: "ready",
                     INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump()},
             VectorParams(size=3, distance=Distance.DOT)),
            ("named", {INDEX_STATUS_METADATA_KEY: "ready",
                       INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump()},
             {"text": VectorParams(size=3, distance=Distance.COSINE)}),
        ):
            qdrant.create_collection(name, vectors_config=vectors, metadata=metadata)
            with self.subTest(name=name), self.assertRaises(ValueError):
                ensure_document_index_compatible(qdrant, name, 3, self.settings)
            self.assertEqual(qdrant.get_collection(name).config.metadata, metadata)

        with self.assertRaises(ValueError):
            ensure_document_index_compatible(qdrant, "absent", 3, self.settings)
        self.assertFalse(qdrant.collection_exists("absent"))

    def test_document_compatibility_propagates_lookup_errors(self) -> None:
        """Compatibility lookup failures propagate without modifying the collection."""
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_document_index_compatible(qdrant, "documents", 3, self.settings)
        qdrant.create_collection.assert_not_called()
        qdrant.update_collection.assert_not_called()
        qdrant.delete_collection.assert_not_called()

    def test_settings_check_rejects_differences_without_modifying_collection(self) -> None:
        """A mismatch error identifies the collection and differing model and chunk size.

        The stored settings and point remain unchanged.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.upsert("documents", points=[
            PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
        ])
        current = IndexSettings(**(self.settings.model_dump() | {
            "embedding_model": "other-model",
            "max_chunk_chars": 500,
        }))

        with self.assertRaisesRegex(
            IndexSettingsMismatchError,
            "documents.*embedding_model, max_chunk_chars.*explicitly rebuild",
        ) as raised:
            ensure_collection_index_settings(qdrant, "documents", current)

        self.assertIsInstance(raised.exception, ValueError)
        self.assertEqual(raised.exception.collection_name, "documents")
        self.assertEqual(raised.exception.differences, ("embedding_model", "max_chunk_chars"))
        self.assertEqual(get_collection_index_settings(qdrant, "documents"), self.settings)
        self.assertEqual(qdrant.count("documents").count, 1)

    def test_settings_check_propagates_missing_invalid_and_lookup_errors(self) -> None:
        """Distinguish missing or invalid settings from a settings mismatch.

        Check the error reason and cause, preserve metadata, and propagate
        service failures without updating or deleting.
        """
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        for name, metadata in (
            ("missing", None),
            ("invalid", {INDEX_SETTINGS_METADATA_KEY: {}}),
        ):
            qdrant.create_collection(
                name,
                vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                metadata=metadata,
            )
            with self.subTest(name=name), self.assertRaisesRegex(
                IndexSettingsMetadataError, "index settings"
            ) as raised:
                ensure_collection_index_settings(qdrant, name, self.settings)
            self.assertNotIsInstance(raised.exception, IndexSettingsMismatchError)
            self.assertIsInstance(raised.exception, ValueError)
            self.assertEqual(raised.exception.collection_name, name)
            self.assertEqual(raised.exception.reason, name)
            if name == "invalid":
                self.assertIsInstance(raised.exception.__cause__, ValueError)
            else:
                self.assertIsNone(raised.exception.__cause__)
            self.assertEqual(qdrant.get_collection(name).config.metadata, metadata)

        unavailable = MagicMock()
        unavailable.get_collection.side_effect = RuntimeError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection_index_settings(unavailable, "documents", self.settings)
        unavailable.update_collection.assert_not_called()
        unavailable.delete_collection.assert_not_called()

    def test_missing_index_settings_are_rejected_without_modification(self) -> None:
        """Absent settings metadata raises while preserving existing metadata and point."""
        for metadata in (None, {}, {"other": "keep"}):
            with self.subTest(metadata=metadata):
                qdrant = QdrantClient(":memory:")
                self.addCleanup(qdrant.close)
                qdrant.create_collection(
                    "documents",
                    vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                    metadata=metadata,
                )
                qdrant.upsert("documents", points=[
                    PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
                ])

                with self.assertRaisesRegex(ValueError, "documents.*no index settings"):
                    get_collection_index_settings(qdrant, "documents")

                self.assertEqual(qdrant.get_collection("documents").config.metadata, metadata)
                self.assertEqual(qdrant.count("documents").count, 1)

    def test_invalid_index_settings_are_rejected_without_modification(self) -> None:
        """Reject malformed settings values and preserve their original metadata."""
        for value in (None, "invalid", {}, self.settings.model_dump() | {"unexpected": True}):
            with self.subTest(value=value):
                qdrant = QdrantClient(":memory:")
                self.addCleanup(qdrant.close)
                metadata = {INDEX_SETTINGS_METADATA_KEY: value}
                qdrant.create_collection(
                    "documents",
                    vectors_config=VectorParams(size=3, distance=Distance.COSINE),
                    metadata=metadata,
                )

                with self.assertRaisesRegex(ValueError, "documents.*invalid index settings"):
                    get_collection_index_settings(qdrant, "documents")

                self.assertEqual(qdrant.get_collection("documents").config.metadata, metadata)

    def test_index_settings_lookup_errors_propagate(self) -> None:
        """A failed settings lookup does not create, update or delete a collection."""
        qdrant = MagicMock()
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            get_collection_index_settings(qdrant, "documents")
        qdrant.create_collection.assert_not_called()
        qdrant.update_collection.assert_not_called()
        qdrant.delete_collection.assert_not_called()

    def test_settings_creation_does_not_modify_existing_collection(self) -> None:
        """Creating at an occupied name fails and retains its metadata and point."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        qdrant.create_collection(
            "documents",
            vectors_config=VectorParams(size=3, distance=Distance.COSINE),
            metadata={"original": "keep"},
        )
        qdrant.upsert("documents", points=[
            PointStruct(id=1, vector=[1.0, 0.0, 0.0], payload={"text": "keep"})
        ])
        with self.assertRaises(ValueError):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)

        self.assertEqual(qdrant.get_collection("documents").config.metadata, {"original": "keep"})
        self.assertEqual(qdrant.count("documents").count, 1)

    def test_settings_creation_rejects_invalid_dimension(self) -> None:
        """Zero or negative vector dimensions fail before collection creation."""
        qdrant = MagicMock()
        for dimension in (0, -1):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(ValueError, "dimension"):
                create_collection_with_settings(qdrant, "documents", dimension, self.settings)
        qdrant.create_collection.assert_not_called()

    def test_settings_creation_errors_propagate(self) -> None:
        """Creation errors propagate without attempting to verify metadata."""
        qdrant = MagicMock()
        qdrant.create_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.get_collection.assert_not_called()

    def test_settings_creation_rejects_failed_creation_or_missing_metadata(self) -> None:
        """Reject false creation confirmation or incomplete and incorrect metadata.

        A collection is not deleted when verification fails.
        """
        qdrant = MagicMock()
        qdrant.create_collection.return_value = False

        with self.assertRaisesRegex(RuntimeError, "did not create"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.get_collection.assert_not_called()

        qdrant.create_collection.return_value = True
        for metadata in (
            None,
            {},
            {INDEX_SETTINGS_METADATA_KEY: {"wrong": "settings"}},
            {INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump()},
            {
                INDEX_SETTINGS_METADATA_KEY: self.settings.model_dump(),
                INDEX_STATUS_METADATA_KEY: "ready",
            },
        ):
            qdrant.get_collection.return_value.config.metadata = metadata
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                RuntimeError, "metadata could not be verified"
            ):
                create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.delete_collection.assert_not_called()

    def test_settings_verification_errors_propagate_without_deleting_collection(self) -> None:
        """A read failure after creation propagates without deleting the collection."""
        qdrant = MagicMock()
        qdrant.create_collection.return_value = True
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.delete_collection.assert_not_called()

    def test_creates_collection_once(self) -> None:
        """The legacy ensure helper creates a cosine collection, then reuses it."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)

        self.assertTrue(ensure_collection(qdrant, "documents", 3))
        vectors = qdrant.get_collection("documents").config.params.vectors
        self.assertEqual(vectors.size, 3)
        self.assertEqual(vectors.distance, Distance.COSINE)
        self.assertFalse(ensure_collection(qdrant, "documents", 3))

    def test_existing_collection_is_not_recreated(self) -> None:
        """A compatible existing collection is checked without a creation request."""
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.return_value.config.params.vectors = VectorParams(
            size=3, distance=Distance.COSINE
        )

        self.assertFalse(ensure_collection(qdrant, "documents", 3))
        qdrant.get_collection.assert_called_once_with("documents")
        qdrant.create_collection.assert_not_called()

    def test_existing_collection_with_incompatible_configuration_is_preserved(self) -> None:
        """Wrong vector size or distance raises while retaining the config and point."""
        for size, distance in ((4, Distance.COSINE), (3, Distance.DOT)):
            with self.subTest(size=size, distance=distance):
                qdrant = QdrantClient(":memory:")
                self.addCleanup(qdrant.close)
                qdrant.create_collection(
                    "documents", vectors_config=VectorParams(size=size, distance=distance)
                )
                qdrant.upsert("documents", points=[
                    PointStruct(id=1, vector=[1.0] * size, payload={"text": "keep"})
                ])

                with self.assertRaisesRegex(ValueError, "expected size 3 and distance Cosine"):
                    ensure_collection(qdrant, "documents", 3)

                vectors = qdrant.get_collection("documents").config.params.vectors
                self.assertEqual(vectors.size, size)
                self.assertEqual(vectors.distance, distance)
                self.assertEqual(qdrant.count("documents").count, 1)

    def test_rejects_named_vectors(self) -> None:
        """The legacy ensure helper rejects named vectors without deleting the collection."""
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        qdrant.create_collection(
            "documents", vectors_config={"text": VectorParams(size=3, distance=Distance.COSINE)}
        )

        with self.assertRaisesRegex(ValueError, "single unnamed dense vector"):
            ensure_collection(qdrant, "documents", 3)
        self.assertTrue(qdrant.collection_exists("documents"))

    def test_rejects_multivector_configuration(self) -> None:
        """A mocked multivector config is rejected without creating or deleting."""
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.return_value.config.params.vectors = VectorParams(
            size=3, distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        )

        with self.assertRaisesRegex(ValueError, "single unnamed dense vector"):
            ensure_collection(qdrant, "documents", 3)
        qdrant.create_collection.assert_not_called()
        qdrant.delete_collection.assert_not_called()

    def test_invalid_dimension_does_not_contact_qdrant(self) -> None:
        """The legacy ensure helper rejects nonpositive dimensions before contacting Qdrant."""
        qdrant = MagicMock()
        for dimension in (0, -1):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(ValueError, "dimension"):
                ensure_collection(qdrant, "documents", dimension)
        qdrant.collection_exists.assert_not_called()
        qdrant.create_collection.assert_not_called()

    def test_existing_collection_lookup_errors_propagate(self) -> None:
        """An existing collection lookup failure does not trigger collection creation."""
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection(qdrant, "documents", 3)
        qdrant.create_collection.assert_not_called()

    def test_creation_errors_propagate(self) -> None:
        """The legacy ensure helper propagates errors when collection creation fails."""
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = False
        qdrant.create_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection(qdrant, "documents", 3)


if __name__ == "__main__":
    unittest.main()

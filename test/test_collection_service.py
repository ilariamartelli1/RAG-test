"""Tests for Qdrant collection setup."""

import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, MultiVectorComparator, MultiVectorConfig, PointStruct, VectorParams

from src.models.index_models import IndexSettings
from src.storage.collection_service import (
    INDEX_SETTINGS_METADATA_KEY,
    create_collection_with_settings,
    ensure_collection,
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

    def test_settings_creation_does_not_modify_existing_collection(self) -> None:
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
        qdrant = MagicMock()
        for dimension in (0, -1):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(ValueError, "dimension"):
                create_collection_with_settings(qdrant, "documents", dimension, self.settings)
        qdrant.create_collection.assert_not_called()

    def test_settings_creation_errors_propagate(self) -> None:
        qdrant = MagicMock()
        qdrant.create_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.get_collection.assert_not_called()

    def test_settings_creation_rejects_failed_creation_or_missing_metadata(self) -> None:
        qdrant = MagicMock()
        qdrant.create_collection.return_value = False

        with self.assertRaisesRegex(RuntimeError, "did not create"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.get_collection.assert_not_called()

        qdrant.create_collection.return_value = True
        for metadata in (None, {}, {INDEX_SETTINGS_METADATA_KEY: {"wrong": "settings"}}):
            qdrant.get_collection.return_value.config.metadata = metadata
            with self.subTest(metadata=metadata), self.assertRaisesRegex(
                RuntimeError, "metadata could not be verified"
            ):
                create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.delete_collection.assert_not_called()

    def test_settings_verification_errors_propagate_without_deleting_collection(self) -> None:
        qdrant = MagicMock()
        qdrant.create_collection.return_value = True
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            create_collection_with_settings(qdrant, "documents", 3, self.settings)
        qdrant.delete_collection.assert_not_called()

    def test_creates_collection_once(self) -> None:
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)

        self.assertTrue(ensure_collection(qdrant, "documents", 3))
        vectors = qdrant.get_collection("documents").config.params.vectors
        self.assertEqual(vectors.size, 3)
        self.assertEqual(vectors.distance, Distance.COSINE)
        self.assertFalse(ensure_collection(qdrant, "documents", 3))

    def test_existing_collection_is_not_recreated(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.return_value.config.params.vectors = VectorParams(
            size=3, distance=Distance.COSINE
        )

        self.assertFalse(ensure_collection(qdrant, "documents", 3))
        qdrant.get_collection.assert_called_once_with("documents")
        qdrant.create_collection.assert_not_called()

    def test_existing_collection_with_incompatible_configuration_is_preserved(self) -> None:
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
        qdrant = QdrantClient(":memory:")
        self.addCleanup(qdrant.close)
        qdrant.create_collection(
            "documents", vectors_config={"text": VectorParams(size=3, distance=Distance.COSINE)}
        )

        with self.assertRaisesRegex(ValueError, "single unnamed dense vector"):
            ensure_collection(qdrant, "documents", 3)
        self.assertTrue(qdrant.collection_exists("documents"))

    def test_rejects_multivector_configuration(self) -> None:
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
        qdrant = MagicMock()
        for dimension in (0, -1):
            with self.subTest(dimension=dimension), self.assertRaisesRegex(ValueError, "dimension"):
                ensure_collection(qdrant, "documents", dimension)
        qdrant.collection_exists.assert_not_called()
        qdrant.create_collection.assert_not_called()

    def test_existing_collection_lookup_errors_propagate(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection(qdrant, "documents", 3)
        qdrant.create_collection.assert_not_called()

    def test_creation_errors_propagate(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = False
        qdrant.create_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection(qdrant, "documents", 3)


if __name__ == "__main__":
    unittest.main()

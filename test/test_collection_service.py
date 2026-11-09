"""Tests for Qdrant collection setup."""

import unittest
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, MultiVectorComparator, MultiVectorConfig, PointStruct, VectorParams

from src.storage.collection_service import ensure_collection


class CollectionServiceTests(unittest.TestCase):
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

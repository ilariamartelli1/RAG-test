import unittest
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance

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

        self.assertFalse(ensure_collection(qdrant, "documents", 3))
        qdrant.create_collection.assert_not_called()

    def test_creation_errors_propagate(self) -> None:
        qdrant = MagicMock()
        qdrant.collection_exists.return_value = False
        qdrant.create_collection.side_effect = RuntimeError("unavailable")

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ensure_collection(qdrant, "documents", 3)


if __name__ == "__main__":
    unittest.main()

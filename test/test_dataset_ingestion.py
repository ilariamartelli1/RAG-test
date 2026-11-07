import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

from src.processing.dataset_ingestion import ingest_dataset


class DatasetIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qdrant = MagicMock()
        self.embedder = MagicMock()

    @patch("src.processing.dataset_ingestion.store_chunks", side_effect=[2, 1])
    @patch("builtins.print")
    def test_ingests_rows_in_batches(self, print_mock, store) -> None:
        dataframe = pd.DataFrame(
            {"field": ["alpha", "beta", "gamma"], "full_text": ["a", "b", "c"]},
            index=[10, 20, 30],
        )

        ingest_dataset(dataframe, "source", self.qdrant, self.embedder, "data", 2, 8)

        self.assertEqual(store.call_count, 2)
        first = store.call_args_list[0].args
        self.assertEqual(first[:3], (self.qdrant, self.embedder, "data"))
        self.assertEqual(first[4], 8)
        self.assertEqual([chunk.text for chunk in first[3]], [
            "Record: 10\nfield: alpha",
            "Record: 20\nfield: beta",
        ])
        self.assertEqual(first[3][0].metadata, {
            "source": "source", "row_id": "10", "chunk_index": 0, "field_name": None,
        })
        self.assertEqual(store.call_args_list[1].args[3][0].metadata["row_id"], "30")
        self.assertEqual(print_mock.call_args_list, [
            call("Progress: 2/3 (+2 new chunks)"),
            call("Progress: 3/3 (+1 new chunks)"),
        ])

    @patch("src.processing.dataset_ingestion.store_chunks")
    def test_empty_dataset_does_not_store(self, store) -> None:
        ingest_dataset(pd.DataFrame(), "source", self.qdrant, self.embedder, "data", 2, 8)
        store.assert_not_called()

    @patch("src.processing.dataset_ingestion.store_chunks")
    def test_invalid_batch_size_is_rejected(self, store) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size"):
            ingest_dataset(pd.DataFrame(), "source", self.qdrant, self.embedder, "data", 0, 8)
        store.assert_not_called()

    @patch("src.processing.dataset_ingestion.store_chunks", side_effect=RuntimeError("unavailable"))
    def test_storage_errors_propagate(self, store) -> None:
        dataframe = pd.DataFrame({"field": ["alpha"], "full_text": ["a"]})
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            ingest_dataset(dataframe, "source", self.qdrant, self.embedder, "data", 2, 8)


if __name__ == "__main__":
    unittest.main()

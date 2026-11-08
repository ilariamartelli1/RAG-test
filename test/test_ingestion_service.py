import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.chunk_models import ChunkPayload
from src.loaders.dataset_loader import dataframe_to_records
from src.processing.dataset_utils import chunk_dataset_record
from src.processing.ingestion_service import ingest_records
from src.processing.record_models import SourceRecord


class IngestionServiceTests(unittest.TestCase):
    def test_ingests_dataset_records_in_chunk_batches(self) -> None:
        dataframe = pd.DataFrame({"title": ["First", "Second", "Third"]})
        loader = MagicMock(side_effect=lambda: dataframe_to_records(dataframe, "dataset"))
        store = MagicMock(side_effect=[1, 1])

        stored = ingest_records(loader, chunk_dataset_record, store, chunk_batch_size=2)

        self.assertEqual(stored, 2)
        loader.assert_called_once_with()
        self.assertEqual(store.call_count, 2)
        batches = [call.args[0] for call in store.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertEqual(
            [chunk.text for batch in batches for chunk in batch],
            ["Record: 0\ntitle: First", "Record: 1\ntitle: Second", "Record: 2\ntitle: Third"],
        )
        self.assertEqual(batches[0][0].metadata, {
            "source": "dataset", "row_id": "0", "chunk_index": 0, "field_name": None,
        })

    def test_accepts_other_record_and_chunker_types(self) -> None:
        record = SourceRecord(content="Page text", metadata={"source": "file.pdf", "page": 1})
        loader = MagicMock(return_value=iter([record]))
        chunker = MagicMock(return_value=[ChunkPayload(text="Page text", metadata=record.metadata)])
        store = MagicMock(return_value=1)

        self.assertEqual(ingest_records(loader, chunker, store, 2), 1)
        chunker.assert_called_once_with(record)
        self.assertEqual(store.call_args.args[0][0].metadata["page"], 1)

    def test_empty_loader_or_chunker_does_not_store(self) -> None:
        store = MagicMock()
        chunker = MagicMock(return_value=[])

        self.assertEqual(ingest_records(lambda: [], chunker, store, 2), 0)
        record = SourceRecord(content="text", metadata={})
        self.assertEqual(ingest_records(lambda: [record], chunker, store, 2), 0)
        store.assert_not_called()

    def test_invalid_batch_size_does_not_call_loader(self) -> None:
        loader = MagicMock()
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "chunk_batch_size"):
                ingest_records(loader, MagicMock(), MagicMock(), size)
        loader.assert_not_called()

    def test_splits_multiple_chunks_from_one_record_into_batches(self) -> None:
        record = SourceRecord(content="text", metadata={})
        chunks = [ChunkPayload(text=str(index), metadata={}) for index in range(5)]
        store = MagicMock(side_effect=[2, 2, 1])

        self.assertEqual(ingest_records(lambda: [record], lambda _: chunks, store, 2), 5)
        self.assertEqual([len(call.args[0]) for call in store.call_args_list], [2, 2, 1])

    def test_loader_and_chunker_errors_propagate(self) -> None:
        record = SourceRecord(content="text", metadata={})
        loader = MagicMock(side_effect=RuntimeError("load failed"))
        chunker = MagicMock(side_effect=RuntimeError("chunk failed"))
        store = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            ingest_records(loader, chunker, store, 1)
        with self.assertRaisesRegex(RuntimeError, "chunk failed"):
            ingest_records(lambda: [record], chunker, store, 1)
        store.assert_not_called()

    def test_storage_errors_propagate(self) -> None:
        record = SourceRecord(content="text", metadata={})
        chunk = ChunkPayload(text="text", metadata={})
        store = MagicMock(side_effect=RuntimeError("storage unavailable"))

        with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            ingest_records(lambda: [record], lambda _: [chunk], store, 1)


if __name__ == "__main__":
    unittest.main()

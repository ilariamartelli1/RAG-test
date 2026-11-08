import unittest
from unittest.mock import MagicMock, patch

from src.ingestion.document_ingestion import ingest_document
from src.models.record_models import SourceRecord


class DocumentIngestionTests(unittest.TestCase):
    def test_ingests_text_and_markdown_in_chunk_batches(self) -> None:
        for filename in ("notes.txt", "notes.md"):
            with self.subTest(filename=filename):
                store = MagicMock(side_effect=[1, 2])

                stored = ingest_document(
                    b"one two three", filename, store, chunk_batch_size=2, max_chunk_chars=4
                )

                self.assertEqual(stored, 3)
                batches = [call.args[0] for call in store.call_args_list]
                self.assertEqual([len(batch) for batch in batches], [2, 2])
                chunks = [chunk for batch in batches for chunk in batch]
                self.assertEqual([chunk.text for chunk in chunks], ["one ", "two ", "thre", "e"])
                self.assertEqual([chunk.metadata for chunk in chunks], [
                    {"source": filename, "chunk_index": index} for index in range(4)
                ])

    @patch("src.loaders.document_loader.load_pdf_records")
    def test_ingests_pdf_pages_with_page_metadata(self, load_pdf) -> None:
        load_pdf.return_value = [
            SourceRecord(content="first", metadata={"source": "guide.pdf", "page": 1}),
            SourceRecord(content="second", metadata={"source": "guide.pdf", "page": 2}),
        ]
        store = MagicMock(return_value=2)

        stored = ingest_document(b"pdf data", "guide.pdf", store, chunk_batch_size=2)

        self.assertEqual(stored, 2)
        load_pdf.assert_called_once_with(b"pdf data", "guide.pdf")
        chunks = store.call_args.args[0]
        self.assertEqual([chunk.text for chunk in chunks], ["first", "second"])
        self.assertEqual([chunk.metadata for chunk in chunks], [
            {"source": "guide.pdf", "page": 1, "chunk_index": 0},
            {"source": "guide.pdf", "page": 2, "chunk_index": 0},
        ])

    def test_empty_document_does_not_store(self) -> None:
        store = MagicMock()

        self.assertEqual(ingest_document(b"", "empty.txt", store, 2), 0)
        store.assert_not_called()

    @patch("src.ingestion.document_ingestion.load_document_records")
    def test_rejects_invalid_sizes_before_loading(self, load_document) -> None:
        for batch_size, chunk_size in ((0, 10), (-1, 10), (2, 0), (2, -1)):
            with self.subTest(batch_size=batch_size, chunk_size=chunk_size):
                with self.assertRaises(ValueError):
                    ingest_document(b"", "notes.txt", MagicMock(), batch_size, chunk_size)
        load_document.assert_not_called()

    def test_loader_errors_propagate(self) -> None:
        store = MagicMock()
        with self.assertRaisesRegex(ValueError, "extension"):
            ingest_document(b"content", "notes.docx", store, 2)
        with self.assertRaises(UnicodeDecodeError):
            ingest_document(b"\xff", "notes.txt", store, 2)
        store.assert_not_called()

    def test_storage_errors_propagate_after_earlier_batches(self) -> None:
        store = MagicMock(side_effect=[1, RuntimeError("storage unavailable")])

        with self.assertRaisesRegex(RuntimeError, "storage unavailable"):
            ingest_document(b"abcdef", "notes.txt", store, 1, 3)

        self.assertEqual(store.call_count, 2)
        self.assertEqual(store.call_args_list[0].args[0][0].text, "abc")
        self.assertEqual(store.call_args_list[1].args[0][0].text, "def")


if __name__ == "__main__":
    unittest.main()

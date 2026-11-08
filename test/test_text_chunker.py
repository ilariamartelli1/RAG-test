import unittest
from functools import partial
from unittest.mock import MagicMock

from src.processing.ingestion_service import ingest_records
from src.processing.record_models import SourceRecord
from src.processing.text_chunker import chunk_text_record
from src.processing.text_loader import load_text_records


class TextChunkerTests(unittest.TestCase):
    def test_keeps_short_text_and_preserves_metadata(self) -> None:
        record = SourceRecord(
            content="Hello world", metadata={"source": "notes.txt", "page": 1}
        )

        chunks = chunk_text_record(record, max_chunk_chars=11)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello world")
        self.assertEqual(chunks[0].metadata, {
            "source": "notes.txt", "page": 1, "chunk_index": 0,
        })
        self.assertEqual(record.metadata, {"source": "notes.txt", "page": 1})

    def test_prefers_whitespace_and_preserves_all_text(self) -> None:
        text = "alpha beta\ngamma delta!"
        record = SourceRecord(content=text, metadata={"source": "notes.txt"})

        chunks = chunk_text_record(record, max_chunk_chars=11)

        self.assertEqual([chunk.text for chunk in chunks], ["alpha beta\n", "gamma ", "delta!"])
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertTrue(all(len(chunk.text) <= 11 for chunk in chunks))
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 1, 2])
        self.assertTrue(all(chunk.metadata["source"] == "notes.txt" for chunk in chunks))

    def test_splits_long_unbroken_text(self) -> None:
        record = SourceRecord(content="abcdefghij", metadata={})

        chunks = chunk_text_record(record, max_chunk_chars=3)

        self.assertEqual([chunk.text for chunk in chunks], ["abc", "def", "ghi", "j"])

    def test_handles_empty_and_whitespace_text(self) -> None:
        self.assertEqual(chunk_text_record(SourceRecord(content="", metadata={})), [])
        chunks = chunk_text_record(SourceRecord(content=" \n ", metadata={}), 2)
        self.assertEqual("".join(chunk.text for chunk in chunks), " \n ")
        self.assertTrue(all(0 < len(chunk.text) <= 2 for chunk in chunks))

    def test_rejects_invalid_size_content_and_reserved_metadata(self) -> None:
        record = SourceRecord(content="", metadata={})
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "max_chunk_chars"):
                chunk_text_record(record, size)
        with self.assertRaisesRegex(ValueError, "content"):
            chunk_text_record(SourceRecord(content={"field": "value"}, metadata={}))
        with self.assertRaisesRegex(ValueError, "chunk_index"):
            chunk_text_record(SourceRecord(content="text", metadata={"chunk_index": 5}))

    def test_ingests_loaded_text_in_chunk_batches(self) -> None:
        store = MagicMock(side_effect=[2, 1])

        stored = ingest_records(
            loader=partial(load_text_records, b"one two three", "notes.txt"),
            chunker=partial(chunk_text_record, max_chunk_chars=4),
            store=store,
            chunk_batch_size=2,
        )

        self.assertEqual(stored, 3)
        batches = [call.args[0] for call in store.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [2, 2])
        self.assertEqual(
            [chunk.text for batch in batches for chunk in batch],
            ["one ", "two ", "thre", "e"],
        )
        self.assertEqual(
            [chunk.metadata["chunk_index"] for batch in batches for chunk in batch],
            [0, 1, 2, 3],
        )

    def test_loads_and_chunks_markdown_without_changing_markup(self) -> None:
        markdown = (
            "# Guide\n\n"
            "Read **carefully** and see [details](https://example.com).\n\n"
            "- First item\n- Second item\n"
            "```python\nprint('hello')\n```\n"
        )

        records = load_text_records(markdown.encode("utf-8"), "guide.md")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, markdown)
        self.assertEqual(records[0].metadata, {"source": "guide.md"})

        chunks = chunk_text_record(records[0], max_chunk_chars=24)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk.text) <= 24 for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks), markdown)
        self.assertEqual(
            [chunk.metadata for chunk in chunks],
            [{"source": "guide.md", "chunk_index": index} for index in range(len(chunks))],
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for PDF text extraction and ingestion."""

import unittest
from contextlib import closing
from functools import partial
from io import BytesIO
from unittest.mock import MagicMock

from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.ingestion.document_ingestion import ingest_document
from src.ingestion.ingestion_service import ingest_records
from src.loaders.pdf_loader import load_pdf_records
from src.chunkers.text_chunker import chunk_text_record
from src.storage.chunk_service import retrieve_chunks, store_chunks


def make_pdf(pages: list[str], password: str | None = None) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for text in pages:
        page = writer.add_blank_page(width=200, height=200)
        if text:
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class PdfLoaderTests(unittest.TestCase):
    def test_extracts_pages_with_source_and_page_numbers(self) -> None:
        records = load_pdf_records(make_pdf(["First page", "", "Third page"]), "guide.pdf")

        self.assertEqual([record.content for record in records], ["First page", "", "Third page"])
        self.assertEqual([record.metadata for record in records], [
            {"source": "guide.pdf", "page": 1},
            {"source": "guide.pdf", "page": 2},
            {"source": "guide.pdf", "page": 3},
        ])

    def test_loads_and_chunks_pdf_pages_for_ingestion(self) -> None:
        store = MagicMock(return_value=2)

        stored = ingest_records(
            loader=partial(load_pdf_records, make_pdf(["First page", "", "Third page"]), "guide.pdf"),
            chunker=chunk_text_record,
            store=store,
            chunk_batch_size=2,
        )

        self.assertEqual(stored, 2)
        chunks = store.call_args.args[0]
        self.assertEqual([chunk.text for chunk in chunks], ["First page", "Third page"])
        self.assertEqual([chunk.metadata for chunk in chunks], [
            {"source": "guide.pdf", "page": 1, "chunk_index": 0},
            {"source": "guide.pdf", "page": 3, "chunk_index": 0},
        ])

    def test_rejects_empty_source(self) -> None:
        for source in ("", " \t"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "source"):
                load_pdf_records(b"", source)

    def test_rejects_documents_without_extractable_text(self) -> None:
        for pages in ([], ["", ""]):
            with self.subTest(pages=pages), self.assertRaisesRegex(ValueError, "no extractable text"):
                load_pdf_records(make_pdf(pages), "empty.pdf")

    def test_rejects_encrypted_pdf(self) -> None:
        with self.assertRaisesRegex(ValueError, "Encrypted PDF"):
            load_pdf_records(make_pdf(["Secret"], password="password"), "encrypted.pdf")

    def test_propagates_invalid_pdf_error(self) -> None:
        with self.assertRaises(PdfReadError):
            load_pdf_records(b"not a pdf", "invalid.pdf")


class DocumentIngestionIntegrationTests(unittest.TestCase):
    def test_ingests_and_retrieves_document_chunks(self) -> None:
        cases = (
            ("notes.txt", b"First text\nSecond text", [
                ("First text\n", {"source": "notes.txt", "chunk_index": 0}),
                ("Second text", {"source": "notes.txt", "chunk_index": 1}),
            ]),
            ("guide.md", b"# Heading\n\n**Details**", [
                ("# Heading\n\n", {"source": "guide.md", "chunk_index": 0}),
                ("**Details**", {"source": "guide.md", "chunk_index": 1}),
            ]),
            ("guide.pdf", make_pdf(["First page", "", "Third page"]), [
                ("First page", {"source": "guide.pdf", "page": 1, "chunk_index": 0}),
                ("Third page", {"source": "guide.pdf", "page": 3, "chunk_index": 0}),
            ]),
        )

        for filename, data, expected in cases:
            with self.subTest(filename=filename), closing(QdrantClient(":memory:")) as qdrant:
                qdrant.create_collection(
                    collection_name="documents",
                    vectors_config=VectorParams(size=2, distance=Distance.COSINE),
                )
                embedder = MagicMock()
                embedder.embed_documents.return_value = [[1.0, 0.0]]
                store = partial(store_chunks, qdrant, embedder, "documents", embedding_batch_size=8)

                stored = ingest_document(data, filename, store, chunk_batch_size=1, max_chunk_chars=12)

                self.assertEqual(stored, len(expected))
                self.assertEqual(qdrant.count("documents").count, len(expected))
                self.assertEqual(
                    ingest_document(data, filename, store, chunk_batch_size=1, max_chunk_chars=12), 0
                )
                self.assertEqual(embedder.embed_documents.call_count, len(expected))

                embedder.embed_query.return_value = [1.0, 0.0]
                results = retrieve_chunks(qdrant, embedder, "documents", "question", 8)

                actual = [(chunk.text, chunk.metadata) for chunk, _ in results]
                self.assertCountEqual(actual, expected)
                for _, score in results:
                    self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

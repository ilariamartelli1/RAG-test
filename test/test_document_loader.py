import unittest
from unittest.mock import patch

from src.loaders.document_loader import load_document_records
from src.models.record_models import SourceRecord


class DocumentLoaderTests(unittest.TestCase):
    @patch("src.loaders.document_loader.load_text_records")
    def test_selects_text_loader_for_txt_and_markdown(self, load_text) -> None:
        expected = [SourceRecord(content="example", metadata={"source": "file"})]
        load_text.return_value = expected

        for filename in ("notes.txt", "notes.TXT", "guide.md", "guide.MD"):
            with self.subTest(filename=filename):
                result = load_document_records(b"example", filename)
                self.assertIs(result, expected)
                load_text.assert_called_once_with(b"example", filename)
                load_text.reset_mock()

    @patch("src.loaders.document_loader.load_pdf_records")
    def test_selects_pdf_loader_case_insensitively(self, load_pdf) -> None:
        expected = [SourceRecord(content="page", metadata={"source": "guide.PDF", "page": 1})]
        load_pdf.return_value = expected

        result = load_document_records(b"pdf data", "guide.PDF")

        self.assertIs(result, expected)
        load_pdf.assert_called_once_with(b"pdf data", "guide.PDF")

    def test_loads_markdown_with_original_filename(self) -> None:
        records = load_document_records(b"# Heading\nText\n", "guide.md")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "# Heading\nText\n")
        self.assertEqual(records[0].metadata, {"source": "guide.md"})

    def test_rejects_empty_filename_and_unsupported_extensions(self) -> None:
        for filename in ("", " \t"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "filename"):
                load_document_records(b"", filename)
        for filename in ("notes", "notes.docx", "guide.md.exe", ".pdf"):
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "extension"):
                load_document_records(b"", filename)

    def test_propagates_text_decoding_error(self) -> None:
        with self.assertRaises(UnicodeDecodeError):
            load_document_records(b"\xff", "notes.txt")

    @patch("src.loaders.document_loader.load_pdf_records", side_effect=ValueError("invalid PDF"))
    def test_propagates_pdf_loader_errors(self, load_pdf) -> None:
        with self.assertRaisesRegex(ValueError, "invalid PDF"):
            load_document_records(b"invalid", "guide.pdf")
        load_pdf.assert_called_once_with(b"invalid", "guide.pdf")


if __name__ == "__main__":
    unittest.main()

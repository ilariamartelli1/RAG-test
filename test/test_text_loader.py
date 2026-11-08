import unittest

from src.loaders.text_loader import load_text_records


class TextLoaderTests(unittest.TestCase):
    def test_loads_utf8_document_with_source(self) -> None:
        records = load_text_records("Caffè\nSecond line\n".encode("utf-8"), "notes.txt")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "Caffè\nSecond line\n")
        self.assertEqual(records[0].metadata, {"source": "notes.txt"})

    def test_removes_utf8_byte_order_mark(self) -> None:
        records = load_text_records(b"\xef\xbb\xbfHello", "notes.txt")

        self.assertEqual(records[0].content, "Hello")

    def test_preserves_empty_document(self) -> None:
        records = load_text_records(b"", "empty.txt")

        self.assertEqual(records[0].content, "")
        self.assertEqual(records[0].metadata, {"source": "empty.txt"})

    def test_rejects_empty_source(self) -> None:
        for source in ("", " \t"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "source"):
                load_text_records(b"Hello", source)

    def test_rejects_invalid_utf8(self) -> None:
        with self.assertRaises(UnicodeDecodeError):
            load_text_records(b"\xff", "notes.txt")


if __name__ == "__main__":
    unittest.main()

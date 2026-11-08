import unittest
from datetime import date

from pydantic import ValidationError

from src.models.record_models import SourceRecord


class RecordModelsTests(unittest.TestCase):
    def test_document_record_contains_text_and_metadata(self) -> None:
        record = SourceRecord(
            content="Text from a page",
            metadata={"source": "document.pdf", "page": 2},
        )

        self.assertEqual(record.content, "Text from a page")
        self.assertEqual(record.metadata, {"source": "document.pdf", "page": 2})

    def test_dataset_record_preserves_field_structure_and_values(self) -> None:
        fields = {"title": "Example", "tags": ["one", "two"], "date": date(2026, 1, 1)}
        record = SourceRecord(content=fields, metadata={"source": "dataset", "row_id": "5"})

        self.assertEqual(record.content, fields)
        self.assertIsInstance(record.content["date"], date)
        self.assertEqual(record.metadata["row_id"], "5")

    def test_missing_fields_are_rejected(self) -> None:
        for data in ({"content": "text"}, {"metadata": {}}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                SourceRecord.model_validate(data)

    def test_invalid_content_and_metadata_are_rejected(self) -> None:
        for data in (
            {"content": 42, "metadata": {}},
            {"content": "text", "metadata": {"source": object()}},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                SourceRecord.model_validate(data)


if __name__ == "__main__":
    unittest.main()

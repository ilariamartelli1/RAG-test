"""Tests for document index settings."""

import unittest

from pydantic import ValidationError

from src.models.index_models import IndexSettings


class IndexModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "document_fingerprint": "a" * 64,
            "document_filename": "example.txt",
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "max_chunk_chars": 1000,
        }

    def test_settings_round_trip(self) -> None:
        settings = IndexSettings(**self.settings)

        self.assertEqual(settings.model_dump(), self.settings)
        self.assertEqual(IndexSettings.model_validate(settings.model_dump()), settings)

    def test_each_setting_affects_equality(self) -> None:
        original = IndexSettings(**self.settings)
        for field, value in (
            ("document_fingerprint", "b" * 64),
            ("document_filename", "other.txt"),
            ("embedding_provider", "sentence-transformers"),
            ("embedding_model", "other-model"),
            ("max_chunk_chars", 500),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(original, IndexSettings(**(self.settings | {field: value})))

    def test_invalid_settings_are_rejected(self) -> None:
        for field, value in (
            ("document_fingerprint", "not-a-digest"),
            ("document_fingerprint", "A" * 64),
            ("document_fingerprint", "a" * 64 + "\n"),
            ("document_filename", " "),
            ("embedding_provider", ""),
            ("embedding_model", "\t"),
            ("max_chunk_chars", 0),
            ("max_chunk_chars", -1),
            ("max_chunk_chars", True),
            ("max_chunk_chars", "1000"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                IndexSettings(**(self.settings | {field: value}))

    def test_missing_and_extra_fields_are_rejected(self) -> None:
        for field in self.settings:
            data = self.settings.copy()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                IndexSettings(**data)

        with self.assertRaises(ValidationError):
            IndexSettings(**self.settings, extra_setting="unexpected")

    def test_settings_are_immutable(self) -> None:
        settings = IndexSettings(**self.settings)

        with self.assertRaises(ValidationError):
            settings.max_chunk_chars = 500


if __name__ == "__main__":
    unittest.main()

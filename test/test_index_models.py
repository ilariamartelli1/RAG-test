"""Tests for document index settings."""

import unittest

from pydantic import ValidationError

from src.models.index_models import IndexSettings, build_index_settings, compare_index_settings


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

    def test_comparison_returns_no_fields_for_matching_settings(self) -> None:
        saved = IndexSettings(**self.settings)
        current = IndexSettings(**self.settings)

        self.assertEqual(compare_index_settings(saved, current), [])

    def test_comparison_reports_each_changed_setting(self) -> None:
        saved = IndexSettings(**self.settings)
        for field, value in (
            ("document_fingerprint", "b" * 64),
            ("document_filename", "other.txt"),
            ("embedding_provider", "sentence-transformers"),
            ("embedding_model", "other-model"),
            ("max_chunk_chars", 500),
        ):
            with self.subTest(field=field):
                current = IndexSettings(**(self.settings | {field: value}))
                self.assertEqual(compare_index_settings(saved, current), [field])

    def test_comparison_reports_multiple_changes_in_model_order(self) -> None:
        saved = IndexSettings(**self.settings)
        changed = self.settings | {
            "max_chunk_chars": 500,
            "embedding_model": "other-model",
            "document_fingerprint": "b" * 64,
        }
        current = IndexSettings(**changed)
        differences = ["document_fingerprint", "embedding_model", "max_chunk_chars"]

        self.assertEqual(compare_index_settings(saved, current), differences)
        self.assertEqual(compare_index_settings(current, saved), differences)
        self.assertEqual(saved.model_dump(), self.settings)
        self.assertEqual(current.model_dump(), changed)

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

    def test_build_settings_from_document_bytes(self) -> None:
        settings = build_index_settings(
            document_bytes=b"abc",
            document_filename="example.txt",
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            max_chunk_chars=1000,
        )

        self.assertEqual(settings.model_dump(), self.settings | {
            "document_fingerprint":
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        })

    def test_build_settings_reflects_document_changes(self) -> None:
        arguments = {
            "document_filename": "example.txt",
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "max_chunk_chars": 1000,
        }

        first = build_index_settings(document_bytes=b"first", **arguments)
        same = build_index_settings(document_bytes=b"first", **arguments)
        changed = build_index_settings(document_bytes=b"second", **arguments)

        self.assertEqual(first, same)
        self.assertNotEqual(first.document_fingerprint, changed.document_fingerprint)

    def test_build_settings_validates_inputs(self) -> None:
        arguments = {
            "document_bytes": b"example",
            "document_filename": "example.txt",
            "embedding_provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "max_chunk_chars": 1000,
        }
        for field, value in (
            ("document_filename", " "),
            ("embedding_provider", ""),
            ("embedding_model", "\t"),
            ("max_chunk_chars", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                build_index_settings(**(arguments | {field: value}))

        with self.assertRaises(TypeError):
            build_index_settings(**(arguments | {"document_bytes": "not bytes"}))


if __name__ == "__main__":
    unittest.main()

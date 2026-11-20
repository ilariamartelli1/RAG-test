"""Tests for future document and build registry records."""

import unittest

from pydantic import ValidationError

from src.models.index_models import IndexBuildManifest, IndexSettings
from src.models.registry_models import BuildRegistryRecord, DocumentRegistryRecord


class DocumentRegistryRecordTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        """Preserve the logical document identity through JSON."""
        record = DocumentRegistryRecord(document_id="document-1")

        self.assertEqual(
            DocumentRegistryRecord.model_validate_json(record.model_dump_json()), record
        )

    def test_rejects_missing_extra_and_invalid_ids(self) -> None:
        """A document record requires a nonblank string ID and no extra fields."""
        for value in ("", " \t", None, 123):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                DocumentRegistryRecord(document_id=value)
        with self.assertRaises(ValidationError):
            DocumentRegistryRecord()
        with self.assertRaises(ValidationError):
            DocumentRegistryRecord(document_id="document-1", extra_field=True)

    def test_document_record_is_immutable(self) -> None:
        """A document record's identity cannot be reassigned."""
        record = DocumentRegistryRecord(document_id="document-1")

        with self.assertRaises(ValidationError):
            record.document_id = "document-2"


class BuildRegistryRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = IndexBuildManifest(
            document_id="document-1",
            build_id="build-1",
            collection_name="index-1",
            index_settings=IndexSettings(
                document_fingerprint="a" * 64,
                document_filename="example.txt",
                embedding_provider="ollama",
                embedding_model="nomic-embed-text",
                max_chunk_chars=1000,
            ),
        )

    def test_round_trip_preserves_manifest(self) -> None:
        """Preserve the build identity and settings through JSON."""
        record = BuildRegistryRecord(manifest=self.manifest)

        self.assertEqual(
            BuildRegistryRecord.model_validate_json(record.model_dump_json()), record
        )
        self.assertEqual(record.manifest, self.manifest)

    def test_multiple_builds_can_share_a_document_id(self) -> None:
        """A logical document can have distinct build records."""
        first = BuildRegistryRecord(manifest=self.manifest)
        second = BuildRegistryRecord(manifest=IndexBuildManifest(
            **(self.manifest.model_dump() | {
                "build_id": "build-2",
                "collection_name": "index-2",
            })
        ))

        self.assertEqual(first.manifest.document_id, second.manifest.document_id)
        self.assertNotEqual(first.manifest.build_id, second.manifest.build_id)

    def test_rejects_missing_extra_and_invalid_manifest(self) -> None:
        """A build record requires a valid manifest and no extra fields."""
        with self.assertRaises(ValidationError):
            BuildRegistryRecord()
        with self.assertRaises(ValidationError):
            BuildRegistryRecord(manifest=self.manifest, extra_field=True)
        with self.assertRaises(ValidationError):
            BuildRegistryRecord(manifest=self.manifest.model_dump() | {"build_id": " "})
        with self.assertRaises(ValidationError):
            BuildRegistryRecord(manifest=self.manifest.model_dump() | {
                "index_settings": self.manifest.index_settings.model_dump()
                | {"max_chunk_chars": 0},
            })

    def test_build_record_and_manifest_are_immutable(self) -> None:
        """Build identity and settings cannot be reassigned."""
        record = BuildRegistryRecord(manifest=self.manifest)

        with self.assertRaises(ValidationError):
            record.manifest = self.manifest
        with self.assertRaises(ValidationError):
            record.manifest.build_id = "build-2"
        with self.assertRaises(ValidationError):
            record.manifest.index_settings.max_chunk_chars = 500


if __name__ == "__main__":
    unittest.main()

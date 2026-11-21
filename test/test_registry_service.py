"""Tests for the SQLite document and build registry."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.models.registry_models import DocumentRegistryRecord
from src.storage.registry_service import initialize_registry, register_document


class RegistryInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "registry.sqlite"

    def test_creates_persistent_schema_and_can_reopen_it(self) -> None:
        """Create empty registry tables and leave them intact on repeat initialization."""
        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT * FROM documents").fetchall(), [])
            self.assertEqual(connection.execute("SELECT * FROM builds").fetchall(), [])
            connection.execute("INSERT INTO documents VALUES (?)", ("document-1",))
            connection.execute(
                "INSERT INTO builds VALUES (?, ?, ?, ?)",
                ("build-1", "document-1", "collection-1", "{}"),
            )

        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertEqual(
                connection.execute("SELECT * FROM documents").fetchall(),
                [("document-1",)],
            )
            self.assertEqual(
                connection.execute("SELECT * FROM builds").fetchall(),
                [("build-1", "document-1", "collection-1", "{}")],
            )

    def test_schema_creation_rolls_back_on_error(self) -> None:
        """A failed initialization must not leave a partially created registry."""
        with patch("src.storage.registry_service._BUILDS_TABLE", "invalid SQL"):
            with self.assertRaises(sqlite3.OperationalError):
                initialize_registry(self.path)

        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("PRAGMA application_id").fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            ).fetchall(), [])
        initialize_registry(self.path)

    def test_constraints_apply_when_foreign_keys_are_enabled(self) -> None:
        """Reject duplicate identities, missing documents, and referenced deletions."""
        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("INSERT INTO documents VALUES (?)", ("document-1",))
            connection.execute(
                "INSERT INTO builds VALUES (?, ?, ?, ?)",
                ("build-1", "document-1", "collection-1", "{}"),
            )
            for statement, values in (
                ("INSERT INTO documents VALUES (?)", ("document-1",)),
                ("INSERT INTO builds VALUES (?, ?, ?, ?)",
                 ("build-1", "document-1", "collection-2", "{}")),
                ("INSERT INTO builds VALUES (?, ?, ?, ?)",
                 ("build-2", "document-1", "collection-1", "{}")),
                ("INSERT INTO builds VALUES (?, ?, ?, ?)",
                 ("build-2", "missing", "collection-2", "{}")),
                ("DELETE FROM documents WHERE document_id = ?", ("document-1",)),
            ):
                with self.subTest(statement=statement, values=values):
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(statement, values)

    def test_refuses_unrelated_database_without_modification(self) -> None:
        """Do not adopt a database already containing unrelated objects."""
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('keep')")

        with self.assertRaisesRegex(ValueError, "not a supported document registry"):
            initialize_registry(self.path)

        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT * FROM unrelated").fetchall(), [("keep",)])
            self.assertEqual(
                connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type = 'table'"
                ).fetchall(),
                [("unrelated",)],
            )

    def test_refuses_unsupported_version_and_damaged_schema(self) -> None:
        """Never silently recreate or migrate an incompatible registry."""
        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA user_version = 2")
        with self.assertRaisesRegex(ValueError, "not a supported document registry"):
            initialize_registry(self.path)

        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA user_version = 1")
            connection.execute("DROP TABLE builds")
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            initialize_registry(self.path)

        with sqlite3.connect(self.path) as connection:
            self.assertIsNone(connection.execute(
                "SELECT name FROM sqlite_schema WHERE name = 'builds'"
            ).fetchone())

    def test_rejects_temporary_and_uri_database_paths(self) -> None:
        """Require a persistent path instead of SQLite's temporary or URI forms."""
        for path in ("", ":memory:", "file::memory:?cache=shared"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                initialize_registry(path)

    def test_propagates_database_open_errors(self) -> None:
        """Report an invalid parent directory instead of silently falling back."""
        with self.assertRaises(sqlite3.OperationalError):
            initialize_registry(self.path / "missing.sqlite")


class DocumentRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "registry #1?.sqlite"
        self.record = DocumentRegistryRecord(document_id="document-1")

    def test_saves_document_to_existing_registry(self) -> None:
        """Registered IDs survive reopening without creating build records."""
        initialize_registry(self.path)

        register_document(self.path, self.record)
        register_document(self.path, DocumentRegistryRecord(document_id="document-2"))

        with sqlite3.connect(self.path) as connection:
            self.assertEqual(
                connection.execute("SELECT document_id FROM documents ORDER BY document_id").fetchall(),
                [("document-1",), ("document-2",)],
            )
            self.assertEqual(connection.execute("SELECT * FROM builds").fetchall(), [])

    def test_duplicate_id_fails_without_changing_existing_record(self) -> None:
        """A duplicate ID is an error, not a silent replacement."""
        initialize_registry(self.path)
        register_document(self.path, self.record)

        with self.assertRaises(sqlite3.IntegrityError):
            register_document(self.path, self.record)

        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT * FROM documents").fetchall(), [
                ("document-1",),
            ])

    def test_missing_registry_is_not_created(self) -> None:
        """Registration requires explicit initialization first."""
        with self.assertRaises(sqlite3.OperationalError):
            register_document(self.path, self.record)
        self.assertFalse(self.path.exists())

    def test_rejects_uninitialized_and_incompatible_databases(self) -> None:
        """Neither an empty nor an incompatible database is adopted."""
        with sqlite3.connect(self.path):
            pass
        with self.assertRaisesRegex(ValueError, "not a supported document registry"):
            register_document(self.path, self.record)

        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA user_version = 2")
        with self.assertRaisesRegex(ValueError, "not a supported document registry"):
            register_document(self.path, self.record)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT * FROM documents").fetchall(), [])

    def test_rejects_damaged_schema_without_writing(self) -> None:
        """A registry with a missing table is not accepted for registration."""
        initialize_registry(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE builds")
        with self.assertRaisesRegex(ValueError, "schema does not match"):
            register_document(self.path, self.record)
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT * FROM documents").fetchall(), [])

    def test_revalidates_document_id_before_opening_database(self) -> None:
        """Even a record built without Pydantic validation cannot save a blank ID."""
        record = DocumentRegistryRecord.model_construct(document_id=" ")
        with self.assertRaises(ValidationError):
            register_document(self.path, record)
        self.assertFalse(self.path.exists())

    def test_rejects_temporary_and_uri_database_paths(self) -> None:
        """Registration accepts filesystem paths only."""
        for path in ("", ":memory:", "file::memory:?cache=shared"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                register_document(path, self.record)


if __name__ == "__main__":
    unittest.main()

"""Tests for initializing a SQLite document and build registry."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.storage.registry_service import initialize_registry


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


if __name__ == "__main__":
    unittest.main()
